"""extract_features.py - Vast extraction for the AIMO interpretability challenge (Small track).

Phases (each resumable, per-problem atomic writes, safe to re-run):
  preflight  versions / GPU / disk, rendered prompt tails, </think> ids, optional HF batch-throughput bench
  plan       build the (model, problem) plan from the datasets and write plan.csv
  generate   P1: greedy decode (vLLM, or --backend hf), loop cut, stop reason eos|cap|loop.
             Reuses existing generations.jsonl (from --reuse-dirs) when prompt ids match a fresh rebuild
             and the trace is not a truncated-below-cap trace.
  features   P2 + P3 on one HF model load:
               P2 teacher-forced pass: per-token logprob / entropy / top-1 prob, prompt-only features
                  (last-prompt-token state at ALL layers, mean problem-token state, per-token problem logprobs
                  for Min-K%), states at each cutoff (last token + mean of last 64), optional window-averaged
                  trajectory states.
               P3 forced answering at token budgets and trace fractions, two suffix variants, using an
                  ASCENDING-cutoff cache that is deep-copied per branch (no cache.crop, so it also works for
                  hybrid / linear-attention models such as Qwen3.5).
             P2b attention-Laplacian eigenvalues (LapEigvals): per (layer, head) the causal attention matrix A is the
                  adjacency of a graph, L = D - A (D = mean incoming attention per token); L is lower triangular so its
                  eigenvalues are diag(L). We save the sorted top-k per (layer, head), for the prefix at every cutoff
                  ("without forcing") and for prefix + forced suffix ("with forcing"), computed exactly in chunks.
                  Downstream: concatenate over layers/heads -> PCA -> linear classifier.
  extras     forced-suffix-only extras -> <model>/extra/<id>.npz: per-head attention Fourier spectrum, lookback ratio,
             layer-to-layer attention KL (layers at 50-80% depth), and every layer's residual / attention-out / MLP-out
             activations at token distances --extra-budgets (1k, 2k, 3k, ...)
  toha       TOHA-style prompt-response minimum spanning forest per attention head -> <model>/toha/<id>.npz, at --toha-budgets, for
             the last --toha-window trace tokens ('win') and the forced suffix + answer ('ans')
  pearl      PEARL-style sensitivity, prompt only: original vs prose-noise at several intensities (+ semantic variants): hidden-state
             and first-token-distribution shifts -> <model>/pearl/<id>.npz
  lookback   Lookback Lens (teacher-forced): per generated token, layer and head, mean attention on the prompt / (that + mean attention on
             the generated tokens so far); saved as span means -> <model>/lookback/<id>.npz
  check      cache-branch vs fresh-forward equivalence on a few traces
  report     coverage summary + forced-answer agreement curve on finished traces
  all        generate then features for each model (subprocess per phase so vLLM memory is released)

Not implemented here (deferred on purpose): perturbation sensitivity (P4), multi-sample agreement (P5),
lookback ratios (P6). The 16k greedy traces are the "unbudgeted" data; every feature is computed as of an
explicit cutoff so it can be reproduced at the (much smaller) submission budget.

Typical Vast run:
  python extract_features.py --phase preflight --bench
  python extract_features.py --phase all --reuse-dirs saved_trajectories_v2 saved_trajectories_aggregate
  python extract_features.py --phase report
"""

import os
import sys

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import copy
import fnmatch
import gc
import hashlib
import json
import math
import re
import shutil
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# =========================================================================== Config
DEEPSEEK = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
EVAL_MODELS = [  # processing order for --models eval
    DEEPSEEK,
    "Qwen/Qwen3.5-4B",
    "Skywork/Skywork-OR1-Math-7B",
    "allenai/Olmo-3-7B-Think",
]
ALIASES = {  # public sample datasets use API-style names; only this one maps to a local checkpoint
    "qwen3-8b:low": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "qwen3-8b": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
}
# Layers used by the existing saved_trajectories_* .npz files (kept so window states stay comparable).
TRAJ_LAYERS = {
    "Skywork/Skywork-OR1-Math-7B": [10, 13, 16, 19, 22, 25],
    "Qwen/Qwen3.5-4B": [10, 14, 18, 22, 26, 30],
    "allenai/Olmo-3-7B-Think": [10, 14, 18, 22, 26, 30],
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": [12, 16, 20, 24, 28, 32],
}
SEL_LAYER_FRACS = (0.25, 0.40, 0.55, 0.70, 0.85, 1.00)  # layers whose states are saved at cutoffs
DEFAULT_BUDGETS = (512, 1024, 2048, 3072, 4096, 6144, 8192)
DEFAULT_FRACS = (0.25, 0.5, 0.75, 0.9, 1.0)

PROMPT_TEMPLATE = (
    "Solve the following mathematical problem step by step. "
    "Put your final answer within \\boxed{{}}.\n\n"
    "{problem}"
)

SUFFIXES_THINK = {
    "A": "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking "
         "directly now.\n</think>\n\n**Final Answer**\n\\boxed{",
    "B": "\n</think>\n\nThe final answer is $\\boxed{",
}
SUFFIXES_NOTHINK = {
    "A": "\n\nI need to stop here and give my final answer now.\n\n**Final Answer**\n\\boxed{",
    "B": "\n\nThe final answer is $\\boxed{",
}

ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.model", "*.tiktoken", "*.jinja"]
IGNORE_PATTERNS = ["*.bin", "*.pt", "*.pth", "*.gguf", "*.h5", "*.msgpack", "*.onnx", "original/*", "consolidated*"]
DOWNLOAD_MARGIN_GB = 1.5
MIN_FREE_OUTPUT_GB = 1.0


# =========================================================================== Generic helpers
def slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def sha(text: str, n: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def free_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1e9


def log(*a):
    print(*a, flush=True)


def half_np(t) -> np.ndarray:
    return t.float().clamp(-65504, 65504).half().cpu().numpy()


def save_npz_atomic(path: Path, data: dict):
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **data)
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


def save_json_atomic(path: Path, obj):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def load_jsonl_by_id(path: Path) -> dict:
    rows = {}
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    rows[r["id"]] = r
                except (json.JSONDecodeError, KeyError):
                    continue
    return rows


def append_jsonl(path: Path, rec: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()


def log_failure(out_dir: Path, pid: str, phase: str, exc: BaseException):
    append_jsonl(out_dir / "failures.jsonl", {
        "id": pid, "phase": phase, "error": repr(exc), "trace": traceback.format_exc()[-1500:],
    })


def extract_boxed(text: str):
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    start = j = i + len("\\boxed{")
    depth = 1
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[start:j - 1].strip() if depth == 0 else None


def find_loop_end(tokens, window: int = 32, repeats: int = 3):
    """First t such that tokens[:t] ends with `repeats` identical `window`-token blocks (same rule as old extractors)."""
    for t in range(window * repeats, len(tokens) + 1):
        recent = tokens[t - window:t]
        if all(tokens[t - (r + 1) * window:t - r * window] == recent for r in range(1, repeats)):
            return t
    return None


# =========================================================================== Prompt / tokenizer
def build_prompt(tokenizer, problem: str):
    """Returns (token_ids, rendered_text, (a, b) problem-token span or None). Single source of truth for the prompt."""
    user = PROMPT_TEMPLATE.format(problem=problem)
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True
        )
        # Template already inserts BOS/role/<think>; never add specials twice.
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    else:
        text = user
        enc = tokenizer(text, return_offsets_mapping=True)
    ids = enc["input_ids"]
    span = None
    c0 = text.rfind(problem)
    if c0 >= 0:
        c1 = c0 + len(problem)
        toks = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if s >= c0 and e <= c1 and e > s]
        if toks:
            span = (toks[0], toks[-1] + 1)
    return ids, text, span


def think_end_token_id(tokenizer):
    try:
        tid = tokenizer.convert_tokens_to_ids("</think>")
    except Exception:
        return None
    if tid is None or tid == getattr(tokenizer, "unk_token_id", None):
        return None
    if tokenizer.convert_ids_to_tokens(tid) != "</think>":
        return None
    return int(tid)


_ROUNDTRIP = "Solve x + y = 3 for the unknown, then find $n^2$."


def tokenizer_roundtrip_ok(tok) -> bool:
    ids = tok(_ROUNDTRIP, add_special_tokens=False)["input_ids"]
    return tok.decode(ids).strip() == _ROUNDTRIP


def load_tokenizer(model_id_or_path: str):
    """AutoTokenizer, with a loud self-test. In Transformers 5.x the DeepSeek-R1-0528-Qwen3-8B tokenizer is loaded as
    LlamaTokenizer and silently DROPS SPACES ("Solvethefollowingproblem..."). When the round trip fails we rebuild a
    PreTrainedTokenizerFast straight from tokenizer.json (chat template + special tokens copied over)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id_or_path)
    if not tokenizer_roundtrip_ok(tok):
        from transformers import PreTrainedTokenizerFast
        tj = Path(model_id_or_path) / "tokenizer.json"
        if not tj.is_file():
            from huggingface_hub import hf_hub_download
            tj = Path(hf_hub_download(model_id_or_path, "tokenizer.json"))
        fixed = PreTrainedTokenizerFast(
            tokenizer_file=str(tj), bos_token=tok.bos_token, eos_token=tok.eos_token,
            pad_token=tok.pad_token or tok.eos_token, unk_token=tok.unk_token,
        )
        fixed.chat_template = tok.chat_template
        if not tokenizer_roundtrip_ok(fixed):
            raise RuntimeError(f"tokenizer for {model_id_or_path} fails the space round trip even after the fix")
        log(f"  [tokenizer] {model_id_or_path}: {type(tok).__name__} dropped spaces -> using PreTrainedTokenizerFast from tokenizer.json")
        tok = fixed
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or tok.pad_token
    return tok


def find_think_end(tok, gen, tid):
    """Index of the first generated token that starts '</think>' (None if absent). Handles models where the tag is
    several tokens (Olmo-3-7B-Think) via a binary search over decoded prefixes."""
    if tid is not None:
        return gen.index(tid) if tid in gen else None
    if "</think>" not in tok.decode(gen):
        return None
    lo, hi = 0, len(gen)
    while lo < hi:
        mid = (lo + hi) // 2
        if "</think>" in tok.decode(gen[:mid]):
            hi = mid
        else:
            lo = mid + 1
    j = max(lo - 1, 0)
    partial = ("<", "</", "</t", "</th", "</thi", "</thin", "</think")
    while j > 0 and tok.decode(gen[:j]).endswith(partial):
        j -= 1
    return j


# =========================================================================== Weights
def _wanted(name: str) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in ALLOW_PATTERNS) and not any(fnmatch.fnmatch(name, p) for p in IGNORE_PATTERNS)


def ensure_weights(model_id: str, args) -> str:
    if args.local_models and (Path(args.local_models) / slug(model_id)).is_dir():
        return str(Path(args.local_models) / slug(model_id))
    from huggingface_hub import HfApi, snapshot_download
    cache_dir = Path(args.work_dir) / "hf_tmp" / slug(model_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:  # already on disk: use it without touching the network (snapshot_download alone re-checks the hub every call)
        return snapshot_download(model_id, cache_dir=str(cache_dir), allow_patterns=ALLOW_PATTERNS,
                                 ignore_patterns=IGNORE_PATTERNS, local_files_only=True)
    except Exception:
        pass
    if not any(cache_dir.rglob("*.safetensors")):
        try:
            info = HfApi().model_info(model_id, files_metadata=True)
            need = sum((s.size or 0) for s in info.siblings if _wanted(s.rfilename)) / 1e9
        except Exception as e:
            log(f"  [warn] could not query size for {model_id}: {e}")
            need = None
        have = free_gb(cache_dir)
        log(f"[{model_id}] disk {have:.1f} GB free; weights ~{need if need else '?'} GB")
        if need is not None and have < need + DOWNLOAD_MARGIN_GB:
            raise SystemExit(f"Not enough disk for {model_id}: need {need + DOWNLOAD_MARGIN_GB:.1f} GB, have {have:.1f}")
    return snapshot_download(model_id, cache_dir=str(cache_dir), allow_patterns=ALLOW_PATTERNS,
                             ignore_patterns=IGNORE_PATTERNS, max_workers=4)


DTYPE = "auto"   # set from --dtype in main(); "auto" = bf16 on Ampere+ (compute capability >= 8), fp16 on older GPUs (V100/Turing/Pascal)


def resolve_dtype(name: str = None):
    import torch
    name = name or DTYPE
    if name == "auto":
        ok = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8   # native bf16 needs sm_80+
        name = "bf16" if ok else "fp16"
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name], name


def assert_finite(t, what):
    """fp16 can overflow on some models; never write silently corrupted features."""
    import torch
    if not torch.isfinite(t).all():
        raise RuntimeError(f"non-finite values in {what} (fp16 overflow?) - rerun this model with --dtype fp32 or on a bf16-capable GPU")


def phase_prefetch(args):
    """Download (or finish downloading) the weights of every model in --models into <work-dir>/hf_tmp, one after another,
    without touching the GPU. Started in the background by the multi-model driver / launch scripts so the next model is
    already on disk when the current one finishes."""
    for m in args.models:
        t0 = time.time()
        try:
            path = ensure_weights(m, args)
            log(f"[prefetch] {m} ready in {time.time() - t0:.0f}s -> {path}")
        except BaseException as e:   # SystemExit from the disk check included: report and continue with the next model
            log(f"[prefetch] {m} FAILED: {e!r}")


def load_hf_model(path: str):
    import torch
    from transformers import AutoModelForCausalLM
    kw = dict(device_map="cuda" if torch.cuda.is_available() else None, attn_implementation="sdpa")
    try:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=resolve_dtype()[0], **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=resolve_dtype()[0], **kw)
    if torch.cuda.is_available():
        log(f"  [model] {torch.cuda.get_device_name(0)} (sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}) dtype={resolve_dtype()[1]}")
    return model.eval()


def find_decoder_layers(model):
    import torch.nn as nn
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    n = cfg.num_hidden_layers
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) == n and name.endswith("layers") and "vis" not in name:
            return name, mod, n
    raise RuntimeError("Could not locate decoder layers; print(model) and adjust find_decoder_layers().")


class Struct:
    """Decoder layers, final norm and output matrix of a causal LM (handles multimodal wrappers)."""

    def __init__(self, model):
        self.model = model
        name, self.layers, self.n_layers = find_decoder_layers(model)
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        if not hasattr(parent, "norm"):
            raise RuntimeError(f"No .norm next to decoder layers ({name}); adjust Struct.")
        self.norm = parent.norm
        self.W = model.get_output_embeddings().weight
        self.sel_layers = sorted({min(self.n_layers - 1, max(0, int(round(f * self.n_layers)) - 1)) for f in SEL_LAYER_FRACS})


# =========================================================================== Datasets -> plan
def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)


def _hf_parquet(repo: str) -> Path:
    from huggingface_hub import snapshot_download
    d = snapshot_download(repo_id=repo, repo_type="dataset")
    files = sorted(Path(d).glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet in {repo}")
    return files[0]


def build_plan(args) -> pd.DataFrame:
    rows = []

    def add(model_id, problem, source, is_robust=None, max_drop=None, **extra):
        model_id = ALIASES.get(model_id, model_id)
        if model_id not in EVAL_MODELS:
            return
        problem = str(problem).strip()
        rows.append(dict(model_id=model_id, problem=problem, base_id=sha(problem, 8), source=source,
                         is_robust=None if is_robust is None or (isinstance(is_robust, float) and math.isnan(is_robust)) else bool(is_robust),
                         max_drop=None if max_drop is None or (isinstance(max_drop, float) and math.isnan(max_drop)) else float(max_drop),
                         **extra))

    # train-main-v2: native rows for small models + gpt-oss non-robust rows as proxy non-robust problems
    if args.main_v2 != "none":
        p = Path(args.main_v2) if args.main_v2 != "hf" else _hf_parquet("aimo-interp/train-main-v2")
        df = _read_table(p)
        for r in df.itertuples(index=False):
            if r.model_id in EVAL_MODELS:
                if pd.isna(r.is_robust) and not args.include_unlabeled:
                    continue
                add(r.model_id, r.problem, "main_v2", r.is_robust, r.max_drop)
        if args.proxies:
            gpt = df[(df["model_id"] == "openai/gpt-oss-120b") & (df["is_robust"] == False)]  # noqa: E712
            for r in gpt.itertuples(index=False):
                for m in EVAL_MODELS:
                    add(m, r.problem, "gpt_proxy", False, r.max_drop)

    # decay-causing permutations (Gemini-written variants of agg problems, with gold answers)
    perms = None
    if args.perms and Path(args.perms).is_file():
        perms = pd.read_csv(args.perms).rename(columns={
            "problem id": "problem_id", "original problem": "original_problem",
            "decay_casuing_permutations": "perms", "correct answer": "gold"})
    gold_by_id = {} if perms is None else {str(r.problem_id): float(r.gold) for r in perms.itertuples(index=False)}

    # aggregate CSV (DeepSeek only): keep max-decay row per problem
    if args.agg and Path(args.agg).is_file():
        a = pd.read_csv(args.agg)
        a = a.sort_values("absolute_accuracy_decay", ascending=False).drop_duplicates("problem_id")
        for r in a.itertuples(index=False):
            add("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", r.original_problem, "agg", r.model_is_robust,
                r.absolute_accuracy_decay, base_acc=float(r.base_accuracy), perm_acc=float(r.permuted_accuracy),
                parent_id=str(r.problem_id), gold=gold_by_id.get(str(r.problem_id)))

    if perms is not None:
        for r in perms.itertuples(index=False):
            variants = json.loads(r.perms) if isinstance(r.perms, str) and r.perms.strip() else []
            for k, v in enumerate(variants):
                add("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", v, "perm", None, None,
                    parent_id=str(r.problem_id), perm_idx=k, gold=float(r.gold), decay_causing=True)

    # val-sample (only rows whose model maps to a local checkpoint survive `add`)
    if args.val_sample != "none":
        p = Path(args.val_sample) if args.val_sample != "hf" else _hf_parquet("aimo-interp/val-sample")
        v = _read_table(p).sort_values("absolute_accuracy_decay", ascending=False)
        v = v.drop_duplicates(["model_id", "original_problem"])
        for r in v.itertuples(index=False):
            add(r.model_id, r.original_problem, "val", r.model_is_robust, r.absolute_accuracy_decay)

    # extra unlabeled pool (per-model normalisation / pretraining probes); column `problem` or `original_problem`
    if args.extra_problems:
        e = _read_table(Path(args.extra_problems))
        col = "problem" if "problem" in e.columns else "original_problem"
        probs = e[col].dropna().astype(str).drop_duplicates().tolist()[: args.extra_max]
        for m in EVAL_MODELS:
            for t in probs:
                add(m, t, "unlabeled")

    plan = pd.DataFrame(rows)
    if plan.empty:
        raise SystemExit("empty plan: check dataset arguments")
    prio = {"main_v2": 0, "agg": 1, "perm": 1.5, "val": 2, "gpt_proxy": 3, "unlabeled": 4}
    plan["_p"] = plan["source"].map(prio)
    plan = plan.sort_values("_p", kind="stable")
    dup = plan.duplicated(["model_id", "base_id"], keep="first")
    if dup.any():
        log(f"[plan] dropped {int(dup.sum())} duplicate (model, problem) rows (priority main_v2 > agg > val > proxy > unlabeled)")
    plan = plan[~dup].drop(columns="_p").reset_index(drop=True)
    if args.smoke_model:  # pipeline smoke test: relabel DeepSeek rows as a small stand-in model
        plan = plan[plan.model_id == DEEPSEEK].copy()
        plan["model_id"] = args.smoke_model
    plan["id"] = [f"{slug(m)}_{sha(p)}" for m, p in zip(plan.model_id, plan.problem)]
    if args.limit:
        plan = plan.groupby("model_id", sort=False).head(args.limit).reset_index(drop=True)
    return plan


def phase_plan(args) -> pd.DataFrame:
    plan = build_plan(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plan.drop(columns=["problem"]).to_csv(out / "plan.csv", index=False)
    log("\n=== plan: rows per model x source (robust / non-robust / null) ===")
    g = plan.assign(lab=plan.is_robust.map({True: "R", False: "N"}).fillna("?")).groupby(["model_id", "source"]).lab
    for (m, s), lab in g:
        log(f"  {m:40s} {s:10s} n={len(lab):4d} R={int((lab == 'R').sum()):3d} N={int((lab == 'N').sum()):3d} ?={int((lab == '?').sum()):3d}")
    return plan


# =========================================================================== P1: generation
def reuse_records(model_id: str, tokenizer, plan_m: pd.DataFrame, args):
    """Existing generations that match a fresh prompt rebuild and were not cut below the current cap."""
    found, n_mismatch, n_trusted = {}, 0, 0
    by_hash = {sha(p): (i, p) for i, p in zip(plan_m["id"], plan_m["problem"])}
    for d in args.reuse_dirs:
        for gp in [Path(d) / slug(model_id) / "generations.jsonl"]:
            for r in load_jsonl_by_id(gp).values():
                h = sha(str(r.get("problem", "")).strip())
                if h not in by_hash or by_hash[h][0] in found:
                    continue
                pid, prob = by_hash[h]
                gen = r.get("gen_ids") or []
                if not gen:
                    continue
                cap_ok = r.get("stop_reason") != "cap" or len(gen) >= args.max_new_tokens - 1
                if not cap_ok:
                    continue
                matches = build_prompt(tokenizer, prob)[0] == r.get("prompt_ids")
                if not matches:
                    if args.strict_reuse:
                        n_mismatch += 1
                        continue
                    n_trusted += 1
                found[pid] = dict(prompt_ids=r["prompt_ids"], gen_ids=gen[: args.max_new_tokens],
                                  stop_reason=r["stop_reason"], reused_from=str(d), prompt_matches_rebuild=matches)
    if n_mismatch:
        log(f"  [reuse] {n_mismatch} traces skipped: saved prompt ids differ from a fresh rebuild (--strict-reuse) -> will regenerate")
    if n_trusted:
        log(f"  [reuse] WARNING {n_trusted} saved traces use prompt ids that differ from a fresh rebuild "
            f"(e.g. the space-dropping DeepSeek tokenizer); kept as-is, flagged prompt_matches_rebuild=false")
    return found


def _clean(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return v.item() if hasattr(v, "item") else v


def make_gen_record(row, prompt_ids, gen_ids, reason, tokenizer, reused_from=None, prompt_matches_rebuild=True):
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    extra = {k: _clean(getattr(row, k, None)) for k in ("parent_id", "perm_idx", "gold", "decay_causing", "base_acc", "perm_acc")}
    return {**extra, "prompt_matches_rebuild": bool(prompt_matches_rebuild),
        "id": row.id, "model_id": row.model_id, "base_id": row.base_id, "problem": row.problem,
        "source": row.source, "is_robust": row.is_robust, "max_drop": row.max_drop,
        "prompt_ids": prompt_ids, "gen_ids": gen_ids, "stop_reason": reason,
        "boxed": extract_boxed(text), "text": text, "reused_from": reused_from,
    }


def phase_generate(model_id: str, args):
    plan = build_plan(args)
    plan_m = plan[plan.model_id == model_id].reset_index(drop=True)
    out_dir = Path(args.out) / slug(model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_path = out_dir / "generations.jsonl"
    done = set(load_jsonl_by_id(gen_path))
    todo_df = plan_m[~plan_m["id"].isin(done)]
    log(f"\n=== [generate] {model_id}: {len(todo_df)} pending of {len(plan_m)} ===")
    if todo_df.empty:
        return
    tokenizer = load_tokenizer(model_id if not args.local_models else ensure_weights(model_id, args))

    reused = reuse_records(model_id, tokenizer, todo_df, args) if args.reuse_dirs else {}
    for row in todo_df.itertuples(index=False):
        if row.id in reused:
            r = reused[row.id]
            append_jsonl(gen_path, make_gen_record(row, r["prompt_ids"], r["gen_ids"], r["stop_reason"], tokenizer,
                                                   r["reused_from"], r["prompt_matches_rebuild"]))
    if reused:
        log(f"  reused {len(reused)} existing traces")
    todo = [r for r in todo_df.itertuples(index=False) if r.id not in reused]
    if not todo:
        return

    local = ensure_weights(model_id, args)
    prompt_ids = [build_prompt(tokenizer, r.problem)[0] for r in todo]
    if args.backend == "vllm":
        _generate_vllm(model_id, local, tokenizer, todo, prompt_ids, gen_path, args)
    else:
        _generate_hf(model_id, local, tokenizer, todo, prompt_ids, gen_path, args)


def _finish(row, ids, gen_ids, finish_reason, tokenizer, gen_path):
    loop_t = find_loop_end(gen_ids)
    if loop_t is not None:
        gen_ids, reason = gen_ids[:loop_t], "loop"
    elif finish_reason == "stop":
        reason = "eos"
    elif finish_reason == "length":
        reason = "cap"
    else:
        reason = "other"
    append_jsonl(gen_path, make_gen_record(row, ids, gen_ids, reason, tokenizer))
    return len(gen_ids), reason


def _generate_vllm(model_id, local, tokenizer, todo, prompt_ids, gen_path, args):
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind
    max_len = max(len(p) for p in prompt_ids) + args.max_new_tokens + 8
    llm = LLM(model=local, tokenizer=local, dtype={"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[resolve_dtype()[1]], max_model_len=max_len,
              gpu_memory_utilization=args.gpu_util, max_num_seqs=args.max_num_seqs, seed=0,
              enforce_eager=args.enforce_eager)
    params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, detokenize=False,
                            output_kind=RequestOutputKind.FINAL_ONLY)
    engine = llm.llm_engine
    for i, ids in enumerate(prompt_ids):
        engine.add_request(str(i), {"prompt_token_ids": ids}, params)
    t0, total, n = time.time(), 0, 0
    while engine.has_unfinished_requests():
        for out in engine.step():
            if not out.finished:
                continue
            i = int(out.request_id)
            comp = out.outputs[0]
            k, reason = _finish(todo[i], prompt_ids[i], list(comp.token_ids), comp.finish_reason, tokenizer, gen_path)
            n += 1
            total += k
            log(f"[gen {n}/{len(todo)}] {todo[i].id} tokens={k} stop={reason} ({total / max(time.time() - t0, 1e-6):.0f} tok/s agg)")


def _generate_hf(model_id, local, tokenizer, todo, prompt_ids, gen_path, args):
    """Batch-1 fallback (slow); useful for smoke tests or if vLLM does not support an architecture."""
    import torch
    model = load_hf_model(local)
    for n, (row, ids) in enumerate(zip(todo, prompt_ids), 1):
        x = torch.tensor([ids], device=model.device)
        with torch.no_grad():
            out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tokenizer.pad_token_id)
        gen = out[0, len(ids):].tolist()
        eos = {model.generation_config.eos_token_id} if isinstance(model.generation_config.eos_token_id, int) else set(model.generation_config.eos_token_id or [])
        eos.add(tokenizer.eos_token_id)
        finish = "stop" if gen and gen[-1] in eos else "length"
        k, reason = _finish(row, ids, gen, finish, tokenizer, gen_path)
        log(f"[gen-hf {n}/{len(todo)}] {row.id} tokens={k} stop={reason}")


# =========================================================================== Attention-Laplacian eigenvalues (LapEigvals)
# For layer l / head h let A be the causal (lower-triangular) attention matrix over the T tokens seen so far, treated as
# a weighted adjacency matrix, and L = D - A with D the diagonal degree matrix, d_ii = (sum_u A_ui) / (#rows that can
# attend to i) (mean incoming attention; --lap-degree sum uses the plain column sum). L is lower triangular, so its
# eigenvalues are diag(L)_i = d_ii - A_ii. We save the sorted top-k per (layer, head).
# Column sums depend on later rows, so the spectrum of the prefix of length n is NOT a slice of the full-trace
# spectrum. Both variants are computed exactly, chunked, without ever materialising T x T:
#   no forcing  : one teacher-forced pass; running column sums are snapshotted at every cutoff length n
#   with forcing: the suffix queries' column sums are added to the snapshot for that cutoff (branch forward)
# We wrap the model's own SDPA attention function so RoPE / q-k-norm / GQA / sliding windows are the model's.
DIAG = {"mode": None, "layers": frozenset(), "res": {}, "snap": {}, "chunk": 512, "ns": [], "k": 64, "S": 0,
        "degree": "mean", "calls": 0}
_SDPA_ORIG = None


def _prob_chunk(qg, k, scaling, window, qpos, kpos):
    """Attention probabilities for query rows `qpos`: p [Hkv,rep,bq,Tk] (masked entries 0), mask [bq,Tk], selfp [Hkv,rep,bq]."""
    import torch
    Hkv, rep = qg.shape[0], qg.shape[1]
    sc = torch.einsum("grqd,gkd->grqk", qg, k).float() * scaling
    mask = kpos[None, :] > qpos[:, None]
    if window:
        mask = mask | (kpos[None, :] <= qpos[:, None] - window)
    sc.masked_fill_(mask, float("-inf"))
    sc.sub_(torch.logsumexp(sc, dim=-1, keepdim=True)).exp_()
    selfp = sc.gather(-1, (qpos - kpos[0])[None, None, :, None].expand(Hkv, rep, -1, 1))[..., 0]  # kpos[0] > 0 on sliding-window caches
    return sc, mask, selfp


def _lap_eig(colsum, cnt, diag, degree):
    deg = colsum / cnt.clamp(min=1) if degree == "mean" else colsum
    return deg - diag


def _topk_pad(lam, k):
    import torch
    v = lam.topk(min(k, lam.shape[-1]), dim=-1).values
    if v.shape[-1] < k:
        v = torch.cat([v, torch.full(v.shape[:-1] + (k - v.shape[-1],), float("nan"), device=v.device)], -1)
    return v


def lap_prefix(query, key, scaling, window, chunk, ns, k, degree):
    """Teacher-forced pass. Returns (topk [n_cut,H,k], mean [n_cut,H], snaps {n: (colsum[H,n], cnt[n], diag[H,n])})."""
    import torch
    q, kk = query[0], key[0]
    H, T, D = q.shape
    Hkv = kk.shape[0]
    qg = q.reshape(Hkv, H // Hkv, T, D)
    kpos = torch.arange(T, device=q.device)
    colsum = torch.zeros(H, T, device=q.device)
    cnt = torch.zeros(T, device=q.device)
    diag = torch.zeros(H, T, device=q.device)
    cuts = sorted(n for n in set(ns) if 0 < n <= T)
    bounds = sorted({0, T, *cuts})
    snaps, tops, means = {}, [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        for s0 in range(a, b, chunk):
            e0 = min(b, s0 + chunk)
            p, mask, selfp = _prob_chunk(qg[:, :, s0:e0], kk, scaling, window, torch.arange(s0, e0, device=q.device), kpos)
            colsum += p.sum(2).reshape(H, T)
            cnt += (~mask).sum(0)
            diag[:, s0:e0] = selfp.reshape(H, e0 - s0)
            del p
        if b in cuts:
            snaps[b] = (colsum[:, :b].clone(), cnt[:b].clone(), diag[:, :b].clone())
            lam = _lap_eig(colsum[:, :b], cnt[:b], diag[:, :b], degree)
            tops.append(_topk_pad(lam, k))
            means.append(lam.mean(-1))
    return torch.stack(tops), torch.stack(means), snaps


def lap_branch(query, key, scaling, window, chunk, snap, k, degree, S, n):
    """Forced-suffix forward: S suffix queries over the cached keys (all n + S, or only the last Tk < n + S on sliding-window
    layers, whose cache drops tokens outside the window - those get no suffix attention anyway). Combines with the prefix snapshot for n."""
    import torch
    q, kk = query[0][:, -S:], key[0]
    H, _, D = q.shape
    Hkv, Tk = kk.shape[0], kk.shape[1]
    off = n + S - Tk  # absolute position of the first cached key
    qg = q.reshape(Hkv, H // Hkv, S, D)
    kpos = off + torch.arange(Tk, device=q.device)
    colsum = torch.zeros(H, Tk, device=q.device)
    cnt = torch.zeros(Tk, device=q.device)
    diag_s = torch.zeros(H, S, device=q.device)
    for s0 in range(0, S, chunk):
        e0 = min(S, s0 + chunk)
        p, mask, selfp = _prob_chunk(qg[:, :, s0:e0], kk, scaling, window, n + torch.arange(s0, e0, device=q.device), kpos)
        colsum += p.sum(2).reshape(H, Tk)
        cnt += (~mask).sum(0)
        diag_s[:, s0:e0] = selfp.reshape(H, e0 - s0)
        del p
    if off:
        colsum, cnt = torch.nn.functional.pad(colsum, (off, 0)), torch.nn.functional.pad(cnt, (off, 0))
    cs_p, cnt_p, diag_p = snap[n]
    lam_pre = _lap_eig(cs_p + colsum[:, :n], cnt_p + cnt[:n], diag_p, degree)
    lam_suf = _lap_eig(colsum[:, n:], cnt[n:], diag_s, degree)
    lam = torch.cat([lam_pre, lam_suf], -1)
    return _topk_pad(lam, k), lam.mean(-1), lam_suf


def _diag_sdpa(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kw):
    out = _SDPA_ORIG(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kw)
    if DIAG["mode"]:
        li = getattr(module, "layer_idx", None)
        if li in DIAG["layers"]:
            sc = scaling if scaling is not None else query.shape[-1] ** -0.5
            win = getattr(module, "sliding_window", None)
            if DIAG["mode"] == "prefix":
                top, mean, snaps = lap_prefix(query, key, sc, win, DIAG["chunk"], DIAG["ns"], DIAG["k"], DIAG["degree"])
                DIAG["res"][li] = (top, mean)
                DIAG["snap"][li] = snaps
            elif DIAG["mode"] == "extra":
                _extra_layer(li, query, key, sc, win)
            elif DIAG["mode"] == "toha":
                _toha_layer(li, query, key, sc, win)
            elif DIAG["mode"] == "lookback":
                _lookback_layer(li, query, key, sc, win)
            else:
                DIAG["res"][li] = lap_branch(query, key, sc, win, DIAG["chunk"], DIAG["snap"][li], DIAG["k"],
                                             DIAG["degree"], DIAG["S"], DIAG["n"])
            DIAG["calls"] += 1
    return out


def install_diag_hook():
    global _SDPA_ORIG
    if _SDPA_ORIG is None:
        from transformers import AttentionInterface
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        _SDPA_ORIG = sdpa_attention_forward
        AttentionInterface.register("sdpa", _diag_sdpa)


class lap_capture:
    """mode 'prefix' (teacher-forced pass, needs ns) or 'branch' (forced-suffix forward, needs S); snapshots persist
    from a prefix pass until the next prefix pass."""

    def __init__(self, mode, layers, chunk=512, ns=None, k=64, S=0, degree="mean", n=0):
        self.mode, self.layers = mode, list(layers or [])
        self.kw = dict(chunk=chunk, ns=sorted(ns or []), k=k, S=S, degree=degree, n=n)

    def __enter__(self):
        DIAG.update(mode=self.mode if self.layers else None, layers=frozenset(self.layers), res={}, **self.kw)
        if self.mode == "prefix":
            DIAG["snap"] = {}
        return self

    def __exit__(self, *a):
        DIAG["mode"] = None

    def prefix_result(self):
        import torch
        if not DIAG["res"]:
            return None, None, None
        ls = sorted(DIAG["res"])
        return ls, torch.stack([DIAG["res"][l][0] for l in ls], 1), torch.stack([DIAG["res"][l][1] for l in ls], 1)

    def branch_result(self):
        import torch
        if not DIAG["res"]:
            return None
        ls = sorted(DIAG["res"])
        return tuple(torch.stack([DIAG["res"][l][i] for l in ls]) for i in range(3))  # topk[L,H,k], mean[L,H], suf[L,H,S]


def pick_attn_layers(st: "Struct", n: int):
    """n < 0: every layer with softmax self-attention; n > 0: ~n equally spaced ones (hybrids' linear layers skipped)."""
    cand = [i for i, layer in enumerate(st.layers) if getattr(layer, "self_attn", None) is not None]
    if n == 0 or not cand:
        return []
    if n < 0 or n >= len(cand):
        return cand
    idx = sorted({int(round(x)) for x in np.linspace(0, len(cand) - 1, n)})
    return [cand[i] for i in idx]


def attn_summary(pack, sufres, n_p, eff_cuts, variant_names, degree):
    """pack = (layer_ids, topk_noforce [n_cut,L,H,k], mean_noforce [n_cut,L,H]); sufres {'c|v': (topk, mean, suf)}."""
    import torch
    ids, top_no, mean_no = pack
    S = max(v[2].shape[-1] for v in sufres.values())
    stack = lambda i: torch.stack([torch.stack([sufres[f"{c}|{v}"][i] for v in variant_names]) for c in eff_cuts])  # noqa: E731
    L, H = top_no.shape[1], top_no.shape[2]
    suf = np.full((len(eff_cuts), len(variant_names), L, H, S), np.nan, dtype=np.float32)  # variants differ in length
    for ci, c in enumerate(eff_cuts):
        for vi, v in enumerate(variant_names):
            x = sufres[f"{c}|{v}"][2]
            suf[ci, vi, ..., :x.shape[-1]] = x.float().cpu().numpy()
    return {
        "attn_layers": np.array(ids, dtype=np.int16), "cut_eff": np.array(eff_cuts, dtype=np.int32),
        "n_prompt": np.int32(n_p), "variant_names": np.array(variant_names), "degree": np.array(degree),
        "topk_noforce": top_no.float().cpu().numpy(), "mean_noforce": mean_no.float().cpu().numpy(),      # [n_cut,L,H,k] / [n_cut,L,H]
        "topk_force": stack(0).float().cpu().numpy(), "mean_force": stack(1).float().cpu().numpy(),       # [n_cut,n_var,L,H,k] / [..,L,H]
        "suffix_eig": suf,                                                                    # [n_cut,n_var,L,H,S_max]
    }


# =========================================================================== P2: teacher-forced pass
def cutoff_plan(n_gen: int, think_end, args):
    """label -> effective cutoff (number of generated tokens kept). Finished traces are cut before </think>."""
    L = think_end if think_end is not None else n_gen
    L = max(1, L)
    labels = {}
    for b in args.budgets:
        labels[f"b{b}"] = max(1, min(b, L))
    for f in args.fracs:
        labels[f"f{int(round(f * 100))}"] = max(1, min(L, int(round(f * L))))
    return labels, L


def teacher_forced(model, st: Struct, prompt_ids, gen_ids, span, eff_cuts, traj_layers, args, attn_layers=()):
    """Returns (feature dict, (attn layer ids, topk_noforce, mean_noforce) or (None, None, None))."""
    import torch
    n_p, n_g = len(prompt_ids), len(gen_ids)
    ids = torch.tensor([prompt_ids + gen_ids], device=model.device)
    cap = {}
    sel = st.sel_layers
    cut_pos = [n_p + c - 1 for c in eff_cuts]

    def hook_all(l):  # last prompt token at every layer
        def h(_m, _i, o):
            hid = o[0] if isinstance(o, tuple) else o
            cap.setdefault("prompt_last", {})[l] = hid[0, n_p - 1].detach()
        return h

    def hook_sel(l):
        def h(_m, _i, o):
            hid = o[0] if isinstance(o, tuple) else o
            d = cap.setdefault("sel", {})
            pos = torch.tensor(cut_pos, device=hid.device)
            last = hid[0].index_select(0, pos)
            means = torch.stack([hid[0, max(n_p, p - 63):p + 1].float().mean(0) for p in cut_pos])
            pm = hid[0, span[0]:span[1]].float().mean(0) if span else torch.full((hid.shape[-1],), float("nan"), device=hid.device)
            d[l] = (last.detach(), means.detach(), pm.detach())
        return h

    def hook_traj(l):
        def h(_m, _i, o):
            hid = o[0] if isinstance(o, tuple) else o
            g = hid[0, n_p:n_p + n_g].float()
            k = n_g // args.stride
            parts = []
            if k:
                parts.append(g[: k * args.stride].view(k, args.stride, -1).mean(1))
            if n_g % args.stride:
                parts.append(g[k * args.stride:].mean(0, keepdim=True))
            cap.setdefault("traj", {})[l] = torch.cat(parts, 0).detach()
        return h

    def hook_norm(_m, _i, o):
        cap["normed"] = o[0] if isinstance(o, tuple) else o

    handles = [st.layers[l].register_forward_hook(hook_all(l)) for l in range(st.n_layers)]
    handles += [st.layers[l].register_forward_hook(hook_sel(l)) for l in sel]
    if args.save_window_states:
        handles += [st.layers[l].register_forward_hook(hook_traj(l)) for l in traj_layers]
    handles.append(st.norm.register_forward_hook(hook_norm))
    try:
        with torch.no_grad(), lap_capture("prefix", attn_layers, args.attn_chunk, ns=[n_p + c for c in eff_cuts],
                                          k=args.attn_topk, degree=args.lap_degree) as dc:
            try:
                model(input_ids=ids, use_cache=False, logits_to_keep=1)
            except TypeError:
                model(input_ids=ids, use_cache=False)
        pack = dc.prefix_result()
    finally:
        for h in handles:
            h.remove()

    H = cap["normed"][0]  # [L, d]
    assert_finite(H, "final hidden states")
    W = st.W

    def stats(start, targets):
        lps, ents, tops = [], [], []
        for s in range(0, len(targets), args.logit_chunk):
            tg = targets[s:s + args.logit_chunk]
            logits = (H[start + s:start + s + len(tg)] @ W.T).float()
            lp = torch.log_softmax(logits, -1)
            p = lp.exp()
            lps.append(lp.gather(1, torch.tensor(tg, device=lp.device)[:, None])[:, 0])
            ents.append(-(p * lp).sum(-1))
            tops.append(p.max(-1).values)
            del logits, lp, p
        cat = lambda xs: torch.cat(xs).half().cpu().numpy()  # noqa: E731
        return cat(lps), cat(ents), cat(tops)

    g_lp, g_ent, g_top = stats(n_p - 1, gen_ids)
    out = {"gen_logprob": g_lp, "gen_entropy": g_ent, "gen_top1": g_top, "cut_eff": np.array(eff_cuts, dtype=np.int32),
           "sel_layers": np.array(sel, dtype=np.int16)}
    if span and span[1] - span[0] > 1:
        p_lp, p_ent, _ = stats(span[0] - 1, prompt_ids[span[0]:span[1]])
        out["problem_logprob"], out["problem_entropy"] = p_lp, p_ent
    out["prompt_last_all_layers"] = half_np(torch.stack([cap["prompt_last"][l] for l in range(st.n_layers)]))
    out["cut_last"] = half_np(torch.stack([cap["sel"][l][0] for l in sel], 1))   # [n_cut, n_sel, d]
    out["cut_mean64"] = half_np(torch.stack([cap["sel"][l][1] for l in sel], 1))
    out["problem_mean"] = half_np(torch.stack([cap["sel"][l][2] for l in sel]))   # [n_sel, d]
    if args.save_window_states:
        out["traj_layers"] = np.array(traj_layers, dtype=np.int16)
        out["win_states"] = half_np(torch.stack([cap["traj"][l] for l in traj_layers], 1))  # [n_win, n_traj, d]
    del cap, H
    return out, pack


# =========================================================================== P3: forced answering
def _extend(model, cache, chunk_ids, chunk):
    import torch
    out = None
    for s in range(0, len(chunk_ids), chunk):
        x = torch.tensor([chunk_ids[s:s + chunk]], device=model.device)
        kw = dict(input_ids=x, use_cache=True)
        if cache is not None:
            kw["past_key_values"] = cache
        try:
            out = model(logits_to_keep=1, **kw)
        except TypeError:
            out = model(**kw)
        cache = out.past_key_values
    return cache


def _cache_to(cache, device):
    """Move every tensor held by a HF cache's layers to `device` in place (--low-mem keeps the shared cache in CPU RAM)."""
    import torch
    for layer in getattr(cache, "layers", None) or []:
        for k, v in list(vars(layer).items()):
            if isinstance(v, torch.Tensor):
                setattr(layer, k, v.to(device))
    return cache


def _decode_answer(model, tokenizer, cache, first_logits, last_hidden, args):
    """Greedy-decode until the \\boxed{ brace closes. Logits for the first 3 answer tokens are summarised."""
    import torch
    depth, text, toks, lps = 1, "", [], []
    top = []
    logits = first_logits
    eos = {tokenizer.eos_token_id}
    first_entropy = None
    closed = False
    for step in range(args.answer_tokens):
        lp = torch.log_softmax(logits.float(), -1)
        if step == 0:
            first_entropy = float(-(lp.exp() * lp).sum())
        if step < 3:
            v, i = lp.topk(20)
            top.append({"ids": i.tolist(), "probs": v.exp().tolist()})
        nxt = int(lp.argmax())
        lps.append(float(lp[nxt]))
        if nxt in eos:
            break
        piece = tokenizer.decode([nxt])
        toks.append(nxt)
        for ch in piece:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    closed = True
                    break
            if not closed:
                text += ch
        if closed:
            break
        x = torch.tensor([[nxt]], device=model.device)
        out = model(input_ids=x, past_key_values=cache, use_cache=True)
        logits = out.logits[0, -1]
    return {"answer": text.strip(), "closed": closed, "n_tokens": len(toks), "first_entropy": first_entropy,
            "mean_logprob": float(np.mean(lps)) if lps else None, "top": top, "ids": toks}


def _branch(model, tokenizer, st: Struct, cache, prefix_ids, suffix_ids, args, diag_layers=()):
    """Append `suffix_ids` to a COPY of `cache`; returns (result dict, hidden[n_sel, d] at last suffix token, last logits)."""
    import torch
    mode = "copy"
    try:
        c2 = copy.deepcopy(cache)  # --low-mem: the shared cache lives on CPU, so this copy is a CPU copy
        if getattr(args, "low_mem", False):
            _cache_to(c2, model.device)
    except Exception:
        c2, mode = None, "recompute"
    x_ids = suffix_ids if mode == "copy" else prefix_ids + suffix_ids
    kw = dict(input_ids=torch.tensor([x_ids], device=model.device), use_cache=True, output_hidden_states=True)
    if c2 is not None:
        kw["past_key_values"] = c2
    with lap_capture("branch", diag_layers, args.attn_chunk, k=args.attn_topk, S=len(suffix_ids),
                     degree=args.lap_degree, n=len(prefix_ids)) as dc:
        out = model(**kw)
    sdiag = dc.branch_result() if diag_layers else None
    c2 = out.past_key_values
    hs = out.hidden_states
    hidden = torch.stack([hs[l + 1][0, -1] for l in st.sel_layers])
    res = _decode_answer(model, tokenizer, c2, out.logits[0, -1], hidden, args)
    res["branch"] = mode
    del c2, out, hs
    return res, half_np(hidden), sdiag


def forced_answers(model, tokenizer, st: Struct, prompt_ids, gen_ids, eff_cuts, variants, args, diag_layers=()):
    import torch
    full = prompt_ids + gen_ids
    n_p = len(prompt_ids)
    cache, pos = None, 0
    low = getattr(args, "low_mem", False)
    results, hiddens, sufdiags = {}, {}, {}
    for c in sorted(set(eff_cuts)):
        target = n_p + c
        if low and cache is not None:
            _cache_to(cache, model.device)
        cache = _extend(model, cache, full[pos:target], args.prefill_chunk)
        pos = target
        if low:
            _cache_to(cache, "cpu")
            torch.cuda.empty_cache()
        for vname, sids in variants.items():
            res, hid, sd = _branch(model, tokenizer, st, cache, full[:target], sids, args, diag_layers)
            results[f"{c}|{vname}"] = res
            hiddens[f"{c}|{vname}"] = hid
            if sd is not None:
                sufdiags[f"{c}|{vname}"] = sd
        torch.cuda.empty_cache()
    return results, hiddens, sufdiags


def phase_features(model_id: str, args):
    import torch
    torch.set_grad_enabled(False)
    out_dir = Path(args.out) / slug(model_id)
    rows = load_jsonl_by_id(out_dir / "generations.jsonl")
    (out_dir / "tf").mkdir(exist_ok=True)
    (out_dir / "force").mkdir(exist_ok=True)
    (out_dir / "attn").mkdir(exist_ok=True)

    def needs(i):
        return (not args.no_tf and not (out_dir / "tf" / f"{i}.npz").exists(),
                not args.no_force and not (out_dir / "force" / f"{i}.json").exists(),
                args.attn_layers != 0 and not args.no_attn and not (out_dir / "attn" / f"{i}.npz").exists())

    def pending(r):
        return any(needs(r["id"]))

    todo = [r for r in rows.values() if r["gen_ids"] and pending(r)]
    if args.ids:
        todo = [r for r in todo if r["id"] in set(args.ids)]
    log(f"\n=== [features] {model_id}: {len(todo)} pending of {len(rows)} ===")
    if not todo:
        return
    local = ensure_weights(model_id, args)
    tokenizer = load_tokenizer(local)
    model = load_hf_model(local)
    st = Struct(model)
    install_diag_hook()
    attn_layers = pick_attn_layers(st, args.attn_layers)
    tid = think_end_token_id(tokenizer)
    has_think = tid is not None or "<think>" in build_prompt(tokenizer, "x")[1]
    variants_txt = SUFFIXES_THINK if has_think else SUFFIXES_NOTHINK
    variants = {k: tokenizer(v, add_special_tokens=False)["input_ids"] for k, v in variants_txt.items()}
    traj_layers = [l for l in TRAJ_LAYERS.get(model_id, []) if l < st.n_layers] or list(st.sel_layers)
    log(f"  attention-eigenvalue layers={attn_layers}")
    log(f"  layers={st.n_layers} sel={st.sel_layers} </think> id={tid} suffix_set={'think' if has_think else 'no-think'}")

    t0 = time.time()
    for k, r in enumerate(todo, 1):
        if free_gb(out_dir) < MIN_FREE_OUTPUT_GB:
            log("Free disk below limit - stopping. Move results off and rerun.")
            break
        pid = r["id"]
        try:
            gen = r["gen_ids"]
            think_end = find_think_end(tokenizer, gen, tid)
            labels, L = cutoff_plan(len(gen), think_end, args)
            eff = sorted(set(labels.values()))
            rebuilt, _, span = build_prompt(tokenizer, r["problem"])
            prompt_ids = r["prompt_ids"]  # always the ids the trace was actually generated from
            if rebuilt != prompt_ids:
                span = None  # saved ids differ from a fresh rebuild (e.g. old DeepSeek tokenizer): no problem-token span
            need_tf, need_force, need_attn = needs(pid)
            meta = {"prompt_matches_rebuild": rebuilt == prompt_ids,"labels": labels, "L": L, "think_end": think_end, "n_gen": len(gen), "stop_reason": r["stop_reason"],
                    "variants": variants_txt, "sel_layers": st.sel_layers, "n_layers": st.n_layers, "span": span,
                    "final_boxed": extract_boxed(tokenizer.decode(gen[think_end:], skip_special_tokens=True)) if think_end is not None else None}
            pack = (None, None, None)
            if need_tf or need_attn:
                feats, pack = teacher_forced(model, st, prompt_ids, gen, span, eff, traj_layers, args,
                                             attn_layers if need_attn else ())
                if need_tf:
                    save_npz_atomic(out_dir / "tf" / f"{pid}.npz", feats)
                del feats
            if need_force or need_attn:
                res, hid, sufd = forced_answers(model, tokenizer, st, prompt_ids, gen, eff, variants, args,
                                                attn_layers if need_attn else ())
                if need_force:
                    save_npz_atomic(out_dir / "force" / f"{pid}.npz", {kk.replace("|", "_"): v for kk, v in hid.items()})
                    save_json_atomic(out_dir / "force" / f"{pid}.json", {**meta, "results": res})
                if need_attn and pack[0] is not None and sufd:
                    save_npz_atomic(out_dir / "attn" / f"{pid}.npz",
                                    attn_summary(pack, sufd, len(prompt_ids), eff, sorted(variants), args.lap_degree))
                del sufd
            pack = None
            DIAG["snap"] = {}
        except Exception as e:  # keep going; failures are logged, never turned into silent labels
            log(f"  [fail] {pid}: {e!r}")
            log_failure(out_dir, pid, "features", e)
            torch.cuda.empty_cache()
        if k % 5 == 0 or k == len(todo):
            log(f"[features {k}/{len(todo)}] {pid} ({(time.time() - t0) / k:.1f}s/problem)")
    del model, st
    gc.collect()
    torch.cuda.empty_cache()


# =========================================================================== extras (forced-suffix only): Fourier / lookback / KL / activations
# Everything below is computed in the forced-answer branch passes ONLY (the S suffix queries attend over the whole cached
# prefix), so no extra teacher-forced pass and no T x T matrix is needed. Per (cutoff, suffix variant):
#   fft            |rfft| of each suffix query's attention row over key positions (zero-padded to n+S), averaged over the S
#                  queries, then averaged inside nb log-spaced frequency bins (cycles/token, DC dropped) -> [layer, head, nb]
#   lookback       Chuang et al. lookback ratio per head: a_ctx / (a_ctx + a_new), a_ctx = mean attention weight on the prompt
#                  tokens, a_new = mean attention weight on all other (generated + forced) tokens; averaged over suffix queries
#   prompt_mass    summed attention mass on the prompt, averaged over suffix queries (lookback_last: last suffix query only)
#   kl_adj         KL(P_{l-1,h} || P_{l,h}) between a head's attention rows in consecutive attention layers inside --kl-range
#   kl_headmean    KL(P_{l,h} || mean_h' P_{l,h'}): how far each head is from its layer's average attention, same layers
#   acts           residual stream (layer output), token-mixer output and MLP output of every layer, at the last suffix
#                  token (res_last / attn_last / mlp_last) and the residual averaged over the suffix (res_mean)
def _extra_layer(li, query, key, sc, win):
    import torch
    import torch.nn.functional as F
    q, kk = query[0][:, -DIAG["S"]:], key[0]
    H, S, D = q.shape
    Hkv, Tk = kk.shape[0], kk.shape[1]
    n, n_p = DIAG["n"], DIAG["n_p"]
    Tfull = n + S
    off = Tfull - Tk
    qg = q.reshape(Hkv, H // Hkv, S, D)
    kpos = off + torch.arange(Tk, device=q.device)
    nb, edges = DIAG["nb"], DIAG["edges"]
    spec = None
    rows = []
    for s0 in range(0, S, DIAG["chunk"]):
        e0 = min(S, s0 + DIAG["chunk"])
        p, _, _ = _prob_chunk(qg[:, :, s0:e0], kk, sc, win, n + torch.arange(s0, e0, device=q.device), kpos)
        r = p.reshape(H, e0 - s0, Tk)
        del p
        if off:
            r = F.pad(r, (off, 0))
        sp = torch.fft.rfft(r, dim=-1).abs().sum(1)
        spec = sp if spec is None else spec + sp
        rows.append(r)
    R = torch.cat(rows, 1)  # [H,S,Tfull]
    del rows
    freqs = torch.arange(spec.shape[-1], device=q.device) / Tfull
    idx = torch.bucketize(freqs, edges) - 1
    ok = (idx >= 0) & (idx < nb) & (torch.arange(spec.shape[-1], device=q.device) > 0)
    fft = torch.zeros(H, nb, device=q.device).index_add_(1, idx[ok], spec[:, ok] / S)
    cnt = torch.bincount(idx[ok], minlength=nb).float()
    fft = torch.where(cnt > 0, fft / cnt.clamp(min=1), torch.full_like(fft, float("nan")))
    mp = R[..., :n_p].sum(-1)  # [H,S]
    a_ctx = mp / max(n_p, 1)
    a_new = (1 - mp).clamp(min=0) / max(Tfull - n_p, 1)
    lookback = (a_ctx / (a_ctx + a_new).clamp(min=1e-12)).mean(-1)
    DIAG["res"][li] = (fft, lookback, mp.mean(-1), mp[:, -1])
    if li in DIAG["kl_prev"]:
        eps = 1e-10
        hm = R.mean(0, keepdim=True)
        kl_hm = (R * (R.clamp_min(eps).log() - hm.clamp_min(eps).log())).sum(-1).mean(-1)
        pl = DIAG["kl_prev"][li]
        if pl is not None and pl in DIAG["rows"]:
            Q = DIAG["rows"].pop(pl)
            kl_adj = (Q * (Q.clamp_min(eps).log() - R.clamp_min(eps).log())).sum(-1).mean(-1)
        else:
            kl_adj = torch.full((H,), float("nan"), device=q.device)
        DIAG["kl"][li] = (kl_adj, kl_hm)
        DIAG["rows"][li] = R
    del R


class extra_capture:
    def __init__(self, layers, kl_layers, chunk, n, S, n_p, nb, edges):
        self.layers, self.kl_layers = list(layers), list(kl_layers)
        self.kw = dict(chunk=chunk, n=n, S=S, n_p=n_p, nb=nb, edges=edges)

    def __enter__(self):
        DIAG.update(mode="extra", layers=frozenset(self.layers), res={}, kl={}, rows={},
                    kl_prev={l: (self.kl_layers[i - 1] if i else None) for i, l in enumerate(self.kl_layers)}, **self.kw)
        return self

    def __exit__(self, *a):
        DIAG["mode"] = None
        DIAG["rows"] = {}


def _mixer(layer):
    for nme in ("self_attn", "linear_attn", "attn", "mixer", "token_mixer"):
        if getattr(layer, nme, None) is not None:
            return getattr(layer, nme)
    return None


def _ffn(layer):
    for nme in ("mlp", "feed_forward", "ffn"):
        if getattr(layer, nme, None) is not None:
            return getattr(layer, nme)
    return None


def extras_one(model, st: Struct, prompt_ids, gen_ids, eff_cuts, variants, args, attn_layers, kl_layers, edges):
    """Returns the npz dict for one trace, or None if no attention statistics were captured."""
    import torch
    full = prompt_ids + gen_ids
    n_p = len(prompt_ids)
    low = getattr(args, "low_mem", False)
    names = sorted(variants)
    cache, pos = None, 0
    cells = {}
    for c in sorted(set(eff_cuts)):
        target = n_p + c
        if low and cache is not None:
            _cache_to(cache, model.device)
        cache = _extend(model, cache, full[pos:target], args.prefill_chunk)
        pos = target
        if low:
            _cache_to(cache, "cpu")
            torch.cuda.empty_cache()
        for v in names:
            sids = variants[v]
            S = len(sids)
            c2 = copy.deepcopy(cache)
            if low:
                _cache_to(c2, model.device)
            acts = {k: {} for k in ("res_last", "res_mean", "attn_last", "mlp_last")}

            def mk(kind, l):
                def h(_m, _i, o):
                    x = o[0] if isinstance(o, tuple) else o
                    if kind == "res":
                        acts["res_last"][l] = x[0, -1].detach()
                        acts["res_mean"][l] = x[0, -S:].float().mean(0).detach()
                    else:
                        acts[kind][l] = x[0, -1].detach()
                return h

            handles = []
            for l in range(st.n_layers):
                lay = st.layers[l]
                handles.append(lay.register_forward_hook(mk("res", l)))
                if _mixer(lay) is not None:
                    handles.append(_mixer(lay).register_forward_hook(mk("attn_last", l)))
                if _ffn(lay) is not None:
                    handles.append(_ffn(lay).register_forward_hook(mk("mlp_last", l)))
            x = torch.tensor([sids], device=model.device)
            try:
                with extra_capture(attn_layers, kl_layers, args.attn_chunk, c + n_p, S, n_p, args.fft_bins, edges) as dc:
                    try:
                        model(input_ids=x, past_key_values=c2, use_cache=True, logits_to_keep=1)
                    except TypeError:
                        model(input_ids=x, past_key_values=c2, use_cache=True)
                res, kl = dict(DIAG["res"]), dict(DIAG["kl"])
            finally:
                for h in handles:
                    h.remove()
            d = model.config.get_text_config().hidden_size if hasattr(model.config, "get_text_config") else model.config.hidden_size
            zero = torch.zeros(d, device=model.device)
            cell = {k: half_np(torch.stack([acts[k].get(l, zero.to(acts["res_last"][0].dtype)) for l in range(st.n_layers)]))
                    for k in acts}
            if res:
                ls = sorted(res)
                for i, nm in enumerate(("fft", "lookback", "prompt_mass", "lookback_last")):
                    cell[nm] = torch.stack([res[l][i] for l in ls]).float().cpu().numpy()
                cell["_ls"] = ls
            if kl:
                kls = sorted(kl)
                cell["kl_adj"] = torch.stack([kl[l][0] for l in kls]).float().cpu().numpy()
                cell["kl_headmean"] = torch.stack([kl[l][1] for l in kls]).float().cpu().numpy()
                cell["_kls"] = kls
            cells[(c, v)] = cell
            del c2, acts
            DIAG["rows"] = {}
        torch.cuda.empty_cache()
    cuts = sorted(set(eff_cuts))
    first = cells[(cuts[0], names[0])]
    if "fft" not in first:
        return None
    out = {"cut_eff": np.array(cuts, dtype=np.int32), "variant_names": np.array(names), "n_prompt": np.int32(n_p),
           "attn_layers": np.array(first["_ls"], dtype=np.int16), "fft_edges": edges.cpu().numpy()}
    if "kl_adj" in first:
        out["kl_layers"] = np.array(first["_kls"], dtype=np.int16)
    for k in first:
        if k.startswith("_"):
            continue
        out[k] = np.stack([np.stack([cells[(c, v)][k] for v in names]) for c in cuts])  # [cut,var,...]
    return out


def phase_extras(model_id: str, args):
    import torch
    torch.set_grad_enabled(False)
    out_dir = Path(args.out) / slug(model_id)
    rows = load_jsonl_by_id(out_dir / "generations.jsonl")
    (out_dir / "extra").mkdir(exist_ok=True)
    todo = [r for r in rows.values() if r["gen_ids"] and not (out_dir / "extra" / f"{r['id']}.npz").exists()]
    if args.ids:
        todo = [r for r in todo if r["id"] in set(args.ids)]
    log(f"\n=== [extras] {model_id}: {len(todo)} pending of {len(rows)} ===")
    if not todo:
        return
    local = ensure_weights(model_id, args)
    tokenizer = load_tokenizer(local)
    model = load_hf_model(local)
    st = Struct(model)
    install_diag_hook()
    attn_layers = pick_attn_layers(st, -1)
    lo, hi = args.kl_range
    kl_layers = [l for l in attn_layers if lo <= (l + 1) / st.n_layers <= hi]
    tid = think_end_token_id(tokenizer)
    has_think = tid is not None or "<think>" in build_prompt(tokenizer, "x")[1]
    variants = {k: tokenizer(v, add_special_tokens=False)["input_ids"]
                for k, v in (SUFFIXES_THINK if has_think else SUFFIXES_NOTHINK).items()}
    edges = torch.logspace(np.log10(1 / 32768), np.log10(0.5), args.fft_bins + 1, device=model.device)
    log(f"  attention layers={len(attn_layers)} kl layers={kl_layers} budgets={args.extra_budgets} fft bins={args.fft_bins}")
    t0 = time.time()
    for k, r in enumerate(todo, 1):
        if free_gb(out_dir) < MIN_FREE_OUTPUT_GB:
            log("Free disk below limit - stopping. Move results off and rerun.")
            break
        pid = r["id"]
        try:
            gen = r["gen_ids"]
            te = find_think_end(tokenizer, gen, tid)
            L = max(1, te if te is not None else len(gen))
            eff = sorted({max(1, min(b, L)) for b in args.extra_budgets})
            out = extras_one(model, st, r["prompt_ids"], gen, eff, variants, args, attn_layers, kl_layers, edges)
            if out is None:
                raise RuntimeError("attention hook never fired on this model (no fft/lookback captured)")
            save_npz_atomic(out_dir / "extra" / f"{pid}.npz", out)
        except Exception as e:
            log(f"  [fail] {pid}: {e!r}")
            log_failure(out_dir, pid, "extras", e)
            torch.cuda.empty_cache()
        if k % 5 == 0 or k == len(todo):
            log(f"[extras {k}/{len(todo)}] {pid} ({(time.time() - t0) / k:.1f}s/problem)")
    del model, st
    gc.collect()
    torch.cuda.empty_cache()


# =========================================================================== PEARL-style perturbation sensitivity (prompt only)
# Idea adapted from PEARL (memorised vs generalised = abrupt vs gradual fall-off under increasing input noise). No gold answers are used:
# we measure how far the model's reading of the problem moves away from its reading of the ORIGINAL problem as noise increases.
# Per problem: the original prompt, prose-only noise at --pearl-rates x --pearl-seeds (letters inside plain words only; never math,
# LaTeX, digits or number words, so no information is removed), and optional semantic variants (--pearl-variants). One prompt-only
# forward each. Saved: last-prompt-token and problem-mean hidden states at the 6 sampled layers, and the exact KL / JS divergence and
# top-1 agreement of the next-token distribution at the end of the prompt (= first reasoning token) versus the original.
NUMBER_WORDS = set("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen "
                   "nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion first second third fourth "
                   "fifth sixth seventh eighth ninth tenth half twice once double triple".split())
_MATH = re.compile(r"(\$\$.*?\$\$|\$.*?\$|\\\[.*?\\\]|\\\(.*?\\\)|\\begin\{(\w+\*?)\}.*?\\end\{\2\}|\\[a-zA-Z]+|\{[^{}]*\}|https?://\S+)", re.S)


def prose_noise(text: str, rate: float, seed: int):
    """Corrupt a fraction `rate` of eligible prose words (alphabetic, length >= 3, not number words, outside math) with one random
    character edit each (swap / delete / duplicate / substitute). Returns (new_text, n_words_changed)."""
    import random
    rng = random.Random(seed)
    pieces, last = [], 0
    for m in _MATH.finditer(text):
        pieces.append((text[last:m.start()], True)); pieces.append((m.group(0), False)); last = m.end()
    pieces.append((text[last:], True))
    cand = [(pi, w.start(), w.group(0)) for pi, (s, prose) in enumerate(pieces) if prose
            for w in re.finditer(r"[A-Za-z]{3,}", s) if w.group(0).lower() not in NUMBER_WORDS]
    if not cand or rate <= 0:
        return text, 0
    k = min(len(cand), max(1, round(rate * len(cand))))
    pick = sorted(rng.sample(cand, k), key=lambda t: (t[0], -t[1]))
    segs = [s for s, _ in pieces]
    def edit(w):
        j = rng.randrange(len(w)); op = rng.choice(("swap", "delete", "dup", "sub"))
        if op == "swap" and len(w) >= 4:
            j = rng.randrange(1, len(w) - 2); return w[:j] + w[j + 1] + w[j] + w[j + 2:]
        if op == "delete":
            return w[:j] + w[j + 1:]
        if op == "dup":
            return w[:j] + w[j] + w[j:]
        c = rng.choice("abcdefghijklmnopqrstuvwxyz"); return w[:j] + (c.upper() if w[j].isupper() else c) + w[j + 1:]

    for pi, st, w in pick:
        for _ in range(20):  # never create a number word by accident ("there" -> "three", "then" -> "ten")
            nw = edit(w)
            if nw != w and nw.lower() not in NUMBER_WORDS:
                break
        else:
            continue
        segs[pi] = segs[pi][:st] + nw + segs[pi][st + len(w):]
    return "".join(segs), k


def pearl_one(model, tokenizer, st: "Struct", problem: str, variants, args, prefix_ids=None):
    """variants: list of (ptype, text) semantic variants (may be empty). prefix_ids: the ORIGINAL problem's first reasoning tokens
    (from its saved trace); if None they are greedy-generated once here. Every prompt (original and perturbed) is followed by the same
    prefix under teacher forcing, and we measure how much the model's next-token distributions over that prefix move away from the
    original's: the prompt-end token alone is a fixed template token (<think>) and barely moves, the reasoning prefix does.
    Returns the npz dict for one problem."""
    import torch
    prompts = [("original", 0.0, -1, problem, 0)]
    for rate in args.pearl_rates:
        for s in range(args.pearl_seeds):
            t, n = prose_noise(problem, rate, seed=1000 * s + int(rate * 1000))
            prompts.append(("noise", float(rate), s, t, n))
    for ptype, t in variants:
        prompts.append((str(ptype), -1.0, -1, t, -1))
    P = args.pearl_prefix
    if P and prefix_ids is None:
        ids0, _, _ = build_prompt(tokenizer, problem)
        g = model.generate(input_ids=torch.tensor([ids0], device=model.device), attention_mask=torch.ones(1, len(ids0), dtype=torch.long, device=model.device),
                           max_new_tokens=P, do_sample=False, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        prefix_ids = g[0, len(ids0):].tolist()
    prefix_ids = list(prefix_ids or [])[:P] if P else []
    lasts, means, ends, ntok = [], [], [], []
    ref = None
    kl0, js0, top10, ent0, klp, jsp, top1p, nllp, nll_orig_tok = [], [], [], [], [], [], [], [], []
    for _, _, _, txt, _ in prompts:
        ids, _, span = build_prompt(tokenizer, txt)
        n = len(ids)
        x = torch.tensor([ids + prefix_ids], device=model.device)
        out = model(input_ids=x, use_cache=False, output_hidden_states=True)
        hs = out.hidden_states
        lasts.append(torch.stack([hs[l + 1][0, n - 1] for l in st.sel_layers]).float().cpu())
        a, b = span if span and span[1] - span[0] > 0 else (0, n)
        means.append(torch.stack([hs[l + 1][0, a:b].float().mean(0) for l in st.sel_layers]).cpu())
        ends.append(torch.stack([hs[l + 1][0, -1] for l in st.sel_layers]).float().cpu())   # state after the reasoning prefix
        assert_finite(out.logits[0, n - 1:], "logits")
        lp = torch.log_softmax(out.logits[0, n - 1:].float(), -1)        # [1 + P, V]: prompt-end distribution, then each prefix position
        if ref is None:
            ref = lp
        pp, po = lp.exp(), ref.exp(); mm = 0.5 * (po + pp); lm = mm.clamp_min(1e-30).log()
        kl = (po * (ref - lp)).sum(-1); js = 0.5 * (po * (ref - lm)).sum(-1) + 0.5 * (pp * (lp - lm)).sum(-1)
        top1 = (lp.argmax(-1) == ref.argmax(-1)).float(); ent = -(pp * lp).sum(-1)
        kl0.append(float(kl[0])); js0.append(float(js[0])); top10.append(float(top1[0])); ent0.append(float(ent[0]))
        if prefix_ids:
            tgt = torch.tensor(prefix_ids, device=lp.device)
            klp.append(float(kl[:-1].mean())); jsp.append(float(js[:-1].mean())); top1p.append(float(top1[:-1].mean()))
            nllp.append(float(-lp[:-1].gather(1, tgt[:, None]).mean()))        # surprise of the ORIGINAL reasoning prefix under this prompt
        ntok.append(n)
        del out, hs
    d = {"kind": np.array([p[0] for p in prompts]), "rate": np.array([p[1] for p in prompts], dtype=np.float32),
         "seed": np.array([p[2] for p in prompts], dtype=np.int16), "n_words_changed": np.array([p[4] for p in prompts], dtype=np.int16),
         "n_tokens": np.array(ntok, dtype=np.int32), "sel_layers": np.array(st.sel_layers, dtype=np.int16),
         "last_state": torch.stack(lasts).half().numpy(), "mean_state": torch.stack(means).half().numpy(),   # [n_prompts, 6, d]
         "kl_vs_original": np.array(kl0, dtype=np.float32), "js_vs_original": np.array(js0, dtype=np.float32),   # prompt-end token
         "top1_same": np.array(top10, dtype=np.float32), "entropy": np.array(ent0, dtype=np.float32),
         "texts": np.array([p[3] for p in prompts]), "prefix_len": np.int32(len(prefix_ids))}
    if prefix_ids:
        d.update(prefix_state=torch.stack(ends).half().numpy(),                       # [n_prompts, 6, d] after the reasoning prefix
                 prefix_kl=np.array(klp, dtype=np.float32), prefix_js=np.array(jsp, dtype=np.float32),   # mean over prefix positions
                 prefix_top1_same=np.array(top1p, dtype=np.float32), prefix_nll=np.array(nllp, dtype=np.float32),
                 prefix_ids=np.array(prefix_ids, dtype=np.int32))
    return d


def phase_pearl(model_id: str, args):
    import torch
    torch.set_grad_enabled(False)
    out_dir = Path(args.out) / slug(model_id)
    out_dir.mkdir(parents=True, exist_ok=True); (out_dir / "pearl").mkdir(exist_ok=True)
    variants = {}
    _norm = lambda t: re.sub(r"[^a-z0-9]", "", str(t).lower())[:200]
    if args.pearl_variants:
        V = pd.read_csv(args.pearl_variants)
        for pid, g in V.groupby("pid"):
            variants[str(pid)] = list(zip(g.ptype, g.text))
        if args.pearl_variant_problems:  # also key the variants by the original problem text, so they attach to matching trace rows
            Q = pd.read_csv(args.pearl_variant_problems)
            for r in Q.itertuples():
                if str(r.pid) in variants:
                    variants[_norm(r.problem)] = variants[str(r.pid)]
    if args.pearl_input:  # e.g. olympiad/selected_problems.csv with columns pid, problem
        T = pd.read_csv(args.pearl_input)
        items = [(str(r.pid), str(r.problem), None) for r in T.itertuples()]          # prefix greedy-generated per problem
    else:                 # every problem in this model's generations.jsonl; the reasoning prefix is the start of its saved trace
        items = [(r["id"], r["problem"], r.get("gen_ids")) for r in load_jsonl_by_id(out_dir / "generations.jsonl").values()]
        for i, prob, _ in items:  # variants keyed by problem text (e.g. the main-v2 problems that also appear in olympiad/)
            if i not in variants and _norm(prob) in variants:
                variants[i] = variants[_norm(prob)]
    todo = [it for it in items if not (out_dir / "pearl" / f"{it[0]}.npz").exists()]
    if args.ids:
        todo = [it for it in todo if it[0] in set(args.ids)]
    log(f"\n=== [pearl] {model_id}: {len(todo)} pending of {len(items)} (rates {args.pearl_rates} x {args.pearl_seeds} seeds, "
        f"semantic variants for {sum(it[0] in variants for it in items)} problems, reasoning prefix {args.pearl_prefix} tokens) ===")
    if not todo:
        return
    local = ensure_weights(model_id, args)
    tokenizer = load_tokenizer(local)
    model = load_hf_model(local)
    st = Struct(model)
    t0 = time.time()
    for k, (pid, prob, gen) in enumerate(todo, 1):
        try:
            save_npz_atomic(out_dir / "pearl" / f"{pid}.npz", pearl_one(model, tokenizer, st, prob, variants.get(pid, []), args,
                                                                     prefix_ids=gen[:args.pearl_prefix] if gen else None))
        except Exception as e:
            log(f"  [fail] {pid}: {e!r}")
            log_failure(out_dir, pid, "pearl", e)
            torch.cuda.empty_cache()
        if k % 10 == 0 or k == len(todo):
            log(f"[pearl {k}/{len(todo)}] {pid} ({(time.time() - t0) / k:.2f}s/problem)")
    del model, st
    gc.collect()
    torch.cuda.empty_cache()


# =========================================================================== Lookback Lens (Chuang et al., 2024), teacher-forced
# For every generated token t (query position p >= n_prompt), layer l, head h:
#   A_ctx = mean attention weight on the context (prompt) tokens, A_new = mean attention weight on the generated tokens so far
#   (positions n_prompt..p, i.e. the previous tokens plus the token itself, which is what the query at p can see);
#   lookback ratio LR = A_ctx / (A_ctx + A_new).
# Means use the keys the query can actually see (on sliding-window layers prompt keys can drop out of view -> LR = 0 there).
# Saved as means over consecutive spans of --lookback-window generated tokens (the paper's span features are such means), plus the
# raw prompt attention mass per span, so any span / budget can be pooled offline.
def _lookback_layer(li, query, key, sc, win):
    import torch
    q, kk = query[0], key[0]
    H, T, D = q.shape
    Hkv = kk.shape[0]
    n_p, Wn = DIAG["n_p"], DIAG["W"]
    n_g = T - n_p
    nwin = -(-n_g // Wn)
    qg = q.reshape(Hkv, H // Hkv, T, D)
    kpos = torch.arange(T, device=q.device)
    lr_sum = torch.zeros(H, nwin, device=q.device)
    m_sum = torch.zeros(H, nwin, device=q.device)
    cnt = torch.zeros(nwin, device=q.device)
    for s0 in range(n_p, T, DIAG["chunk"]):
        e0 = min(T, s0 + DIAG["chunk"])
        qpos = torch.arange(s0, e0, device=q.device)
        p, mask, _ = _prob_chunk(qg[:, :, s0:e0], kk, sc, win, qpos, kpos)
        p = p.reshape(H, e0 - s0, T)
        mctx = p[..., :n_p].sum(-1)                                    # [H, bq] attention mass on the prompt
        vis = ~mask                                                     # [bq, T] keys visible to each query
        cctx = vis[:, :n_p].sum(-1).float(); cnew = vis[:, n_p:].sum(-1).float()
        actx = mctx / cctx.clamp(min=1)
        anew = (1 - mctx).clamp(min=0) / cnew.clamp(min=1)
        lr = torch.where(cctx > 0, actx / (actx + anew).clamp(min=1e-12), torch.zeros_like(actx))
        widx = torch.div(qpos - n_p, Wn, rounding_mode="floor")
        lr_sum.index_add_(1, widx, lr); m_sum.index_add_(1, widx, mctx); cnt.index_add_(0, widx, torch.ones_like(widx, dtype=torch.float))
        del p
    DIAG["res"][li] = (lr_sum / cnt.clamp(min=1), m_sum / cnt.clamp(min=1))


class lookback_capture:
    def __init__(self, layers, chunk, n_p, W):
        self.layers, self.kw = list(layers), dict(chunk=chunk, n_p=n_p, W=W)

    def __enter__(self):
        DIAG.update(mode="lookback", layers=frozenset(self.layers), res={}, **self.kw)
        return self

    def __exit__(self, *a):
        DIAG["mode"] = None

    def result(self):
        """-> (layer ids, lr [n_win, L, H], prompt_mass [n_win, L, H]) or (None, None, None)."""
        import torch
        if not DIAG["res"]:
            return None, None, None
        ls = sorted(DIAG["res"])
        lr = torch.stack([DIAG["res"][l][0] for l in ls]).permute(2, 0, 1)
        pm = torch.stack([DIAG["res"][l][1] for l in ls]).permute(2, 0, 1)
        DIAG["res"] = {}
        return ls, lr.float().cpu().numpy(), pm.float().cpu().numpy()


def lookback_one(model, prompt_ids, gen_ids, args, layers):
    import torch
    ids = torch.tensor([prompt_ids + gen_ids], device=model.device)
    with lookback_capture(layers, args.attn_chunk, len(prompt_ids), args.lookback_window) as lc:
        try:
            model(input_ids=ids, use_cache=False, logits_to_keep=1)
        except TypeError:
            model(input_ids=ids, use_cache=False)
    ls, lr, pm = lc.result()
    if lr is not None and not (np.isfinite(lr).all() and np.isfinite(pm).all()):
        raise RuntimeError("non-finite lookback ratios (fp16 overflow?) - rerun with --dtype fp32 or on a bf16-capable GPU")
    return ls, lr, pm


def phase_lookback(model_id: str, args):
    import torch
    torch.set_grad_enabled(False)
    out_dir = Path(args.out) / slug(model_id)
    rows = load_jsonl_by_id(out_dir / "generations.jsonl")
    (out_dir / "lookback").mkdir(exist_ok=True)
    todo = [r for r in rows.values() if r["gen_ids"] and not (out_dir / "lookback" / f"{r['id']}.npz").exists()]
    if args.ids:
        todo = [r for r in todo if r["id"] in set(args.ids)]
    log(f"\n=== [lookback] {model_id}: {len(todo)} pending of {len(rows)} ===")
    if not todo:
        return
    local = ensure_weights(model_id, args)
    tokenizer = load_tokenizer(local)
    model = load_hf_model(local)
    st = Struct(model)
    install_diag_hook()
    layers = pick_attn_layers(st, args.attn_layers)
    tid = think_end_token_id(tokenizer)
    log(f"  attention layers={len(layers)} span={args.lookback_window} tokens")
    t0 = time.time()
    for k, r in enumerate(todo, 1):
        if free_gb(out_dir) < MIN_FREE_OUTPUT_GB:
            log("Free disk below limit - stopping. Move results off and rerun.")
            break
        pid = r["id"]
        try:
            gen = r["gen_ids"]
            ls, lr, pm = lookback_one(model, r["prompt_ids"], gen, args, layers)
            if lr is None:
                raise RuntimeError("attention hook never fired on this model (no lookback ratios captured)")
            te = find_think_end(tokenizer, gen, tid)
            save_npz_atomic(out_dir / "lookback" / f"{pid}.npz", {
                "lookback": lr.astype(np.float16), "prompt_mass": pm.astype(np.float16),      # [n_span, L, H]
                "span": np.int32(args.lookback_window), "attn_layers": np.array(ls, dtype=np.int16),
                "n_prompt": np.int32(len(r["prompt_ids"])), "n_gen": np.int32(len(gen)), "think_end": np.int32(-1 if te is None else te)})
        except Exception as e:
            log(f"  [fail] {pid}: {e!r}")
            log_failure(out_dir, pid, "lookback", e)
            torch.cuda.empty_cache()
        if k % 5 == 0 or k == len(todo):
            log(f"[lookback {k}/{len(todo)}] {pid} ({(time.time() - t0) / k:.1f}s/problem)")
    del model, st
    gc.collect()
    torch.cuda.empty_cache()


# =========================================================================== TOHA: prompt-response minimum spanning forest per head
# For a head, take the response tokens R and the prompt tokens P as a graph with edge length w = 1 - attention (symmetrised: for causal
# attention the pair weight is the later token's attention to the earlier one). Contract P to one node p* (w(r, p*) = 1 - max_p A[r, p]).
# The TOHA score is the total length of the minimum spanning tree on {p*} + R, i.e. the minimum spanning forest that connects every response
# token to the prompt. Attention probabilities are exact (softmax over ALL cached keys, incl. the reasoning trace); only the columns for P and R
# are kept. Two response sets per cutoff c:  'win' = the last --toha-window trace tokens before c (is the reasoning still anchored to the
# problem?), 'ans' = forced suffix + the greedy forced answer at c (is the answer anchored to the problem?). |R| is fixed for 'win', so the
# score does not depend on trace length.
TOHA_STATS = ["total", "mean_edge", "max_edge", "frac_direct_to_prompt", "mean_prompt_dist"]


def _toha_layer(li, query, key, sc, win):
    import torch
    S, n, n_p = DIAG["S"], DIAG["n"], DIAG["n_p"]
    q, kk = query[0][:, -S:], key[0]
    H, _, D = q.shape
    Hkv, Tk = kk.shape[0], kk.shape[1]
    off = n + S - Tk                                   # > 0 on sliding-window caches (dropped keys get no attention anyway)
    qg = q.reshape(Hkv, H // Hkv, S, D)
    kpos = off + torch.arange(Tk, device=q.device)
    p_lo, p_hi = max(0, -off), max(0, n_p - off)       # local columns of the prompt keys still in the cache
    r_lo, r_hi = n - off, n + S - off                  # local columns of the response keys
    maxp = torch.zeros(H, S, device=q.device)
    rr = torch.zeros(H, S, S, device=q.device)
    for s0 in range(0, S, DIAG["chunk"]):
        e0 = min(S, s0 + DIAG["chunk"])
        p, _, _ = _prob_chunk(qg[:, :, s0:e0], kk, sc, win, n + torch.arange(s0, e0, device=q.device), kpos)
        p = p.reshape(H, e0 - s0, Tk)
        if p_hi > p_lo:
            maxp[:, s0:e0] = p[..., p_lo:p_hi].max(-1).values
        rr[:, s0:e0] = p[..., r_lo:r_hi]
        del p
    lo = torch.tril(rr, -1)
    W = torch.full((H, S + 1, S + 1), float("inf"), device=q.device)
    W[:, 0, 1:] = 1 - maxp
    W[:, 1:, 0] = 1 - maxp
    W[:, 1:, 1:] = 1 - (lo + lo.transpose(1, 2))
    W[:, torch.arange(1, S + 1), torch.arange(1, S + 1)] = float("inf")
    DIAG["res"][li] = (W, (1 - maxp).mean(-1))


def msf_stats(W):
    """W [B, N, N] symmetric distances, node 0 = contracted prompt. Prim's algorithm batched over B. Returns [B, 4]:
    total MST length, mean edge, max edge, fraction of response nodes attached directly to the prompt node."""
    import torch
    B, N, _ = W.shape
    ar = torch.arange(B, device=W.device)
    in_t = torch.zeros(B, N, dtype=torch.bool, device=W.device); in_t[:, 0] = True
    key = W[:, 0].clone(); key[:, 0] = float("inf")
    parent = torch.zeros(B, N, dtype=torch.long, device=W.device)
    edges = []
    for _ in range(N - 1):
        j = key.masked_fill(in_t, float("inf")).argmin(1)
        edges.append(key[ar, j]); in_t[ar, j] = True
        nw = W[ar, j]
        upd = (nw < key) & ~in_t
        key = torch.where(upd, nw, key); parent = torch.where(upd, j[:, None], parent)
    E = torch.stack(edges, 1)
    return torch.stack([E.sum(1), E.mean(1), E.max(1).values, (parent[:, 1:] == 0).float().mean(1)], 1)


class toha_capture:
    def __init__(self, layers, chunk, n, S, n_p):
        self.layers, self.kw = list(layers), dict(chunk=chunk, n=n, S=S, n_p=n_p)

    def __enter__(self):
        DIAG.update(mode="toha", layers=frozenset(self.layers), res={}, **self.kw)
        return self

    def __exit__(self, *a):
        DIAG["mode"] = None

    def result(self):
        """-> (layer ids, stats [L, H, 5]) or (None, None) if the hook never fired."""
        import torch
        if not DIAG["res"]:
            return None, None
        ls = sorted(DIAG["res"])
        Ws = torch.stack([DIAG["res"][l][0] for l in ls])            # [L, H, N, N]
        L, H, N, _ = Ws.shape
        st = msf_stats(Ws.reshape(L * H, N, N)).reshape(L, H, 4)
        pd_ = torch.stack([DIAG["res"][l][1] for l in ls])            # [L, H]
        DIAG["res"] = {}
        return ls, torch.cat([st, pd_[..., None]], -1).float().cpu().numpy()


def toha_one(model, tokenizer, prompt_ids, gen_ids, eff_cuts, variants, args, layers):
    import torch
    full = prompt_ids + gen_ids
    n_p, m = len(prompt_ids), args.toha_window
    low = getattr(args, "low_mem", False)
    names = sorted(variants)
    cache, pos = None, 0
    win, ans, wlen, alen, atxt, ls_out = [], [], [], [], [], None
    for c in sorted(set(eff_cuts)):
        target = n_p + c
        wstart = max(n_p, target - m)
        if low and cache is not None:
            _cache_to(cache, model.device)
        if wstart > pos:
            cache = _extend(model, cache, full[pos:wstart], args.prefill_chunk)
        x = torch.tensor([full[wstart:target]], device=model.device)
        with toha_capture(layers, args.attn_chunk, wstart, target - wstart, n_p) as tc:
            out = model(input_ids=x, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
        ls, st = tc.result()
        if st is None:
            return None
        ls_out = ls; win.append(st); wlen.append(target - wstart); pos = target
        if low:
            _cache_to(cache, "cpu"); torch.cuda.empty_cache()
        row, rl, rt = [], [], []
        for v in names:
            sids = variants[v]
            c2 = copy.deepcopy(cache)
            if low:
                _cache_to(c2, model.device)
            o = model(input_ids=torch.tensor([sids], device=model.device), past_key_values=c2, use_cache=True)
            res = _decode_answer(model, tokenizer, c2, o.logits[0, -1], None, args)
            del c2, o
            c3 = copy.deepcopy(cache)
            if low:
                _cache_to(c3, model.device)
            seq = sids + res["ids"]
            with toha_capture(layers, args.attn_chunk, target, len(seq), n_p) as tc:
                model(input_ids=torch.tensor([seq], device=model.device), past_key_values=c3, use_cache=True, logits_to_keep=1)
            _, st2 = tc.result()
            del c3
            row.append(st2); rl.append(len(seq)); rt.append(res["answer"])
        ans.append(np.stack(row)); alen.append(rl); atxt.append(rt)
        torch.cuda.empty_cache()
    return {"cut_eff": np.array(sorted(set(eff_cuts)), dtype=np.int32), "attn_layers": np.array(ls_out, dtype=np.int16),
            "variant_names": np.array(names), "stat_names": np.array(TOHA_STATS), "n_prompt": np.int32(n_p),
            "win": np.stack(win).astype(np.float32),                      # [cut, L, H, 5]
            "win_len": np.array(wlen, dtype=np.int32),
            "ans": np.stack(ans).astype(np.float32),                      # [cut, var, L, H, 5]
            "ans_len": np.array(alen, dtype=np.int32), "ans_text": np.array(atxt)}


def phase_toha(model_id: str, args):
    import torch
    torch.set_grad_enabled(False)
    out_dir = Path(args.out) / slug(model_id)
    rows = load_jsonl_by_id(out_dir / "generations.jsonl")
    (out_dir / "toha").mkdir(exist_ok=True)
    todo = [r for r in rows.values() if r["gen_ids"] and not (out_dir / "toha" / f"{r['id']}.npz").exists()]
    if args.ids:
        todo = [r for r in todo if r["id"] in set(args.ids)]
    log(f"\n=== [toha] {model_id}: {len(todo)} pending of {len(rows)} ===")
    if not todo:
        return
    local = ensure_weights(model_id, args)
    tokenizer = load_tokenizer(local)
    model = load_hf_model(local)
    st = Struct(model)
    install_diag_hook()
    layers = pick_attn_layers(st, args.attn_layers)
    tid = think_end_token_id(tokenizer)
    has_think = tid is not None or "<think>" in build_prompt(tokenizer, "x")[1]
    allv = {k: tokenizer(v, add_special_tokens=False)["input_ids"] for k, v in (SUFFIXES_THINK if has_think else SUFFIXES_NOTHINK).items()}
    variants = {k: allv[k] for k in args.toha_variants}
    log(f"  attention layers={len(layers)} budgets={args.toha_budgets} window={args.toha_window} variants={list(variants)}")
    t0 = time.time()
    for k, r in enumerate(todo, 1):
        if free_gb(out_dir) < MIN_FREE_OUTPUT_GB:
            log("Free disk below limit - stopping. Move results off and rerun.")
            break
        pid = r["id"]
        try:
            gen = r["gen_ids"]
            te = find_think_end(tokenizer, gen, tid)
            L = max(1, te if te is not None else len(gen))
            eff = sorted({max(1, min(b, L)) for b in args.toha_budgets})
            out = toha_one(model, tokenizer, r["prompt_ids"], gen, eff, variants, args, layers)
            if out is None:
                raise RuntimeError("attention hook never fired on this model (no TOHA statistics captured)")
            save_npz_atomic(out_dir / "toha" / f"{pid}.npz", out)
        except Exception as e:
            log(f"  [fail] {pid}: {e!r}")
            log_failure(out_dir, pid, "toha", e)
            torch.cuda.empty_cache()
        if k % 5 == 0 or k == len(todo):
            log(f"[toha {k}/{len(todo)}] {pid} ({(time.time() - t0) / k:.1f}s/problem)")
    del model, st
    gc.collect()
    torch.cuda.empty_cache()


# =========================================================================== check / report / preflight
def phase_check(model_id: str, args):
    """Branch-from-cache vs fresh full forward: same final-position distribution and same forced answer."""
    import torch
    torch.set_grad_enabled(False)
    out_dir = Path(args.out) / slug(model_id)
    rows = [r for r in load_jsonl_by_id(out_dir / "generations.jsonl").values() if len(r["gen_ids"]) > 200][: args.check_n]
    if not rows:
        raise SystemExit("no generations to check; run generate first")
    local = ensure_weights(model_id, args)
    tokenizer = load_tokenizer(local)
    model = load_hf_model(local)
    st = Struct(model)
    tid = think_end_token_id(tokenizer)
    sufs = SUFFIXES_THINK if (tid is not None or "<think>" in build_prompt(tokenizer, "x")[1]) else SUFFIXES_NOTHINK
    sids = tokenizer(sufs["B"], add_special_tokens=False)["input_ids"]
    for r in rows:
        prompt_ids = r["prompt_ids"]
        gen = r["gen_ids"]
        te = find_think_end(tokenizer, gen, tid)
        L = te if te is not None else len(gen)
        cuts = sorted({min(256, L), max(1, L // 2)})
        results, _, _ = forced_answers(model, tokenizer, st, prompt_ids, gen, cuts, {"B": sids}, args)
        for c in cuts:
            full = prompt_ids + gen[:c] + sids
            with torch.no_grad():
                lg = model(input_ids=torch.tensor([full], device=model.device), use_cache=False, logits_to_keep=1).logits[0, -1].float()
            fresh_top = int(lg.argmax())
            got = results[f"{c}|B"]
            log(f"  {r['id']} cut={c}: branch={got['branch']} answer={got['answer']!r} first-token entropy={got['first_entropy']:.3f} "
                f"| fresh top1 token={tokenizer.decode([fresh_top])!r} (branch top1 id={got['top'][0]['ids'][0]}, match={fresh_top == got['top'][0]['ids'][0]})")
    log("check done: every 'match=True' means the cache-branch path reproduces a fresh forward at the cutoff.")


def _norm_ans(s):
    if s is None:
        return None
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", str(s))
    s = s.strip().strip("$").replace(",", "").replace("\\!", "").replace("\\,", "").replace("\\dfrac", "\\frac")
    return re.sub(r"\s+", "", s).rstrip(".")


def phase_audit(args):
    """Trace-integrity audit: per model x source stop-reason mix, and flags for truncation artifacts.
    'cap' must be exactly --max-new-tokens long; every 'eos' trace must end in the model's EOS token; 'loop' = post-hoc cut."""
    for m in EVAL_MODELS:
        gens = load_jsonl_by_id(Path(args.out) / slug(m) / "generations.jsonl")
        if not gens:
            continue
        eos_last = pd.Series([g["gen_ids"][-1] for g in gens.values() if g["stop_reason"] == "eos" and g["gen_ids"]])
        modal = int(eos_last.mode().iloc[0]) if len(eos_last) else None
        log(f"\n== audit {m} (modal EOS token id {modal})")
        by = {}
        for g in gens.values():
            by.setdefault(g["source"], []).append(g)
        for src, gs in sorted(by.items()):
            n = len(gs)
            c = {r: sum(g["stop_reason"] == r for g in gs) for r in ("eos", "cap", "loop", "other")}
            lens = sorted(len(g["gen_ids"]) for g in gs)
            bad_cap = sum(g["stop_reason"] == "cap" and len(g["gen_ids"]) != args.max_new_tokens for g in gs)
            bad_eos = sum(g["stop_reason"] == "eos" and g["gen_ids"][-1] != modal for g in gs)
            fresh = sum(g.get("reused_from") is None for g in gs)
            log(f"  {src:10s} n={n:4d} eos={c['eos']:4d} cap={c['cap']:4d} loop={c['loop']:3d} other={c['other']:3d} "
                f"median_len={lens[n // 2]:6d} | freshly generated={fresh:4d} | FLAGS cap!=max:{bad_cap} eos-not-ending-in-EOS:{bad_eos}")
        for d in args.compare_dirs:  # old vs new traces for the same problem text (same model)
            old = {sha(str(g.get("problem", "")).strip()): g for g in load_jsonl_by_id(Path(d) / slug(m) / "generations.jsonl").values()}
            pairs = [(old[sha(g["problem"])], g) for g in gens.values() if g.get("reused_from") is None and sha(g["problem"]) in old]
            if pairs:
                trans = pd.Series([f"{o['stop_reason']}->{g['stop_reason']}" for o, g in pairs]).value_counts().to_dict()
                ratio = float(np.median([len(g["gen_ids"]) / max(1, len(o["gen_ids"])) for o, g in pairs]))
                same_prefix = float(np.mean([g["gen_ids"][:64] == o["gen_ids"][:64] for o, g in pairs]))
                log(f"  vs {d}: {len(pairs)} regenerated problems; stop transitions {trans}; median new/old length {ratio:.2f}; "
                    f"first-64-token match {same_prefix:.2f}")


def phase_report(args):
    phase_audit(args)
    phase_report_gold(args)
    for m in EVAL_MODELS:
        d = Path(args.out) / slug(m)
        gens = load_jsonl_by_id(d / "generations.jsonl")
        if not gens:
            continue
        n_tf = len(list((d / "tf").glob("*.npz"))) if (d / "tf").is_dir() else 0
        forces = {p.stem: json.load(open(p)) for p in (d / "force").glob("*.json")} if (d / "force").is_dir() else {}
        stops = pd.Series([g["stop_reason"] for g in gens.values()]).value_counts().to_dict()
        fails = len(load_jsonl_by_id(d / "failures.jsonl")) if (d / "failures.jsonl").exists() else 0
        log(f"\n== {m}: {len(gens)} traces, stop={stops}, tf={n_tf}, force={len(forces)}, failure_lines={fails}")
        agree = {}
        for pid, f in forces.items():
            if f.get("final_boxed") is None:
                continue
            fin = _norm_ans(f["final_boxed"])
            for lab, c in f["labels"].items():
                for v in f["variants"]:
                    res = f["results"].get(f"{c}|{v}")
                    if res:
                        agree.setdefault((lab, v), []).append(_norm_ans(res["answer"]) == fin)
        if agree:
            log("  forced-vs-final agreement on finished traces (suffix variant A/B):")
            for lab in sorted({k[0] for k in agree}, key=lambda s: (s[0], int(s[1:]))):
                parts = [f"{v}={np.mean(agree[(lab, v)]):.2f} (n={len(agree[(lab, v)])})" for v in ("A", "B") if (lab, v) in agree]
                log(f"    {lab:>6s}: " + "  ".join(parts))
            log("  (f90/f100 agreement near 0 means the suffix is wrong for this model; f25 near the base rate is expected.)")


def _matches_gold(boxed, gold) -> bool:
    if boxed is None or gold is None:
        return False
    b = _norm_ans(boxed)
    try:
        return abs(float(b) - float(gold)) < 1e-6 * max(1.0, abs(float(gold)))
    except (TypeError, ValueError):
        return b == _norm_ans(gold)


def phase_report_gold(args):
    """Final-answer accuracy against gold answers (agg originals vs their decay-causing permutations)."""
    d = Path(args.out) / slug("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    gens = load_jsonl_by_id(d / "generations.jsonl")
    rows = [g for g in gens.values() if g.get("gold") is not None and g["stop_reason"] == "eos"]
    if not rows:
        return
    log("\n== final-answer accuracy vs gold (finished traces only):")
    for src in ("agg", "perm"):
        rs = [g for g in rows if g["source"] == src]
        if rs:
            log(f"  {src:5s}: {sum(_matches_gold(g['boxed'], g['gold']) for g in rs)}/{len(rs)} correct")


def phase_preflight(args):
    import torch
    import transformers
    info = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "python": sys.version.split()[0], "torch": torch.__version__,
            "transformers": transformers.__version__, "cuda": torch.cuda.is_available()}
    try:
        import vllm
        info["vllm"] = vllm.__version__
    except Exception as e:
        info["vllm"] = f"unavailable ({e.__class__.__name__})"
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["gpu"], info["vram_gb"] = p.name, round(p.total_memory / 1e9, 1)
    info["free_disk_gb"] = round(free_gb(Path(args.out)), 1)
    log(json.dumps(info, indent=2))
    sample = "What is $1+1$?"
    models_info = {}
    for m in args.models:
        tok = load_tokenizer(m if not args.local_models else ensure_weights(m, args))
        ids, text, span = build_prompt(tok, sample)
        tid = think_end_token_id(tok)
        models_info[m] = {"prompt_tail": text[-120:], "n_prompt_tokens": len(ids), "think_end_id": tid,
                          "template_has_think": "<think>" in text, "bos_in_ids": (tok.bos_token_id in ids) if tok.bos_token_id is not None else None,
                          "problem_span_found": span is not None, "eos_token": tok.eos_token,
                          "tokenizer_class": type(tok).__name__, "roundtrip_ok": tokenizer_roundtrip_ok(tok),
                          "think_tag_tokens": tok.convert_ids_to_tokens(tok("</think>", add_special_tokens=False)["input_ids"])}
        log(f"\n[{m}] " + json.dumps(models_info[m], indent=2))
    if args.bench:
        for m in args.models:
            models_info[m]["bench_tok_per_s"] = _bench(m, args)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    save_json_atomic(Path(args.out) / "manifest.json", {"env": info, "models": models_info, "args": vars(args)})


def _bench(model_id, args):
    """HF batched-generate throughput (what the submission will actually use), left padded."""
    import torch
    local = ensure_weights(model_id, args)
    tok = load_tokenizer(local)
    tok.padding_side = "left"
    model = load_hf_model(local)
    probs = ["Find the number of positive integers $n \\le 100$ such that $n^2+n+41$ is prime."] * 5
    res = {}
    for bs in (1, 2, 4, 5):
        texts = [build_prompt(tok, p)[1] for p in probs[:bs]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            model.generate(**enc, max_new_tokens=args.bench_tokens, min_new_tokens=args.bench_tokens, do_sample=False,
                           pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize()
        dt = time.time() - t0
        res[bs] = round(bs * args.bench_tokens / dt, 1)
        log(f"  [{model_id}] batch={bs}: {res[bs]} tok/s aggregate ({dt:.1f}s for {args.bench_tokens} tokens)")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return res


# =========================================================================== Orchestration
def child_cmd(phase, model_id, args):
    skip = {"phase", "models", "child"}
    cmd = [sys.executable, "-u", __file__, "--phase", phase, "--models", model_id, "--child"]
    for k, v in vars(args).items():
        if k in skip or v is None:
            continue
        flag = "--" + k.replace("_", "-")
        if v is False:
            if k in ("proxies", "include_unlabeled"):
                cmd.append("--no-" + k.replace("_", "-"))
            continue
        if v is True:
            cmd.append(flag)
        elif isinstance(v, (list, tuple)):
            if v:
                cmd += [flag, *map(str, v)]
        else:
            cmd += [flag, str(v)]
    return cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["preflight", "plan", "prefetch", "generate", "features", "extras", "toha", "lookback", "pearl", "check", "report", "all"], default="plan")
    ap.add_argument("--models", nargs="+", default=["eval"], help="model ids or 'eval' (the 4 Small-track models)")
    ap.add_argument("--out", default="extraction_out")
    ap.add_argument("--work-dir", default=".")
    ap.add_argument("--local-models", default=None, help="dir containing <slug(model_id)>/ checkpoints (skips download)")
    # data
    ap.add_argument("--main-v2", default="hf", help="'hf', path to parquet/csv, or 'none'")
    ap.add_argument("--agg", default="augmented-sample-math-agg.csv")
    ap.add_argument("--perms", default="decay_causing_permutations_summary.csv",
                    help="decay-causing permutations csv (DeepSeek traces are generated for every variant); 'none' to skip")
    ap.add_argument("--strict-reuse", action="store_true",
                    help="regenerate saved traces whose prompt ids differ from a fresh rebuild (default: trust saved ids)")
    ap.add_argument("--val-sample", default="none", help="'hf', path, or 'none' (almost all rows are non-local models)")
    ap.add_argument("--extra-problems", default=None, help="csv/parquet/jsonl of unlabeled problems for every model")
    ap.add_argument("--extra-max", type=int, default=200)
    ap.add_argument("--proxies", action=argparse.BooleanOptionalAction, default=True,
                    help="reuse gpt-oss-120b non-robust problems as proxy non-robust rows (source=gpt_proxy)")
    ap.add_argument("--include-unlabeled", action=argparse.BooleanOptionalAction, default=True,
                    help="keep borderline main-v2 rows with null is_robust (aux targets via max_drop)")
    ap.add_argument("--limit", type=int, default=None, help="max problems per model (smoke tests)")
    ap.add_argument("--ids", nargs="*", default=None, help="restrict features phase to these ids")
    ap.add_argument("--reuse-dirs", nargs="*", default=[], help="dirs with <slug>/generations.jsonl to reuse (prompt ids verified)")
    # generation
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--max-new-tokens", type=int, default=16384)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--gpu-util", type=float, default=0.92)
    ap.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
                    help="weights/activations dtype; auto = bf16 on Ampere+ GPUs, fp16 on V100 / Turing (TITAN RTX) / Pascal (no native bf16)")
    ap.add_argument("--low-mem", action="store_true", help="24 GB GPUs (e.g. Olmo on a 3090): keep the KV cache in CPU RAM between branches "
                    "(one GPU copy at a time), small chunks, vLLM eager with few sequences")
    ap.add_argument("--enforce-eager", action="store_true", help="vLLM: skip CUDA-graph capture (saves memory, fixes some startup OOMs)")
    # features
    ap.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    ap.add_argument("--fracs", type=float, nargs="+", default=list(DEFAULT_FRACS))
    ap.add_argument("--extra-budgets", type=int, nargs="+", default=[1024, 2048, 3072, 4096, 6144, 8192, 12288],
                    help="extras phase: token distances at which Fourier/lookback/KL/activations are taken (clipped to the pre-</think> length)")
    ap.add_argument("--kl-range", type=float, nargs=2, default=[0.5, 0.8], help="extras: layer-depth range for the attention KL features")
    ap.add_argument("--fft-bins", type=int, default=32)
    ap.add_argument("--no-extras", action="store_true", help="--phase all: skip the extras phase")
    ap.add_argument("--toha-budgets", type=int, nargs="+", default=[1024, 2048, 4096, 8192],
                    help="toha phase: cutoffs (tokens, clipped to the pre-</think> length) at which the spanning forests are computed")
    ap.add_argument("--pearl-rates", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2, 0.3],
                    help="pearl phase: fractions of eligible prose words corrupted (one character edit each)")
    ap.add_argument("--pearl-seeds", type=int, default=4, help="pearl phase: random corruptions per rate")
    ap.add_argument("--pearl-prefix", type=int, default=128,
                    help="pearl phase: tokens of the ORIGINAL problem's reasoning teacher-forced after every prompt (0 = prompt only)")
    ap.add_argument("--pearl-input", default=None, help="pearl phase: csv with pid, problem (default: the model's generations.jsonl)")
    ap.add_argument("--pearl-variants", default=None, help="pearl phase: csv with pid, ptype, text semantic variants (e.g. olympiad/perturbations.csv)")
    ap.add_argument("--pearl-variant-problems", default=None,
                    help="pearl phase: csv with pid, problem for --pearl-variants; variants then also attach to trace rows with the same problem text")
    ap.add_argument("--lookback-window", type=int, default=16,
                    help="lookback phase: generated tokens per saved span mean (the Lookback Lens paper uses 8-token spans; 16k tokens / 16 = 1024 spans)")
    ap.add_argument("--toha-window", type=int, default=128, help="toha phase: number of trace tokens before each cutoff used as the response set")
    ap.add_argument("--toha-variants", nargs="+", default=["B"], choices=["A", "B"], help="toha phase: forced-answer suffix variants")
    ap.add_argument("--answer-tokens", type=int, default=24)
    ap.add_argument("--prefill-chunk", type=int, default=2048)
    ap.add_argument("--logit-chunk", type=int, default=512)
    ap.add_argument("--stride", type=int, default=16, help="window length for --save-window-states")
    ap.add_argument("--save-window-states", action="store_true", help="~50 MB per 16k trace; check disk first")
    ap.add_argument("--attn-layers", type=int, default=-1,
                    help="attention-Laplacian eigenvalues: -1 = every softmax-attention layer, N = N equally spaced layers, 0 = off")
    ap.add_argument("--attn-topk", type=int, default=64, help="top-k Laplacian eigenvalues kept per (layer, head, cutoff)")
    ap.add_argument("--lap-degree", choices=["mean", "sum"], default="mean",
                    help="degree d_ii: mean incoming attention (LapEigvals) or plain column sum")
    ap.add_argument("--attn-chunk", type=int, default=512, help="query chunk for the Laplacian computation (~2 GB at 512 for 32 heads x 16k keys)")
    ap.add_argument("--no-attn", action="store_true")
    ap.add_argument("--no-tf", action="store_true")
    ap.add_argument("--no-force", action="store_true")
    ap.add_argument("--check-n", type=int, default=3)
    ap.add_argument("--compare-dirs", nargs="*", default=[], help="old trace dirs to compare regenerated traces against (report/audit)")
    ap.add_argument("--keep-weights", action="store_true")
    ap.add_argument("--no-prefetch", action="store_true",
                    help="multi-model runs: do NOT download the remaining models' weights in the background while the first one runs")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--bench-tokens", type=int, default=1024)
    ap.add_argument("--smoke-model", default=None, help="stand-in model id for pipeline smoke tests")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    global DTYPE
    DTYPE = args.dtype
    if args.low_mem:
        args.attn_chunk = min(args.attn_chunk, 128)
        args.prefill_chunk = min(args.prefill_chunk, 1024)
        args.logit_chunk = min(args.logit_chunk, 128)
        args.max_num_seqs = min(args.max_num_seqs, 2)
        args.enforce_eager = True
    args.models = EVAL_MODELS if args.models == ["eval"] else args.models
    for m in args.models:
        if m not in EVAL_MODELS and m != args.smoke_model:
            raise SystemExit(f"{m} is not a Small-track model")
    Path(args.out).mkdir(parents=True, exist_ok=True)

    if args.phase == "plan":
        phase_plan(args)
    elif args.phase == "prefetch":
        phase_prefetch(args)
    elif args.phase == "preflight":
        phase_preflight(args)
    elif args.phase == "report":
        phase_report(args)
    elif args.child or (args.phase in ("generate", "features", "extras", "toha", "lookback", "pearl", "check") and len(args.models) == 1):
        m = args.models[0]
        {"generate": phase_generate, "features": phase_features, "extras": phase_extras, "toha": phase_toha, "lookback": phase_lookback, "pearl": phase_pearl, "check": phase_check}[args.phase](m, args)
    else:  # multi-model driver: one subprocess per (model, phase) so GPU memory is fully released
        phases = (["generate", "features"] + ([] if args.no_extras else ["extras"])) if args.phase == "all" else [args.phase]
        phase_plan(args)
        prefetch = None
        if len(args.models) > 1 and not args.no_prefetch and not args.local_models:
            cmd = [sys.executable, "-u", __file__, "--phase", "prefetch", "--models", *args.models[1:], "--work-dir", args.work_dir]
            prefetch = subprocess.Popen(cmd, stdout=open(Path(args.out) / "prefetch.log", "a"), stderr=subprocess.STDOUT)
            log(f"[prefetch] downloading {args.models[1:]} in the background (log: {Path(args.out) / 'prefetch.log'})")
        for m in args.models:
            ok = True
            for ph in phases:
                rc = subprocess.call(child_cmd(ph, m, args))
                log(f"##### {m} [{ph}] exit code {rc}")
                if rc != 0:
                    ok = False
                    break
            if ok and args.phase == "all" and not args.keep_weights:
                shutil.rmtree(Path(args.work_dir) / "hf_tmp" / slug(m), ignore_errors=True)
        if prefetch is not None and prefetch.poll() is None:
            prefetch.terminate()
        phase_report(args)


if __name__ == "__main__":
    main()
