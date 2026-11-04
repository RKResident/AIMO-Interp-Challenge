"""extract_trajectories_v3.py - trajectory extraction on the new perturbed dataset (Vast.ai, RTX 3090).

WHAT IT DOES
  * Reads train_perturbed__2_.csv (columns: id, problem, answer, perturbation).
  * By default runs ALL variants of every problem (original, distraction statement, injected wrong answer, rename,
    rephrase, typos, unsolvable, value changed). `--types original` restricts to the unperturbed problems
    (the only thing the competition ever shows the model). Variants are processed problem by problem, so an
    interrupted run still has complete groups; `--problems N` limits to the first N base problems.
  * Greedy-generates up to 4096 new tokens per problem for each of the four small eval models and records the
    last-token hidden state of 6 layers every `stride` forward steps (plus step 1 and the final step).
  * Also records the generated text, the \\boxed{} answer, whether it is correct, and why generation stopped
    (eos / cap / loop) -> results.jsonl per model. 'correct' means: matches `answer` for every type except
    'unsolvable', where it means the model FLAGGED the problem as unsolvable (keyword heuristic, see `grading`).
    'injected wrong answer' rows also get injected_answer / followed_injected.  (With a 4096 cap many reasoning traces will stop at
    'cap' with no answer; that is expected and worth looking at.)

OUTPUT
  <output-dir>/<model>/<id>.npz       layer_XX [waypoints, hidden] float16, tokens_generated, stop_reason, correct
  <output-dir>/<model>/results.jsonl  one line per problem (deduplicate by "id" if a run was interrupted)
  <output-dir>/plan.csv               problems x models that were planned
  ./completed_4k/<model>.txt          resume manifest (written last, outside output-dir)

TWO MODELS AT ONCE (24 GB card)
  python3 extract_trajectories_v3.py --plan-parallel     # measures free VRAM, estimates which pairs fit
  python3 extract_trajectories_v3.py --parallel          # runs models in fitting groups as separate processes
  Without --parallel the models run one after another in this process.

Usage
  python3 -u extract_trajectories_v3.py --dry-run               # show the plan only
  python3 -u extract_trajectories_v3.py --problems 2 --models Qwen/Qwen3.5-4B   # smoke test: 2 problems x all variants, one model
  python3 -u extract_trajectories_v3.py --types original "distraction statement" --models Qwen/Qwen3-8B
  python3 -u extract_trajectories_v3.py                         # everything, sequentially
"""

import os

os.environ["HF_HUB_DISABLE_XET"] = "1"                                   # before huggingface_hub import
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # less fragmentation

import argparse
import fnmatch
import gc
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi, snapshot_download
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

LAYER_CONFIGS = {
    "Skywork/Skywork-OR1-Math-7B": [10, 13, 16, 19, 22, 25],
    "Qwen/Qwen3.5-4B": [10, 14, 18, 22, 26, 30],
    "allenai/Olmo-3-7B-Think": [10, 14, 18, 22, 26, 30],
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": [12, 16, 20, 24, 28, 32],
    "Qwen/Qwen3-8B": [12, 16, 20, 24, 28, 32],        # not an eval model; only with --models all
}
EVAL_MODELS = ["Qwen/Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B", "allenai/Olmo-3-7B-Think",
               "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"]

ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.model", "*.tiktoken", "*.jinja"]
IGNORE_PATTERNS = ["*.bin", "*.pt", "*.pth", "*.gguf", "*.h5", "*.msgpack", "*.onnx", "original/*", "consolidated*"]
PROMPT = "You are a precise math problem solver. Solve the given problem step by step, then output your final answer on the last line in the exact format:\nANSWER: <your answer>\nThis result should be an integer value or NaN if the given problem does not have a solution. Here is the problem {problem}"

USE_CHAT_TEMPLATE = True
DOWNLOAD_MARGIN_GB = 1.5
MIN_FREE_OUTPUT_GB = 0.5
GIB = 1024 ** 3
CUDA_CONTEXT_GIB = 0.6        # per-process CUDA context (estimate)
ACTIVATION_SLACK_GIB = 0.4    # activations / allocator slack per process (estimate)
PROMPT_ALLOWANCE_TOKENS = 600
FIT_MARGIN_GIB = 1.0          # keep this much of the card free when judging whether a group fits


def slug(model_id):
    return model_id.replace("/", "__")


# =========================================================================== disk helpers
def free_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1e9


def _wanted(name):
    return (any(fnmatch.fnmatch(name, p) for p in ALLOW_PATTERNS)
            and not any(fnmatch.fnmatch(name, p) for p in IGNORE_PATTERNS))


def remote_download_size_gb(model_id):
    try:
        info = HfApi().model_info(model_id, files_metadata=True)
        return sum((s.size or 0) for s in info.siblings if _wanted(s.rfilename)) / 1e9
    except Exception as e:
        print(f"  [warn] could not query model size for {model_id} ({e})")
        return None


# =========================================================================== VRAM planning
def gpu_memory_gib():
    """(total, used) in GiB via nvidia-smi - deliberately NOT torch, so this process holds no CUDA context."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout.strip().splitlines()[0]
    total, used = (float(x) for x in out.split(","))
    return total / 1024, used / 1024


def kv_bytes_per_token(model_id):
    """Rough KV-cache bytes per token (bf16). Sliding-window layers are counted as full attention."""
    cfg = AutoConfig.from_pretrained(model_id)
    t = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    n_layers = t.num_hidden_layers
    layer_types = getattr(t, "layer_types", None)
    n_attn = sum(1 for x in layer_types if "linear" not in str(x)) if layer_types else n_layers
    kv_heads = getattr(t, "num_key_value_heads", None) or t.num_attention_heads
    head_dim = getattr(t, "head_dim", None) or t.hidden_size // t.num_attention_heads
    return 2 * n_attn * kv_heads * head_dim * 2


def estimate_need_gib(model_id, max_new_tokens):
    w = remote_download_size_gb(model_id)
    if w is None:
        raise SystemExit(f"Cannot estimate size of {model_id} (no Hub access?).")
    kv = kv_bytes_per_token(model_id) * (max_new_tokens + PROMPT_ALLOWANCE_TOKENS) / GIB
    return {"weights": w * 1e9 / GIB, "kv": kv, "need": w * 1e9 / GIB + kv + CUDA_CONTEXT_GIB + ACTIVATION_SLACK_GIB}


def make_groups(needs, free_gib, max_parallel=2):
    """Greedy: biggest first, put each model in the first group whose total still fits."""
    groups = []
    for m in sorted(needs, key=needs.get, reverse=True):
        for g in groups:
            if len(g) < max_parallel and sum(needs[x] for x in g) + needs[m] <= free_gib - FIT_MARGIN_GIB:
                g.append(m)
                break
        else:
            groups.append([m])
    return groups


def plan_parallel(models, max_new_tokens, max_parallel=2, verbose=True):
    total, used = gpu_memory_gib()
    free = total - used
    est = {m: estimate_need_gib(m, max_new_tokens) for m in models}
    needs = {m: e["need"] for m, e in est.items()}
    if verbose:
        print(f"GPU: {total:.1f} GiB total, {used:.1f} GiB in use -> {free:.1f} GiB free "
              f"(keeping {FIT_MARGIN_GIB} GiB margin)\n")
        print(f"{'model':42s} {'weights':>8s} {'KV@cap':>7s} {'need':>7s}  (GiB, estimates)")
        for m, e in est.items():
            print(f"{m:42s} {e['weights']:8.1f} {e['kv']:7.2f} {e['need']:7.1f}")
        print()
        for i, a in enumerate(models):
            for b in models[i + 1:]:
                s = needs[a] + needs[b]
                print(f"  {a.split('/')[-1]} + {b.split('/')[-1]}: {s:5.1f} GiB -> "
                      f"{'FITS' if s <= free - FIT_MARGIN_GIB else 'does NOT fit'}")
    groups = make_groups(needs, free, max_parallel)
    if verbose:
        print("\nSuggested run order (models in a group run concurrently):")
        for k, g in enumerate(groups, 1):
            print(f"  group {k}: {g}")
    return groups


# =========================================================================== model loading
def load_model(model_id, cache_dir: Path, keep_weights):
    need = remote_download_size_gb(model_id)
    have = free_gb(cache_dir)
    print(f"[{model_id}] disk: {have:.1f} GB free; needs ~{need if need else '?'} GB")
    if need is not None and have < need + DOWNLOAD_MARGIN_GB:
        raise SystemExit(f"Not enough free disk for {model_id}: need ~{need + DOWNLOAD_MARGIN_GB:.1f} GB, have {have:.1f} GB.")
    local = snapshot_download(model_id, cache_dir=str(cache_dir), allow_patterns=ALLOW_PATTERNS,
                              ignore_patterns=IGNORE_PATTERNS, max_workers=4)
    tokenizer = AutoTokenizer.from_pretrained(local)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.pad_token
    try:
        model = AutoModelForCausalLM.from_pretrained(local, dtype=torch.bfloat16, device_map="cuda")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(local, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    if not keep_weights:
        shutil.rmtree(cache_dir, ignore_errors=True)
        print(f"[{model_id}] deleted weights from disk; free now {free_gb(cache_dir):.1f} GB")
    return model, tokenizer


def find_decoder_layers(model):
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    n = cfg.num_hidden_layers
    for name, mod in model.named_modules():
        if (isinstance(mod, torch.nn.ModuleList) and len(mod) == n and name.endswith("layers") and "vis" not in name):
            return mod, n
    raise RuntimeError("Could not locate decoder layers; print(model) and adjust find_decoder_layers().")


# =========================================================================== generation
class LoopStoppingCriteria(StoppingCriteria):
    def __init__(self, window_size=32, max_repeats=3):
        super().__init__()
        self.window_size, self.max_repeats, self.triggered = window_size, max_repeats, False

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        tokens = input_ids[0].tolist()
        if len(tokens) < self.window_size * self.max_repeats:
            return False
        recent = tokens[-self.window_size:]
        for r in range(1, self.max_repeats):
            if tokens[-(r + 1) * self.window_size:-r * self.window_size] != recent:
                return False
        self.triggered = True
        return True


def build_inputs(tokenizer, problem, device):
    prompt = PROMPT.format(problem=problem)
    if USE_CHAT_TEMPLATE and getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                             add_generation_prompt=True)
        return tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    return tokenizer(prompt, return_tensors="pt").to(device)


def to_store(vec):
    return vec.float().clamp(-65504, 65504).half().cpu().numpy()[0]


def extract_boxed(text):
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    start = j = i + len("\\boxed{")
    depth = 1
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[start:j - 1].strip() if depth == 0 else None


def _norm(s):
    s = str(s).strip().strip("$").replace(",", "").replace("\\!", "").replace("\\,", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    return re.sub(r"\s+", "", s).rstrip(".")


def answers_match(pred, truth):
    if pred is None:
        return False
    a, b = _norm(pred), _norm(truth)
    if a == b:
        return True
    try:
        return float(a) == float(b)
    except ValueError:
        return False


UNSOLVABLE_KEYS = ("unsolvable", "cannot be determined", "can't be determined", "not enough information",
                   "insufficient information", "missing information", "not enough data", "ill-posed",
                   "cannot be solved", "underdetermined", "cannot determine", "no unique")
INJECT_RE = re.compile(r"I think it(?:'s|s| is)\s*(-?\d+)", re.I)


def grade(ptype, boxed, text, truth, problem):
    """-> (correct, extra_fields). For 'unsolvable' rows, correct = the model flagged the problem (heuristic)."""
    if ptype == "unsolvable":
        tail = text[-800:].lower()
        flagged = (boxed is not None and "unsolvable" in boxed.lower()) or any(k in tail for k in UNSOLVABLE_KEYS)
        return bool(flagged), {"grading": "heuristic: model flagged the problem as unsolvable"}
    correct = answers_match(boxed, truth)
    extra = {}
    if ptype == "injected wrong answer":
        m = INJECT_RE.search(problem)
        if m:
            extra = {"injected_answer": m.group(1), "followed_injected": answers_match(boxed, m.group(1))}
    return bool(correct), extra


def extract_one(model, tokenizer, layers_module, target_layers, problem, stride, max_new_tokens):
    lead = target_layers[0]
    state = {"step": 0}
    latest, recorded = {}, {l: [] for l in target_layers}
    last_rec = {l: 0 for l in target_layers}

    def make_hook(l):
        def hook(module, inp, outp):
            hidden = outp[0] if isinstance(outp, tuple) else outp
            vec = hidden[:, -1, :].detach()
            if l == lead:
                state["step"] += 1
            latest[l] = vec
            s = state["step"]
            if s == 1 or s % stride == 0:
                recorded[l].append(to_store(vec))
                last_rec[l] = s
        return hook

    handles = [layers_module[l].register_forward_hook(make_hook(l)) for l in target_layers]
    stopper = LoopStoppingCriteria()
    t0 = time.time()
    try:
        inputs = build_inputs(tokenizer, problem, model.device)
        n_prompt = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id,
                                 stopping_criteria=StoppingCriteriaList([stopper]))
    finally:
        for h in handles:
            h.remove()
    seconds = time.time() - t0

    for l in target_layers:
        if l in latest and last_rec[l] != state["step"]:
            recorded[l].append(to_store(latest[l]))

    n_new = int(out.shape[1] - n_prompt)
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) | {tokenizer.eos_token_id}
    if stopper.triggered:
        reason = "loop"
    elif int(out[0, -1]) in eos_ids:
        reason = "eos"
    elif n_new >= max_new_tokens:
        reason = "cap"
    else:
        reason = "other"
    text = tokenizer.decode(out[0, n_prompt:], skip_special_tokens=True)
    arrays = {f"layer_{l:02d}": np.stack(recorded[l]) for l in target_layers}
    return arrays, {"tokens": n_new, "stop_reason": reason, "text": text, "boxed": extract_boxed(text),
                    "seconds": round(seconds, 1), "forward_steps": state["step"]}


def save_atomic(path: Path, data: dict):
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **data)
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


# =========================================================================== dataset / plan
def _ptype(label):
    return re.sub(r"\s*\(.*\)", "", str(label)).strip().lower()


def load_dataset(path, types, problems=None, limit=None):
    df = pd.read_csv(path, dtype=str)
    df["ptype"] = df["perturbation"].map(_ptype)
    df["base_id"] = df["id"].str.split(".").str[0]
    df["variant"] = df["id"].str.split(".").str[1].fillna("0").astype(int)
    df["base_n"] = pd.to_numeric(df["base_id"], errors="coerce")
    if types and [t.lower() for t in types] != ["all"]:
        want = [t.lower().replace("_", " ") for t in types]
        df = df[df["ptype"].isin(want)]
    if problems:
        keep = sorted(df["base_n"].dropna().unique())[:problems]
        df = df[df["base_n"].isin(keep)]
    df = df.sort_values(["base_n", "variant"])
    if limit:
        df = df.head(limit)
    return df.rename(columns={"id": "problem_id"})[["problem_id", "base_id", "ptype", "problem", "answer", "perturbation"]]


def load_manifest(path: Path):
    return set(path.read_text().split()) if path.exists() else set()


# =========================================================================== per-model run
def run_model(model_id, df, args):
    out_dir = Path(args.output_dir) / slug(model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = Path(args.work_dir) / "completed_4k" / f"{slug(model_id)}.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    done = load_manifest(manifest)
    todo = [r for r in df.itertuples(index=False) if r.problem_id not in done]
    print(f"\n=== {model_id}: {len(todo)} pending of {len(df)} problems ===", flush=True)
    if not todo:
        return

    cache_dir = Path(args.work_dir) / "hf_tmp" / slug(model_id)     # per model: workers never delete each other's files
    model, tokenizer = load_model(model_id, cache_dir, args.keep_weights)
    try:
        layers_module, n_layers = find_decoder_layers(model)
        target_layers = LAYER_CONFIGS[model_id]
        assert max(target_layers) < n_layers, f"layer index out of range (model has {n_layers})"
        results_path = out_dir / "results.jsonl"
        for k, r in enumerate(todo, 1):
            if free_gb(out_dir) < MIN_FREE_OUTPUT_GB:
                print(f"Free disk below {MIN_FREE_OUTPUT_GB} GB - stopping. Move results off and rerun.")
                break
            arrays, meta = extract_one(model, tokenizer, layers_module, target_layers, r.problem,
                                       args.stride, args.max_new_tokens)
            correct, extra = grade(r.ptype, meta["boxed"], meta["text"], r.answer, r.problem)
            save_atomic(out_dir / f"{r.problem_id}.npz",
                        {**arrays, "tokens_generated": meta["tokens"], "forward_steps": meta["forward_steps"],
                         "stop_reason": np.array(meta["stop_reason"]), "correct": bool(correct)})
            with open(results_path, "a") as f:
                f.write(json.dumps({"id": r.problem_id, "base_id": r.base_id, "ptype": r.ptype, "answer": r.answer,
                                    "perturbation": r.perturbation, "correct": bool(correct), **extra, **meta}) + "\n")
            with open(manifest, "a") as mf:
                mf.write(f"{r.problem_id}\n")
            print(f"[{model_id.split('/')[-1]} {k}/{len(todo)}] id={r.problem_id} tokens={meta['tokens']} "
                  f"[{r.ptype}] stop={meta['stop_reason']} boxed={meta['boxed']!r} truth={r.answer} correct={correct} "
                  f"({meta['seconds']}s, {meta['tokens'] / max(meta['seconds'], 1e-9):.0f} tok/s)", flush=True)
            if k % 10 == 0:
                torch.cuda.empty_cache()
    finally:
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        if not args.keep_weights:
            shutil.rmtree(cache_dir, ignore_errors=True)


# =========================================================================== main
def run_parallel_groups(groups, args, passthrough):
    Path("logs").mkdir(exist_ok=True)
    for gi, group in enumerate(groups, 1):
        print(f"\n##### group {gi}/{len(groups)}: {group}", flush=True)
        procs = []
        for m in group:
            log = open(f"logs/{slug(m)}.log", "a")
            cmd = [sys.executable, "-u", __file__, "--models", m, "--worker"] + passthrough
            procs.append((m, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
            time.sleep(20)          # stagger loads so the two models don't allocate at the same instant
        for m, p, log in procs:
            rc = p.wait()
            log.close()
            print(f"  {m}: exit code {rc}  (log: logs/{slug(m)}.log)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="train_perturbed__2_.csv")
    ap.add_argument("--types", nargs="+", default=["all"],
                    help="perturbation types to run, e.g. original \"distraction statement\" typos (default: all)")
    ap.add_argument("--problems", type=int, default=None, help="only the first N base problems (all their variants)")
    ap.add_argument("--models", nargs="+", default=["eval"], help="HF ids, 'eval' (the four) or 'all'")
    ap.add_argument("--output-dir", default="saved_trajectories_4k")
    ap.add_argument("--work-dir", default=".")
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None, help="only the first N rows (after --problems)")
    ap.add_argument("--keep-weights", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--plan-parallel", action="store_true", help="estimate which models fit on the GPU together; exit")
    ap.add_argument("--parallel", action="store_true", help="run models in VRAM-fitting groups as parallel processes")
    ap.add_argument("--max-parallel", type=int, default=2)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    models = (EVAL_MODELS if args.models == ["eval"] else list(LAYER_CONFIGS) if args.models == ["all"] else args.models)
    for m in models:
        if m not in LAYER_CONFIGS:
            raise SystemExit(f"Unknown model {m}; add it to LAYER_CONFIGS.")

    if args.plan_parallel:
        plan_parallel(models, args.max_new_tokens, args.max_parallel)
        return

    df = load_dataset(args.dataset, args.types, args.problems, args.limit)
    print(f"{len(df)} rows from {df['base_id'].nunique()} base problems (types={args.types}); "
          f"models={[m.split('/')[-1] for m in models]}")
    if not args.worker:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"model_id": m, "problem_id": r.problem_id, "base_id": r.base_id, "ptype": r.ptype,
                       "answer": r.answer} for m in models
                      for r in df.itertuples(index=False)]).to_csv(Path(args.output_dir) / "plan.csv", index=False)
    if args.dry_run:
        print(df.head(3).to_string())
        return

    if args.parallel and not args.worker:
        groups = plan_parallel(models, args.max_new_tokens, args.max_parallel)
        passthrough = ["--dataset", args.dataset, "--types", *args.types, "--output-dir", args.output_dir,
                       "--work-dir", args.work_dir, "--stride", str(args.stride),
                       "--max-new-tokens", str(args.max_new_tokens)]
        if args.problems:
            passthrough += ["--problems", str(args.problems)]
        if args.limit:
            passthrough += ["--limit", str(args.limit)]
        if args.keep_weights:
            passthrough += ["--keep-weights"]
        run_parallel_groups(groups, args, passthrough)
        return

    for m in models:
        run_model(m, df, args)
    print(f"\nDone. Free disk: {free_gb(Path(args.output_dir)):.1f} GB")


if __name__ == "__main__":
    main()
