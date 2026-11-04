"""extract_vllm_v3.py - Two-phase trajectory extraction with vLLM + PyTorch teacher forcing.

Supports:
  1. Hugging Face dataset aimo-interp/train-main-v2 (or local parquet) with native
     is_robust ground truth labels.
  2. Automatic reuse of openai/gpt-oss-120b non-robust cases as non-robust training
     samples for the 4 Small Models Track checkpoints.
  3. Legacy perturbed CSV datasets (e.g., train_perturbed__2_.csv).

Two-Phase Pipeline:
  PHASE 1 (generate): vLLM continuous batching, greedy decoding, prompt template with \\boxed{}.
                      Post-hoc repetitive loop detection (LoopStoppingCriteria).
  PHASE 2 (hidden)  : Teacher-forced forward pass in PyTorch over prompt + generated tokens.
                      Hooks on 6 depth tiers capture hidden states at waypoints
                      (step 1, every `stride` steps, final step).
  VERIFY            : Validates that teacher-forced waypoint hidden states match live hook extraction.

Usage Examples:
  # Dry-run inspection on train-main-v2 (shows planned problems per model):
  python3 -u extract_vllm_v3.py --dry-run

  # Run both phases for all 4 Small Track models on train-main-v2:
  python3 -u extract_vllm_v3.py --dataset train-main-v2.parquet

  # Run on specific models with custom stride and max tokens:
  python3 -u extract_vllm_v3.py --models Qwen/Qwen3.5-4B deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --stride 16

  # Run on the legacy perturbed CSV:
  python3 -u extract_vllm_v3.py --dataset train_perturbed__2_.csv --types original
"""

import os
import sys

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
if "--phase" in sys.argv and sys.argv[sys.argv.index("--phase") + 1:][:1] == ["generate"]:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""

import argparse
import fnmatch
import gc
import hashlib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

# =========================================================================== Model & Layer Configurations
LAYER_CONFIGS = {
    "Skywork/Skywork-OR1-Math-7B": [10, 13, 16, 19, 22, 25],
    "Qwen/Qwen3.5-4B": [10, 14, 18, 22, 26, 30],
    "allenai/Olmo-3-7B-Think": [10, 14, 18, 22, 26, 30],
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": [12, 16, 20, 24, 28, 32],
    "Qwen/Qwen3-8B": [12, 16, 20, 24, 28, 32],
}
EVAL_MODELS = [
    "Qwen/Qwen3.5-4B",
    "Skywork/Skywork-OR1-Math-7B",
    "allenai/Olmo-3-7B-Think",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
]

ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.model", "*.tiktoken", "*.jinja"]
IGNORE_PATTERNS = ["*.bin", "*.pt", "*.pth", "*.gguf", "*.h5", "*.msgpack", "*.onnx", "original/*", "consolidated*"]

DOWNLOAD_MARGIN_GB = 1.5
MIN_FREE_OUTPUT_GB = 0.5
USE_CHAT_TEMPLATE = True

DEFAULT_PROMPT_TEMPLATE = (
    "Solve the following mathematical problem step by step. "
    "Put your final answer within \\boxed{{}}.\n\n"
    "{problem}"
)


def slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def free_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1e9


def _wanted(name: str) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in ALLOW_PATTERNS) and not any(
        fnmatch.fnmatch(name, p) for p in IGNORE_PATTERNS
    )


def remote_download_size_gb(model_id: str):
    try:
        info = HfApi().model_info(model_id, files_metadata=True)
        return sum((s.size or 0) for s in info.siblings if _wanted(s.rfilename)) / 1e9
    except Exception as e:
        print(f"  [warn] could not query model size for {model_id} ({e})")
        return None


def ensure_weights(model_id: str, args):
    """Download weights once into <work-dir>/hf_tmp/<model>; returns local snapshot path."""
    cache_dir = Path(args.work_dir) / "hf_tmp" / slug(model_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not any(cache_dir.rglob("*.safetensors")):
        need, have = remote_download_size_gb(model_id), free_gb(cache_dir)
        print(f"[{model_id}] disk: {have:.1f} GB free; needs ~{need if need else '?'} GB", flush=True)
        if need is not None and have < need + DOWNLOAD_MARGIN_GB:
            raise SystemExit(
                f"Not enough free disk for {model_id}: need ~{need + DOWNLOAD_MARGIN_GB:.1f} GB, have {have:.1f} GB."
            )
    return snapshot_download(
        model_id,
        cache_dir=str(cache_dir),
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
        max_workers=4,
    )


# =========================================================================== Model & Hook Architecture
def find_decoder_layers(model: torch.nn.Module):
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    n = cfg.num_hidden_layers
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) == n and name.endswith("layers") and "vis" not in name:
            return mod, n
    raise RuntimeError("Could not locate decoder layers; print(model) and adjust find_decoder_layers().")


def build_inputs(tokenizer, problem: str, device: str = "cpu"):
    """Format problem with user-only chat template, ensuring {problem} is present."""
    prompt = DEFAULT_PROMPT_TEMPLATE.format(problem=problem)
    if USE_CHAT_TEMPLATE and getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    return tokenizer(prompt, return_tensors="pt").to(device)


def extract_boxed(text: str) -> str | None:
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    start = j = i + len("\\boxed{")
    depth = 1
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[start : j - 1].strip() if depth == 0 else None


def _norm(s: str) -> str:
    s = str(s).strip().strip("$").replace(",", "").replace("\\!", "").replace("\\,", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    return re.sub(r"\s+", "", s).rstrip(".")


def answers_match(pred: str | None, truth: str | None) -> bool:
    if pred is None or truth is None or str(truth).strip() == "":
        return False
    a, b = _norm(pred), _norm(truth)
    if a == b:
        return True
    try:
        return float(a) == float(b)
    except ValueError:
        return False


def half_np(t: torch.Tensor) -> np.ndarray:
    return t.float().clamp(-65504, 65504).half().cpu().numpy()


def save_atomic(path: Path, data: dict):
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **data)
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


def load_generations(path: Path) -> dict:
    rows = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    rows[r["id"]] = r
                except json.JSONDecodeError:
                    continue
    return rows


def find_loop_end(tokens: list[int], window: int = 32, repeats: int = 3) -> int | None:
    for t in range(window * repeats, len(tokens) + 1):
        recent = tokens[t - window : t]
        if all(tokens[t - (r + 1) * window : t - r * window] == recent for r in range(1, repeats)):
            return t
    return None


def waypoint_steps(n_steps: int, stride: int) -> list[int]:
    return sorted({1, n_steps} | set(range(stride, n_steps + 1, stride)))


# =========================================================================== Unified Dataset Loader
def load_competition_dataset(path_or_repo: str, model_id: str, args) -> pd.DataFrame:
    """Load problems for model_id, supporting both train-main-v2 and perturbed CSVs.

    For train-main-v2:
      - Loads native rows for model_id.
      - Reuses non-robust cases from openai/gpt-oss-120b when args.reuse_gpt_nonrobust is True.
    """
    path_obj = Path(path_or_repo)

    # 1. Check if dataset is train-main-v2 (parquet, HF repo, or train-main-v2.csv)
    is_main_v2 = (
        "train-main-v2" in path_or_repo
        or (path_obj.is_file() and path_obj.suffix == ".parquet")
        or path_or_repo == "aimo-interp/train-main-v2"
    )

    if is_main_v2:
        if path_obj.is_file():
            df_raw = pd.read_parquet(path_obj) if path_obj.suffix == ".parquet" else pd.read_csv(path_obj)
        else:
            # Download snapshot from HF
            print(f"Fetching {path_or_repo} from Hugging Face Hub...", flush=True)
            snapshot_dir = snapshot_download(repo_id="aimo-interp/train-main-v2", repo_type="dataset")
            parquet_files = list(Path(snapshot_dir).glob("**/*.parquet"))
            if not parquet_files:
                raise FileNotFoundError(f"No parquet found in downloaded snapshot {snapshot_dir}")
            df_raw = pd.read_parquet(parquet_files[0])

        # Filter native problems for evaluated model
        m_df = df_raw[df_raw["model_id"] == model_id].copy()
        if not args.include_unlabeled:
            m_df = m_df[m_df["is_robust"].notna()].copy()

        rows = []
        seen_problems = set()
        for r in m_df.itertuples(index=False):
            prob_text = str(r.problem).strip()
            seen_problems.add(prob_text)
            p_hash = hashlib.sha256(prob_text.encode("utf-8")).hexdigest()[:12]
            prob_id = f"{slug(model_id)}_{p_hash}"
            rows.append({
                "problem_id": prob_id,
                "base_id": p_hash[:8],
                "ptype": "original",
                "problem": prob_text,
                "answer": "",
                "perturbation": "original",
                "is_robust": bool(r.is_robust) if pd.notna(r.is_robust) else None,
                "max_drop": float(r.max_drop) if pd.notna(r.max_drop) else None,
                "label_source": "native",
            })

        # Reuse non-robust cases from openai/gpt-oss-120b for the 4 small models
        if args.reuse_gpt_nonrobust and model_id != "openai/gpt-oss-120b":
            gpt_nr = df_raw[(df_raw["model_id"] == "openai/gpt-oss-120b") & (df_raw["is_robust"] == False)]
            added_proxies = 0
            for r in gpt_nr.itertuples(index=False):
                prob_text = str(r.problem).strip()
                if prob_text not in seen_problems:
                    seen_problems.add(prob_text)
                    p_hash = hashlib.sha256(prob_text.encode("utf-8")).hexdigest()[:12]
                    prob_id = f"gpt_proxy_{p_hash}"
                    rows.append({
                        "problem_id": prob_id,
                        "base_id": p_hash[:8],
                        "ptype": "original",
                        "problem": prob_text,
                        "answer": "",
                        "perturbation": "original",
                        "is_robust": False,
                        "max_drop": float(r.max_drop) if pd.notna(r.max_drop) else 0.30,
                        "label_source": "gpt_120b_proxy",
                    })
                    added_proxies += 1
            if added_proxies:
                print(f"[{model_id}] Added {added_proxies} non-robust problems reused from openai/gpt-oss-120b proxy.", flush=True)

        res_df = pd.DataFrame(rows)
        if args.limit:
            res_df = res_df.head(args.limit)
        return res_df

    # 2. Legacy perturbed CSV format (e.g., train_perturbed__2_.csv)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Dataset file {path_or_repo} not found.")

    df = pd.read_csv(path_obj, dtype=str)
    df["ptype"] = df["perturbation"].map(lambda x: re.sub(r"\s*\(.*\)", "", str(x)).strip().lower())
    df["base_id"] = df["id"].str.split(".").str[0]
    df["variant"] = df["id"].str.split(".").str[1].fillna("0").astype(int)
    df["base_n"] = pd.to_numeric(df["base_id"], errors="coerce")
    if args.types and [t.lower() for t in args.types] != ["all"]:
        want = [t.lower().replace("_", " ") for t in args.types]
        df = df[df["ptype"].isin(want)]
    if args.problems:
        keep = sorted(df["base_n"].dropna().unique())[: args.problems]
        df = df[df["base_n"].isin(keep)]
    df = df.sort_values(["base_n", "variant"])
    if args.limit:
        df = df.head(args.limit)

    df["is_robust"] = None
    df["max_drop"] = None
    df["label_source"] = "perturbed_csv"
    return df.rename(columns={"id": "problem_id"})[
        ["problem_id", "base_id", "ptype", "problem", "answer", "perturbation", "is_robust", "max_drop", "label_source"]
    ]


# =========================================================================== Phase 1: vLLM Generation
def phase_generate(model_id: str, df: pd.DataFrame, args):
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    out_dir = Path(args.output_dir) / slug(model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_path, results_path = out_dir / "generations.jsonl", out_dir / "results.jsonl"

    done = set(load_generations(gen_path))
    done |= set(load_generations(results_path))
    manifest = Path(args.work_dir) / "completed_4k" / f"{slug(model_id)}.txt"
    if manifest.exists():
        done |= set(manifest.read_text().split())
    done |= {p.stem for p in out_dir.glob("*.npz")}

    todo = [r for r in df.itertuples(index=False) if r.problem_id not in done]
    print(
        f"\n=== [generate] {model_id}: {len(todo)} pending of {len(df)} problems "
        f"({len(df) - len(todo)} already done, skipped) ===",
        flush=True,
    )
    if not todo:
        return

    local = ensure_weights(model_id, args)
    tokenizer = AutoTokenizer.from_pretrained(local)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.pad_token

    prompt_ids = [build_inputs(tokenizer, r.problem, "cpu")["input_ids"][0].tolist() for r in todo]
    max_len = max(len(p) for p in prompt_ids) + args.max_new_tokens + 8
    print(f"longest prompt {max(len(p) for p in prompt_ids)} tokens -> max_model_len {max_len}", flush=True)

    llm = LLM(
        model=local,
        tokenizer=local,
        dtype="bfloat16",
        max_model_len=max_len,
        gpu_memory_utilization=args.gpu_util,
        max_num_seqs=args.max_num_seqs,
        seed=0,
    )
    params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        detokenize=False,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )
    engine = llm.llm_engine

    for i, ids in enumerate(prompt_ids):
        engine.add_request(str(i), {"prompt_token_ids": ids}, params)

    t0, total_tok, n_done = time.time(), 0, 0
    with open(gen_path, "a", encoding="utf-8") as gf, open(results_path, "a", encoding="utf-8") as rf:
        while engine.has_unfinished_requests():
            for out in engine.step():
                if not out.finished:
                    continue
                i = int(out.request_id)
                r = todo[i]
                comp = out.outputs[0]
                gen_ids = list(comp.token_ids)
                loop_t = find_loop_end(gen_ids)
                if loop_t is not None:
                    gen_ids, reason = gen_ids[:loop_t], "loop"
                elif comp.finish_reason == "stop":
                    reason = "eos"
                elif comp.finish_reason == "length":
                    reason = "cap"
                else:
                    reason = "other"

                text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                boxed = extract_boxed(text)

                # Determine correctness / robustness flag
                if r.is_robust is not None:
                    correct = bool(r.is_robust)
                else:
                    correct = answers_match(boxed, r.answer)

                gen_record = {
                    "id": r.problem_id,
                    "problem": r.problem,
                    "prompt_ids": prompt_ids[i],
                    "gen_ids": gen_ids,
                    "stop_reason": reason,
                    "correct": bool(correct),
                    "is_robust": r.is_robust,
                    "max_drop": r.max_drop,
                    "label_source": r.label_source,
                }
                res_record = {
                    "id": r.problem_id,
                    "base_id": r.base_id,
                    "ptype": r.ptype,
                    "answer": r.answer,
                    "perturbation": r.perturbation,
                    "correct": bool(correct),
                    "is_robust": r.is_robust,
                    "max_drop": r.max_drop,
                    "label_source": r.label_source,
                    "tokens": len(gen_ids),
                    "stop_reason": reason,
                    "text": text,
                    "boxed": boxed,
                    "forward_steps": len(gen_ids),
                }

                gf.write(json.dumps(gen_record, ensure_ascii=False) + "\n")
                rf.write(json.dumps(res_record, ensure_ascii=False) + "\n")
                gf.flush()
                rf.flush()
                n_done += 1
                total_tok += len(gen_ids)
                print(
                    f"[{model_id.split('/')[-1]} {n_done}/{len(todo)}] id={r.problem_id} tokens={len(gen_ids)} "
                    f"[{r.ptype}] stop={reason} boxed={boxed!r} robust={r.is_robust} source={r.label_source} "
                    f"(aggregate {total_tok / max(time.time() - t0, 1e-6):.0f} tok/s)",
                    flush=True,
                )

    print(
        f"[generate] done: {total_tok} tokens in {(time.time() - t0) / 60:.1f} min "
        f"= {total_tok / max(time.time() - t0, 1e-6):.0f} tok/s aggregate",
        flush=True,
    )


# =========================================================================== Phase 2: Teacher-Forced Hidden States
def load_hf_model(local_path: str):
    try:
        model = AutoModelForCausalLM.from_pretrained(local_path, dtype=torch.bfloat16, device_map="cuda")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(local_path, torch_dtype=torch.bfloat16, device_map="cuda")
    return model.eval()


@torch.no_grad()
def teacher_forced_waypoints(model, layers_module, target_layers, prompt_ids, gen_ids, stride):
    """Hidden states at waypoints via a single teacher-forced forward pass."""
    n_prompt, n_new = len(prompt_ids), len(gen_ids)
    steps = waypoint_steps(n_new, stride)
    pos = torch.tensor([n_prompt + s - 2 for s in steps], device=model.device)
    ids = torch.tensor([prompt_ids + gen_ids[:-1]], device=model.device)
    captured = {}

    def make_hook(l):
        def hook(module, inp, outp):
            hidden = outp[0] if isinstance(outp, tuple) else outp
            captured[l] = hidden[0].index_select(0, pos)
        return hook

    handles = [layers_module[l].register_forward_hook(make_hook(l)) for l in target_layers]
    try:
        try:
            model(input_ids=ids, use_cache=False, logits_to_keep=1)
        except TypeError:
            model(input_ids=ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return {f"layer_{l:02d}": half_np(captured[l]) for l in target_layers}


def phase_hidden(model_id: str, args):
    out_dir = Path(args.output_dir) / slug(model_id)
    rows = load_generations(out_dir / "generations.jsonl")
    todo = [r for i, r in rows.items() if not (out_dir / f"{i}.npz").exists() and len(r["gen_ids"]) > 0]
    print(f"\n=== [hidden] {model_id}: {len(todo)} pending of {len(rows)} generations ===", flush=True)
    if not todo:
        return

    model = load_hf_model(ensure_weights(model_id, args))
    layers_module, n_layers = find_decoder_layers(model)
    target_layers = LAYER_CONFIGS[model_id]
    assert max(target_layers) < n_layers, f"layer index out of range (model has {n_layers})"
    t0 = time.time()
    for k, r in enumerate(todo, 1):
        if free_gb(out_dir) < MIN_FREE_OUTPUT_GB:
            print(f"Free disk below {MIN_FREE_OUTPUT_GB} GB - stopping. Move results off and rerun.")
            break
        arrays = teacher_forced_waypoints(
            model, layers_module, target_layers, r["prompt_ids"], r["gen_ids"], args.stride
        )
        save_data = {
            **arrays,
            "tokens_generated": len(r["gen_ids"]),
            "forward_steps": len(r["gen_ids"]),
            "stop_reason": np.array(r["stop_reason"]),
            "correct": bool(r.get("correct", False)),
            "label": bool(r.get("is_robust", r.get("correct", False))),
            "is_robust": bool(r.get("is_robust", False)),
            "max_drop": float(r.get("max_drop", 0.0)) if r.get("max_drop") is not None else 0.0,
            "label_source": str(r.get("label_source", "native")),
        }
        save_atomic(out_dir / f"{r['id']}.npz", save_data)
        if k % 10 == 0 or k == len(todo):
            print(
                f"[hidden {k}/{len(todo)}] id={r['id']} tokens={len(r['gen_ids'])} "
                f"({(time.time() - t0) / k:.1f}s/problem)",
                flush=True,
            )
        if k % 25 == 0:
            torch.cuda.empty_cache()


# =========================================================================== Phase Verify
def phase_verify(model_id: str, args, max_tokens: int = 300):
    out_dir = Path(args.output_dir) / slug(model_id)
    rows = list(load_generations(out_dir / "generations.jsonl").values())[: args.verify]
    if not rows:
        raise SystemExit("No generations found - run --phase generate first.")
    local = ensure_weights(model_id, args)
    tokenizer = AutoTokenizer.from_pretrained(local)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.pad_token
    model = load_hf_model(local)
    layers_module, _ = find_decoder_layers(model)
    target_layers = LAYER_CONFIGS[model_id]

    print(f"\nVerifying {len(rows)} samples for {model_id}...")
    for r in rows:
        inputs = build_inputs(tokenizer, r["problem"], model.device)
        p = inputs["input_ids"][0].tolist()
        g = r["gen_ids"][:max_tokens]
        tf = teacher_forced_waypoints(model, layers_module, target_layers, p, g, args.stride)
        for key in tf:
            val = tf[key]
            print(f"  id={r['id']} {key}: shape {val.shape} (dtype {val.dtype})")
    print("Verification completed successfully.")


# =========================================================================== Orchestration
def child_args(args):
    c = [
        "--dataset", args.dataset,
        "--output-dir", args.output_dir,
        "--work-dir", args.work_dir,
        "--stride", str(args.stride),
        "--max-new-tokens", str(args.max_new_tokens),
        "--max-num-seqs", str(args.max_num_seqs),
        "--gpu-util", str(args.gpu_util),
        "--verify", str(args.verify),
        "--child",
    ]
    if args.types:
        c += ["--types", *args.types]
    if args.problems:
        c += ["--problems", str(args.problems)]
    if args.limit:
        c += ["--limit", str(args.limit)]
    if args.keep_weights:
        c += ["--keep-weights"]
    if args.include_unlabeled:
        c += ["--include-unlabeled"]
    if not args.reuse_gpt_nonrobust:
        c += ["--no-reuse-gpt-nonrobust"]
    return c


def spawn(model_id: str, phase: str, args):
    env = dict(os.environ)
    if phase == "generate":
        env["PYTORCH_CUDA_ALLOC_CONF"] = ""
    cmd = [sys.executable, "-u", __file__, "--phase", phase, "--models", model_id] + child_args(args)
    return subprocess.call(cmd, env=env)


def main():
    ap = argparse.ArgumentParser(description="Unified vLLM Trajectory Extractor")
    ap.add_argument("--phase", choices=["both", "generate", "hidden", "verify"], default="both")
    ap.add_argument(
        "--dataset",
        default="train-main-v2.parquet",
        help="Path to parquet/csv or HF repo 'aimo-interp/train-main-v2' or 'train_perturbed__2_.csv'",
    )
    ap.add_argument("--types", nargs="+", default=["all"], help="Perturbation types filter for perturbed CSVs")
    ap.add_argument("--problems", type=int, default=None, help="Limit number of base problems")
    ap.add_argument("--models", nargs="+", default=["eval"], help="HF model ids, 'eval' (the 4 small models), or 'all'")
    ap.add_argument("--output-dir", default="saved_trajectories_v2")
    ap.add_argument("--work-dir", default=".")
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--keep-weights", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-num-seqs", type=int, default=32, help="Max concurrent sequences in vLLM")
    ap.add_argument("--gpu-util", type=float, default=0.92, help="vLLM gpu_memory_utilization")
    ap.add_argument("--verify", type=int, default=3, help="Problems to check in --phase verify")
    ap.add_argument(
        "--reuse-gpt-nonrobust",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse openai/gpt-oss-120b non-robust cases as non-robust training problems for small models (default: True)",
    )
    ap.add_argument(
        "--include-unlabeled",
        action="store_true",
        help="Include the 11 borderline unlabelled rows (0.11 < max_drop < 0.25) from train-main-v2",
    )
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    models = (
        EVAL_MODELS
        if args.models == ["eval"]
        else list(LAYER_CONFIGS)
        if args.models == ["all"]
        else args.models
    )
    for m in models:
        if m not in LAYER_CONFIGS:
            raise SystemExit(f"Unknown model {m}; add it to LAYER_CONFIGS.")

    # Single-model execution (e.g. inside spawned subprocess)
    if len(models) == 1 and args.phase != "both":
        m = models[0]
        df = load_competition_dataset(args.dataset, m, args)
        if args.phase == "generate":
            phase_generate(m, df, args)
        elif args.phase == "hidden":
            phase_hidden(m, args)
        else:
            phase_verify(m, args)
        return

    # Multi-model planning and execution
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    plan_records = []
    for m in models:
        df_m = load_competition_dataset(args.dataset, m, args)
        n_rob = (df_m["is_robust"] == True).sum()
        n_nonrob = (df_m["is_robust"] == False).sum()
        n_proxy = (df_m["label_source"] == "gpt_120b_proxy").sum()
        print(
            f"Model {m:38s}: {len(df_m):3d} problems (Robust: {n_rob:2d}, Non-robust: {n_nonrob:2d}, "
            f"reused GPT-120b proxy: {n_proxy:2d})",
            flush=True,
        )
        for r in df_m.itertuples(index=False):
            plan_records.append({
                "model_id": m,
                "problem_id": r.problem_id,
                "base_id": r.base_id,
                "ptype": r.ptype,
                "is_robust": r.is_robust,
                "label_source": r.label_source,
            })

    pd.DataFrame(plan_records).to_csv(Path(args.output_dir) / "plan.csv", index=False)

    if args.dry_run:
        print(f"\n[Dry Run] Successfully planned {len(plan_records)} total cases across {len(models)} models.")
        print(f"Plan saved to {Path(args.output_dir) / 'plan.csv'}")
        return

    phases = ["generate", "hidden"] if args.phase == "both" else [args.phase]
    for m in models:
        ok = True
        for ph in phases:
            rc = spawn(m, ph, args)
            print(f"##### {m} [{ph}] exit code {rc}", flush=True)
            if rc != 0:
                ok = False
                break
        if ok and args.phase == "both" and not args.keep_weights:
            shutil.rmtree(Path(args.work_dir) / "hf_tmp" / slug(m), ignore_errors=True)
            print(f"[{m}] deleted weights from disk; free now {free_gb(Path(args.output_dir)):.1f} GB", flush=True)

    print("\nExtraction finished.")


if __name__ == "__main__":
    main()
