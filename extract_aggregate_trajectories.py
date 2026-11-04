#!/usr/bin/env python3
"""extract_aggregate_trajectories.py - Trajectory extraction for the aggregate math dataset
(augmented-sample-math-agg.csv: 141 rows, 137 unique problem_ids), with vLLM generation.

Two-phase pipeline (default backend: vllm):
  PHASE 1 (generate): vLLM continuous batching, greedy decoding, chat-templated prompt asking
                      for \\boxed{}. Post-hoc repetitive-loop truncation. Writes results.jsonl
                      and generations.jsonl (prompt_ids / gen_ids) incrementally.
  PHASE 2 (hidden)  : Teacher-forced forward pass in PyTorch over prompt + generated tokens.
                      Hooks on the depth tiers capture hidden states at waypoints
                      (step 1, every `stride` steps, final step) -> <problem_id>.npz.
  VERIFY            : Generates a short live trajectory with HF hooks, teacher-forces the same
                      tokens, and checks the waypoint hidden states agree.

--backend hf keeps the original single-process path (HF generate + live hooks) for comparison.
vLLM and the HF model both want the GPU, so `--phase both` runs each phase in its own subprocess.

Output Layout:
  <output_dir>/<model_slug>/<problem_id>.npz     # float16 layer arrays + labels (completion marker)
  <output_dir>/<model_slug>/results.jsonl        # Decoded reasoning text, boxed answer, stop reason
  <output_dir>/<model_slug>/generations.jsonl    # Raw prompt_ids and gen_ids for teacher-forcing

Usage Examples:
  # 1. Preview the plan and check dataset integrity (no GPU/model loading):
  python3 -u extract_aggregate_trajectories.py --dry-run

  # 2. Full run (vLLM generate, then teacher-forced hidden states):
  python3 -u extract_aggregate_trajectories.py --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B

  # 3. Run phases separately (e.g. generate now, hidden states later):
  python3 -u extract_aggregate_trajectories.py --phase generate --max-num-seqs 64
  python3 -u extract_aggregate_trajectories.py --phase hidden

  # 4. Sanity-check hook alignment between live extraction and teacher forcing:
  python3 -u extract_aggregate_trajectories.py --phase verify

  # 5. Smoke test on 2 problems:
  python3 -u extract_aggregate_trajectories.py --limit 2 --max-new-tokens 2048

  # 6. Original HF-only path:
  python3 -u extract_aggregate_trajectories.py --backend hf
"""

from __future__ import annotations

import os
import sys

# vLLM spawns workers; its memory pool is incompatible with expandable_segments.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
if "--phase" in sys.argv and sys.argv[sys.argv.index("--phase") + 1 :][:1] == ["generate"]:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""

import argparse
import gc
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    StoppingCriteria,
    StoppingCriteriaList,
)

# =========================================================================== Configuration
LAYER_CONFIGS: Dict[str, List[int]] = {
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": [12, 16, 20, 24, 28, 32],
    "Qwen/Qwen3-8B": [12, 16, 20, 24, 28, 32],
    "qwen3-8b:low": [12, 16, 20, 24, 28, 32],
    "Qwen/Qwen3.5-4B": [10, 14, 18, 22, 26, 30],
    "Skywork/Skywork-OR1-Math-7B": [10, 13, 16, 19, 22, 25],
    "allenai/Olmo-3-7B-Think": [10, 14, 18, 22, 26, 30],
}

DEFAULT_PROMPT_TEMPLATE = (
    "Solve the following mathematical problem step by step. "
    "Put your final answer within \\boxed{{}}.\n\n"
    "{problem}"
)


# =========================================================================== Small helpers
def slug(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "_")


def extract_boxed(text: str) -> Optional[str]:
    """Extract content of the last \\boxed{...} in text, handling nested braces."""
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    start = j = i + len("\\boxed{")
    depth = 1
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[start : j - 1].strip() if depth == 0 else None


def half_np(t: torch.Tensor) -> np.ndarray:
    """float16 conversion that clamps instead of overflowing to inf."""
    return t.float().clamp(-65504, 65504).half().cpu().numpy()


def build_prompt_ids(tokenizer: PreTrainedTokenizerBase, problem: str, use_chat_template: bool = True) -> List[int]:
    """Prompt token ids, identical for vLLM generation, HF generation and teacher forcing."""
    prompt = DEFAULT_PROMPT_TEMPLATE.format(problem=problem)
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        # The template already inserts BOS/role/<think> tokens, so don't add specials twice.
        return tokenizer(text, add_special_tokens=False)["input_ids"]
    return tokenizer(prompt)["input_ids"]


def find_loop_end(tokens: List[int], window: int = 32, repeats: int = 3) -> Optional[int]:
    """First index t such that tokens[:t] ends with `repeats` identical `window`-token blocks."""
    for t in range(window * repeats, len(tokens) + 1):
        recent = tokens[t - window : t]
        if all(tokens[t - (r + 1) * window : t - r * window] == recent for r in range(1, repeats)):
            return t
    return None


def waypoint_steps(n_steps: int, stride: int) -> List[int]:
    """Forward steps that get recorded: step 1, every `stride`, and the final step."""
    return sorted({1, n_steps} | set(range(stride, n_steps + 1, stride)))


class LoopStoppingCriteria(StoppingCriteria):
    """(HF backend) Halts generation if an identical 32-token window repeats 3 times in a row."""

    def __init__(self, prompt_len: int, window_size: int = 32, max_repeats: int = 3):
        super().__init__()
        self.prompt_len = prompt_len
        self.window_size = window_size
        self.max_repeats = max_repeats
        self.triggered = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        tokens = input_ids[0].tolist()
        gen_tokens = tokens[self.prompt_len :]
        min_len = self.window_size * self.max_repeats
        if len(gen_tokens) < min_len:
            return False

        recent = gen_tokens[-self.window_size :]
        if all(
            gen_tokens[-(r + 1) * self.window_size : -r * self.window_size] == recent
            for r in range(1, self.max_repeats)
        ):
            self.triggered = True
            return True
        return False


def find_decoder_layers(model: nn.Module) -> Tuple[nn.ModuleList, int]:
    """Locate the decoder block list (works for nested text configs, e.g. multimodal wrappers)."""
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    n = cfg.num_hidden_layers
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) == n and name.endswith("layers") and "vis" not in name:
            return mod, n
    raise RuntimeError("Could not locate decoder layers; print(model) and adjust find_decoder_layers().")


def get_eos_ids(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> Set[int]:
    """All token ids that can terminate generation (generation_config may list several)."""
    ids: Set[int] = set()
    gen_cfg = getattr(model, "generation_config", None)
    for src in (getattr(gen_cfg, "eos_token_id", None), tokenizer.eos_token_id):
        if src is None:
            continue
        if isinstance(src, (list, tuple, set)):
            ids.update(int(i) for i in src)
        else:
            ids.add(int(src))
    return ids


# =========================================================================== Dataset
def load_dataset(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    df = pd.read_csv(dataset_path)
    required = ["problem_id", "original_problem", "model_is_robust", "absolute_accuracy_decay"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {dataset_path}")
    return dedupe_problems(df, dataset_path)


def dedupe_problems(df: pd.DataFrame, dataset_path: Path) -> pd.DataFrame:
    """Collapse to one row per problem_id (outputs are written one file per problem).

    In augmented-sample-math-agg.csv a non-robust problem can appear on several rows,
    one per permutation_type that hurt it, each with its own accuracy decay. The problem
    text and robustness label must agree across those rows (otherwise we raise); we keep
    the row with the largest absolute_accuracy_decay, matching the 'max_drop' field
    saved with each trajectory.
    """
    key = df["problem_id"].astype(str)
    if not key.duplicated().any():
        return df

    dup_ids = sorted(key[key.duplicated(keep=False)].unique())
    conflicting = [
        pid for pid in dup_ids
        if len(df.loc[key == pid, ["original_problem", "model_is_robust"]].drop_duplicates()) > 1
    ]
    if conflicting:
        raise ValueError(
            f"{len(conflicting)} problem_id(s) in {dataset_path} have rows that disagree on "
            f"problem text or model_is_robust, e.g. {conflicting[:5]}. Fix the CSV first."
        )

    n_before = len(df)
    keep_idx = df.groupby(key, sort=False)["absolute_accuracy_decay"].idxmax()
    df = df.loc[sorted(keep_idx.values)].reset_index(drop=True)
    print(
        f"[warn] {len(dup_ids)} problem_id(s) had multiple rows ({n_before} rows -> "
        f"{len(df)} unique problems); kept the max-decay row for each: {dup_ids}"
    )
    return df


def problem_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    recs = []
    for row in df.itertuples(index=False):
        decay = float(row.absolute_accuracy_decay)
        base_acc = float(getattr(row, "base_accuracy", 1.0))
        perm_acc = float(getattr(row, "permuted_accuracy", base_acc - decay))
        recs.append(
            {
                "id": str(row.problem_id),
                "problem": str(row.original_problem),
                "is_robust": bool(row.model_is_robust),
                "decay": decay,
                "base_acc": base_acc,
                "perm_acc": perm_acc,
            }
        )
    return recs


# =========================================================================== I/O helpers
def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def load_jsonl_by_id(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    rows[str(r["id"])] = r
                except (json.JSONDecodeError, KeyError):
                    continue
    return rows


def prune_jsonl(path: Path, keep_ids: Optional[Set[str]]) -> int:
    """Keep only the first record per id (and only ids in keep_ids, when given).

    Drops unparseable lines and lines belonging to interrupted problems so that
    redoing them can't create duplicate records. Returns the number of dropped lines.
    """
    if not path.is_file():
        return 0
    kept, seen, dropped = [], set(), 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rid = str(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                dropped += 1
                continue
            if rid not in seen and (keep_ids is None or rid in keep_ids):
                seen.add(rid)
                kept.append(line if line.endswith("\n") else line + "\n")
            else:
                dropped += 1
    if dropped:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(kept)
        os.replace(tmp, path)
    return dropped


def save_npz(
    out_dir: Path,
    prob_id: str,
    layer_arrays: Dict[str, np.ndarray],
    tokens_generated: int,
    forward_steps: int,
    stop_reason: str,
    is_robust: bool,
    decay: float,
    base_acc: float,
    perm_acc: float,
) -> None:
    """Atomic .npz write. Goes through a file handle because np.savez_compressed appends
    '.npz' to bare filenames lacking that suffix, which would break os.replace()."""
    out_npz = out_dir / f"{prob_id}.npz"
    tmp_npz = out_dir / f"{prob_id}.tmp"
    try:
        with open(tmp_npz, "wb") as fh:
            np.savez_compressed(
                fh,
                **layer_arrays,
                tokens_generated=np.int64(tokens_generated),
                forward_steps=np.int64(forward_steps),
                stop_reason=np.str_(stop_reason),
                is_robust=np.bool_(is_robust),
                label=np.bool_(is_robust),
                correct=np.bool_(is_robust),
                max_drop=np.float64(decay),
                absolute_accuracy_decay=np.float64(decay),
                base_accuracy=np.float64(base_acc),
                permuted_accuracy=np.float64(perm_acc),
                label_source=np.str_("native"),
            )
        os.replace(tmp_npz, out_npz)
    except OSError:
        tmp_npz.unlink(missing_ok=True)
        raise


# =========================================================================== HF model + hooks
def pick_dtype(device: str) -> torch.dtype:
    if device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_hf_model(model_id: str, device: str) -> PreTrainedModel:
    dtype = pick_dtype(device)
    kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if device.startswith("cuda"):
        kwargs["device_map"] = device
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, **kwargs)
    if not device.startswith("cuda"):
        model = model.to(device)
    return model.eval()


def load_tokenizer(model_id: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.pad_token
    return tokenizer


@torch.no_grad()
def teacher_forced_waypoints(
    model: PreTrainedModel,
    layers_module: nn.ModuleList,
    target_layers: List[int],
    prompt_ids: List[int],
    gen_ids: List[int],
    stride: int,
) -> Dict[str, np.ndarray]:
    """Hidden states at waypoints via a single teacher-forced forward pass.

    Step 1 is the last prompt token; step s>=2 is the hidden state of input token
    gen_ids[s-2]. The final generated token is never fed back in, matching live decoding.
    """
    n_prompt, n_new = len(prompt_ids), len(gen_ids)
    steps = waypoint_steps(n_new, stride)
    pos = torch.tensor([n_prompt + s - 2 for s in steps], device=model.device)
    ids = torch.tensor([prompt_ids + gen_ids[:-1]], device=model.device)
    captured: Dict[int, torch.Tensor] = {}

    def make_hook(l: int):
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


def extract_problem_trajectory(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    all_layers: nn.ModuleList,
    target_layers: List[int],
    prompt_ids: List[int],
    stride: int,
    max_new_tokens: int,
    eos_ids: Set[int],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], List[int]]:
    """(HF backend / verify) Generate with HF and record hidden states live via hooks."""
    lead_tier = 0
    state: Dict[str, Any] = {"step": 0, "last_rec": 0, "latest": {}}
    tier_trajectories: List[List[torch.Tensor]] = [[] for _ in range(len(target_layers))]
    hooks = []

    def make_hook(tier_idx: int):
        def hook_fn(module: nn.Module, inp: Any, outp: Any) -> None:
            hidden = outp[0] if isinstance(outp, tuple) else outp
            # clone() so we don't keep the whole prefill tensor alive via a view;
            # stays on-device (no sync) unless this step is actually recorded.
            last = hidden[:, -1, :].detach().clone()
            if tier_idx == lead_tier:
                state["step"] += 1
            s = state["step"]
            state["latest"][tier_idx] = last
            if s == 1 or s % stride == 0:
                tier_trajectories[tier_idx].append(last.float().cpu())
                if tier_idx == lead_tier:
                    state["last_rec"] = s

        return hook_fn

    for tier_idx, layer_idx in enumerate(target_layers):
        hooks.append(all_layers[layer_idx].register_forward_hook(make_hook(tier_idx)))

    device = model.device
    input_ids = torch.tensor([prompt_ids], device=device)
    attention_mask = torch.ones_like(input_ids)
    prompt_len = len(prompt_ids)

    stopper = LoopStoppingCriteria(prompt_len=prompt_len)
    t0 = time.time()

    try:
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([stopper]),
            )
    finally:
        for h in hooks:
            h.remove()

    elapsed = time.time() - t0

    for tier_idx in range(len(target_layers)):
        if tier_idx in state["latest"] and state["last_rec"] != state["step"]:
            tier_trajectories[tier_idx].append(state["latest"][tier_idx].float().cpu())

    gen_ids = out[0, prompt_len:].tolist()
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    n_tokens = len(gen_ids)

    if gen_ids and gen_ids[-1] in eos_ids:
        stop_reason = "eos"  # checked first so an EOS landing exactly on the cap isn't called "cap"
    elif stopper.triggered:
        stop_reason = "loop"
    elif n_tokens >= max_new_tokens:
        stop_reason = "cap"
    else:
        stop_reason = "other"

    layer_arrays = {}
    for tier_idx, layer_idx in enumerate(target_layers):
        if tier_trajectories[tier_idx]:
            arr = half_np(torch.cat(tier_trajectories[tier_idx], dim=0))
        else:
            arr = np.zeros((1, model.config.get_text_config().hidden_size), dtype=np.float16)
        layer_arrays[f"layer_{layer_idx:02d}"] = arr

    meta = {
        "tokens_generated": n_tokens,
        "forward_steps": state["step"],
        "stop_reason": stop_reason,
        "seconds": round(elapsed, 2),
        "boxed": extract_boxed(gen_text),
        "text": gen_text,
    }
    return layer_arrays, meta, gen_ids


# =========================================================================== Phase 1: vLLM generation
def phase_generate(args, records: List[Dict[str, Any]], model_out_dir: Path) -> None:
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    gen_path = model_out_dir / "generations.jsonl"
    results_path = model_out_dir / "results.jsonl"

    # generations.jsonl is this phase's completion marker (written after results.jsonl).
    done = set(load_jsonl_by_id(gen_path)) | {p.stem for p in model_out_dir.glob("*.npz")}
    pruned = prune_jsonl(results_path, done) + prune_jsonl(gen_path, None)
    if pruned:
        print(f"Pruned {pruned} stale/duplicate jsonl line(s)")

    todo = [r for r in records if r["id"] not in done]
    print(
        f"\n=== [generate] {args.model}: {len(todo)} pending of {len(records)} problems "
        f"({len(records) - len(todo)} already done, skipped) ===",
        flush=True,
    )
    if not todo:
        return

    tokenizer = load_tokenizer(args.model)
    prompt_ids = [build_prompt_ids(tokenizer, r["problem"], not args.no_chat_template) for r in todo]
    max_len = max(len(p) for p in prompt_ids) + args.max_new_tokens + 8
    print(f"longest prompt {max(len(p) for p in prompt_ids)} tokens -> max_model_len {max_len}", flush=True)

    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="bfloat16",
        max_model_len=max_len,
        gpu_memory_utilization=args.gpu_util,
        max_num_seqs=args.max_num_seqs,
        seed=0,
        trust_remote_code=True,
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

            append_jsonl(
                results_path,
                {
                    "id": r["id"],
                    "is_robust": r["is_robust"],
                    "max_drop": r["decay"],
                    "base_accuracy": r["base_acc"],
                    "permuted_accuracy": r["perm_acc"],
                    "label_source": "native",
                    "tokens": len(gen_ids),
                    "forward_steps": len(gen_ids),
                    "stop_reason": reason,
                    "seconds": None,  # per-request timing isn't meaningful under batching
                    "boxed": boxed,
                    "text": text,
                },
            )
            # Written last: this line marks the problem as generated.
            append_jsonl(
                gen_path,
                {
                    "id": r["id"],
                    "problem": r["problem"],
                    "prompt_ids": prompt_ids[i],
                    "gen_ids": gen_ids,
                    "stop_reason": reason,
                    "is_robust": r["is_robust"],
                    "max_drop": r["decay"],
                    "base_accuracy": r["base_acc"],
                    "permuted_accuracy": r["perm_acc"],
                    "label_source": "native",
                },
            )

            n_done += 1
            total_tok += len(gen_ids)
            print(
                f"[{n_done}/{len(todo)}] id={r['id']} | tokens={len(gen_ids):5d} | stop={reason:<4s} | "
                f"robust={r['is_robust']!s:<5s} | decay={r['decay']:.2f} | boxed={boxed!r} "
                f"(aggregate {total_tok / max(time.time() - t0, 1e-6):.0f} tok/s)",
                flush=True,
            )

    print(
        f"[generate] done: {total_tok} tokens in {(time.time() - t0) / 60:.1f} min "
        f"= {total_tok / max(time.time() - t0, 1e-6):.0f} tok/s aggregate",
        flush=True,
    )


# =========================================================================== Phase 2: teacher-forced hidden states
def phase_hidden(args, target_layers: List[int], model_out_dir: Path) -> None:
    rows = load_jsonl_by_id(model_out_dir / "generations.jsonl")
    todo = [r for pid, r in rows.items() if not (model_out_dir / f"{pid}.npz").exists() and len(r["gen_ids"]) > 0]
    print(f"\n=== [hidden] {args.model}: {len(todo)} pending of {len(rows)} generations ===", flush=True)
    if not todo:
        return

    model = load_hf_model(args.model, args.device)
    layers_module, n_layers = find_decoder_layers(model)
    assert max(target_layers) < n_layers, f"layer index out of range (model has {n_layers})"

    t0 = time.time()
    for k, r in enumerate(todo, 1):
        arrays = teacher_forced_waypoints(
            model, layers_module, target_layers, r["prompt_ids"], r["gen_ids"], args.stride
        )
        save_npz(
            model_out_dir,
            r["id"],
            arrays,
            tokens_generated=len(r["gen_ids"]),
            forward_steps=len(r["gen_ids"]),
            stop_reason=r["stop_reason"],
            is_robust=bool(r["is_robust"]),
            decay=float(r["max_drop"]),
            base_acc=float(r["base_accuracy"]),
            perm_acc=float(r["permuted_accuracy"]),
        )
        print(
            f"[hidden {k}/{len(todo)}] id={r['id']} tokens={len(r['gen_ids'])} "
            f"({(time.time() - t0) / k:.1f}s/problem)",
            flush=True,
        )
        if k % 25 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# =========================================================================== Verify: live hooks vs teacher forcing
def phase_verify(args, records: List[Dict[str, Any]], target_layers: List[int]) -> None:
    tokenizer = load_tokenizer(args.model)
    model = load_hf_model(args.model, args.device)
    layers_module, n_layers = find_decoder_layers(model)
    assert max(target_layers) < n_layers, f"layer index out of range (model has {n_layers})"
    eos_ids = get_eos_ids(model, tokenizer)

    min_cos_allowed = 0.98  # bf16 incremental-KV decoding vs one full forward won't be bit-identical
    failed = False
    print(f"\nVerifying {min(args.verify, len(records))} problem(s), {args.verify_tokens} tokens each...")
    for r in records[: args.verify]:
        prompt_ids = build_prompt_ids(tokenizer, r["problem"], not args.no_chat_template)
        live, meta, gen_ids = extract_problem_trajectory(
            model, tokenizer, layers_module, target_layers, prompt_ids,
            args.stride, args.verify_tokens, eos_ids,
        )
        tf = teacher_forced_waypoints(model, layers_module, target_layers, prompt_ids, gen_ids, args.stride)
        for key in live:
            a, b = live[key].astype(np.float32), tf[key].astype(np.float32)
            if a.shape != b.shape:
                print(f"  id={r['id']} {key}: FAIL shape live {a.shape} vs teacher-forced {b.shape}")
                failed = True
                continue
            cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)
            rel = np.linalg.norm(a - b, axis=1) / (np.linalg.norm(a, axis=1) + 1e-8)
            ok = cos.min() >= min_cos_allowed
            failed |= not ok
            print(
                f"  id={r['id']} {key}: {'ok  ' if ok else 'FAIL'} waypoints={a.shape[0]} "
                f"min cos={cos.min():.5f} max rel err={rel.max():.4f}"
            )
    if failed:
        raise SystemExit("Verification FAILED: live and teacher-forced hidden states disagree.")
    print("Verification passed: teacher-forced waypoints match live hook extraction.")


# =========================================================================== HF backend (original single-process path)
def run_hf_backend(args, records: List[Dict[str, Any]], target_layers: List[int], model_out_dir: Path) -> None:
    results_path = model_out_dir / "results.jsonl"
    generations_path = model_out_dir / "generations.jsonl"

    # Here the .npz is the completion marker (written last), as in the single-process flow.
    existing_npz = {p.stem for p in model_out_dir.glob("*.npz")}
    pruned = prune_jsonl(results_path, existing_npz) + prune_jsonl(generations_path, existing_npz)
    if pruned:
        print(f"Pruned {pruned} stale/duplicate jsonl line(s)")
    todo = [r for r in records if r["id"] not in existing_npz]
    if not todo:
        print("All problems already extracted!")
        return

    print(f"\nLoading model {args.model} on {args.device}...")
    tokenizer = load_tokenizer(args.model)
    model = load_hf_model(args.model, args.device)
    all_layers, n_layers = find_decoder_layers(model)
    assert max(target_layers) < n_layers, f"layer index out of range (model has {n_layers})"
    eos_ids = get_eos_ids(model, tokenizer)
    print(f"EOS token ids: {sorted(eos_ids)}")

    print("\nStarting generation and waypoint recording...")
    start_time = time.time()
    for idx, r in enumerate(todo, 1):
        prompt_ids = build_prompt_ids(tokenizer, r["problem"], not args.no_chat_template)
        layer_arrays, meta, gen_ids = extract_problem_trajectory(
            model, tokenizer, all_layers, target_layers, prompt_ids,
            args.stride, args.max_new_tokens, eos_ids,
        )

        append_jsonl(
            results_path,
            {
                "id": r["id"],
                "is_robust": r["is_robust"],
                "max_drop": r["decay"],
                "base_accuracy": r["base_acc"],
                "permuted_accuracy": r["perm_acc"],
                "label_source": "native",
                "tokens": meta["tokens_generated"],
                "forward_steps": meta["forward_steps"],
                "stop_reason": meta["stop_reason"],
                "seconds": meta["seconds"],
                "boxed": meta["boxed"],
                "text": meta["text"],
            },
        )
        append_jsonl(
            generations_path,
            {
                "id": r["id"],
                "problem": r["problem"],
                "prompt_ids": prompt_ids,
                "gen_ids": gen_ids,
                "stop_reason": meta["stop_reason"],
                "is_robust": r["is_robust"],
                "max_drop": r["decay"],
                "base_accuracy": r["base_acc"],
                "permuted_accuracy": r["perm_acc"],
                "label_source": "native",
            },
        )
        save_npz(
            model_out_dir, r["id"], layer_arrays,
            tokens_generated=meta["tokens_generated"], forward_steps=meta["forward_steps"],
            stop_reason=meta["stop_reason"], is_robust=r["is_robust"], decay=r["decay"],
            base_acc=r["base_acc"], perm_acc=r["perm_acc"],
        )

        speed = meta["tokens_generated"] / max(0.1, meta["seconds"])
        print(
            f"[{idx}/{len(todo)}] id={r['id']} | tokens={meta['tokens_generated']:5d} | "
            f"stop={meta['stop_reason']:<4s} | robust={r['is_robust']!s:<5s} | decay={r['decay']:.2f} | "
            f"boxed={meta['boxed']!r} ({meta['seconds']}s, {speed:.1f} tok/s)",
            flush=True,
        )
        if idx % 5 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nExtraction completed in {(time.time() - start_time) / 60:.1f} minutes.")


# =========================================================================== Orchestration
def child_args(args) -> List[str]:
    c = [
        "--dataset", str(args.dataset),
        "--model", args.model,
        "--output-dir", str(args.output_dir),
        "--stride", str(args.stride),
        "--max-new-tokens", str(args.max_new_tokens),
        "--max-num-seqs", str(args.max_num_seqs),
        "--gpu-util", str(args.gpu_util),
        "--device", args.device,
        "--backend", "vllm",
        "--child",
    ]
    if args.limit:
        c += ["--limit", str(args.limit)]
    if args.no_chat_template:
        c += ["--no-chat-template"]
    return c


def spawn(phase: str, args) -> int:
    """Run one phase in a fresh process so vLLM's GPU memory is fully released before the next."""
    env = dict(os.environ)
    if phase == "generate":
        env["PYTORCH_CUDA_ALLOC_CONF"] = ""
    cmd = [sys.executable, "-u", __file__, "--phase", phase] + child_args(args)
    return subprocess.call(cmd, env=env)


def main():
    parser = argparse.ArgumentParser(description="Trajectory extraction (vLLM generation + teacher-forced hidden states).")
    parser.add_argument("--dataset", type=Path, default=Path("augmented-sample-math-agg.csv"))
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--output-dir", type=Path, default=Path("saved_trajectories_aggregate"))
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm",
                        help="vllm: two-phase (vLLM generate + teacher-forced hidden); hf: HF generate with live hooks")
    parser.add_argument("--phase", choices=["both", "generate", "hidden", "verify"], default="both",
                        help="vllm backend: which phase(s) to run; 'verify' works with either backend")
    parser.add_argument("--max-new-tokens", type=int, default=16384, help="Max generated tokens (default 16384)")
    parser.add_argument("--stride", type=int, default=16, help="Waypoint sampling stride (default 16)")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of problems")
    parser.add_argument("--dry-run", action="store_true", help="Preview dataset stats and plan; no model loading")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for HF model (hidden/verify phases and hf backend)")
    parser.add_argument("--no-chat-template", action="store_true",
                        help="Feed the raw prompt string instead of applying the tokenizer's chat template")
    parser.add_argument("--max-num-seqs", type=int, default=32, help="Max concurrent sequences in vLLM")
    parser.add_argument("--gpu-util", type=float, default=0.92, help="vLLM gpu_memory_utilization")
    parser.add_argument("--verify", type=int, default=3, help="Problems to check in --phase verify")
    parser.add_argument("--verify-tokens", type=int, default=256, help="Tokens to generate per verify problem")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    df = load_dataset(args.dataset)
    if args.limit:
        df = df.head(args.limit)
    records = problem_records(df)

    model_id = args.model
    target_layers = LAYER_CONFIGS.get(model_id)
    if target_layers is None:
        target_layers = [12, 16, 20, 24, 28, 32]
        print(f"[warn] {model_id} not in LAYER_CONFIGS; using fallback layers {target_layers}")

    model_out_dir = args.output_dir / slug(model_id)

    if not args.child:
        print("=" * 80)
        print("AGGREGATE DATASET TRAJECTORY EXTRACTION")
        print("=" * 80)
        print(f"Dataset:            {args.dataset} ({len(df)} problems)")
        print(f"Target Model:       {model_id}")
        print(f"Backend / Phase:    {args.backend} / {args.phase}")
        print(f"Target Layers:      {target_layers}")
        print(f"Max New Tokens:     {args.max_new_tokens}")
        print(f"Waypoint Stride:    {args.stride}")
        print(f"Output Directory:   {model_out_dir}")
        print(f"Robust Problems:    {sum(df['model_is_robust'] == True)} / {len(df)}")
        print(f"Non-Robust:         {sum(df['model_is_robust'] == False)} / {len(df)}")
        print(f"Mean Decay:         {df['absolute_accuracy_decay'].mean():.4f}")
        print("=" * 80)

    # Progress report (read-only; the output directory is only created for real runs)
    existing_npz = {p.stem for p in model_out_dir.glob("*.npz")}
    generated = set(load_jsonl_by_id(model_out_dir / "generations.jsonl")) | existing_npz
    ids = {r["id"] for r in records}
    n_gen_pending = len(ids - generated)
    n_hidden_pending = len(ids - existing_npz)
    if args.backend == "vllm":
        print(f"Progress: generate pending {n_gen_pending}, hidden pending {n_hidden_pending} of {len(ids)}.")
    else:
        print(f"Progress: {len(existing_npz & ids)} already extracted, {n_hidden_pending} remaining.")

    if args.dry_run:
        print("\n[DRY RUN] Plan verified successfully. Exiting without model execution.")
        return

    model_out_dir.mkdir(parents=True, exist_ok=True)

    if args.phase == "verify":
        phase_verify(args, records, target_layers)
        return

    if args.backend == "hf":
        run_hf_backend(args, records, target_layers, model_out_dir)
        return

    # vLLM backend
    if args.phase == "generate":
        phase_generate(args, records, model_out_dir)
    elif args.phase == "hidden":
        phase_hidden(args, target_layers, model_out_dir)
    else:  # both: separate processes so vLLM releases the GPU before the HF model loads
        for ph in ("generate", "hidden"):
            rc = spawn(ph, args)
            print(f"##### {model_id} [{ph}] exit code {rc}", flush=True)
            if rc != 0:
                raise SystemExit(rc)
        print("\n" + "=" * 80)
        print(f"Extraction finished. Outputs saved to: {model_out_dir}")
        print("=" * 80)


if __name__ == "__main__":
    main()