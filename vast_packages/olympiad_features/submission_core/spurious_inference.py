"""Submission-side inference for the spurious-reasoning signals.

Per call (one model, a batch of problems):
1. Greedy generation, one problem at a time, up to the training token cap (16,384), with the same prompt and chat template as
   our extraction; generation stops early only on a degenerate loop (3 identical 32-token blocks).
2. The features named in spurious_artifacts/recipe.json (see spurious_features.py for the formulas):
     fa    forced-answer confidence at min(6144, thinking length), suffix B
     ab    max over 12 cutoffs of |entropy(suffix A) - entropy(suffix B)| of the first answer token
     seg   first-PC variance share of the 1024-token segment at 25% of thinking
     traj  late-vs-early cosine of the 16-token window means
     pearl drift of the mean problem-token state under prose typos (area over 5 rates)
     ge    global efficiency of the 5-NN graph over 64-token mean states of the first 4096 tokens
     tda   H0 Betti-curve spread of the Vietoris-Rips persistence of the 64-token mean states of thinking (4th trajectory layer)
     cr    cycle rank of the walk of the 64-token states (first 4096 tokens) over this model's 64 K-means clusters
     chk   share of the thinking's sentences that check / verify / confirm something (text only)
     len   -log(number of generated tokens)
3. Each feature -> percentile among that model's main_v2 values (flipped where lower = robust), minus 0.5, averaged;
   robust = score > the recipe threshold. A feature that cannot be computed for a trace counts as neutral.
Time: the whole prediction run has 3600 s (budget 3500 s). Each model's problems are generated together in one batched generate call
(left padding; rows stop independently on eos / loop), allowed all remaining time minus a reserve of 45 s per problem still to come
(expected_problems in the recipe) and 30 s per problem of this call for its features, so it only bites if the run is about to overrun.
"""

from __future__ import annotations

import copy
import gc
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase,
                          PreTrainedTokenizerFast, StoppingCriteria, StoppingCriteriaList)

from spurious_features import (SEL_LAYER_FRACS, STRIDE, SUFFIXES_NOTHINK, SUFFIXES_THINK, TRAJ_LAYERS, build_prompt, cutoff_labels,
                               SHIFT_BUDGETS, decide_absolute, decide_batch, ends_in_loop, ensemble_score, f_check_share, f_forced_saturated_share, f_forced_slope, f_pc1_change, f_cluster_cycle_rank, f_global_eff, score_trace, f_late_vs_early, f_pearl_area, f_seg25_pc1, f_tda_h0_spread,
                               find_think_end, pearl_texts, think_end_token_id)

ARTIFACT_DIR_NAME = "spurious_artifacts"
_START = time.time()
_DONE = [0]          # problems processed so far in this run
_ROUNDTRIP = "Solve x + y = 3 for the unknown, then find $n^2$."

# ---------------------------------------------------------------- logging (always on: ingestion hides participant exceptions, so
# stdout / stderr are the only record of a run)
DIAG = True
_STOPS: dict[str, int] = {}          # stop reasons over the whole run


def _log(*parts) -> None:
    if DIAG:
        msg = f"[spurious {time.time() - _START:8.1f}s] " + " ".join(str(p) for p in parts)
        print(msg, flush=True)
        print(msg, file=sys.stderr, flush=True)


def _gpu() -> str:
    if not torch.cuda.is_available():
        return "cuda unavailable"
    free, total = torch.cuda.mem_get_info()
    return (f"gpu free {free / 2**30:.1f}/{total / 2**30:.1f} GiB | this process allocated {torch.cuda.memory_allocated() / 2**30:.1f} "
            f"reserved {torch.cuda.memory_reserved() / 2**30:.1f} GiB")


def _environment_report() -> None:
    import atexit, faulthandler, os, platform, signal, subprocess
    faulthandler.enable(all_threads=True)          # a hard crash (segfault / abort) still dumps Python stacks to stderr
    atexit.register(lambda: _log("interpreter exiting normally"))

    def on_signal(signum, _frame):
        _log(f"received signal {signum} - dumping stacks")
        faulthandler.dump_traceback(all_threads=True)
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, on_signal)

    import transformers, huggingface_hub
    _log("python", sys.version.replace("\n", " "), "|", platform.platform(), "| cpus", os.cpu_count())
    _log("torch", torch.__version__, "cuda build", torch.version.cuda, "| cuda available", torch.cuda.is_available(),
         "| devices", torch.cuda.device_count())
    if torch.cuda.is_available():
        _log("device 0:", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0),
             "| bf16", torch.cuda.is_bf16_supported(), "|", _gpu())
    import importlib.util
    _log("transformers", transformers.__version__, "| huggingface_hub", huggingface_hub.__version__, "| numpy", np.__version__,
         "| installed:", {m: importlib.util.find_spec(m) is not None for m in ("accelerate", "safetensors", "sklearn", "pandas", "joblib")})
    _log("env:", {k: os.environ.get(k) for k in ("HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "CUDA_VISIBLE_DEVICES",
                                                   "PYTORCH_CUDA_ALLOC_CONF", "TOKENIZERS_PARALLELISM")})
    _log("cwd", os.getcwd(), "| sys.path[:3]", sys.path[:3])
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20).stdout
        _log("nvidia-smi:\n" + out)
    except Exception as e:
        _log("nvidia-smi failed:", repr(e))
    from huggingface_hub import try_to_load_from_cache
    for m in ("Qwen/Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B", "allenai/Olmo-3-7B-Think", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"):
        try:
            cfg = try_to_load_from_cache(m, "config.json")
            if isinstance(cfg, str):
                snap = Path(cfg).parent
                files = sorted(p.name for p in snap.iterdir())
                size = sum(p.stat().st_size for p in snap.iterdir() if p.suffix == ".safetensors") / 2**30
                _log(f"cache {m}: {snap} | {len(files)} files, safetensors {size:.1f} GiB | {files}")
            else:
                _log(f"cache {m}: config.json NOT found in the cache ({cfg!r})")
        except Exception as e:
            _log(f"cache {m}: lookup failed {e!r}")


if DIAG:
    try:
        _environment_report()
    except Exception:
        _log("environment report failed:\n" + traceback.format_exc())


COMPETITION_MODEL_ALIASES: dict[str, str] = {
    "qwen3-8b:low": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "qwen3-8b": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "Qwen/Qwen3-8B": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
}


def resolve_checkpoint_model_id(model_id: str) -> str:
    """Resolve a competition alias, or strip a reasoning-effort tag (':low' / ':medium'), to the Hugging Face checkpoint ID."""
    if model_id in COMPETITION_MODEL_ALIASES:
        return COMPETITION_MODEL_ALIASES[model_id]
    base_id = model_id.split(":")[0]
    return COMPETITION_MODEL_ALIASES.get(base_id, base_id)


def load_recipe() -> dict[str, Any]:
    path = Path(__file__).resolve().parent / ARTIFACT_DIR_NAME / "recipe.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing artifact {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tokenizer(model_id: str, local_files_only: bool = True) -> PreTrainedTokenizerBase:
    """Transformers 5.x loads the DeepSeek-R1-0528-Qwen3-8B tokenizer as a LlamaTokenizer that drops spaces; in that case it is
    rebuilt from tokenizer.json, exactly as in our extraction, so prompts match the training traces token for token."""
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only, trust_remote_code=False)
    _log(f"tokenizer {type(tokenizer).__name__} loaded in {time.time() - t0:.1f}s")
    roundtrip = lambda t: t.decode(t(_ROUNDTRIP, add_special_tokens=False)["input_ids"]).strip() == _ROUNDTRIP   # noqa: E731
    if not roundtrip(tokenizer):
        from huggingface_hub import hf_hub_download
        fixed = PreTrainedTokenizerFast(
            tokenizer_file=hf_hub_download(model_id, "tokenizer.json", local_files_only=local_files_only),
            bos_token=tokenizer.bos_token, eos_token=tokenizer.eos_token,
            pad_token=tokenizer.pad_token or tokenizer.eos_token, unk_token=tokenizer.unk_token,
        )
        fixed.chat_template = tokenizer.chat_template
        if not roundtrip(fixed):
            raise RuntimeError(f"tokenizer for {model_id} fails the space round trip")
        tokenizer = fixed
        _log("tokenizer dropped spaces -> rebuilt as PreTrainedTokenizerFast from tokenizer.json")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.pad_token
    return tokenizer


def load_model_and_tokenizer(model_id: str, local_files_only: bool = True) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load evaluation model and tokenizer offline."""
    tokenizer = load_tokenizer(model_id, local_files_only)

    # No device_map: the evaluation image does not ship `accelerate`, and transformers refuses any device_map without it.
    # Load on the CPU in the target dtype, then move the whole model to the single GPU.
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        _log(f"loading model {model_id} dtype {dtype} (no device_map, then .to('cuda')) | before: {_gpu()}")
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, local_files_only=local_files_only, trust_remote_code=False)
        model.to("cuda")
        _log(f"model {type(model).__name__} loaded in {time.time() - t0:.1f}s | on {model.device} | after: {_gpu()}")
    else:
        _log("CUDA NOT AVAILABLE - loading on CPU in float32")
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, local_files_only=local_files_only, trust_remote_code=False)
        model.to("cpu")
    model.eval()
    return model, tokenizer


def find_decoder_layers(model: torch.nn.Module) -> tuple[torch.nn.ModuleList, int]:
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    n = cfg.num_hidden_layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) == n and name.endswith("layers") and "vis" not in name:
            return module, n
    raise RuntimeError("could not locate the decoder layers")


def kv_bytes_per_token(model) -> int:
    """KV-cache bytes per token in bf16, counting every attention layer that keeps a cache (linear-attention layers excluded)."""
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    types = getattr(cfg, "layer_types", None)
    n_attn = sum(1 for t in types if "linear" not in str(t)) if types else cfg.num_hidden_layers
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    kvh = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    return 2 * n_attn * kvh * hd * 2


def forward_last_logits(model, **kwargs):
    """Forward pass keeping only the last position's logits (falls back for wrappers without logits_to_keep)."""
    try:
        return model(logits_to_keep=1, **kwargs)
    except TypeError:
        return model(**kwargs)


class LoopStoppingCriteria(StoppingCriteria):
    """Per-row stop once a row's generated tokens end in 3 identical 32-token blocks (the rule our extraction cut traces with).
    Records each row's generated length at the moment it first looped (rows stopped by generate are padded afterwards)."""

    def __init__(self, prompt_len: int, batch: int):
        super().__init__()
        self.prompt_len = prompt_len
        self.stop_len = [None] * batch

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        n = input_ids.shape[1] - self.prompt_len
        done = []
        for r in range(input_ids.shape[0]):
            if self.stop_len[r] is None and n >= 96 and ends_in_loop(input_ids[r, -96:].tolist()):
                self.stop_len[r] = n
            done.append(self.stop_len[r] is not None)
        return torch.tensor(done, device=input_ids.device, dtype=torch.bool)


def generate_batch(model, tokenizer, prompts: list[list[int]], max_new_tokens: int, max_time: float) -> list[tuple[list[int], str]]:
    """Greedy traces for several prompts in one generate call (left padding) and why each stopped: 'eos', 'loop' (3 identical
    32-token blocks), 'cap' (max_new_tokens) or 'time' (the call's time allowance ran out first)."""
    gcfg = model.generation_config.eos_token_id
    eos = set([gcfg] if isinstance(gcfg, int) else (gcfg or [])) | {tokenizer.eos_token_id}
    eos.discard(None)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else min(eos)
    n = max(len(p) for p in prompts)
    ids = torch.full((len(prompts), n), pad, dtype=torch.long)
    att = torch.zeros((len(prompts), n), dtype=torch.long)
    for r, p in enumerate(prompts):
        ids[r, n - len(p):] = torch.tensor(p)
        att[r, n - len(p):] = 1
    stopper = LoopStoppingCriteria(n, len(prompts))
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            input_ids=ids.to(model.device), attention_mask=att.to(model.device), max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=pad, eos_token_id=sorted(eos), max_time=max(10.0, max_time), stopping_criteria=StoppingCriteriaList([stopper]),
        )
    elapsed = time.time() - t0
    res = []
    for r in range(len(prompts)):
        gen = out[r, n:].tolist()
        lim = stopper.stop_len[r]
        e = next((k for k, t in enumerate(gen[:lim]) if t in eos), None)
        if e is not None:
            res.append((gen[:e + 1], "eos"))
        elif lim is not None:
            res.append((gen[:lim], "loop"))
        elif len(gen) >= max_new_tokens:
            res.append((gen[:max_new_tokens], "cap"))
        else:                                             # generation stopped by max_time: drop the padding of finished rows
            res.append((gen, "time" if elapsed >= max_time - 1 else "cap"))
    return res


def window_states(model, layers, traj_layers, prompt_ids: list[int], gen_ids: list[int]) -> np.ndarray:
    """[n_windows, 6, d]: exact means of consecutive 16-token windows of the generated part (teacher forced)."""
    n_p, n_g = len(prompt_ids), len(gen_ids)
    captured = {}

    def hook(layer_idx):
        def fn(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            g = hidden[0, n_p:n_p + n_g].float()
            k = n_g // STRIDE
            parts = [g[:k * STRIDE].view(k, STRIDE, -1).mean(1)] if k else []
            if n_g % STRIDE:
                parts.append(g[k * STRIDE:].mean(0, keepdim=True))
            captured[layer_idx] = torch.cat(parts, 0).cpu()
        return fn

    handles = [layers[l].register_forward_hook(hook(l)) for l in traj_layers]
    try:
        with torch.no_grad():
            forward_last_logits(model, input_ids=torch.tensor([prompt_ids + gen_ids], device=model.device), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return torch.stack([captured[l] for l in traj_layers], 1).numpy()


def forced_answers(model, prompt_ids, gen_ids, cuts, suffixes) -> dict:
    """{(cut, suffix): (top-1 probability, entropy) of the first answer token}; the prefix cache is extended cut by cut and copied
    for every suffix."""
    full = prompt_ids + gen_ids
    n_p = len(prompt_ids)
    cache, pos, results = None, 0, {}
    with torch.no_grad():
        for c in sorted(set(cuts)):
            target = n_p + c
            for s in range(pos, target, 4096):
                out = forward_last_logits(model, input_ids=torch.tensor([full[s:min(target, s + 4096)]], device=model.device),
                                          past_key_values=cache, use_cache=True)
                cache = out.past_key_values
            pos = target
            for name, sids in suffixes.items():
                try:
                    branch = copy.deepcopy(cache)
                    out = forward_last_logits(model, input_ids=torch.tensor([sids], device=model.device), past_key_values=branch, use_cache=True)
                except (RuntimeError, TypeError, AttributeError):   # a cache type that cannot be copied: recompute prefix + suffix
                    out = forward_last_logits(model, input_ids=torch.tensor([full[:target] + sids], device=model.device), use_cache=False)
                lp = torch.log_softmax(out.logits[0, -1].float(), -1)
                p = lp.exp()
                results[(c, name)] = (float(p.max()), float(-(p * lp).sum()))
                del out
    del cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def pearl_mean_states(model, tokenizer, sel_layers, problem: str) -> np.ndarray:
    """[21, 6, d] mean hidden state over the problem tokens for the original and its 20 prose-typo copies."""
    states = []
    with torch.no_grad():
        for text in pearl_texts(problem):
            ids, _, span = build_prompt(tokenizer, text)
            a, b = span if span and span[1] - span[0] > 0 else (0, len(ids))
            out = forward_last_logits(model, input_ids=torch.tensor([ids], device=model.device), use_cache=False, output_hidden_states=True)
            states.append(torch.stack([out.hidden_states[l + 1][0, a:b].float().mean(0) for l in sel_layers]).cpu().numpy())
            del out
    return np.array(states)


def predict_robustness(model_id: str, problem_texts: Sequence[str]) -> list[bool]:
    """Predict robustness for one model's batch of problems."""
    _log(f"===== call: model {model_id!r}, {len(list(problem_texts))} problems | {_gpu()}")
    try:
        out = _predict_robustness(model_id, problem_texts)
        _log(f"===== call done: {model_id} -> {out}")
        return out
    except BaseException:
        _log(f"===== call FAILED for {model_id}:\n" + traceback.format_exc())
        raise


def _predict_robustness(model_id: str, problem_texts: Sequence[str]) -> list[bool]:
    problems = list(problem_texts)
    if not problems:
        return []
    recipe = load_recipe()
    feats = recipe["features"]
    model_id = resolve_checkpoint_model_id(model_id)
    if model_id not in recipe["ref"][feats[0]]:
        raise ValueError(f"{model_id} has no calibration in {ARTIFACT_DIR_NAME}/recipe.json")
    deadline = _START + recipe["time_budget_s"]
    _log(f"recipe {recipe.get('name')} features {feats} | resolved id {model_id} | {deadline - time.time():.0f}s left in the run")

    X = {f: np.full(len(problems), np.nan) for f in feats}
    call_stops: dict[str, int] = {}
    model = None
    tokenizer = None
    try:
        model, tokenizer = load_model_and_tokenizer(model_id, local_files_only=True)
        layers, n_layers = find_decoder_layers(model)
        _log(f"{n_layers} decoder layers | input device {model.device}")
        sel_layers = sorted({min(n_layers - 1, max(0, int(round(f * n_layers)) - 1)) for f in SEL_LAYER_FRACS})
        traj_layers = [l for l in TRAJ_LAYERS.get(model_id, []) if l < n_layers] or sel_layers
        tid = think_end_token_id(tokenizer)
        has_think = tid is not None or "<think>" in build_prompt(tokenizer, "x")[1]
        suffix_text = SUFFIXES_THINK if has_think else SUFFIXES_NOTHINK
        suffixes = {k: tokenizer(v, add_special_tokens=False)["input_ids"] for k, v in suffix_text.items()}
        mu = centroids = None
        art = Path(__file__).resolve().parent / ARTIFACT_DIR_NAME
        if "ge" in feats or "cr" in feats:
            mu = np.load(art / f"mu_{model_id.replace('/', '__')}.npy")
        if "cr" in feats:
            centroids = np.load(art / f"km64_{model_id.replace('/', '__')}.npy")
            _log(f"cluster centroids {centroids.shape} loaded")

        prompts = [build_prompt(tokenizer, p)[0] for p in problems]
        # all of this model's problems in one batched generate call; the allowance keeps a reserve for the problems still to come
        left_after = max(0, recipe["expected_problems"] - _DONE[0] - len(problems))
        allowance = (deadline - time.time()) - 45 * left_after - 30 * len(problems)
        per_seq = kv_bytes_per_token(model) * (max(len(p) for p in prompts) + recipe["cap"])
        free = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 8e9
        bs = int(max(1, min(len(prompts), (0.6 * free) // per_seq)))
        chunks = [list(range(s0, min(len(prompts), s0 + bs))) for s0 in range(0, len(prompts), bs)]
        _log(f"generating {len(prompts)} problems in {len(chunks)} batch(es) of up to {bs} | prompts {[len(p) for p in prompts]} tokens | "
             f"allowance {allowance:.0f}s | {_gpu()}")
        gens = [None] * len(prompts)
        for ci, ch in enumerate(chunks):
            t0 = time.time()
            share = (allowance - (time.time() - t_gen0 if ci else 0)) / (len(chunks) - ci) if ci else allowance / len(chunks)
            for r, g in zip(ch, generate_batch(model, tokenizer, [prompts[r] for r in ch], recipe["cap"], share)):
                gens[r] = g
            dt = time.time() - t0
            _log(f"  batch {ci + 1}/{len(chunks)}: {len(ch)} traces in {dt:.1f}s ({sum(len(gens[r][0]) for r in ch) / max(1e-6, dt):.0f} tok/s total) | "
                 f"{_gpu()}")
            if ci == 0:
                t_gen0 = t0
        _DONE[0] += len(problems)
        for i, problem in enumerate(problems):
            prompt_ids = prompts[i]
            gen, reason = gens[i]
            t0 = time.time()
            call_stops[reason] = call_stops.get(reason, 0) + 1
            _STOPS[reason] = _STOPS.get(reason, 0) + 1
            _log(f"problem {i + 1}/{len(problems)}: prompt {len(prompt_ids)} tokens | generated {len(gen)} tokens | stop: {reason} | "
                 f"problems done in run: {_DONE[0]}/{recipe['expected_problems']}")
            te = find_think_end(tokenizer, gen, tid)
            L = max(1, te if te is not None else len(gen))
            if "len" in feats:
                X["len"][i] = -math.log(max(1, len(gen)))
            if not gen or time.time() > deadline - 60:
                _log(f"  features SKIPPED ({'empty trace' if not gen else 'less than 60 s left in the run'})")
                continue
            if any(f in feats for f in ("seg", "traj", "ge", "tda", "cr", "pcr")):
                ws = window_states(model, layers, traj_layers, prompt_ids, gen)
                if "seg" in feats:
                    X["seg"][i] = f_seg25_pc1(ws, L)
                if "traj" in feats:
                    X["traj"][i] = f_late_vs_early(ws)
                if "ge" in feats:
                    X["ge"][i] = f_global_eff(ws[:, 3], mu, L)
                if "cr" in feats:
                    X["cr"][i] = f_cluster_cycle_rank(ws[:, 3], mu, centroids, L)
                if "tda" in feats:
                    t1 = time.time()
                    X["tda"][i] = f_tda_h0_spread(ws[:, 3], L)
                    _log(f"  TDA persistence ({min(len(ws), max(1, L // 16)) // 4} nodes) in {time.time() - t1:.1f}s")
                if "pcr" in feats:                                  # every generated window within the cap, as in exp44
                    X["pcr"][i] = f_pc1_change(ws)
                del ws
            if "fas" in feats or "fsat" in feats:                  # within-trace forced confidence over budgets 512 ... 6144 (exp44)
                lab = cutoff_labels(L); cuts = [lab[f"b{b}"] for b in SHIFT_BUDGETS]
                r = forced_answers(model, prompt_ids, gen, cuts, {"B": suffixes["B"]}); p1s = [r[(c, "B")][0] for c in cuts]
                _log(f"  forced p1 at budgets {dict(zip(SHIFT_BUDGETS, [round(v, 4) for v in p1s]))}")
                if "fas" in feats:
                    X["fas"][i] = f_forced_slope(p1s)
                if "fsat" in feats:
                    X["fsat"][i] = f_forced_saturated_share(p1s)
            if "fa" in feats or "ab" in feats:
                lab = cutoff_labels(L)
                if "ab" in feats:
                    r = forced_answers(model, prompt_ids, gen, list(lab.values()), suffixes)
                    X["ab"][i] = max(abs(r[(c, "A")][1] - r[(c, "B")][1]) for c in lab.values())
                else:
                    r = forced_answers(model, prompt_ids, gen, [lab["b6144"]], {"B": suffixes["B"]})
                if "fa" in feats:
                    X["fa"][i] = r[(lab["b6144"], "B")][0]
            if "pearl" in feats:
                X["pearl"][i] = f_pearl_area(pearl_mean_states(model, tokenizer, sel_layers, problem))
            if "chk" in feats:                                     # decoded as our extraction stored traces (special tokens skipped)
                X["chk"][i] = f_check_share(tokenizer.decode(gen, skip_special_tokens=True))
            _log(f"  thinking {L} tokens ({'ended' if te is not None else 'never ended'}) | features " + " ".join(f"{f}={X[f][i]:.4f}" for f in feats) + f" | {time.time() - t0:.1f}s total")
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _log(f"model released | {_gpu()}")

    scores = [score_trace({f: X[f][i] for f in feats}, recipe, model_id) for i in range(len(problems))]
    if recipe.get("abs_feature"):
        preds = [decide_absolute({f: X[f][i] for f in feats}, recipe) for i in range(len(problems))]
        scores = [float(X[recipe["abs_feature"]][i]) for i in range(len(problems))]
        _log(f"absolute cut: {recipe['abs_feature']} > {recipe['abs_cut']:.6g} -> robust (missing counts as robust)")
    else:
        preds = decide_batch(scores, recipe["threshold"], recipe.get("per_model_split", False))
    missing = {f: int(np.isnan(X[f]).sum()) for f in feats}
    _log(f"call summary {model_id}: {len(problems)} problems | stops {call_stops} (run so far {_STOPS}) | missing features {missing} | "
         f"scores {[round(v, 4) for v in scores]} vs threshold {recipe['threshold']:.4f}"
         f"{' (per-model half split)' if recipe.get('per_model_split') else ''} | raw features {dict((f, [round(float(x), 6) for x in X[f]]) for f in feats)} | "
         f"robust {sum(preds)}/{len(preds)} | "
         f"{_DONE[0]}/{recipe['expected_problems']} problems done, {_START + recipe['time_budget_s'] - time.time():.0f}s left")
    return preds
