"""Direct perturbation test + activation patching (Vast, one process per model; run.sh). No generation: every pass teacher-forces traces
we already have and forces an answer with suffix B ("</think> The final answer is $\\boxed{").

A  direct test (olympiad; DeepSeek + Skywork): for each labelled problem, the original and its 5 perturbed versions (distract, domain, rename,
   rephrase, typos) with the friend's greedy completions: the answer forced at 2048 / 4096 / 6144 tokens (or at the end of thinking if
   earlier). Locally: robust iff the perturbed answers agree with the original's - what a submission could do at the 6144 cap.
B  activation patching on the white-box run's 118 labelled greedy traces (cut at end of thinking / 6144). The clean run is prompt + trace +
   suffix; the answer token is its top-1.
   B1 attention knockout: in the suffix pass (the tokens that produce the answer), block attention to (a) the problem's tokens, (b) the
      reasoning trace, at every softmax-attention layer -> probability of the clean answer. A large drop for (a) = the answer reads the
      problem directly (shortcut); for (b) = it rests on the reasoning.
   B2 rewording patching: run the trace after each of 3 automatic rewordings of the problem; then, layer by layer, copy the reworded run's
      residual stream at the answer position into the clean run -> probability of the clean answer per layer.
   B3 reasoning patching: the same with the no-reasoning run (prompt + suffix only) as the source.
Outputs <out>/<model>/: direct.jsonl, patch.jsonl (resumable)."""
from __future__ import annotations

import argparse, json, re, sys, time, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "submission_core"))
import numpy as np
import torch

import spurious_inference as SI
from spurious_features import SUFFIXES_NOTHINK, SUFFIXES_THINK, auto_variants, build_prompt, find_think_end, think_end_token_id
from run_t1 import gpu, log, prefetch, final_norm
from run_whitebox import prefix_cache, suffix_pass, continuation

ORDER = ["Qwen/Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B", "allenai/Olmo-3-7B-Think", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"]


def parse_answer(s):
    s = s.split("}")[0].replace(",", "").replace("$", "").strip()
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return m.group(0) if m else s


# ----------------------------------------------------------------------------------------------------------------- B1 knockout hook
_KO = {"on": False, "block": None, "orig": None, "S": 0}


def _ko_sdpa(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kw):
    if _KO["on"] and _KO["block"] is not None:
        q, T = query.shape[2], key.shape[2]
        kpos = torch.arange(T, device=query.device); qpos = T - q + torch.arange(q, device=query.device)
        allowed = kpos[None, :] <= qpos[:, None]
        win = getattr(module, "sliding_window", None)
        if win:
            allowed = allowed & (kpos[None, :] > qpos[:, None] - win)
        blk = _KO["block"][_KO["block"] < T]
        rows_ = slice(max(0, q - _KO["S"]), q)                         # only the suffix queries (also on a full-sequence fallback pass)
        allowed[rows_, blk] = False
        m = torch.zeros(q, T, dtype=query.dtype, device=query.device).masked_fill(~allowed, float("-inf"))[None, None]
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            m = m + attention_mask[..., -T:].to(m.dtype)
        return _KO["orig"](module, query, key, value, m, dropout=dropout, scaling=scaling, **{k: v for k, v in kw.items() if k != "is_causal"})
    return _KO["orig"](module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kw)


def install_ko():
    if _KO["orig"] is None:
        from transformers import AttentionInterface
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        _KO["orig"] = sdpa_attention_forward; AttentionInterface.register("sdpa", _ko_sdpa)


def capture_hook(layers, store):
    """forward hooks storing every decoder layer's raw output at the LAST position (hidden_states[-1] is post-final-norm, so not used)"""
    def make():
        hs = []
        for i, layer in enumerate(layers):
            def fn(m, a, o, i=i):
                store[i] = (o[0] if isinstance(o, tuple) else o)[0, -1].detach().clone()
            hs.append(layer.register_forward_hook(fn))
        return hs
    return make


def patch_hook(layer, vec):
    """forward hook replacing the layer's output at the LAST position with vec"""
    def make():
        def fn(m, a, o):
            h = o[0] if isinstance(o, tuple) else o
            h = h.clone(); h[0, -1] = vec.to(h.dtype)
            return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
        return layer.register_forward_hook(fn)
    return make


# ----------------------------------------------------------------------------------------------------------------- A direct test
def direct_stage(model, tok, ctx, rows, odir):
    out = odir / "direct.jsonl"; done = {json.loads(l)["key"] for l in open(out)} if out.exists() else set()
    sfx = ctx["suffixes"]["B"]; todo = [(r, it) for r in rows for it in r["items"] if f"{r['pid']}|{it['ptype']}" not in done]
    log(f"  direct test: {sum(len(r['items']) for r in rows)} completions, {len(todo)} to do")
    for n, (r, it) in enumerate(todo):
        t = time.time(); prompt = build_prompt(tok, it["text"])[0]; gen = it["gen_ids"]
        te = find_think_end(tok, gen, ctx["tid"]); L = te if te is not None else len(gen)
        rec = dict(key=f"{r['pid']}|{it['ptype']}", pid=r["pid"], ptype=it["ptype"], label=r["label"], think_end=te, n_tokens=it["n_tokens"],
                   finish=it["finish"], final_answer=it["final_answer"], correct=it["correct"], forced={})
        for cut in (2048, 4096, 6144):
            c = max(1, min(cut, L))
            cache = prefix_cache(model, prompt + gen[:c])
            lp, _, kv = suffix_pass(model, cache, prompt + gen[:c], sfx); top = int(lp.argmax())
            ans = continuation(model, tok, kv, top, n=12); del kv, cache
            rec["forced"][str(cut)] = dict(cut=c, answer=parse_answer(ans), raw=ans[:30], p1=float(lp.exp()[top]))
        with open(out, "a") as f: f.write(json.dumps(rec) + "\n")
        log(f"    [{n + 1}/{len(todo)}] {rec['key']} {r['label']}: forced@6144 {rec['forced']['6144']['answer']} (p {rec['forced']['6144']['p1']:.2f}) | "
            f"final {it['final_answer']} ({'finished' if te is not None else 'unfinished'} at {L}) | {time.time() - t:.1f}s | {gpu()}")
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------------------------- B patching
def patch_stage(model, tok, ctx, rows, odir):
    install_ko()
    out = odir / "patch.jsonl"; done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    sfx = ctx["suffixes"]["B"]; layers = ctx["layers"]; nL = len(layers)
    todo = [r for r in rows if r["id"] not in done]; log(f"  patching: {len(rows)} rows, {len(todo)} to do")
    for n, r in enumerate(todo):
        t = time.time(); prompt = r["prompt_ids"]; gen = r["greedy_gen_ids"][:6144]
        te = find_think_end(tok, gen, ctx["tid"]); c = min(max(1, te if te is not None else len(gen)), 6144); trace = gen[:c]
        span = build_prompt(tok, r["problem"])[2]; n_p = len(prompt)
        cache = prefix_cache(model, prompt + trace)
        lp0, _, kv = suffix_pass(model, cache, prompt + trace, sfx); del kv
        top = int(lp0.argmax()); p0 = float(lp0.exp()[top])
        rec = dict(id=r["id"], dataset=r["dataset"], robust=r["robust"], cut=c, think_end=te, clean_top=top, clean_p=p0, n_layers=nL)
        # B1 knockout from the suffix queries
        ko = {}
        for nm, blk in (("problem", np.arange(span[0], span[1]) if span else np.arange(0, n_p)), ("trace", np.arange(n_p, n_p + c)),
                        ("prompt_all", np.arange(0, n_p))):
            _KO.update(on=True, block=torch.tensor(blk, device=model.device), S=len(sfx))
            try:
                lp, _, kv = suffix_pass(model, cache, prompt + trace, sfx); del kv
            finally:
                _KO.update(on=False, block=None)
            ko[nm] = dict(p_clean=float(lp.exp()[top]), flip=bool(int(lp.argmax()) != top))
        rec["knockout"] = ko
        # B2 / B3 sources: reworded prompts with the same trace; the no-reasoning run
        sources = {}
        for k, v in enumerate(auto_variants(r["problem"])):
            pv = build_prompt(tok, v)[0]; cv = prefix_cache(model, pv + trace)
            hs = {}; lpv, _, kv = suffix_pass(model, cv, pv + trace, sfx, hook=capture_hook(layers, hs)); del kv, cv
            sources[f"reword{k}"] = (hs, float(lpv.exp()[top]))
        cn = prefix_cache(model, prompt)
        hs = {}; lpn, _, kv = suffix_pass(model, cn, prompt, sfx, hook=capture_hook(layers, hs)); del kv, cn
        sources["noreason"] = (hs, float(lpn.exp()[top]))
        rec["patch"] = {}
        for nm, (hs, p_src) in sources.items():
            curve = []
            for li in range(nL):
                lp, _, kv = suffix_pass(model, cache, prompt + trace, sfx, hook=patch_hook(layers[li], hs[li])); del kv
                curve.append(round(float(lp.exp()[top]), 5))
            rec["patch"][nm] = dict(source_p_clean=p_src, curve=curve)
        del cache, sources; torch.cuda.empty_cache()
        with open(out, "a") as f: f.write(json.dumps(rec) + "\n")
        pc = rec["patch"]
        log(f"    [{n + 1}/{len(todo)}] {r['id']} ({'robust' if r['robust'] else 'spurious'}): clean p {p0:.3f} | knockout problem {ko['problem']['p_clean']:.3f} "
            f"trace {ko['trace']['p_clean']:.3f} | no-reasoning source {pc['noreason']['source_p_clean']:.3f}, patched mid-layer "
            f"{pc['noreason']['curve'][nL // 2]:.3f} | {time.time() - t:.1f}s | {gpu()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=""); ap.add_argument("--out", default=str(HERE / "out_patching")); ap.add_argument("--stages", default="A,B")
    ap.add_argument("--direct_inputs", default=str(HERE / "direct_inputs.jsonl")); ap.add_argument("--patch_inputs", default=str(HERE / "patch_inputs.jsonl"))
    a = ap.parse_args()
    D = [json.loads(l) for l in open(a.direct_inputs)]; P = [json.loads(l) for l in open(a.patch_inputs)]
    models = [m for m in ORDER if not a.models or m in a.models.split(",")]; stages = set(a.stages.split(","))
    import transformers
    log(f"models {models} | stages {sorted(stages)} | {len(D)} olympiad problems, {len(P)} patching rows | torch {torch.__version__} transformers {transformers.__version__} | {gpu()}")
    prefetch(ORDER[ORDER.index(models[-1]) + 1:ORDER.index(models[-1]) + 2] if models else [])
    for m in models:
        odir = Path(a.out) / m.replace("/", "__"); odir.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import snapshot_download
        snapshot_download(m)
        model, tok = SI.load_model_and_tokenizer(m, local_files_only=True); layers, n_layers = SI.find_decoder_layers(model); tid = think_end_token_id(tok)
        has_think = tid is not None or "<think>" in build_prompt(tok, "x")[1]
        sfx = {k: tok(v, add_special_tokens=False)["input_ids"] for k, v in (SUFFIXES_THINK if has_think else SUFFIXES_NOTHINK).items()}
        ctx = dict(layers=layers, n_layers=n_layers, tid=tid, suffixes=sfx, norm=final_norm(model))
        log(f"=== {m}: {n_layers} layers | {gpu()}"); T0 = time.time()
        try:
            if "A" in stages:
                rows = [r for r in D if r["model_id"] == m]
                if rows: direct_stage(model, tok, ctx, rows, odir)
            if "B" in stages:
                patch_stage(model, tok, ctx, [r for r in P if r["model_id"] == m], odir)
        except Exception:                                                       # noqa: BLE001 - log, go on with the next model
            log(f"FAILED {m}:\n{traceback.format_exc()}")
        log(f"=== {m} done in {(time.time() - T0) / 60:.1f} min"); del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
