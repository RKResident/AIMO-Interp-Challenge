"""fp8 vs bf16 KV cache, measured on token trajectories (user, 2026-10-29). 10 problems per model (novel set), stored bf16 greedy screening
traces (6144 tokens). Three levels:
 1. TEACHER-FORCED, same trajectory: per-token log-prob of the stored trace tokens under a bf16 cache and an fp8 cache (prompt_logprobs; prefill
    chunked into 512-token pieces so later chunks READ the cache - one 6k chunk would never touch fp8). Noise floor: bf16 with 2048-token chunks
    vs bf16 with 512. Reported: mean / p99 |delta log p|, mean KL between top-k-approximated next-token distributions, top-1 flip rate.
 2. FREE-RUNNING greedy 6144 tokens (bf16 rerun = batching-noise control, fp8): first divergence from the stored trace, 8-gram Jaccard with
    the stored trace after the divergence point, length, loops, thinking ended.
 3. ANSWER: forced answer (suffix B, 16 greedy tokens) at the cut of each free-running trace vs the stored trace's.
Writes fp8_check2_<model>.json and prints a summary."""
import argparse, gc, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "submission_core"))
import numpy as np
import torch
from spurious_features import SUFFIXES_THINK, ends_in_loop, find_think_end, think_end_token_id  # noqa: E402
from screen import boxed, log, norm_int  # noqa: E402

K = 20                                                                                # top-k for the approximate KL / flip rate


def engine(model, kv, chunk, maxlen):
    from vllm import LLM
    free, total = torch.cuda.mem_get_info()
    import os
    extra = {"attention_backend": os.environ["ATTN_BACKEND"]} if os.environ.get("ATTN_BACKEND") else {}   # same kernels for fp8 and bf16
    return LLM(model=model, dtype="bfloat16", max_model_len=maxlen, gpu_memory_utilization=round(0.88 * free / total, 3), kv_cache_dtype=kv, seed=0, **extra,
               enable_chunked_prefill=True, max_num_batched_tokens=chunk, limit_mm_per_prompt={"image": 0, "video": 0} if "Qwen3.5" in model else None)


def release(llm):
    del llm; gc.collect(); torch.cuda.empty_cache()


def teacher_forced(llm, seqs):
    from vllm import SamplingParams, TokensPrompt
    out = llm.generate([TokensPrompt(prompt_token_ids=s) for s in seqs], SamplingParams(max_tokens=1, prompt_logprobs=K, temperature=0.0), use_tqdm=False)
    res = []
    for s, o in zip(seqs, out):
        pl = o.prompt_logprobs; tok_lp, tops = [], []
        for t, d in zip(s[1:], pl[1:]):
            tok_lp.append(d[t].logprob if t in d else float("nan")); tops.append({k: v.logprob for k, v in d.items()})
        res.append((np.array(tok_lp), tops))
    return res


def free_run(llm, prompts, sfx):
    from vllm import SamplingParams, TokensPrompt
    out = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], SamplingParams(temperature=0.0, max_tokens=6144, detokenize=False), use_tqdm=False)
    gens = [list(o.outputs[0].token_ids) for o in out]
    fo = llm.generate([TokensPrompt(prompt_token_ids=p + g + sfx) for p, g in zip(prompts, gens)], SamplingParams(temperature=0.0, max_tokens=16, detokenize=False), use_tqdm=False)
    return gens, [list(o.outputs[0].token_ids) for o in fo]


def grams(x, n=8): return {tuple(x[i:i + n]) for i in range(len(x) - n + 1)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--keep", required=True); ap.add_argument("--screen", required=True)
    ap.add_argument("--n", type=int, default=10); ap.add_argument("--phase", required=True, help="bf16_c2048 | bf16_c512 | fp8_c512 | compare")
    a = ap.parse_args()
    import pickle
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model); tid = think_end_token_id(tok); sfx = tok(SUFFIXES_THINK["B"], add_special_tokens=False)["input_ids"]
    keep = [l.strip() for l in open(a.keep) if l.strip()]; scr = {json.loads(l)["pid"]: json.loads(l) for l in open(a.screen)}
    rows = [scr[p] for p in keep if p in scr][:a.n]; prompts = [r["prompt_ids"] for r in rows]; stored = [r["gen_ids"] for r in rows]
    seqs = [p + g for p, g in zip(prompts, stored)]; maxlen = max(len(s) for s in seqs) + len(sfx) + 32
    log(f"{a.model}: {len(rows)} problems, stored traces {[len(g) for g in stored]}")
    CFG = {"bf16_c2048": ("auto", 2048), "bf16_c512": ("auto", 512), "fp8_c512": ("fp8", 512)}
    pk = lambda nm: Path(f"fp8_{a.model.replace('/', '__')}_{nm}.pkl")
    if a.phase in CFG:
        kv, chunk = CFG[a.phase]; t0 = time.time(); llm = engine(a.model, kv, chunk, maxlen); tf_ = teacher_forced(llm, seqs)
        fr_ = free_run(llm, prompts, sfx) if a.phase != "bf16_c2048" else None
        pickle.dump((tf_, fr_), open(pk(a.phase), "wb")); log(f"  {a.phase}: done in {time.time() - t0:.0f}s"); return
    TF, FR = {}, {}
    for nm in CFG:
        TF[nm], fr_ = pickle.load(open(pk(nm), "rb"))
        if fr_ is not None: FR[nm] = fr_
    def compare(x, y, nps):
        """trace tokens only: index i of the arrays scores token s[i + 1], so the trace starts at index n_p - 1"""
        d, kl, flip = [], [], []
        for (lx, tx), (ly, ty), n_p in zip(x, y, nps):
            lx, ly = lx[n_p - 1:], ly[n_p - 1:]; m = np.isfinite(lx) & np.isfinite(ly); d.append(np.abs(lx - ly)[m])
            for a_, b_ in zip(tx[n_p - 1:], ty[n_p - 1:]):
                flip.append(max(a_, key=a_.get) != max(b_, key=b_.get)); ks = set(a_) & set(b_)
                if ks:
                    pa = np.array([np.exp(a_[k]) for k in ks]); pb = np.array([np.exp(b_[k]) for k in ks])
                    pa, pb = pa / pa.sum(), pb / pb.sum(); kl.append(float((pa * (np.log(pa + 1e-12) - np.log(pb + 1e-12))).sum()))
        d = np.concatenate(d); return dict(mean_abs_dlogp=float(d.mean()), p99_abs_dlogp=float(np.quantile(d, .99)), max_abs_dlogp=float(d.max()),
                                           mean_kl_topk=float(np.mean(kl)), top1_flip_rate=float(np.mean(flip)))
    nps = [len(p) for p in prompts]
    tf = {"noise floor (bf16 2048 vs 512 chunks)": compare(TF["bf16_c2048"], TF["bf16_c512"], nps), "fp8 vs bf16 (both 512 chunks)": compare(TF["fp8_c512"], TF["bf16_c512"], nps)}
    log("  TEACHER-FORCED on the stored trajectories (trace tokens only):")
    for k, v in tf.items(): log(f"     {k:38s}: mean |dlogp| {v['mean_abs_dlogp']:.5f} | p99 {v['p99_abs_dlogp']:.4f} | max {v['max_abs_dlogp']:.3f} | "
                                 f"KL(top-{K}) {v['mean_kl_topk']:.6f} | top-1 flips {v['top1_flip_rate']:.4%}")

    def divergence(x, y):
        n = min(len(x), len(y))
        for i in range(n):
            if x[i] != y[i]: return i
        return n
    fr = {}
    for nm, (gens, fans) in FR.items():
        dv = [divergence(g, s) for g, s in zip(gens, stored)]
        jac = [len(grams(g[d:]) & grams(s[d:])) / max(1, len(grams(g[d:]) | grams(s[d:]))) if d < min(len(g), len(s)) - 8 else 1.0 for g, s, d in zip(gens, stored, dv)]
        loops = [bool(ends_in_loop(g)) for g in gens]; ended = [find_think_end(tok, g, tid) is not None for g in gens]
        ans = [norm_int(boxed("\\boxed{" + tok.decode(f, skip_special_tokens=True))) for f in fans]
        fr[nm] = dict(divergence=dv, median_div=float(np.median(dv)), jaccard8_after_div=[round(j, 3) for j in jac], mean_jaccard=float(np.mean(jac)),
                      loops=int(sum(loops)), ended=int(sum(ended)), lengths=[len(g) for g in gens], answers=ans)
        log(f"  FREE-RUNNING {nm:10s}: first divergence from stored median {np.median(dv):.0f} tokens {dv} | 8-gram Jaccard after it {np.mean(jac):.3f} | "
            f"loops {sum(loops)}/{len(gens)} | thinking ended {sum(ended)}/{len(gens)}")
    a0, a1 = fr["bf16_c512"]["answers"], fr["fp8_c512"]["answers"]
    same = sum(x == y for x, y in zip(a0, a1)); log(f"  ANSWERS (forced at the cut): bf16 rerun {a0} | fp8 {a1} | identical {same}/{len(a0)}")
    json.dump(dict(model=a.model, teacher_forced=tf, free_running=fr), open(f"fp8_check2_{a.model.replace('/', '__')}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
