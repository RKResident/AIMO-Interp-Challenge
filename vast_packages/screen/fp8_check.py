"""fp8-vs-bf16 KV-cache check for Olmo before its labelling (user, 2026-10-29). Greedy decoding is deterministic up to batching noise, so:
10 of Olmo's set50 problems, greedy up to 6144 tokens, three ways: the STORED bf16 screening trace, a fresh bf16 RERUN (batching-noise
control) and an fp8 run. Per trace: first token where it diverges from the stored trace, whether it ends in a repetition loop (the
submission's own ends_in_loop rule), and whether thinking ended. Decision (written to fp8_decision.txt): 'fp8' unless the fp8 run loops in
>= 3 more traces than the bf16 rerun, or its median divergence point is below a quarter of the rerun's -> then 'auto' (bf16)."""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "submission_core"))
import numpy as np
from spurious_features import ends_in_loop, find_think_end, think_end_token_id  # noqa: E402
from screen import log  # noqa: E402


def run(model, prompts, kv):
    from vllm import LLM, SamplingParams, TokensPrompt
    import torch, gc
    free, total = torch.cuda.mem_get_info()
    llm = LLM(model=model, dtype="bfloat16", max_model_len=max(len(p) for p in prompts) + 6160, gpu_memory_utilization=round(0.9 * free / total, 3),
              kv_cache_dtype=kv, seed=0)
    out = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], SamplingParams(temperature=0.0, max_tokens=6144, detokenize=False), use_tqdm=False)
    res = [list(o.outputs[0].token_ids) for o in out]
    del llm; gc.collect(); torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="allenai/Olmo-3-7B-Think"); ap.add_argument("--keep", default="set50_olmo_novelfirst.txt")
    ap.add_argument("--screen", required=True); ap.add_argument("--n", type=int, default=10); a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model); tid = think_end_token_id(tok)
    keep = [l.strip() for l in open(a.keep) if l.strip()]
    scr = {json.loads(l)["pid"]: json.loads(l) for l in open(a.screen)}
    rows = [scr[p] for p in keep if p in scr][:a.n]
    prompts = [r["prompt_ids"] for r in rows]; stored = [r["gen_ids"] for r in rows]
    t0 = time.time(); bf = run(a.model, prompts, "auto"); log(f"bf16 rerun done in {time.time() - t0:.0f}s")
    t0 = time.time(); f8 = run(a.model, prompts, "fp8"); log(f"fp8 run done in {time.time() - t0:.0f}s")

    def div(x, y):
        n = min(len(x), len(y))
        for i in range(n):
            if x[i] != y[i]: return i
        return n

    stat = {}
    for nm, G in (("stored bf16", stored), ("bf16 rerun", bf), ("fp8", f8)):
        loops = [bool(ends_in_loop(g)) for g in G]; ended = [find_think_end(tok, g, tid) is not None for g in G]
        d = [div(g, s) for g, s in zip(G, stored)]
        stat[nm] = dict(loops=int(sum(loops)), ended=int(sum(ended)), median_div=float(np.median(d)), div=d)
        log(f"  {nm:12s}: loops {sum(loops)}/{len(G)} | thinking ended {sum(ended)}/{len(G)} | first divergence from stored (tokens) median "
            f"{np.median(d):.0f}, per trace {d}")
    bad = stat["fp8"]["loops"] - stat["bf16 rerun"]["loops"] >= 3 or stat["fp8"]["median_div"] < 0.25 * max(1.0, stat["bf16 rerun"]["median_div"])
    dec = "auto" if bad else "fp8"
    Path("fp8_decision.txt").write_text(dec + "\n"); json.dump(stat, open("fp8_check.json", "w"))
    log(f"FP8 CHECK DECISION: Olmo labelling KV cache = {dec} ({'fp8 looks pathological -> bf16' if bad else 'fp8 within the bf16 noise / no gross failure'})")


if __name__ == "__main__":
    main()
