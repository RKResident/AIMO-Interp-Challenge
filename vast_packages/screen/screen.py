"""Screening pass: one greedy trace per problem, capped at 6144 generated tokens (the submission's cap), with the submission's own prompt
(submission_core.build_prompt). Records whether the model is still thinking at the cap (no </think> within 6144) and, if it finished, its
boxed answer and whether that is correct. The trace is also what the features are computed on later.
Resumable: problems already in the output are skipped. Every record is appended to OUT and copied (appended) to PERSIST as it is written.
  python screen.py --model Qwen/Qwen3.5-4B --pool pool.jsonl --out $W/out/screen --persist $PERSIST/screen [--kv-cache-dtype fp8]"""
import argparse, json, os, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "submission_core"))
from spurious_features import build_prompt, find_think_end, think_end_token_id  # noqa: E402


def log(msg): print(time.strftime("%H:%M:%S"), msg, flush=True)


def boxed(text):
    """content of the last \\boxed{...} (balanced braces)"""
    i = text.rfind("\\boxed{")
    if i < 0: return None
    j, depth = i + 7, 1
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0); j += 1
    return text[i + 7:j - 1].strip() if depth == 0 else None


def norm_int(s):
    if s is None: return None
    s = s.replace(",", "").replace("$", "").replace("\\!", "").replace(" ", "")
    m = re.fullmatch(r"-?\d+", s)
    return str(int(s)) if m else s


def gpu():
    import torch
    f, t = torch.cuda.mem_get_info(); return f"GPU free {f / 2**30:.1f}/{t / 2**30:.1f} GiB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--persist", default="", help="second copy of every record (a persistent disk); empty = none")
    ap.add_argument("--cap", type=int, default=6144); ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--gpu-util", type=float, default=0.85); ap.add_argument("--kv-cache-dtype", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="first N problems only (benchmark)")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt
    tag = a.model.replace("/", "__")
    out = Path(a.out) / f"{tag}.jsonl"; per = Path(a.persist) / f"{tag}.jsonl" if a.persist else Path(os.devnull)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.persist: per.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["pid"] for l in open(out)} if out.exists() else set()
    pool = [json.loads(l) for l in open(a.pool)]
    if a.limit: pool = pool[:a.limit]
    todo = [p for p in pool if p["pid"] not in done]
    log(f"{a.model}: {len(pool)} problems, {len(done)} done, {len(todo)} to do | cap {a.cap} | kv cache {a.kv_cache_dtype} | {gpu()}")
    if not todo: return
    tok = AutoTokenizer.from_pretrained(a.model); tid = think_end_token_id(tok)
    prompts = {p["pid"]: build_prompt(tok, p["problem"])[0] for p in todo}
    longest = max(len(v) for v in prompts.values())
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=longest + a.cap + 16, gpu_memory_utilization=a.gpu_util,
              kv_cache_dtype=a.kv_cache_dtype, seed=0, limit_mm_per_prompt={"image": 0, "video": 0} if "Qwen3.5" in a.model else None)
    sp = SamplingParams(temperature=0.0, max_tokens=a.cap, detokenize=False)
    t0, ntok, nrec = time.time(), 0, 0
    for c in range(0, len(todo), a.chunk):
        batch = todo[c:c + a.chunk]; tc = time.time()
        res = llm.generate([TokensPrompt(prompt_token_ids=prompts[p["pid"]]) for p in batch], sp, use_tqdm=False)
        with open(out, "a") as fo, open(per, "a") as fp:
            for p, r in zip(batch, res):
                g = list(r.outputs[0].token_ids); te = find_think_end(tok, g, tid)
                ans = norm_int(boxed(tok.decode(g[te:], skip_special_tokens=True))) if te is not None else None
                rec = dict(pid=p["pid"], source=p["source"], model=a.model, answer=p["answer"], prompt_ids=prompts[p["pid"]], gen_ids=g,
                           n_gen=len(g), think_end=te, still_thinking=te is None, finish=r.outputs[0].finish_reason,
                           final_answer=ans, correct=(ans == norm_int(p["answer"])) if ans is not None else None)
                line = json.dumps(rec) + "\n"; fo.write(line); fp.write(line)
                ntok += len(g); nrec += 1
        el = time.time() - t0
        st = sum(1 for p, r in zip(batch, res) if find_think_end(tok, list(r.outputs[0].token_ids), tid) is None)
        log(f"  [{nrec}/{len(todo)}] chunk {len(batch)} in {time.time() - tc:.0f}s | still thinking at cap {st}/{len(batch)} | "
            f"{ntok / el:.0f} tok/s overall | {gpu()}")
    log(f"done: {nrec} problems in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
