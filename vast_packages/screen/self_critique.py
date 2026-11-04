"""Self-Critique (Tao et al., ICLR 2026, arXiv 2510.09259; user 2026-10-30: try it for memorization AND robustness).
Per problem:
  r1 = the model's greedy response (our stored 6144-token screening trace when there is one, else generated greedy here, cap 6144);
  E1 = its per-token entropy, from top-K log-probs (paper Table 4: top-K entropy ~ exact), teacher-forced over the stored trace;
  r2 = the response to  q (+) I_critique(r1)  - the paper's Prompt 2 verbatim, r1's text pasted in - generated greedy (the paper's best
       setting, App. C.2) AND at T=1 (the user's variant), cap 6144, with its entropies E2;
  score = cos(pad(E1), pad(E2)) * min(|E1|,|E2|) / max(|E1|,|E2|)  (paper eq. 9; higher = contaminated / RL-memorized). Other alignments
  are computed offline from the saved sequences.
Entropy H = -sum_topK p log p (unnormalised, the paper's top-K masses) and also renormalised over the top K; both are saved.
Output: <out>/<tag>.jsonl (one summary line per problem, resumable) + <out>/<tag>/<pid>.npz (E1, E2 per temperature, r2 token ids).
  python self_critique.py --model Qwen/Qwen3.5-4B --pool pool.jsonl --keep set100.txt --screen out/screen/Qwen__Qwen3.5-4B.jsonl --out out/sc"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "submission_core"))
import numpy as np
from spurious_features import PROMPT_TEMPLATE, build_prompt, find_think_end, think_end_token_id  # noqa: E402
from screen import boxed, gpu, log, norm_int  # noqa: E402

K = 20
CRITIQUE = ("A possible answer is provided below (it may or may not be correct). Please provide a response that follows a different reasoning "
            "path or provides an alternative solution:\n—\n{r1}\n—\nPlease now provide your new, different response:")


def ent(d):
    """(unnormalised, renormalised) top-K entropy of one position's logprob dict"""
    lp = np.array(sorted((v.logprob for v in d.values()), reverse=True)[:K], float); p = np.exp(lp)
    q = p / p.sum(); return float(-(p * lp).sum()), float(-(q * np.log(q + 1e-30)).sum())


def cos_pen(a, b):
    n = max(len(a), len(b))
    if n == 0 or min(len(a), len(b)) == 0: return float("nan")
    A = np.zeros(n); B = np.zeros(n); A[:len(a)] = a; B[:len(b)] = b
    return float(A @ B / (np.linalg.norm(A) * np.linalg.norm(B) + 1e-12) * min(len(a), len(b)) / n)


def chat(tok, user):
    if getattr(tok, "chat_template", None):
        text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
        return list(tok(text, add_special_tokens=False)["input_ids"])
    return list(tok(user)["input_ids"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--pool", required=True); ap.add_argument("--keep", default="")
    ap.add_argument("--screen", default=""); ap.add_argument("--out", required=True); ap.add_argument("--cap", type=int, default=6144)
    ap.add_argument("--temps", default="0,1"); ap.add_argument("--chunk", type=int, default=16); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu-util", type=float, default=0.88); ap.add_argument("--kv-cache-dtype", default="auto")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt
    tag = a.model.replace("/", "__"); out = Path(a.out) / f"{tag}.jsonl"; nd = Path(a.out) / tag; nd.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["pid"] for l in open(out)} if out.exists() else set()
    pool = {json.loads(l)["pid"]: json.loads(l) for l in open(a.pool)}
    keep = [l.strip() for l in open(a.keep) if l.strip()] if a.keep else list(pool)
    scr = {json.loads(l)["pid"]: json.loads(l) for l in open(a.screen)} if a.screen and os.path.exists(a.screen) else {}
    todo = [p for p in keep if p in pool and p not in done]
    if a.limit: todo = todo[:a.limit]
    temps = [float(t) for t in a.temps.split(",")]
    log(f"{a.model}: {len(todo)} problems to do ({len(done)} done; {sum(p in scr for p in todo)} with a stored greedy trace) | temps {temps} | {gpu()}")
    if not todo: return
    tok = AutoTokenizer.from_pretrained(a.model); tid = think_end_token_id(tok)
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=2 * a.cap + 2048, gpu_memory_utilization=a.gpu_util, kv_cache_dtype=a.kv_cache_dtype,
              seed=0, enable_prefix_caching=True, limit_mm_per_prompt={"image": 0, "video": 0} if "Qwen3.5" in a.model else None)
    t0 = time.time(); last = t0; nrec = 0
    for c in range(0, len(todo), a.chunk):
        batch = todo[c:c + a.chunk]
        P1 = {p: (scr[p]["prompt_ids"] if p in scr else build_prompt(tok, pool[p]["problem"])[0]) for p in batch}
        R1 = {p: scr[p]["gen_ids"] for p in batch if p in scr}; E1 = {}
        gen = [p for p in batch if p not in R1]                                   # no stored trace -> greedy r1 here, entropies from its logprobs
        if gen:
            res = llm.generate([TokensPrompt(prompt_token_ids=P1[p]) for p in gen], SamplingParams(temperature=0.0, max_tokens=a.cap, logprobs=K, detokenize=False), use_tqdm=False)
            for p, r in zip(gen, res):
                R1[p] = list(r.outputs[0].token_ids); E1[p] = np.array([ent(d) for d in r.outputs[0].logprobs])
        tf = [p for p in batch if p not in E1]                                    # stored trace -> teacher-forced entropies
        if tf:
            res = llm.generate([TokensPrompt(prompt_token_ids=P1[p] + R1[p]) for p in tf], SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=K, detokenize=False), use_tqdm=False)
            for p, r in zip(tf, res):
                n1 = len(P1[p]); E1[p] = np.array([ent(d) for d in r.prompt_logprobs[n1:]])   # entry i scores token i: the trace = positions n1..
        P2 = {p: chat(tok, PROMPT_TEMPLATE.format(problem=pool[p]["problem"]) + "\n\n" + CRITIQUE.format(r1=tok.decode(R1[p], skip_special_tokens=True).strip())) for p in batch}
        E2, G2 = {}, {}
        for T in temps:
            sp = SamplingParams(temperature=T, top_p=1.0, max_tokens=a.cap, logprobs=K, detokenize=False, seed=0 if T > 0 else None)
            res = llm.generate([TokensPrompt(prompt_token_ids=P2[p]) for p in batch], sp, use_tqdm=False)
            for p, r in zip(batch, res):
                G2[(p, T)] = list(r.outputs[0].token_ids); E2[(p, T)] = np.array([ent(d) for d in r.outputs[0].logprobs])
        with open(out, "a") as fo:
            for p in batch:
                e1 = E1[p]; rec = dict(pid=p, model=a.model, source=pool[p].get("source"), stored_r1=p in scr, n_r1=len(R1[p]), n_p2=len(P2[p]),
                                       r1_still_thinking=find_think_end(tok, R1[p], tid) is None, mean_e1=float(e1[:, 0].mean()) if len(e1) else None)
                arr = {"E1": e1.astype(np.float32), "R1": np.array(R1[p], np.int32)}
                for T in temps:
                    e2, g = E2[(p, T)], G2[(p, T)]; te = find_think_end(tok, g, tid)
                    ans = norm_int(boxed(tok.decode(g[te:], skip_special_tokens=True))) if te is not None else None
                    rec[f"T{T:g}"] = dict(n_r2=len(g), still_thinking=te is None, answer=ans, correct=(ans == norm_int(pool[p]["answer"])) if ans is not None and pool[p].get("answer") is not None else None,
                                          mean_e2=float(e2[:, 0].mean()) if len(e2) else None, score=cos_pen(e1[:, 0], e2[:, 0]) if len(e2) else None,
                                          score_renorm=cos_pen(e1[:, 1], e2[:, 1]) if len(e2) else None)
                    arr[f"E2_T{T:g}"] = e2.astype(np.float32); arr[f"R2_T{T:g}"] = np.array(g, np.int32)
                np.savez_compressed(nd / f"{p}.npz", **arr); fo.write(json.dumps(rec) + "\n"); nrec += 1
        if time.time() - last > 300 or c + a.chunk >= len(todo):
            el = time.time() - t0; last = time.time(); s0 = [json.loads(l)["T0"]["score"] for l in open(out) if "T0" in json.loads(l)]
            log(f"  [{nrec}/{len(todo)}] {el / 60:.1f} min | ETA {el / nrec * (len(todo) - nrec) / 60:.0f} min | greedy-r2 score so far: median "
                f"{np.nanmedian([s for s in s0 if s is not None]):.3f} | {gpu()}")
    log(f"done: {nrec} problems in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
