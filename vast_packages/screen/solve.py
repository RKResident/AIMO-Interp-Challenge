"""Solvability check: k temperature-1 samples per problem (the labelling sampler: T=1, top_k 40, top_p 0.95, as in olympiad/label_olympiad.py),
capped at --cap tokens, for the problems a model is still thinking about at 6144 tokens in its screening trace. A problem with at least one
correct sample is solvable for that model. Samples that hit the cap get a FORCED answer (the submission's suffix B,
"</think> The final answer is $\\boxed{", then 16 greedy tokens) - recorded separately from the natural answer (finished samples). The samples are kept (token ids + answers): they are
also the first k samples of the original in the labelling stage.
Resumable per problem; every record also goes to --persist.
  python solve.py --model Qwen/Qwen3.5-4B --screen $W/out/screen/Qwen__Qwen3.5-4B.jsonl --pool pool.jsonl --out $W/out/solve --persist $PERSIST/solve"""
import argparse, datetime, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "submission_core"))
from spurious_features import SUFFIXES_THINK, find_think_end, think_end_token_id  # noqa: E402
from screen import boxed, gpu, log, norm_int  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--screen", required=True); ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--persist", default="")
    ap.add_argument("--k", type=int, default=4); ap.add_argument("--cap", type=int, default=32768); ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--gpu-util", type=float, default=0.0, help="0 = 90% of the memory free at start")
    ap.add_argument("--deadline-utc", default="", help="HH:MM - no new chunk starts after this (UTC)")
    ap.add_argument("--morning-utc", default="23:30", help="HH:MM - start of the next GPU window (UTC); the window may cross midnight"); ap.add_argument("--kv-cache-dtype", default="auto"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stream", action="store_true", help="request-level loop: write each problem as soon as it is done")
    ap.add_argument("--pids", default="", help="file with one pid per line: sample only these (still thinking at 6144)")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt
    tag = a.model.replace("/", "__")
    out = Path(a.out) / f"{tag}.jsonl"; out.parent.mkdir(parents=True, exist_ok=True)
    per = Path(a.persist) / f"{tag}.jsonl" if a.persist else Path(os.devnull)
    if a.persist: per.parent.mkdir(parents=True, exist_ok=True)
    pool = {json.loads(l)["pid"]: json.loads(l) for l in open(a.pool)}
    scr = [json.loads(l) for l in open(a.screen)]
    surv = [s for s in scr if s["still_thinking"]]
    if a.pids:
        want = {l.strip() for l in open(a.pids) if l.strip()}; surv = [s for s in surv if s["pid"] in want]
    if a.limit: surv = surv[:a.limit]
    done = {json.loads(l)["pid"] for l in open(out)} if out.exists() else set()
    todo = [s for s in surv if s["pid"] not in done]
    log(f"{a.model}: {len(scr)} screened, {len(surv)} still thinking at 6144 -> {len(todo)} to sample | k {a.k} | cap {a.cap} | "
        f"kv cache {a.kv_cache_dtype} | {gpu()}")
    if not todo: return
    tok = AutoTokenizer.from_pretrained(a.model); tid = think_end_token_id(tok)
    longest = max(len(s["prompt_ids"]) for s in todo)
    import torch
    free, total = torch.cuda.mem_get_info(); util = a.gpu_util or round(0.9 * free / total, 3)
    sfx = tok(SUFFIXES_THINK["B"], add_special_tokens=False)["input_ids"]
    log(f"  vLLM memory fraction {util} of {total / 2**30:.0f} GiB (free at start {free / 2**30:.1f} GiB) | forced-answer suffix {len(sfx)} tokens")
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=longest + a.cap + len(sfx) + 32, gpu_memory_utilization=util, kv_cache_dtype=a.kv_cache_dtype,
              seed=0, scheduling_policy="priority", limit_mm_per_prompt={"image": 0, "video": 0} if "Qwen3.5" in a.model else None)
    t0, ntok, nrec, nsolv = time.time(), 0, 0, 0
    forced_sp = SamplingParams(temperature=0.0, max_tokens=16, detokenize=False)

    def write(s, outs, forced):
        """one problem's record: outs = its k sample outputs, forced = {sample index: forced answer}"""
        nonlocal ntok, nrec, nsolv
        gold = norm_int(pool[s["pid"]]["answer"]); samples = []
        for j, o in enumerate(outs):
            g = list(o.token_ids); te = find_think_end(tok, g, tid)
            ans = norm_int(boxed(tok.decode(g[te:], skip_special_tokens=True))) if te is not None else None
            fa = forced.get(j)
            samples.append(dict(gen_ids=g, n_gen=len(g), think_end=te, finish=o.finish_reason, answer=ans, correct=ans == gold,
                                forced_answer=fa, forced_correct=(fa == gold) if fa is not None else None,
                                answer_any=ans if ans is not None else fa, correct_any=(ans if ans is not None else fa) == gold))
            ntok += len(g)
        nc = sum(x["correct"] for x in samples); nca = sum(x["correct_any"] for x in samples); nsolv += nca > 0; nrec += 1
        rec = dict(pid=s["pid"], source=s["source"], model=a.model, answer=gold, k=a.k, cap=a.cap, n_correct=nc, n_correct_any=nca,
                   n_truncated=sum(x["finish"] == "length" for x in samples), samples=samples)
        with open(out, "a") as fo, open(per, "a") as fp:
            line = json.dumps(rec) + "\n"; fo.write(line); fp.write(line)

    if a.stream and a.k > 1:
        sys.exit("--stream is broken for k > 1 (vLLM returns each of the n samples separately; only the first was kept) - use chunks")
    if a.stream:
        # request-level loop: every problem is written the moment its k samples (and their forced answers) are done
        eng = llm.llm_engine; byid = {s["pid"]: s for s in todo}; pend = {}; last = time.time()
        for q, s in enumerate(todo):
            eng.add_request(s["pid"], TokensPrompt(prompt_token_ids=s["prompt_ids"]),
                            SamplingParams(n=a.k, temperature=1.0, top_k=40, top_p=0.95, max_tokens=a.cap, seed=1000 + int(s["pid"][2:]), detokenize=False),
                            priority=q)
        while eng.has_unfinished_requests():
            hm = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M")
            if a.deadline_utc and a.deadline_utc <= hm < a.morning_utc:
                log(f"deadline {a.deadline_utc} UTC reached - stopping ({nrec} done this run; unfinished problems resume next run)"); sys.exit(3)
            for o in eng.step():
                if not o.finished: continue
                if "|f|" not in o.request_id:                                            # a problem's k samples are done
                    s = byid[o.request_id]; outs = list(o.outputs); cut = [j for j, x in enumerate(outs) if x.finish_reason == "length"]
                    if not cut: write(s, outs, {}); continue
                    pend[s["pid"]] = dict(outs=outs, left=set(cut), forced={})
                    for j in cut:
                        eng.add_request(f"{s['pid']}|f|{j}", TokensPrompt(prompt_token_ids=s["prompt_ids"] + list(outs[j].token_ids) + sfx), forced_sp,
                                        priority=-1)                                       # forced answers first
                else:                                                                     # one forced answer is done
                    pid, _, j = o.request_id.split("|"); j = int(j); P = pend[pid]
                    P["forced"][j] = norm_int(boxed("\\boxed{" + tok.decode(list(o.outputs[0].token_ids), skip_special_tokens=True))); P["left"].discard(j)
                    if not P["left"]: write(byid[pid], P["outs"], P["forced"]); del pend[pid]
            if time.time() - last > 300:
                last = time.time(); el = time.time() - t0
                log(f"  [{nrec}/{len(todo)}] problems written | in flight {eng.get_num_unfinished_requests()} requests ({len(pend)} awaiting forced answers) | "
                    f"solvable so far {nsolv}/{nrec} | {ntok / el:.0f} tok/s (finished samples) | ETA ~{(len(todo) - nrec) * el / max(1, nrec) / 3600:.1f} h | {gpu()}")
        log(f"done: {nrec} problems, {nsolv} with >= 1 correct sample, in {(time.time() - t0) / 60:.1f} min"); return

    for c in range(0, len(todo), a.chunk):
        hm = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M")
        if a.deadline_utc and a.deadline_utc <= hm < a.morning_utc:                     # outside the window [morning, deadline) across midnight
            log(f"deadline {a.deadline_utc} UTC reached - stopping cleanly before chunk {c // a.chunk + 1} ({nrec} done this run; re-run resumes)")
            sys.exit(3)                                                                   # 3 = stopped by the deadline (run_solve.sh resumes next morning)
        batch = todo[c:c + a.chunk]; tc = time.time()
        sps = [SamplingParams(n=a.k, temperature=1.0, top_k=40, top_p=0.95, max_tokens=a.cap, seed=1000 + int(s["pid"][2:]), detokenize=False) for s in batch]
        res = llm.generate([TokensPrompt(prompt_token_ids=s["prompt_ids"]) for s in batch], sps, use_tqdm=False)
        # forced answers for the samples that hit the cap (one extra greedy pass each)
        cut = [(i, j) for i, r in enumerate(res) for j, o in enumerate(r.outputs) if o.finish_reason == "length"]
        fres = llm.generate([TokensPrompt(prompt_token_ids=batch[i]["prompt_ids"] + list(res[i].outputs[j].token_ids) + sfx) for i, j in cut],
                            forced_sp, use_tqdm=False) if cut else []
        forced = {ij: norm_int(boxed("\\boxed{" + tok.decode(list(f.outputs[0].token_ids), skip_special_tokens=True))) for ij, f in zip(cut, fres)}
        with open(out, "a") as fo, open(per, "a") as fp:
            for i, (s, r) in enumerate(zip(batch, res)):
                gold = norm_int(pool[s["pid"]]["answer"]); samples = []
                for j, o in enumerate(r.outputs):
                    g = list(o.token_ids); te = find_think_end(tok, g, tid)
                    ans = norm_int(boxed(tok.decode(g[te:], skip_special_tokens=True))) if te is not None else None
                    fa = forced.get((i, j))
                    samples.append(dict(gen_ids=g, n_gen=len(g), think_end=te, finish=o.finish_reason, answer=ans, correct=ans == gold,
                                        forced_answer=fa, forced_correct=(fa == gold) if fa is not None else None,
                                        answer_any=ans if ans is not None else fa, correct_any=(ans if ans is not None else fa) == gold))
                    ntok += len(g)
                nc = sum(x["correct"] for x in samples); nca = sum(x["correct_any"] for x in samples); nsolv += nca > 0; nrec += 1
                rec = dict(pid=s["pid"], source=s["source"], model=a.model, answer=gold, k=a.k, cap=a.cap, n_correct=nc, n_correct_any=nca,
                           n_truncated=sum(x["finish"] == "length" for x in samples), samples=samples)
                line = json.dumps(rec) + "\n"; fo.write(line); fp.write(line)
        log(f"  [{nrec}/{len(todo)}] chunk {len(batch)} in {time.time() - tc:.0f}s | solvable so far {nsolv}/{nrec} (natural or forced answer) | {len(cut)} forced | "
            f"{ntok / (time.time() - t0):.0f} tok/s overall | {gpu()}")
    log(f"done: {nrec} problems, {nsolv} with >= 1 correct sample, in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
