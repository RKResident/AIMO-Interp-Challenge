"""Labelling samples for the kept problems (after the solvability pass): temperature-1 samples (T=1, top_k 40, top_p 0.95; cap --cap) of
  - the original: --k-orig more samples on top of the solvability pass's k (default 4 + 4 = 8)
  - each perturbation (perturbations.jsonl: pid, ptype, text): round 1 = --k-pert samples; round 2 = --k-pert more, only where the drop is
    still undecided under EITHER scoring rule (natural answers only / natural-or-forced): decided = no drop (perturbation accuracy >= the
    original's) or a clear drop (>= 0.5).
Every sample that hits the cap gets a FORCED answer (suffix B, 16 greedy tokens), stored next to the natural answer, so labels can be built
per class later (pure = natural answers only; forced = natural-or-forced). Token ids of these samples are not kept (answers, lengths,
finish reasons only). Resumable per (pid, ptype, round); every record also goes to --persist. Deadline / morning as in solve.py (exit 3).
  python label.py --model Qwen/Qwen3.5-4B --keep keep_Qwen.txt --solve $W/out/solve/Qwen__Qwen3.5-4B.jsonl --perts perturbations.jsonl \
                  --pool pool.jsonl --out $W/out/label --persist $PERSIST/label"""
import re, argparse, datetime, json, os, sys, time, zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "submission_core"))
from spurious_features import SUFFIXES_THINK, build_prompt, find_think_end, think_end_token_id  # noqa: E402
from screen import boxed, gpu, log, norm_int  # noqa: E402
import spurious_features as _sf  # noqa: E402
_PS = {"raw": "{problem}", "r1": "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."}   # exp112e prompt-format test
if os.environ.get("PROMPT_STYLE") in _PS: _sf.PROMPT_TEMPLATE = _PS[os.environ["PROMPT_STYLE"]]


def acc(samples, rule):
    k = "correct" if rule == "pure" else "correct_any"
    return sum(s[k] for s in samples) / len(samples) if samples else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--keep", required=True); ap.add_argument("--solve", default="")
    ap.add_argument("--perts", required=True); ap.add_argument("--pool", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--persist", default=""); ap.add_argument("--n-orig-total", type=int, default=8, help="original samples wanted in total (solvability + labelling)"); ap.add_argument("--k-pert", type=int, default=4)
    ap.add_argument("--cap", type=int, default=32768); ap.add_argument("--chunk", type=int, default=48)
    ap.add_argument("--gpu-util", type=float, default=0.0); ap.add_argument("--kv-cache-dtype", default="auto")
    ap.add_argument("--deadline-utc", default=""); ap.add_argument("--morning-utc", default="23:30")
    ap.add_argument("--limit", type=int, default=0, help="first N kept problems only (smoke test)")
    ap.add_argument("--orig-only", action="store_true", help="originals only (olympiad redo 2026-10-30: re-sample cut-off originals before spending perturbation samples)")
    ap.add_argument("--force-at", default="", help="comma list of budgets < cap (user 2026-11-02): each sample longer than B also gets a forced answer on its first B tokens, stored in sample['at'][B] = what a cap-B run would have given")
    ap.add_argument("--no-ids", action="store_true", help="do NOT save token ids (default: saved to <out>/ids_<tag>/<pid>__<ptype>__r<round>.npz)")
    ap.add_argument("--rounds", default="1,2", help="which rounds to run (user 2026-10-29: round 1 only, to fit one day)")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt
    import torch
    tag = a.model.replace("/", "__")
    if "Olmo" in a.model and a.kv_cache_dtype == "auto": a.kv_cache_dtype = "fp8"   # user 2026-11-02: Olmo ALWAYS runs with fp8 KV cache (fp8_check2 passed)
    out = Path(a.out) / f"{tag}.jsonl"; out.parent.mkdir(parents=True, exist_ok=True); ids_dir = Path(a.out) / f"ids_{tag}"
    per = Path(a.persist) / f"{tag}.jsonl" if a.persist else Path(os.devnull)
    if a.persist: per.parent.mkdir(parents=True, exist_ok=True)
    keep = [l.strip() for l in open(a.keep) if l.strip()]
    pool = {json.loads(l)["pid"]: json.loads(l) for l in open(a.pool)}
    solve = {json.loads(l)["pid"]: json.loads(l) for l in open(a.solve)} if a.solve and os.path.exists(a.solve) else {}
    perts = {}
    for l in open(a.perts):
        d = json.loads(l); perts.setdefault(d["pid"], {})[d["ptype"]] = d["text"]
    keep = [p for p in keep if p in perts or (a.orig_only and p in pool)]
    if a.limit: keep = keep[:a.limit]
    recs = [json.loads(l) for l in open(out)] if out.exists() else []
    have = {(r["pid"], r["ptype"], r["round"]) for r in recs}
    log(f"{a.model}: {len(keep)} kept problems with perturbations ({sum(p in solve for p in keep)} with solvability samples) | {len(recs)} label records so far | cap {a.cap} | "
        f"kv cache {a.kv_cache_dtype} | {gpu()}")
    from spurious_inference import load_tokenizer                         # DeepSeek: Transformers 5.x LlamaTokenizer drops spaces -> rebuilt (as the submission)
    tok = load_tokenizer(a.model, local_files_only=False); tid = think_end_token_id(tok)
    sfx = tok(SUFFIXES_THINK["B"], add_special_tokens=False)["input_ids"]

    def jobs_round1():
        J = []
        for p in keep:
            need = a.n_orig_total - len(solve.get(p, {}).get("samples", []))
            if (p, "original", 1) not in have and need > 0: J.append((p, "original", 1, pool[p]["problem"], need))
            for t, txt in ([] if a.orig_only else sorted(perts.get(p, {}).items())):
                if (p, t, 1) not in have: J.append((p, t, 1, txt, a.k_pert))
        return J

    def jobs_round2():
        by = {}
        for r in recs: by.setdefault((r["pid"], r["ptype"]), []).extend(r["samples"])
        J = []
        for p in keep:
            orig = solve.get(p, {}).get("samples", []) + by.get((p, "original"), [])
            for t, txt in ([] if a.orig_only else sorted(perts.get(p, {}).items())):
                s = by.get((p, t))
                if not s or (p, t, 2) in have: continue
                undecided = any(0 < acc(orig, rule) - acc(s, rule) < 0.5 for rule in ("pure", "forced"))
                if undecided: J.append((p, t, 2, txt, a.k_pert))
        return J

    free, total = torch.cuda.mem_get_info(); util = a.gpu_util or round(0.9 * free / total, 3)
    longest = max(len(build_prompt(tok, txt)[0]) for p in keep for txt in [pool[p]["problem"], *perts.get(p, {}).values()])
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=longest + a.cap + len(sfx) + 32, gpu_memory_utilization=util,
              kv_cache_dtype=a.kv_cache_dtype, seed=0, scheduling_policy="priority", limit_mm_per_prompt={"image": 0, "video": 0} if "Qwen3.5" in a.model else None)
    forced_sp = SamplingParams(temperature=0.0, max_tokens=16, detokenize=False)
    eng = llm.llm_engine; t0 = time.time(); st = dict(tok=0, samples=0, capped=0, written=0)

    FB = [int(b) for b in a.force_at.split(",") if b.strip() and int(b) < a.cap]

    def write(job, outs, forced, forcedB):
        p, t, r, _, k = job; gold = norm_int(pool[p]["answer"]); S = []
        for j, o in enumerate(outs):
            g = list(o.token_ids); te = find_think_end(tok, g, tid)
            ans = norm_int(boxed(tok.decode(g[te:], skip_special_tokens=True))) if te is not None else None
            fa = forced.get(j); anyv = ans if ans is not None else fa
            S.append(dict(n_gen=len(g), think_end=te, finish=o.finish_reason, answer=ans, correct=ans == gold, forced_answer=fa,
                          forced_correct=(fa == gold) if fa is not None else None, answer_any=anyv, correct_any=anyv == gold))
            if FB: S[-1]["at"] = {str(B): (dict(capped=False, answer=ans, correct=ans == gold, forced_answer=None, answer_any=ans, correct_any=ans == gold) if len(g) <= B else
                                           dict(capped=True, answer=None, correct=False, forced_answer=forcedB.get((j, B)), answer_any=forcedB.get((j, B)), correct_any=forcedB.get((j, B)) == gold)) for B in FB}
            st["tok"] += len(g); st["samples"] += 1; st["capped"] += o.finish_reason == "length"
        rec = dict(pid=p, ptype=t, round=r, model=a.model, answer=gold, cap=a.cap, kv_cache_dtype=a.kv_cache_dtype, samples=S)
        if not a.no_ids:                                                 # user 2026-11-02: keep generated token ids (prompt + each sample) so a later cap raise can CONTINUE samples
            import numpy as np; ids_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(ids_dir / f"{p}__{t}__r{r}.npz", prompt=np.asarray(prom[f"{p}|{t}|{r}"], np.int32),
                                **{f"s{j}": np.asarray(list(o.token_ids), np.int32) for j, o in enumerate(outs)}, seeds=np.asarray([(zlib.crc32(f"{p}|{t}|{r}".encode()) + 7919 * j) % 2**31 for j in range(len(outs))], np.int64))
            rec["ids_file"] = f"ids_{tag}/{p}__{t}__r{r}.npz"
        recs.append(rec); st["written"] += 1
        with open(out, "a") as fo, open(per, "a") as fp:
            line = json.dumps(rec) + "\n"; fo.write(line); fp.write(line)

    def progress(rnd, J, done_jobs, t_rnd):
        el = time.time() - t_rnd; probs = {j[0] for j in J}; full = sum(1 for q in probs if all(x in done_jobs for x in J if x[0] == q))
        eta = (len(J) - len(done_jobs)) * el / max(1, len(done_jobs)) / 3600
        log(f"  round {rnd}: {len(done_jobs)}/{len(J)} versions written ({full}/{len(probs)} problems complete) | in flight "
            f"{eng.get_num_unfinished_requests()} sample requests | samples done {st['samples']} ({st['capped']} hit the cap, forced) | "
            f"{st['tok'] / (time.time() - t0):.0f} tok/s | ETA ~{eta:.1f} h at this rate | {gpu()}")

    for rnd, make in [(r_, m_) for r_, m_ in ((1, jobs_round1), (2, jobs_round2)) if str(r_) in a.rounds.split(",")]:
        J = make(); log(f"  round {rnd}: {len(J)} (problem, version) jobs, submitted problem by problem (they finish roughly in that order)")
        byid = {f"{j[0]}|{j[1]}|{j[2]}": j for j in J}; prom = {}; pend = {}; done_jobs = set(); t_rnd = time.time(); last = 0
        # one request per SAMPLE (n=1, own seed): a version is written only when all its k samples + forced answers are in
        for q, (rid, (p, t, r, txt, k)) in enumerate(byid.items()):
            prom[rid] = build_prompt(tok, txt)[0]; pend[rid] = dict(outs=[None] * k, left_s=set(range(k)), left_f=set(), forced={}, forcedB={})
            for j in range(k):
                eng.add_request(f"{rid}#s#{j}", TokensPrompt(prompt_token_ids=prom[rid]), SamplingParams(n=1, temperature=1.0, top_k=40, top_p=0.95,
                                max_tokens=a.cap, detokenize=False, seed=(zlib.crc32(f"{p}|{t}|{r}".encode()) + 7919 * j) % 2**31), priority=q)
        while eng.has_unfinished_requests():
            hm = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M")
            if a.deadline_utc and a.deadline_utc <= hm < a.morning_utc:
                progress(rnd, J, done_jobs, t_rnd); log(f"deadline {a.deadline_utc} UTC reached - stopping (unfinished versions resume next run)"); sys.exit(3)
            for o in eng.step():
                if not o.finished: continue
                if "#s#" in o.request_id:                                                 # one sample done
                    rid, j = o.request_id.split("#s#"); j = int(j); P = pend[rid]; x = o.outputs[0]; P["outs"][j] = x; P["left_s"].discard(j)
                    for B in FB:
                        if len(x.token_ids) > B:
                            P["left_f"].add((j, B)); eng.add_request(f"{rid}#f{B}#{j}", TokensPrompt(prompt_token_ids=prom[rid] + list(x.token_ids)[:B] + sfx), forced_sp, priority=-1)
                    if x.finish_reason == "length":
                        P["left_f"].add((j, 0))
                        eng.add_request(f"{rid}#f#{j}", TokensPrompt(prompt_token_ids=prom[rid] + list(x.token_ids) + sfx), forced_sp, priority=-1)
                else:                                                                     # one forced answer done
                    m_ = re.match(r"(.*)#f(\d*)#(\d+)$", o.request_id); rid, B, j = m_.group(1), int(m_.group(2) or 0), int(m_.group(3)); P = pend[rid]
                    v_ = norm_int(boxed("\\boxed{" + tok.decode(list(o.outputs[0].token_ids), skip_special_tokens=True))); P["left_f"].discard((j, B))
                    if B: P["forcedB"][(j, B)] = v_
                    else: P["forced"][j] = v_
                if not P["left_s"] and not P["left_f"]:
                    assert all(x is not None for x in P["outs"]) and len(P["outs"]) == byid[rid][4]
                    write(byid[rid], P["outs"], P["forced"], P["forcedB"]); done_jobs.add(byid[rid]); del pend[rid]
            if time.time() - last > 300: last = time.time(); progress(rnd, J, done_jobs, t_rnd)
        progress(rnd, J, done_jobs, t_rnd)
    log(f"done in {(time.time() - t0) / 60:.1f} min")

if __name__ == "__main__":
    main()
