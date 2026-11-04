"""Self-perturbation test (user 2026-10-31, new idea 2): would a submission that perturbs the problem ITSELF and checks whether its answer
survives predict robustness? Per problem, greedy traces capped at --cap (6144, as the submission generates) for
  original                      (the stored screening trace when given, else generated here)
  code_typos / code_rename / code_distract   perturbations made by deterministic code (deployable in a submission, no other model)
  sonnet_<ptype>                the labelling perturbations (rename, typos, distract) - an upper bound on what code can do
Answer of a trace = its boxed answer if thinking ended within the cap, else the forced answer (suffix B, 16 greedy tokens), as in labelling.
Writes <out>/<tag>.jsonl, one record per problem (resumable), with every version's answer, length and whether thinking ended.
  python direct_test.py --model Qwen/Qwen3.5-4B --pool pool.jsonl --perts perturbations.jsonl --keep set100.txt --screen <screen.jsonl> --out out/direct"""
import argparse, json, os, random, re, sys, time, zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "submission_core"))
from spurious_features import SUFFIXES_THINK, build_prompt, find_think_end, think_end_token_id  # noqa: E402
from screen import boxed, gpu, log, norm_int  # noqa: E402

DISTRACT = ["Note: this problem was posed on a rainy Tuesday afternoon.", "A student's younger brother is learning to play the violin.",
            "The library where this problem was found closes at 6 pm.", "Fun fact: honey never spoils if it is stored properly."]
SKIPV = set("eidEI")
MATH = re.compile(r"(\$\$.*?\$\$|\$[^$]*\$|\\\(.*?\\\)|\\\[.*?\\\])", re.S)   # $$..$$, $..$, \(..\), \[..\]
ism = lambda p: bool(MATH.fullmatch(p))                                                   # e, i, d: constants / differentials; E, I: often sets / identity


def code_typos(text, rng):
    """swap two inner letters in ~12% of plain words of length >= 5, outside math"""
    parts = MATH.split(text)
    def sw(w):
        if len(w) < 5 or not w.isalpha() or rng.random() > 0.12: return w
        j = rng.randrange(1, len(w) - 2); return w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return "".join(p if ism(p) else re.sub(r"(?<!\\)\b[A-Za-z]+", lambda m: sw(m.group(0)), p) for p in parts)


def code_rename(text, rng):
    """consistently rename standalone single-letter variables inside $...$ to unused letters"""
    parts = MATH.split(text); pat = re.compile(r"(?<![\\A-Za-z])([A-Za-z])(?![A-Za-z])")
    used = {m for p in parts if ism(p) for m in pat.findall(p)}
    free_l = [c for c in "abcghjkmnpqrstuvwxyz" if c not in used]; free_u = [c for c in "ABCDFGHJKLMNPQRSTUVWXYZ" if c not in used]
    rng.shuffle(free_l); rng.shuffle(free_u); mp = {}
    for v in sorted(used):
        if v in SKIPV: continue
        pool = free_l if v.islower() else free_u
        if pool: mp[v] = pool.pop()
    return "".join(pat.sub(lambda m: mp.get(m.group(1), m.group(1)), p) if ism(p) else p for p in parts)


def code_distract(text, rng): return rng.choice(DISTRACT) + " " + text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--pool", required=True); ap.add_argument("--perts", required=True)
    ap.add_argument("--keep", required=True); ap.add_argument("--screen", default=""); ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=6144); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--kv-cache-dtype", default="auto")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams, TokensPrompt
    from spurious_inference import load_tokenizer
    tag = a.model.replace("/", "__"); out = Path(a.out) / f"{tag}.jsonl"; out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["pid"] for l in open(out)} if out.exists() else set()
    pool = {json.loads(l)["pid"]: json.loads(l) for l in open(a.pool)}
    perts = {}
    for l in open(a.perts): d = json.loads(l); perts.setdefault(d["pid"], {})[d["ptype"]] = d["text"]
    scr = {json.loads(l)["pid"]: json.loads(l) for l in open(a.screen)} if a.screen and os.path.exists(a.screen) else {}
    keep = [p for p in (l.strip() for l in open(a.keep)) if p and p in pool and p not in done]
    if a.limit: keep = keep[:a.limit]
    tok = load_tokenizer(a.model, local_files_only=False); tid = think_end_token_id(tok); sfx = tok(SUFFIXES_THINK["B"], add_special_tokens=False)["input_ids"]
    jobs = {}                                                          # (pid, version) -> prompt ids; stored original traces skip generation
    texts = {}
    for p in keep:
        rng = random.Random(zlib.crc32(p.encode())); q = pool[p]["problem"]
        V = {"code_typos": code_typos(q, rng), "code_rename": code_rename(q, rng), "code_distract": code_distract(q, rng)}
        V.update({f"sonnet_{t}": perts[p][t] for t in ("rename", "typos", "distract") if t in perts.get(p, {})})
        if p not in scr: V["original"] = q
        texts[p] = V
        for v, t in V.items(): jobs[(p, v)] = build_prompt(tok, t)[0]
    log(f"{a.model}: {len(keep)} problems ({len(done)} done) | {len(jobs)} traces to generate (cap {a.cap}) | {sum(p in scr for p in keep)} stored originals | {gpu()}")
    if not keep: return
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=max(len(v) for v in jobs.values()) + a.cap + len(sfx) + 32, gpu_memory_utilization=0.9,
              kv_cache_dtype=a.kv_cache_dtype, seed=0, limit_mm_per_prompt={"image": 0, "video": 0} if "Qwen3.5" in a.model else None)
    gsp = SamplingParams(temperature=0.0, max_tokens=a.cap, detokenize=False); fsp = SamplingParams(temperature=0.0, max_tokens=16, detokenize=False)
    t0 = time.time(); B = 24
    for c in range(0, len(keep), B):
        ps = keep[c:c + B]; keys = [k for k in jobs if k[0] in ps]
        res = llm.generate([TokensPrompt(prompt_token_ids=jobs[k]) for k in keys], gsp, use_tqdm=False)
        gens = {k: list(r.outputs[0].token_ids) for k, r in zip(keys, res)}
        for p in ps:
            if p in scr: gens[(p, "original")] = scr[p]["gen_ids"][:a.cap]; jobs[(p, "original")] = scr[p]["prompt_ids"]
        allk = [k for k in gens if k[0] in ps]; te = {k: find_think_end(tok, gens[k], tid) for k in allk}
        need = [k for k in allk if te[k] is None]
        fres = llm.generate([TokensPrompt(prompt_token_ids=jobs[k] + gens[k] + sfx) for k in need], fsp, use_tqdm=False)
        forced = {k: norm_int(boxed("\\boxed{" + tok.decode(list(r.outputs[0].token_ids), skip_special_tokens=True))) for k, r in zip(need, fres)}
        with open(out, "a") as fo:
            for p in ps:
                rec = dict(pid=p, model=a.model, gold=norm_int(pool[p]["answer"]), versions={})
                for (pp, v) in [k for k in allk if k[0] == p]:
                    g = gens[(pp, v)]; e = te[(pp, v)]
                    ans = norm_int(boxed(tok.decode(g[e:], skip_special_tokens=True))) if e is not None else forced[(pp, v)]
                    rec["versions"][v] = dict(answer=ans, ended=e is not None, n=len(g))
                o = rec["versions"]["original"]["answer"]
                rec["agree"] = {v: (d["answer"] == o) for v, d in rec["versions"].items() if v != "original"}
                fo.write(json.dumps(rec) + "\n")
        n = c + len(ps); el = time.time() - t0
        log(f"  [{n}/{len(keep)}] {el / 60:.1f} min | ETA {el / n * (len(keep) - n) / 60:.0f} min | {gpu()}")
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
