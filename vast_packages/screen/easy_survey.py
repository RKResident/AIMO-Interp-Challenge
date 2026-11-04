"""EASIER problems for the pool (user 2026-11-01: train-main-v2 / leaderboard problems are much easier for the models than ours).
Integer answers, no figures, deduplicated against the existing pool by normalised text. Sources are tried one by one; a failing source is
skipped and logged. Writes easy_pool.jsonl (pid ez0000.., source, problem, answer) and easy_all.txt.
  python easy_survey.py --existing pool.jsonl --out easy_pool.jsonl"""
import argparse, json, random, re
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

FIG = re.compile(r"\[asy\]|\\includegraphics|figure below|shown below|diagram below|pictured|in the figure|shown in the figure|as shown", re.I)
# (repo, per-source cap, optional filter on a column)
SRC = [("MathArena/aime_2025", 30, None), ("MathArena/hmmt_feb_2025", 30, None), ("AI-MO/aimo-validation-amc", 83, None),
       ("di-zhang-fdu/AIME_1983_2024", 90, ("Year", lambda y: int(y) <= 2022)), ("nlile/hendrycks-MATH-benchmark", 80, ("level", lambda v: str(v) == "5")),
       ("MathArena/brumo_2025", 30, None), ("MathArena/cmimc_2025", 30, None)]


def as_int(a):
    if not isinstance(a, str) and hasattr(a, "__len__"): a = a[0] if len(a) else None
    s = str(a).strip().replace(",", "").replace("$", "").replace("\\boxed{", "").rstrip("}").strip()
    if re.fullmatch(r"-?\d+\.0+", s): s = s.split(".")[0]
    return int(s) if re.fullmatch(r"-?\d+", s) else None


def norm(t): return re.sub(r"\W+", "", str(t).lower())[:200]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--existing", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    seen = {norm(json.loads(l)["problem"]) for l in open(a.existing)}; api = HfApi(); rng = random.Random(0); rows = []
    for repo, cap, flt in SRC:
        try:
            files = [f for f in api.list_repo_files(repo, repo_type="dataset") if f.endswith((".parquet", ".jsonl", ".json", ".csv"))]
            parts = []
            for f in files:
                p = hf_hub_download(repo, f, repo_type="dataset", cache_dir="hf")
                parts.append(pd.read_parquet(p) if f.endswith(".parquet") else pd.read_csv(p) if f.endswith(".csv") else pd.read_json(p, lines=f.endswith(".jsonl")))
            T = pd.concat(parts, ignore_index=True); cols = {c.lower(): c for c in T.columns}
            pc = next(cols[k] for k in ("problem", "question", "prompt") if k in cols); ac = next(cols[k] for k in ("answer", "final_answer", "solution") if k in cols)
            if flt and flt[0].lower() in cols:
                T = T[[bool(flt[1](v)) if pd.notna(v) else False for v in T[cols[flt[0].lower()]]]]
            cand = []
            for _, r in T.iterrows():
                q, ans = str(r[pc]), as_int(r[ac]) if ac != cols.get("solution") else as_int(re.findall(r"\\boxed\{([^}]*)\}", str(r[ac]))[-1:] or [None])
                if ans is None or FIG.search(q) or len(q) < 40 or norm(q) in seen: continue
                cand.append((q, ans)); seen.add(norm(q))
            rng.shuffle(cand); cand = cand[:cap]
            rows += [dict(source=repo, problem=q, answer=str(x)) for q, x in cand]
            print(f"{repo}: {len(T)} rows -> {len(cand)} kept", flush=True)
        except Exception as e:                                           # noqa: BLE001 - skip the source
            print(f"{repo}: SKIPPED ({type(e).__name__}: {str(e)[:120]})", flush=True)
    with open(a.out, "w") as fo:
        for i, r in enumerate(rows): fo.write(json.dumps(dict(pid=f"ez{i:04d}", **r)) + "\n")
    open(a.out.replace("easy_pool.jsonl", "easy_all.txt"), "w").write("\n".join(f"ez{i:04d}" for i in range(len(rows))) + "\n")
    print(f"easy pool: {len(rows)} problems", flush=True)


if __name__ == "__main__":
    main()
