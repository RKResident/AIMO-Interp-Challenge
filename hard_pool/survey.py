"""Survey harder public problem sources: integer answers, no figures. Reads only the needed parquet columns (DeepMath has big solution columns)."""
import re, pandas as pd, pyarrow.parquet as pq, pyarrow as pa
from huggingface_hub import HfApi, hf_hub_download
pa.set_cpu_count(4); pa.set_io_thread_count(2)
api = HfApi()
def as_int(a):
    if not isinstance(a, str) and hasattr(a, "__len__"): a = a[0] if len(a) else None
    s = str(a).strip().replace(",", "").replace("$", "").replace("\\boxed{", "").rstrip("}").strip()
    return int(s) if re.fullmatch(r"-?\d+", s) else None
FIG = re.compile(r"\[asy\]|\\includegraphics|figure below|shown below|diagram below|pictured|in the figure|shown in the figure|as shown", re.I)
SRC = [("ByteDance-Seed/BeyondAIME", None), ("MathArena/hmmt_nov_2025", None), ("MathArena/hmmt_feb_2023", None), ("MathArena/hmmt_feb_2024", None),
       ("MathArena/apex_2025", None), ("MathArena/apex-shortlist", None), ("MathArena/aime_2023_I", None), ("MathArena/aime_2023_II", None),
       ("MathArena/aime_2024_I", None), ("MathArena/aime_2024_II", None), ("MathArena/final_answer_comps", None), ("MathArena/brumo_2025", None),
       ("MathArena/smt_2025", None), ("MathArena/cmimc_2025", None), ("Hothan/OlympiadBench", "OE_TO_maths_en_COMP"), ("KbsdJames/Omni-MATH", None),
       ("zwhe99/DeepMath-103K", None)]
out = []
for ds, cfg in SRC:
    try:
        allf = api.list_repo_files(ds, repo_type="dataset")
        fs = [f for f in allf if f.endswith(".parquet") and (cfg is None or cfg in f)]
        parts = []
        if not fs:                                                       # e.g. Omni-MATH ships jsonl
            for f in [f for f in allf if f.endswith((".jsonl", ".json"))]:
                t = pd.read_json(hf_hub_download(ds, f, repo_type="dataset", cache_dir="hf"), lines=f.endswith(".jsonl"))
                if "problem" in t and "answer" in t: parts.append(t[["problem", "answer"] + (["difficulty"] if "difficulty" in t else [])])
        for f in fs:
            p = hf_hub_download(ds, f, repo_type="dataset", cache_dir="hf")
            names = pq.read_schema(p).names
            pc = next(c for c in ("problem", "question", "Problem") if c in names); ac = next(c for c in ("answer", "final_answer", "Answer") if c in names)
            dc = next((c for c in ("difficulty", "level") if c in names), None)
            t = pq.read_table(p, columns=[pc, ac] + ([dc] if dc else [])).to_pandas()
            t.columns = ["problem", "answer"] + (["difficulty"] if dc else []); parts.append(t)
        d = pd.concat(parts, ignore_index=True)
    except Exception as e:
        print(f"{ds}: skipped ({type(e).__name__}: {str(e)[:80]})"); continue
    a = d.answer.map(as_int); ok = a.notna() & ~d.problem.astype(str).str.contains(FIG)
    dd = pd.to_numeric(d["difficulty"], errors="coerce") if "difficulty" in d else None
    extra = f" | difficulty of usable: median {dd[ok].median():.1f}, >= 7: {(ok & (dd >= 7)).sum()}, >= 8: {(ok & (dd >= 8)).sum()}" if dd is not None else ""
    print(f"{ds:30s} rows {len(d):6d} | integer answer, no figure: {ok.sum():5d}{extra}", flush=True)
    for i in d.index[ok]: out.append(dict(source=ds, problem=str(d.problem[i]), answer=str(int(a[i])), difficulty=(float(dd[i]) if dd is not None else None)))
    del d, parts
pd.DataFrame(out).to_parquet("pool_raw.parquet"); print(len(out), "rows saved to hard_pool/pool_raw.parquet")
