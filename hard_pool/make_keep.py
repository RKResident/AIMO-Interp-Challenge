"""Source-level filter (user, 2026-10-28): drop only sources the model has effectively no hope on; keep every problem of the other sources,
solved or not (unsolved ones are flagged in the labels later, not removed).
Per source, from the solvability pass (k T=1 samples, 32k cap, natural or forced answer): share of problems with >= 1 correct sample.
A source is dropped if that share is below --min-solvable (default 0.10).
Writes keep_<model>.txt (one pid per line: still thinking at 6144 AND from a kept source) and prints the per-source table.
  python make_keep.py --model Qwen/Qwen3.5-4B"""
import argparse, json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent; RES = HERE.parent / "ec2_results/big/out"
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--min-solvable", type=float, default=0.10)
a = ap.parse_args(); tag = a.model.replace("/", "__")
rows = []
for l in open(RES / "solve" / f"{tag}.jsonl"):
    r = json.loads(l); S = r["samples"]
    rows.append(dict(pid=r["pid"], source=r["source"].split("/")[-1], k=len(S), solved_pure=r["n_correct"] > 0, solved_any=r["n_correct_any"] > 0,
                     acc_pure=r["n_correct"] / len(S), acc_any=r["n_correct_any"] / len(S), truncated=r["n_truncated"] / len(S),
                     mean_len=sum(s["n_gen"] for s in S) / len(S)))
D = pd.DataFrame(rows)
T = D.groupby("source").agg(n=("pid", "size"), solvable_pure=("solved_pure", "mean"), solvable_any=("solved_any", "mean"), acc_pure=("acc_pure", "mean"),
                            acc_any=("acc_any", "mean"), truncated=("truncated", "mean"), mean_len=("mean_len", "mean"))
T["kept"] = T.solvable_any >= a.min_solvable
print(T.sort_values("solvable_any", ascending=False).round(2).to_string())
keep = D[D.source.map(T.kept)]
print(f"\n{len(D)} problems sampled | kept sources {int(T.kept.sum())}/{len(T)} -> {len(keep)} problems "
      f"({int(keep.solved_any.sum())} solved at least once, {int((~keep.solved_any).sum())} never solved: flagged, not dropped)")
(HERE / f"keep_{tag.split('__')[0]}.txt").write_text("\n".join(keep.pid) + "\n")
T.to_csv(HERE / f"source_solvability_{tag}.csv")
