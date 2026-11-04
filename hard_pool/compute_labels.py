"""Robustness labels for the set100 problems from the synced labelling samples (ec2_results/big/out/{solve,label}).
Per problem and model: original accuracy over its samples (solvability pass + labelling top-up, 8 in total) and accuracy of each perturbation
(4, or 8 if round 2 ran). max_drop = acc(original) - min over perturbations acc(perturbation), as in main_v2.
Two scoring classes, kept separate (user, 2026-10-28):
  pure    natural answers only (a sample that hit the 32k cap counts as wrong)
  forced  natural answer, else the forced answer of a capped sample
Label per class (main_v2's thresholds: its robust rows have max_drop <= 0.08, its non-robust rows >= 0.25):
  robust if max_drop <= 0.10 | non_robust if max_drop >= 0.25 | ambiguous otherwise | unsolved if the original is never right (flagged:
  its drop is 0 by construction, so "robust" would say nothing).
Only problems with ALL versions written are labelled. Writes set100_labels_<model>.csv and prints yield per class.
  python compute_labels.py [--model Qwen/Qwen3.5-4B]"""
import argparse, json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent; RES = HERE.parent / "ec2_results/big/out"
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="Qwen/Qwen3.5-4B"); a = ap.parse_args(); tag = a.model.replace("/", "__")
S100 = [l.strip() for l in open(HERE / "set100.txt") if l.strip()]
P = pd.read_csv(HERE / "pool_screen.csv").set_index("pid")
TYPES = ["distract", "domain", "rename", "rephrase", "typos"]
samp = {}
sv = RES / "solve" / f"{tag}.jsonl"
if sv.exists():
    for l in open(sv):
        r = json.loads(l)
        if r["pid"] in S100: samp.setdefault((r["pid"], "original"), []).extend(r["samples"])
seen = set()                                                                 # big box + A10G records (the same version is never sampled twice)
for lb in (RES / "label" / f"{tag}.jsonl", HERE.parent / "ec2_results/a10g/out/label" / f"{tag}.jsonl"):
    if not lb.exists(): continue
    for l in open(lb):
        r = json.loads(l); key = (r["pid"], r["ptype"], r["round"])
        if key in seen: continue
        seen.add(key); samp.setdefault((r["pid"], r["ptype"]), []).extend(r["samples"])


def acc(S, rule): return sum(s["correct"] if rule == "pure" else s["correct_any"] for s in S) / len(S)


rows = []
for p in S100:
    o = samp.get((p, "original"), [])
    if len(o) < 8 or any((p, t) not in samp for t in TYPES): continue       # not complete yet
    d = dict(pid=p, source=P.loc[p, "source"].split("/")[-1], n_orig=len(o), capped_share=sum(s["finish"] == "length" for s in o) / len(o))
    for rule in ("pure", "forced"):
        ao = acc(o, rule); ap_ = {t: acc(samp[(p, t)], rule) for t in TYPES}; md = ao - min(ap_.values())
        d[f"acc_orig_{rule}"] = ao; d[f"max_drop_{rule}"] = md; d[f"worst_{rule}"] = min(ap_, key=ap_.get)
        d[f"label_{rule}"] = "unsolved" if ao == 0 else "robust" if md <= 0.10 else "non_robust" if md >= 0.25 else "ambiguous"
    rows.append(d)
D = pd.DataFrame(rows)
print(f"{a.model}: {len(D)} of {len(S100)} problems complete")
if len(D):
    for rule in ("pure", "forced"):
        vc = D[f"label_{rule}"].value_counts(); lab = int(vc.get("robust", 0) + vc.get("non_robust", 0))
        print(f"  {rule:6s}: " + ", ".join(f"{k} {v}" for k, v in vc.items()) + f" | labellable (robust + non_robust) {lab}/{len(D)} = {lab / len(D):.0%}")
    print("  forced class by source:\n" + D.groupby("source")["label_forced"].value_counts().unstack(fill_value=0).to_string())
    D.to_csv(HERE / f"set100_labels_{tag}.csv", index=False)
