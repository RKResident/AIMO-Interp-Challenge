"""exp113 (autonomous session 2026-11-02): main-v2 labels agree across models in 35/35 pairs (exp112c), so non-robustness looks problem-level.
Test: score each (model, problem) with the shipped recipe's margin (v8; ent4096 override for Qwen; Olmo: familiar branch, and olmoseg25),
turn margins into a per-model rank percentile over that model's main-v2 traces (label-free), then for each problem AVERAGE the percentiles of
every model that has a trace of it. Compare per-model calls vs pooled calls on the labelled rows (finished -> robust in both).
  python analysis/exp113_crossmodel.py"""
import json, numpy as np, pandas as pd
src = open("/home/sabyasachi19/PycharmProjects/AimoAGAIN/analysis/exp109_mainv2.py").read(); exec(src[:src.index("rows, corr = [], []")])
SEG = json.load(open("/home/sabyasachi19/PycharmProjects/aimo/getting-started/solutions/spurious-permodel-v8-ent4096-olmoseg25/spurious_artifacts/recipe.json"))["per_model"]
recs = []
for m, tag in TAGS.items():
    ship = dict(np.load(A + f"analysis/wb2_ship/wb2_ship_{tag}.npz")); M = table(tag, "wb_mv2", "wb2_mv2", "entropy_mv2", ship)
    F = {json.loads(l)["pid"]: json.loads(l) for l in open(A + f"ec2_results/feats_mv2/{tag}.jsonl")}
    pm = REC["per_model"][HF[m]]; rec = {**REC, **pm}; thr = REC["threshold_by_model"][HF[m]]
    for p in F:
        if "router" in pm: b = pm["router"]["familiar"]; mg = score_trace({f: F[p].get(f, np.nan) for f in b["features"]}, {**rec, **b}, HF[m]) - b["threshold"]
        else: mg = score_trace({f: F[p].get(f, np.nan) for f in pm["features"]}, rec, HF[m]) - thr
        O = pm.get("override") or (SEG[HF[m]].get("override") if m == "allenai" else None)
        if O and p in M and np.isfinite(M[p].get(O["feature"], np.nan)): mg = O["sign"] * (M[p][O["feature"]] - O["threshold"])
        lab = MV2.get((HF[m], p), {}).get("is_robust")
        recs.append(dict(model=m, pid=p, margin=float(mg) if np.isfinite(mg) else np.nan, finished=not F[p]["still_thinking"], y=None if lab is None else int(lab)))
D = pd.DataFrame(recs)
D["pct"] = D.groupby("model").margin.rank(pct=True)                                         # label-free, per model
D["pool"] = D.groupby("pid").pct.transform("mean"); D["n_models"] = D.groupby("pid").pct.transform("count")
D["pool_other"] = (D.groupby("pid").pct.transform("sum") - D.pct) / (D.n_models - 1).replace(0, np.nan)   # other models only
L = D[D.y.notna()].copy()
for nm, col in (("per-model (shipped recipe, sign as shipped)", "pct"), ("pooled over models", "pool"), ("other models only", "pool_other")):
    c = np.where(L.finished, 1, (L[col].fillna(L.pct) > .5).astype(int))
    print(f"{nm:45s} acc {np.mean(c == L.y):.2f}  (n {len(L)}, {int((L.y == 0).sum())} NR) | per model: " +
          ", ".join(f"{m} {np.mean(c[L.model.values == m] == L.y.values[L.model.values == m]):.2f}" for m in TAGS))
print(f"all-robust baseline acc {L.y.mean():.2f}")
print("\nproblems seen by >1 model:", int((D.drop_duplicates('pid').n_models > 1).sum()), "of", D.pid.nunique())
print(L.sort_values(["pid", "model"])[["pid", "model", "y", "finished", "pct", "pool"]].round(2).to_string(index=False))
