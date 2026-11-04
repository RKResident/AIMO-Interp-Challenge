"""exp78b: the formula-recall direction (recall a formula in symbolic n vs compute an instance) came out REVERSED on Qwen, Skywork and
DeepSeek in exp78 (more formula-recall-like trace -> spurious). Check it as ONE fixed feature family across models, not the best of 39:
formula x {ans, last} position x {mean, last quarter}, each direction at its recall-chosen layer; per-model AUC on unfinished rows, pooled
within-model AUC with a within-model permutation p, and overlap with the existing features (Spearman with fa / chk / fas)."""
import glob, json
import numpy as np, pandas as pd
R = "/home/sabyasachi19/PycharmProjects/AimoAGAIN/ec2_results/a10g/recall/"; rng = np.random.default_rng(0)
F = pd.read_csv("exp56_rows.csv")[["id", "fa", "chk", "fas"]].set_index("id")


def auc(s, y):
    p, n = s[y == 1], s[y == 0]
    return float(((p[:, None] > n[None]) + .5 * (p[:, None] == n[None])).mean()) if len(p) and len(n) else np.nan


rows = []
for d in sorted(glob.glob(R + "*/")):
    m = d.rstrip("/").split("/")[-1][:8]; info = json.load(open(d + "directions.json"))
    for r in map(json.loads, open(d + "projections.jsonl")):
        if r.get("robust") is None or r["think_end"] is not None: continue
        x = dict(model=m, id=r["id"], y=int(r["robust"]))
        for t in ("ans", "last"):
            k = r["names"].index(f"formula__{t}"); b = info["formula"]["best_layer"][t]
            x[f"{t}_mean"] = r["mean"][b - 1][k]; x[f"{t}_lastq"] = r["last_q"][b - 1][k]
        rows.append(x)
D = pd.DataFrame(rows).join(F, on="id")
C = ["ans_mean", "ans_lastq", "last_mean", "last_lastq"]
for c in C: D[c + "_p"] = D.groupby("model")[c].rank(pct=True)
print("per-model AUC (higher projection -> robust; < 0.5 = reversed) on unfinished rows:")
print("   " + f"{'':12s}" + "".join(f"{m:>10s}" for m in D.model.unique()) + "   pooled   perm p")
for c in C:
    pooled = auc(D[c + "_p"].values, D.y.values)
    null = []
    for _ in range(4000):
        yy = D.y.values.copy()
        for m in D.model.unique():
            k = (D.model == m).values; yy[k] = rng.permutation(yy[k])
        null.append(abs(auc(D[c + "_p"].values, yy) - .5))
    p = np.mean(np.array(null) >= abs(pooled - .5) - 1e-12)
    print(f"   {c:12s}" + "".join(f"{auc(g[c].values, g.y.values):10.2f}" for _, g in D.groupby('model', sort=False)) + f"   {pooled:6.2f}   {p:.3f}")
print("\nSpearman with existing features (all models pooled, within-model ranks):")
for c in C:
    print(f"   {c:12s} " + "  ".join(f"{e} {D[c + '_p'].corr(D.groupby('model')[e].rank(pct=True), method='spearman'):+.2f}" for e in ("fa", "chk", "fas")))
print(f"\nrows: {len(D)} unfinished ({D.groupby('model').size().to_dict()})")
