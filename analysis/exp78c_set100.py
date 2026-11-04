"""exp78c: recall directions on the NEW set100 labels (Qwen; labels from hard_pool/set100_labels_*.csv, forced and pure classes; all set100
traces are still thinking at 6144). Same protocol as exp78: each direction at its recall-chosen layer, AUC vs robust, permutation p."""
import json, sys
import numpy as np, pandas as pd
rng = np.random.default_rng(0); M = "Qwen__Qwen3.5-4B"; R = f"/home/sabyasachi19/PycharmProjects/AimoAGAIN/ec2_results/a10g/recall/{M}/"
info = json.load(open(R + "directions.json"))
Pj = {r["id"].split("_")[-1]: r for r in map(json.loads, open(R + "projections.jsonl")) if "_hp" in r["id"]}
L = pd.read_csv(f"/home/sabyasachi19/PycharmProjects/AimoAGAIN/hard_pool/set100_labels_{M}.csv")


def auc(s, y):
    p, n = s[y == 1], s[y == 0]
    return float(((p[:, None] > n[None]) + .5 * (p[:, None] == n[None])).mean()) if len(p) and len(n) else np.nan


def perm(s, y, n=4000): a = abs(auc(s, y) - .5); return float(np.mean([abs(auc(s, rng.permutation(y)) - .5) >= a - 1e-12 for _ in range(n)]))


for rule in ("forced", "pure"):
    X = L[L[f"label_{rule}"].isin(["robust", "non_robust"]) & L.pid.isin(Pj)]; y = (X[f"label_{rule}"] == "robust").astype(int).values
    print(f"\n{rule} class: {len(X)} labelled set100 problems ({y.sum()} robust)")
    if len(X) < 6 or y.min() == y.max(): print("   too few"); continue
    names = next(iter(Pj.values()))["names"]; res = []
    for k, nm in enumerate(names):
        fam, t = nm.split("__"); b = info[fam]["best_layer"][t]
        for stat in ("mean", "first_q", "last_q"):
            s = np.array([Pj[p][stat][b - 1][k] for p in X.pid]); a = auc(s, y); res.append((abs(a - .5), nm, stat, a, perm(s, y)))
    for _, nm, stat, a, p in sorted(res, reverse=True)[:8]: print(f"   {nm:16s} {stat:8s} AUC {a:.2f}  p {p:.3f}")
    ps = np.array([r[-1] for r in res]); print(f"   {len(res)} tests | smallest p {ps.min():.3f} | share p<0.05 {np.mean(ps < .05):.2f} (null 0.05)")
    fm = [r for r in res if r[1].startswith("formula__ans")]
    print("   pre-registered formula/ans (reversed on the old labels): " + ", ".join(f"{s} {a:.2f} (p {p:.2f})" for _, n, s, a, p in fm))
