"""exp78: do the recall-vs-computation directions (A10G run, run_recall.py) separate robust from spurious on our labelled traces?
Per model: projections of each labelled greedy trace (cut at end of thinking / 6144) on every family x position-type direction, at the
layer chosen on the RECALL data alone (best CV layer in directions.json; no label fitting). Statistic: mean projection over the trace,
first quarter, last quarter. AUC (higher projection = more 'recall-like' -> robust?) on the unfinished rows, with a permutation p."""
import glob, json, sys
import numpy as np
R = "/home/sabyasachi19/PycharmProjects/AimoAGAIN/ec2_results/a10g/recall/"
rng = np.random.default_rng(0)


def auc(s, y):
    p, n = s[y == 1], s[y == 0]
    return float(((p[:, None] > n[None]) + .5 * (p[:, None] == n[None])).mean()) if len(p) and len(n) else np.nan


def perm_p(s, y, n=4000):
    a = abs(auc(s, y) - .5)
    return float(np.mean([abs(auc(s, rng.permutation(y)) - .5) >= a - 1e-12 for _ in range(n)]))


for d in sorted(glob.glob(R + "*/")):
    m = d.rstrip("/").split("/")[-1]
    try:
        info = json.load(open(d + "directions.json")); P = [json.loads(l) for l in open(d + "projections.jsonl")]
    except FileNotFoundError:
        continue
    P = [r for r in P if r.get("robust") is not None]                  # set100 traces (labels joined later) are skipped here
    U = [r for r in P if r["think_end"] is None]; y = np.array([int(r["robust"]) for r in U]); names = P[0]["names"]
    print(f"\n{m}: {len(P)} labelled traces, {len(U)} still thinking at the cut ({y.sum()} robust)")
    if len(U) < 4 or y.min() == y.max(): print("   too few unfinished rows of both classes"); continue
    res = []
    for k, nm in enumerate(names):
        fam, t = nm.split("__"); b = info[fam]["best_layer"][t]; cv = info[fam]["cv_auc"][t][b]
        for stat in ("mean", "first_q", "last_q"):
            s = np.array([r[stat][b - 1][k] for r in U])              # projections index layer l-1 = output of decoder layer l-1 = hidden index l
            a = auc(s, y); res.append((abs(a - .5), nm, stat, b, cv, a, perm_p(s, y)))
    print(f"   {'direction':16s} {'stat':8s} layer  recall-CV  AUC(robust)  perm p")
    for _, nm, stat, b, cv, a, p in sorted(res, reverse=True)[:12]:
        print(f"   {nm:16s} {stat:8s} {b:5d}  {cv:9.2f}  {a:11.2f}  {p:6.3f}")
    ps = np.array([r[-1] for r in res])
    print(f"   {len(res)} tests | smallest p {ps.min():.3f} | expected smallest p under the null with {len(res)} tests ~ {1 / (len(res) + 1):.3f} "
          f"| share with p < 0.05: {np.mean(ps < .05):.2f} (null ~0.05)")
