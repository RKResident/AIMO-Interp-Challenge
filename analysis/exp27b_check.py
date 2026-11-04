import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "2"
_src = open("exp27_memlens_mainv2.py").read().split("def cv(")[0]
exec(_src)
from scipy.stats import rankdata
NAMES = ["emergence depth", "entropy mean", "entropy late", "final max conf", "mid max conf", "final-minus-mid conf"]
mm = allm.model.values[rows]; prob = allm.base_id.values[rows]; up = np.unique(prob)
fin = (allm.stop.values[rows] == "eos").astype(float)
print(f"rows {len(rows)}, problems {len(up)}; finished (eos) robust {fin[y[rows]==1].mean():.2f} vs non-robust {fin[y[rows]==0].mean():.2f}")
for k, tag in ((1, "last cutoff"), (0, "first cutoff")):
    S = np.stack([SUM[i][k] for i in rows])
    Sz = zgroup(np.pad(S, ((0, 0), (0, 0))), allm.grp.values[rows])             # z within model
    Lz = zgroup(L3[rows], allm.grp.values[rows]); Fz = np.c_[Lz, fin]
    print(f"\n=== {tag} ===")
    for j, n in enumerate(NAMES):
        x = Sz[:, j]; xr = x - LinearRegression().fit(Fz, x).predict(Fz)       # remove lengths + finished-or-not
        a, ar = roc_auc_score(y[rows], x), roc_auc_score(y[rows], xr)
        # problem level: mean over the models that solved the problem, label is problem-level
        xp = np.array([x[prob == p].mean() for p in up]); xpr = np.array([xr[prob == p].mean() for p in up]); yp = np.array([y[rows][prob == p][0] for p in up])
        ap, apr = roc_auc_score(yp, xp), roc_auc_score(yp, xpr)
        rng = np.random.default_rng(0); null = np.array([roc_auc_score(rng.permutation(yp), xpr) for _ in range(20000)])
        pval = np.mean(np.abs(null - .5) >= abs(apr - .5))
        print(f"  {n:22s} rows: AUC {a:.2f} -> lengths+finish removed {ar:.2f} | problems (n={len(up)}): {ap:.2f} -> removed {apr:.2f}  two-sided permutation p {pval:.3f}")
