"""exp43: does a whole FAMILY of correlated features beat one chosen feature? (main_v2 only, 43 rows, 22 problems, 7 non-robust)
Leave-pair-out (one robust + one non-robust problem per fold, 105 folds). Inside every fold, using the training problems only:
  composite   every feature of the family, signed by its training AUC, averaged (per-model percentile coordinates, as the submissions)
  best single the one feature with the highest training AUC (honest selection)
  fixed single the feature we currently ship for that family (sign fixed), as a reference
Held-out score = the pair's rows; AUC over all (robust row, non-robust row) comparisons of the held-out pair (per-model percentiles
make models comparable). Also the composite's in-sample AUC and its controlled problem-level AUC with all-data signs."""
import os, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "2"
from common import *
from scipy.stats import rankdata
sys.path.insert(0, str(ROOT / "submission_core"))
from spurious_features import percentile

allm, _ = load_all(); y = allm.y.values; rows = datasets(allm)["main_v2"]; model = allm.model.values; prob = allm.base_id.values
fin = (allm.stop.values == "eos").astype(float); lgn = np.log(allm.n_gen.astype(float).values)
F28 = pickle.load(open(A / "exp28_features_cache.pkl", "rb")); F32 = pickle.load(open(A / "exp32_features.pkl", "rb"))
F33 = pickle.load(open(A / "exp33_features.pkl", "rb")); F35 = pickle.load(open(A / "exp35_features.pkl", "rb")); F36 = pickle.load(open(A / "exp36_features.pkl", "rb"))
FAM = {
    "forced answer (all budgets / suffixes / stats)": ({k: v for k, v in F28.items() if k.startswith("meta:fa_b")}, "meta:fa_b6144_B_p1"),
    "kNN graph, 64-token, 0-4k (k5/k10)": ({k: v for k, v in F32.items() if k.startswith("step64|") and "|4k|" in k}, "step64|k5|4k|global_efficiency"),
    "cluster graph, 64-token, 0-4k (K16/32/64)": ({k: v for k, v in F33.items() if k.startswith("step64|K") and "|4k|" in k}, "step64|K64|4k|cluster_cycle_rank_per_step"),
    "trajectory loops (cycle rank, self-loops, returns, 3-cycles, recurrence)": (
        {k: v for k, v in F33.items() if "|4k|" in k and any(t in k for t in ("cycle_rank", "self_loop", "return_rate", "repeated_3gram", "recurrence_rate", "mean_max_backsim"))},
        "step64|K64|4k|cluster_cycle_rank_per_step"),
    "TDA, 64-token raw, whole thinking": ({k: v for k, v in F35.items() if k.startswith("step64|raw|think|")}, "step64|raw|think|H0_betti_spread"),
    "PEARL (all curves / summaries)": ({k: v for k, v in F28.items() if k.startswith("pearl:")}, "pearl:mean_state:area"),
    "trajectory dynamics, averaged (exp36)": ({k: v for k, v in F36.items() if k.startswith(("sentence|", "ma64|"))}, "sentence|full|straightness"),
}
yy = y[rows] == 1; mods = model[rows]; pbs = prob[rows]; probs = np.unique(pbs); plab = {p: yy[pbs == p][0] for p in probs}
FOLDS = [(r, n) for r in probs if plab[r] for n in probs if not plab[n]]


def pct(v):
    out = np.zeros(len(rows))
    for m in np.unique(mods):
        idx = np.where(mods == m)[0]; ref = [x for x in v[rows][idx] if np.isfinite(x)]
        for k in idx:
            x = v[rows][k]; out[k] = percentile(x, ref) - 0.5 if np.isfinite(x) and ref else 0.0
    return out


def auc(s, lab):
    if lab.all() or (~lab).all(): return 0.5
    rk = rankdata(s); n1 = lab.sum(); n0 = len(lab) - n1
    return (rk[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def pair_auc(s, te, r, n):
    sr = s[te][pbs[te] == r]; sn = s[te][pbs[te] == n]
    return sum((a > b) + 0.5 * (a == b) for a in sr for b in sn), len(sr) * len(sn)


up = np.unique(pbs); ypb = np.array([plab[p] for p in up])
fp = np.array([fin[rows][pbs == p].mean() for p in up]); lp = np.array([lgn[rows][pbs == p].mean() for p in up])
print(f"{len(FOLDS)} leave-pair-out folds. AUC > 0.5 = ranks robust above non-robust.\n")
print(f"{'family':72s} {'n':>3s} | {'composite':>9s} {'best single':>11s} {'fixed single':>12s} | composite in-sample, controlled")
for name, (fd, fixed) in FAM.items():
    keys = [k for k, v in fd.items() if np.isfinite(v[rows]).sum() >= 20 and len(np.unique(v[rows][np.isfinite(v[rows])])) >= 3]
    P = np.stack([pct(fd[k]) for k in keys], 1)
    fixed_j = keys.index(fixed) if fixed in keys else None
    acc = {"comp": [0, 0], "best": [0, 0], "fixed": [0, 0]}
    for r, n in FOLDS:
        te = np.where((pbs == r) | (pbs == n))[0]; tr = np.setdiff1d(np.arange(len(rows)), te)
        a_tr = np.array([auc(P[tr, j], yy[tr]) for j in range(P.shape[1])]); sg = np.sign(a_tr - .5)
        comp = (P * sg).mean(1); best = P[:, int(np.argmax(np.abs(a_tr - .5)))] * sg[int(np.argmax(np.abs(a_tr - .5)))]
        for k_, s in (("comp", comp), ("best", best)) + ((("fixed", P[:, fixed_j] * np.sign(auc(P[tr, fixed_j], yy[tr]) - .5)),) if fixed_j is not None else ()):
            a, b = pair_auc(s, te, r, n); acc[k_][0] += a; acc[k_][1] += b
    sg_all = np.sign(np.array([auc(P[:, j], yy) for j in range(P.shape[1])]) - .5); comp_all = (P * sg_all).mean(1)
    xp = np.array([comp_all[pbs == p].mean() for p in up]); C = np.c_[np.ones(len(up)), fp, lp]
    ctrl = auc(xp - C @ np.linalg.lstsq(C, xp, rcond=None)[0], ypb)
    res = {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in acc.items()}
    print(f"{name:72s} {len(keys):3d} | {res['comp']:9.2f} {res['best']:11.2f} {res['fixed']:12.2f} | {auc(comp_all, yy):.2f}, {ctrl:.2f}")
