"""exp41 (main_v2 only, 43 rows, 22 problems, 12 non-robust rows): two ways of turning our signals into decisions.
 1. LINEAR SEPARATION in signal space: each trace = a point whose coordinates are its per-model percentile scores (as the submissions
    use) for several signals; a regularised logistic regression / shrinkage LDA draws the separating plane. Compared with the
    equal-weight average (all weights fixed at 1). Fitted leave-one-PROBLEM-out, so every held-out row is scored by a plane fitted
    without it; balanced accuracy uses the per-model half split (as the submissions decide).
 2. ABSOLUTE CUTS: one raw-value cut per signal, shared by all models (fa is a probability; ge and tda are scale-free graph /
    topology summaries), chosen to maximise main_v2 balanced accuracy - in-sample, and leave-one-problem-out (cut chosen without the
    held-out problem) for an honest estimate. Missing values (e.g. ge when thinking ended before 4096 tokens) count as robust."""
import os, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "2"
from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
sys.path.insert(0, str(ROOT / "submission_core"))
from spurious_features import decide_batch, percentile

allm, _ = load_all(); y = allm.y.values; rows = datasets(allm)["main_v2"]; model = allm.model.values; prob = allm.base_id.values
F28 = pickle.load(open(A / "exp28_features_cache.pkl", "rb")); F32 = pickle.load(open(A / "exp32_features.pkl", "rb")); F38 = pickle.load(open(A / "exp38_features.pkl", "rb"))
RAW = {"fa": F28["meta:fa_b6144_B_p1"], "ge": F32["step64|k5|4k|global_efficiency"], "tda": F38["C_layer|H0_betti_spread|L3"],
       "pearl": F28["pearl:mean_state:area"], "ab": F28["ab:dent_max"], "seg": F28["subsample:seg25:f11"], "traj": F28["traj:k1|full:late_vs_early_cos"]}
yy = (y[rows] == 1); mods = model[rows]; pbs = prob[rows]


def pct_matrix(feats):
    """per-model percentile of each feature among that model's main_v2 values, minus 0.5 (all signs +: higher -> robust); NaN -> 0"""
    M = np.zeros((len(rows), len(feats)))
    for j, f in enumerate(feats):
        for m in np.unique(mods):
            idx = np.where(mods == m)[0]; ref = [v for v in RAW[f][rows][idx] if np.isfinite(v)]
            for k in idx:
                x = RAW[f][rows][k]; M[k, j] = percentile(x, ref) - 0.5 if np.isfinite(x) else 0.0
    return M


def bal_acc(pred): return 0.5 * (pred[yy].mean() + (~pred[~yy]).mean())


def split_pred(score):
    pred = np.zeros(len(rows), bool)
    for m in np.unique(mods):
        idx = np.where(mods == m)[0]; pred[idx] = decide_batch(score[idx], np.median(score), True)
    return pred


def loo(feats, fit):
    M = pct_matrix(feats); s = np.zeros(len(rows))
    for p in np.unique(pbs):
        te = pbs == p; tr = ~te
        s[te] = fit(M[tr], yy[tr], M[te])
    return s


eq = lambda Mtr, ytr, Mte: Mte.mean(1)                                                   # noqa: E731
lr = lambda C: (lambda Mtr, ytr, Mte: LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(Mtr, ytr).decision_function(Mte))  # noqa: E731
lda = lambda Mtr, ytr, Mte: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Mtr, ytr).decision_function(Mte)  # noqa: E731
SETS = {"fa+ge": ["fa", "ge"], "fa+ge+tda": ["fa", "ge", "tda"], "fa+ge+tda+pearl": ["fa", "ge", "tda", "pearl"],
        "all 7": ["fa", "ge", "tda", "pearl", "ab", "seg", "traj"]}
print("1. LINEAR SEPARATION in per-model-percentile space, leave-one-problem-out (held-out AUC | balanced acc with per-model half split)")
print(f"   {'signals':18s} {'equal weights':>16s} {'logistic C=0.1':>16s} {'logistic C=1':>16s} {'shrinkage LDA':>16s}")
for nm, fs in SETS.items():
    cells = []
    for fit in (eq, lr(0.1), lr(1.0), lda):
        s = loo(fs, fit); cells.append(f"{roc_auc_score(yy, s):.2f} | {bal_acc(split_pred(s)):.2f}")
    w = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(pct_matrix(fs), yy).coef_[0]
    print(f"   {nm:18s} " + " ".join(f"{c:>16s}" for c in cells) + f"   | full-data logistic weights {dict(zip(fs, np.round(w, 2)))}")
rng = np.random.default_rng(0)
s_obs = loo(["fa", "ge", "tda"], lr(1.0)); a_obs = roc_auc_score(yy, s_obs); null = []
for _ in range(200):                                                       # permute labels across problems, refit everything
    perm = dict(zip(np.unique(pbs), rng.permutation([yy[pbs == p][0] for p in np.unique(pbs)])))
    yp_ = np.array([perm[p] for p in pbs]); M = pct_matrix(["fa", "ge", "tda"]); s = np.zeros(len(rows))
    for p in np.unique(pbs):
        te = pbs == p
        s[te] = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(M[~te], yp_[~te]).decision_function(M[te]) if len(np.unique(yp_[~te])) > 1 else 0
    null.append(roc_auc_score(yp_, s))
print(f"   permutation check, logistic fa+ge+tda: held-out AUC {a_obs:.2f}, null 95% {np.quantile(null, .025):.2f}-{np.quantile(null, .975):.2f}, p {np.mean(np.array(null) >= a_obs):.3f}")

print("\n2. ABSOLUTE CUTS (one raw cut per signal for all models; robust = value above the cut; missing -> robust)")
for f in ("fa", "ge", "tda", "pearl"):
    x = RAW[f][rows].astype(float)
    def best_cut(idx):
        v = np.unique(x[idx][np.isfinite(x[idx])]); cand = (v[1:] + v[:-1]) / 2 if len(v) > 1 else v
        sc = []
        for c in cand:
            pr = np.where(np.isfinite(x[idx]), x[idx] > c, True); a, b = yy[idx], pr
            sc.append(0.5 * (b[a].mean() + (~b[~a]).mean()) if a.any() and (~a).any() else 0)
        return cand[int(np.argmax(sc))], max(sc)
    c_all, ba_in = best_cut(np.arange(len(rows)))
    pred_loo = np.zeros(len(rows), bool); cuts = []
    for p in np.unique(pbs):
        te = pbs == p; c, _ = best_cut(np.where(~te)[0]); cuts.append(c)
        pred_loo[te] = np.where(np.isfinite(x[te]), x[te] > c, True)
    per = {m.split("__")[1][:8]: f"{np.where(np.isfinite(x[mods == m]), x[mods == m] > c_all, True).sum()}/{(mods == m).sum()}" for m in np.unique(mods)}
    print(f"   {f:6s} cut {c_all:.5g} | in-sample balanced acc {ba_in:.2f} | leave-one-problem-out {bal_acc(pred_loo):.2f} "
          f"(cut ranges {min(cuts):.4g}-{max(cuts):.4g}) | robust per model {per}")
