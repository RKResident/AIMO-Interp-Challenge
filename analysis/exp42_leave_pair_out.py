"""exp42: leave-PAIR-out cross-validation (one robust + one non-robust PROBLEM held out per fold; 15 x 7 = 105 folds) for every method
that fits something, next to leave-one-problem-out (LOO). main_v2 only (43 rows, 22 problems, 7 non-robust).
Why: AUC = P(robust case ranked above non-robust case); leave-pair-out estimates exactly that, keeps the training class balance fixed
(LOO shifts the training class means against every held-out point - the small-sample anticorrelation bias seen earlier).
Held-out scores of different models are put on one scale using the TRAINING rows of the same model (z-score of training scores),
then every (robust row, non-robust row) comparison of the held-out pair counts.
  planes : logistic regression (C 0.1 / 1) and shrinkage LDA on per-model percentile coordinates (exp41), equal weights as reference
  cuts   : one absolute cut per signal, chosen on the training problems by balanced accuracy (exp41) -> held-out pair accuracy
  direction: per-model diff-of-means direction of the mean reasoning state (exp39), each of the 6 layers"""
import os, sys, itertools
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "2"
from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
sys.path.insert(0, str(ROOT / "submission_core"))
from spurious_features import percentile

P2 = ROOT / "extraction_out_generations_p2" / "extraction_out_generations"
allm, _ = load_all(); y = allm.y.values; rows = datasets(allm)["main_v2"]; model = allm.model.values; prob = allm.base_id.values
F28 = pickle.load(open(A / "exp28_features_cache.pkl", "rb")); F32 = pickle.load(open(A / "exp32_features.pkl", "rb")); F38 = pickle.load(open(A / "exp38_features.pkl", "rb"))
RAW = {"fa": F28["meta:fa_b6144_B_p1"], "ge": F32["step64|k5|4k|global_efficiency"], "tda": F38["C_layer|H0_betti_spread|L3"],
       "pearl": F28["pearl:mean_state:area"], "ab": F28["ab:dent_max"], "seg": F28["subsample:seg25:f11"], "traj": F28["traj:k1|full:late_vs_early_cos"]}
yy = (y[rows] == 1); mods = model[rows]; pbs = prob[rows]
probs = np.unique(pbs); plab = {p: yy[pbs == p][0] for p in probs}
ROB = [p for p in probs if plab[p]]; NON = [p for p in probs if not plab[p]]
FOLDS_LPO = [(r, n) for r in ROB for n in NON]
print(f"{len(rows)} rows, {len(probs)} problems ({len(ROB)} robust, {len(NON)} non-robust) -> {len(FOLDS_LPO)} leave-pair-out folds")


def pct_matrix(feats):
    M = np.zeros((len(rows), len(feats)))
    for j, f in enumerate(feats):
        for m in np.unique(mods):
            idx = np.where(mods == m)[0]; ref = [v for v in RAW[f][rows][idx] if np.isfinite(v)]
            for k in idx:
                x = RAW[f][rows][k]; M[k, j] = percentile(x, ref) - 0.5 if np.isfinite(x) else 0.0
    return M


def model_scale(score_tr, tr, score_te, te):
    """z-score held-out scores with the mean / sd of the same model's training scores (makes models comparable)"""
    out = np.zeros(len(te))
    for k, i in enumerate(te):
        same = [j for j, t in enumerate(tr) if mods[t] == mods[i]]
        mu, sd = (np.mean(score_tr[same]), np.std(score_tr[same]) + 1e-9) if len(same) >= 2 else (0.0, 1.0)
        out[k] = (score_te[k] - mu) / sd
    return out


def evaluate(fit_score):
    """fit_score(train_idx, test_idx) -> scores for test rows (comparable across models). Returns (LPO AUC, LOO AUC)."""
    num = den = 0.0
    for r, n in FOLDS_LPO:
        te = np.where((pbs == r) | (pbs == n))[0]; tr = np.setdiff1d(np.arange(len(rows)), te)
        s = fit_score(tr, te); sr = s[pbs[te] == r]; sn = s[pbs[te] == n]
        num += sum((a > b) + 0.5 * (a == b) for a in sr for b in sn); den += len(sr) * len(sn)
    loo = np.zeros(len(rows))
    for p in probs:
        te = np.where(pbs == p)[0]; tr = np.setdiff1d(np.arange(len(rows)), te); loo[te] = fit_score(tr, te)
    return num / den, roc_auc_score(yy, loo)


# ---------------------------------------------------------------- planes
SETS = {"fa+ge": ["fa", "ge"], "fa+ge+tda": ["fa", "ge", "tda"], "fa+ge+tda+pearl": ["fa", "ge", "tda", "pearl"], "all 7": list(RAW)}
print("\nPLANES in per-model percentile space: AUC leave-pair-out | leave-one-problem-out")
print(f"   {'signals':16s} {'equal weights':>16s} {'logistic C=0.1':>16s} {'logistic C=1':>16s} {'shrinkage LDA':>16s}")
for nm, fs in SETS.items():
    M = pct_matrix(fs); cells = []
    fits = {"eq": lambda tr, te: M[te].mean(1),
            "lr01": lambda tr, te: (lambda c: model_scale(c.decision_function(M[tr]), tr, c.decision_function(M[te]), te))(
                LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000).fit(M[tr], yy[tr])),
            "lr1": lambda tr, te: (lambda c: model_scale(c.decision_function(M[tr]), tr, c.decision_function(M[te]), te))(
                LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(M[tr], yy[tr])),
            "lda": lambda tr, te: (lambda c: model_scale(c.decision_function(M[tr]), tr, c.decision_function(M[te]), te))(
                LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(M[tr], yy[tr]))}
    for k in ("eq", "lr01", "lr1", "lda"):
        a, b = evaluate(fits[k]); cells.append(f"{a:.2f} | {b:.2f}")
    print(f"   {nm:16s} " + " ".join(f"{c:>16s}" for c in cells))

# ---------------------------------------------------------------- absolute cuts
print("\nABSOLUTE CUTS (cut chosen on the training problems): held-out balanced accuracy, leave-pair-out | leave-one-problem-out")
for f in ("fa", "ge", "tda", "pearl"):
    x = RAW[f][rows].astype(float)

    def choose(tr):
        v = np.unique(x[tr][np.isfinite(x[tr])]); cand = (v[1:] + v[:-1]) / 2
        best, bc = -1, None
        for c in cand:
            pr = np.where(np.isfinite(x[tr]), x[tr] > c, True); a = yy[tr]
            ba = 0.5 * (pr[a].mean() + (~pr[~a]).mean())
            if ba > best: best, bc = ba, c
        return bc
    pred = lambda te, c: np.where(np.isfinite(x[te]), x[te] > c, True)   # noqa: E731
    accs = []
    for r, n in FOLDS_LPO:
        te = np.where((pbs == r) | (pbs == n))[0]; tr = np.setdiff1d(np.arange(len(rows)), te); c = choose(tr)
        pr = pred(te, c); a = yy[te]; accs.append(0.5 * (pr[a].mean() + (~pr[~a]).mean()))
    loo = np.zeros(len(rows), bool)
    for p in probs:
        te = np.where(pbs == p)[0]; tr = np.setdiff1d(np.arange(len(rows)), te); loo[te] = pred(te, choose(tr))
    print(f"   {f:6s} {np.mean(accs):.2f} | {0.5 * (loo[yy].mean() + (~loo[~yy]).mean()):.2f}")

# ---------------------------------------------------------------- diff-of-means direction (exp39), mean reasoning state, 6 layers
Lthink = {i: force_json(model[i], allm.id.iloc[i])["L"] for i in rows}
Xs = {}
for k, i in enumerate(rows):
    z = np.load(P2 / model[i] / "tf" / f"{allm.id.iloc[i]}.npz"); ws = z["win_states"].astype(np.float32); nw = min(len(ws), max(1, Lthink[i] // 16))
    for l in range(ws.shape[1]):
        v = ws[:nw, l].mean(0); Xs.setdefault(l, np.zeros((len(rows), v.shape[0]), np.float32) if False else {})[k] = v / (np.linalg.norm(v) + 1e-9)


def dir_fit(l):
    def f(tr, te):
        s_te = np.zeros(len(te)); s_tr = np.zeros(len(tr))
        for m in np.unique(mods):
            trm = [t for t in tr if mods[t] == m]; tem = [k for k, t in enumerate(te) if mods[t] == m]
            if not tem: continue
            pos = [Xs[l][t] for t in trm if yy[t]]; neg = [Xs[l][t] for t in trm if not yy[t]]
            if not pos or not neg: continue
            mu = np.mean([Xs[l][t] for t in trm], 0); w = np.mean(pos, 0) - np.mean(neg, 0); w /= np.linalg.norm(w) + 1e-9
            proj_tr = np.array([(Xs[l][t] - mu) @ w for t in trm]); sd = proj_tr.std() + 1e-9
            for k in tem: s_te[k] = ((Xs[l][te[k]] - mu) @ w - proj_tr.mean()) / sd
        return s_te
    return f


print("\nDIFF-OF-MEANS DIRECTION (mean reasoning state over thinking): AUC leave-pair-out | leave-one-problem-out")
print("   " + "  ".join(f"layer {l}: {a:.2f} | {b:.2f}" for l, (a, b) in ((l, evaluate(dir_fit(l))) for l in sorted(Xs))))
