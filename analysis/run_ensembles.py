"""Ensembles over feature families, nested CV. agg and main_v2 are cross-validated SEPARATELY. No gpt_proxy rows.
Ensemble types (all selection / weighting is done inside the training folds only):
  mean_all   average of z-scored family scores
  top5       average of the 5 families with the best inner-CV AUC
  stack      logistic regression on the families' inner-CV scores
  rf         random forest on the low-dimensional scalar features (a tree ensemble, for contrast)"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
__file__ = str(HERE / "run_pooled_families.py")                      # so the exec'd scripts find their siblings
exec(open(HERE / "run_pooled_families.py").read().split("res = []")[0])
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
src = meta.source.values


def folds_for(rows, y_, groups, seed, k=5):
    """list of (train_idx, test_idx) positions into `rows`; grouped by problem when groups repeat."""
    g = groups[rows]
    if len(np.unique(g)) == len(rows):
        return [(a, b) for a, b in StratifiedKFold(k, shuffle=True, random_state=seed).split(rows, y_[rows])]
    ug = np.unique(g); order = np.random.default_rng(seed).permutation(ug); fo = {p: i % k for i, p in enumerate(order)}
    f = np.array([fo[p] for p in g]); return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k) if (f == i).any()]


def fit_scores(X, kind, ytr, tr_rows, te_rows, mk):
    return mk(kind).fit(X[tr_rows], ytr).decision_function(X[te_rows])


def nested(FAMX, rows, y_, groups, reps, seed0, mk, rf_X=None, n_jobs=12):
    """FAMX: {name: (X (all rows), kind)}. Returns dict ensemble-name -> mean OOF score over rows (positions in `rows`)."""
    names = list(FAMX); out = {k: np.zeros(len(rows)) for k in ("mean_all", "top5", "stack", "rf")}; cnt = np.zeros(len(rows))
    for r in range(reps):
        for tr, te in folds_for(rows, y_, groups, seed0 + r):
            tr_rows, te_rows = rows[tr], rows[te]; ytr = y_[tr_rows]
            if len(np.unique(ytr)) < 2: continue
            inner = folds_for(tr_rows, y_, groups, 1000 + seed0 + r)

            def one(nm):
                X, kind = FAMX[nm]; oo = np.zeros(len(tr_rows))
                for a, b in inner:
                    if len(np.unique(ytr[a])) < 2: oo[b] = 0; continue
                    oo[b] = fit_scores(X, kind, ytr[a], tr_rows[a], tr_rows[b], mk)
                return oo, fit_scores(X, kind, ytr, tr_rows, te_rows, mk)
            res = Parallel(n_jobs=n_jobs)(delayed(one)(nm) for nm in names)
            OO = np.stack([a for a, _ in res], 1); TE = np.stack([b for _, b in res], 1)
            mu, sd = OO.mean(0), OO.std(0) + 1e-9; OOz, TEz = (OO - mu) / sd, (TE - mu) / sd
            auc_in = np.array([roc_auc_score(ytr, OO[:, j]) if len(np.unique(ytr)) == 2 else .5 for j in range(len(names))])
            top = np.argsort(-auc_in)[:5]
            st = LogisticRegression(C=0.1, max_iter=2000).fit(OOz, ytr)
            out["mean_all"][te] += TEz.mean(1); out["top5"][te] += TEz[:, top].mean(1); out["stack"][te] += st.decision_function(TEz)
            if rf_X is not None:
                rf = RandomForestClassifier(300, min_samples_leaf=3, max_features="sqrt", random_state=0, n_jobs=n_jobs).fit(rf_X[tr_rows], ytr)
                out["rf"][te] += rf.predict_proba(rf_X[te_rows])[:, 1]
            cnt[te] += 1
            print(f"    outer fold done (rep {r}, {int(cnt.max())} folds so far, {len(names)} families)", flush=True)
    return {k: v / np.maximum(cnt, 1) for k, v in out.items()}


def report(tag, S, yy, nulls, ctrl):
    print(f"\n[{tag}]")
    for k, s in S.items():
        if k == "rf" and not np.any(s): continue
        a, lo, hi = auc_ci(yy, s)
        nl = np.mean([roc_auc_score(*(n_y, n_s[k])) for n_y, n_s in nulls]) if nulls else np.nan
        print(f"  {k:9s} AUC {a:.3f} [{lo:.3f},{hi:.3f}]   shuffled-label null {nl:.2f}")
    print(f"  length-only control (same protocol): {ctrl:.3f}")


mkm = lambda kind: make_model(kind)
# ============================================================================== A. agg (DeepSeek), CV on agg only
Z = np.hstack([log_("n_prompt"), log_("n_gen"), log_("L_think")]); Z = (Z - Z.mean(0)) / Z.std(0)


def resid(X):
    X = np.nan_to_num(X.copy()); lr = LinearRegression().fit(Z[agg], X[agg]); out = X.copy(); out[agg] = X[agg] - lr.predict(Z[agg]); return out


famA = {n: (src_standardize(np.nan_to_num(X), src), k) for n, (X, k) in FAM.items() if not n.startswith("ctrl:")}
famA_r = {n: (src_standardize(resid(X), src), k) for n, (X, k) in FAM.items() if not n.startswith("ctrl:")}
yA = np.where(np.isnan(y), 0, y).astype(int)
scal_rf = src_standardize(np.hstack([FAM["traj:scalars"][0], FAM["forced:all"][0]]), src)
ctrlX = src_standardize(FAM["ctrl:len_both"][0], src)
print(f"A. agg: {len(agg)} problems ({int(yA[agg].sum())} robust); {len(famA)} families")
SA = {}
for lbl, fm in (("raw", famA), ("length-residualised", famA_r)):
    S = nested(fm, agg, yA, np.arange(len(meta)), reps=2, seed0=0, mk=mkm, rf_X=scal_rf if lbl == "raw" else None)
    nulls = []
    for sd in range(2):
        y2 = yA.copy(); y2[agg] = np.random.default_rng(sd).permutation(yA[agg])
        n_ = nested(fm, agg, y2, np.arange(len(meta)), reps=1, seed0=50 + sd, mk=mkm, rf_X=None)
        nulls.append((y2[agg], n_))
    ctrl = roc_auc_score(yA[agg], cv_oof(ctrlX, "low", agg)[0])
    report(f"agg CV, {lbl} families", S, yA[agg], nulls, ctrl); SA[lbl] = S
pickle.dump(SA, open(HERE / "ens_agg.pkl", "wb"))

# ============================================================================== B. main_v2 (4 models pooled), CV on main_v2 only, grouped by problem
fam_m = {k: (v, "low") for k, v in Xz.items() if k not in ("length (control)", "scalars+hid+lap", "hid:all_cuts", "lap:force_all_cuts", "lap:mean_force_all_cuts")}
pb = np.unique(prob, return_inverse=True)[1]
print(f"\nB. main_v2: {len(mv)} rows, {len(set(prob[mv]))} problems, {len(fam_m)} families")
rf_m = Xz["scalars"]
S = nested(fam_m, mv, yv.astype(int), pb, reps=6, seed0=0, mk=lambda kind: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)), rf_X=rf_m, n_jobs=4)
nulls = []
for sd in range(4):
    up = np.unique(prob[mv]); lab = {p: yv[mv][prob[mv] == p][0] for p in up}; pm_ = dict(zip(up, np.random.default_rng(sd).permutation(up)))
    y2 = yv.copy(); y2[mv] = [lab[pm_[p]] for p in prob[mv]]
    nulls.append((y2[mv].astype(int), nested(fam_m, mv, y2.astype(int), pb, reps=2, seed0=70 + sd, mk=lambda kind: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)), n_jobs=4)))
lc = Xz["length (control)"]
sc_ = np.zeros(len(mv)); cc = np.zeros(len(mv))
for r in range(6):
    for a, b in folds_for(mv, yv.astype(int), pb, r):
        m_ = make_pipeline(StandardScaler(), LogisticRegression(C=0.05)).fit(lc[mv[a]], yv[mv[a]]); sc_[b] += m_.decision_function(lc[mv[b]]); cc[b] += 1
report("main_v2 CV (problem-grouped, 4 models pooled)", S, yv[mv], nulls, roc_auc_score(yv[mv], sc_ / np.maximum(cc, 1)))
mm = allm.model.values[mv]
for k in ("mean_all", "top5", "stack"):
    print(f"  {k:9s} per model:", {m.split('__')[-1][:7]: round(roc_auc_score(yv[mv][mm == m], S[k][mm == m]), 2) for m in np.unique(mm) if len(np.unique(yv[mv][mm == m])) == 2})
pickle.dump(S, open(HERE / "ens_main.pkl", "wb"))
