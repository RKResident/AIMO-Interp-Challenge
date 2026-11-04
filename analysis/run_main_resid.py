"""main_v2 only (4 models pooled, problem-grouped CV): ensemble with log prompt/trace/think length regressed out of every family."""
from pathlib import Path
HERE = Path(__file__).resolve().parent
__file__ = str(HERE / "run_ensembles.py")
exec(open(HERE / "run_ensembles.py").read().split("mkm = lambda")[0])
from sklearn.linear_model import LinearRegression
grp_all = (allm.model + allm.source).values
Lcols = src_standardize(np.log(allm[["n_prompt", "n_gen", "L_think"]].astype(float).values + 1), grp_all)
lr_cache = {}


def resid_rows(X):
    X = np.nan_to_num(X.astype(np.float64)); out = X.copy()
    out[mv] = X[mv] - LinearRegression().fit(Lcols[mv], X[mv]).predict(Lcols[mv])      # label-free, main_v2 rows only
    return out


fam_r = {k: (src_standardize(resid_rows(fam[k]), grp_all), "low") for k in
         [k for k in fam if k not in ("length (control)", "scalars+hid+lap", "hid:all_cuts", "lap:force_all_cuts", "lap:mean_force_all_cuts")]}
pb = np.unique(prob, return_inverse=True)[1]
mk_ = lambda kind: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))
print(f"main_v2 (length removed): {len(mv)} rows, {len(set(prob[mv]))} problems, {len(fam_r)} families", flush=True)
S = nested(fam_r, mv, yv.astype(int), pb, reps=6, seed0=0, mk=mk_, rf_X=resid_rows(fam["scalars"]), n_jobs=4)
nulls = []
for sd in range(4):
    up = np.unique(prob[mv]); lab = {p: yv[mv][prob[mv] == p][0] for p in up}; pm_ = dict(zip(up, np.random.default_rng(sd).permutation(up)))
    y2 = yv.copy(); y2[mv] = [lab[pm_[p]] for p in prob[mv]]
    nulls.append((y2[mv].astype(int), nested(fam_r, mv, y2.astype(int), pb, reps=2, seed0=70 + sd, mk=mk_, n_jobs=4)))
lc = Lcols
sc_ = np.zeros(len(mv)); cc = np.zeros(len(mv))
for r in range(6):
    for a, b in folds_for(mv, yv.astype(int), pb, r):
        m_ = mk_("low").fit(lc[mv[a]], yv[mv[a]]); sc_[b] += m_.decision_function(lc[mv[b]]); cc[b] += 1
report("main_v2 CV, LENGTH REGRESSED OUT", S, yv[mv], nulls, roc_auc_score(yv[mv], sc_ / np.maximum(cc, 1)))
mm = allm.model.values[mv]
for k in ("mean_all", "top5"):
    print(f"  {k:9s} per model:", {m.split('__')[-1][:7]: round(roc_auc_score(yv[mv][mm == m], S[k][mm == m]), 2) for m in np.unique(mm) if len(np.unique(yv[mv][mm == m])) == 2})
# how much of each family's signal is left: single-family AUCs (same folds), before/after
for tag, FM in (("with length", {k: (src_standardize(np.nan_to_num(fam[k]), grp_all), "low") for k in fam_r}), ("length removed", fam_r)):
    r_ = {}
    for k, (X, kind) in FM.items():
        s = np.zeros(len(mv)); c = np.zeros(len(mv))
        for rep in range(4):
            for a, b in folds_for(mv, yv.astype(int), pb, rep):
                if len(np.unique(yv[mv[a]])) < 2: continue
                s[b] += mk_(kind).fit(X[mv[a]], yv[mv[a]].astype(int)).decision_function(X[mv[b]]); c[b] += 1
        r_[k] = roc_auc_score(yv[mv], s / np.maximum(c, 1))
    print(f"  single-family AUC, {tag}: mean {np.mean(list(r_.values())):.2f}, max {max(r_, key=r_.get)} {max(r_.values()):.2f}, min {min(r_.values()):.2f}")
