"""Same families, but with log prompt length / log trace length / log think length regressed out of every column
(fit label-free, per dataset). Anything that still beats chance carries information beyond length."""
import re
src_code = open(__file__.replace("run_resid.py", "run_experiments.py")).read().split("# ----------------------------------------------------------------------------------------- 2/3")[0]
exec(src_code)
from sklearn.linear_model import LinearRegression
src = meta.source.values
Z = np.hstack([log_("n_prompt"), log_("n_gen"), log_("L_think")])
Z = (Z - Z.mean(0)) / Z.std(0)


def residualize(X, sources):
    X = np.nan_to_num(X.copy()); out = X.copy()
    for s in np.unique(sources):
        m = sources == s
        if m.sum() < 8:
            continue
        lr = LinearRegression().fit(Z[m], X[m])
        out[m] = X[m] - lr.predict(Z[m])
    return out


rows = []
for name, (X, kind) in FAM.items():
    if name.startswith("ctrl:"):
        continue
    Xr = src_standardize(residualize(X, src), src)
    oof, folds = cv_oof(Xr, kind, agg)
    a, lo, hi = auc_ci(y[agg], oof)
    # label-shuffle null for this exact pipeline (3 shuffles) to show the chance spread
    nulls = []
    for sd in range(3):
        yy = y.copy(); yy[agg] = np.random.default_rng(sd).permutation(y[agg])
        keep = y; y = yy
        o2, _ = cv_oof(Xr, kind, agg, reps=2, seed=sd); nulls.append(roc_auc_score(y[agg], o2)); y = keep
    m = make_model(kind).fit(Xr[agg], y[agg])
    g = roc_auc_score(y[main], m.decision_function(Xr[main]))
    rows.append((name, X.shape[1], a, lo, hi, np.mean(nulls), g))
    print(f"{name:32s} len-residualised agg-CV AUC {a:.3f} [{lo:.3f},{hi:.3f}]  shuffled-label null ~{np.mean(nulls):.2f}  gate(main)={g:.2f}", flush=True)
pd.DataFrame(rows, columns=["family", "dim", "auc", "lo", "hi", "null", "gate_main"]).to_csv(ROOT / "results_deepseek_len_residualised.csv", index=False)
