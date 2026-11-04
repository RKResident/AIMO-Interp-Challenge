"""Pooled over the 4 models, problem-grouped CV (a problem is held out from ALL models at once). No gpt_proxy rows."""
src_code = open(__file__.replace("run_pooled_grouped.py", "run_experiments.py")).read().split("# ----------------------------------------------------------------------------------------- 2/3")[0]
exec(src_code)
from sklearn.model_selection import GroupKFold
others = {s: load(s)[0] for s in ["Qwen__Qwen3.5-4B", "Skywork__Skywork-OR1-Math-7B", "allenai__Olmo-3-7B-Think"]}
allm = pd.concat([meta.assign(model=DS)] + [m.assign(model=s) for s, m in others.items()], ignore_index=True)
allm = allm[allm.source.isin(["main_v2", "agg"])].reset_index(drop=True)          # no gpt_proxy, no unlabeled perm
allm["y"] = allm.is_robust.map({True: 1, False: 0, "True": 1, "False": 0}).astype(float)
allm = allm[allm.y.notna()].reset_index(drop=True)
common = [c for c in tcols + fc + sc if c in allm.columns and allm[c].notna().mean() > 0.8]
grp = (allm.model + allm.source).values
Xs = src_standardize(allm[common].astype(float).values, grp)
Xl = src_standardize(np.log(allm[["n_gen", "n_prompt"]].astype(float).values + 1), grp)
Xm = np.hstack([Xs, pd.get_dummies(allm.model).values.astype(float)])             # + model indicator


def run(X, rows, C, reps=10):
    yy, g = allm.y.values[rows], allm.base_id.values[rows]
    oof, cnt = np.zeros(len(rows)), np.zeros(len(rows))
    for r in range(reps):
        order = np.random.default_rng(r).permutation(len(np.unique(g)))
        gid = pd.Series(g).map(dict(zip(np.unique(g), order))).values
        for tr, te in GroupKFold(n_splits=5).split(rows, yy, groups=gid % 5 if False else gid):
            if len(np.unique(yy[tr])) < 2: continue
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(X[rows[tr]], yy[tr])
            oof[te] += m.decision_function(X[rows[te]]); cnt[te] += 1
    return oof / np.maximum(cnt, 1)


mv = np.where(allm.source == "main_v2")[0]
av = np.arange(len(allm))
print(f"main_v2 rows {len(mv)} ({allm.y[mv].sum():.0f} robust), problems {allm.base_id[mv].nunique()}; +agg rows {len(av)-len(mv)}")
for tag, rows in (("train+test on main_v2 only", mv), ("main_v2 test, agg rows always in train", None)):
    for nm, X in (("scalar features", Xs), ("scalar + model id", Xm), ("length only (control)", Xl)):
        for C in (0.01, 0.1):
            if rows is not None:
                s = run(X, rows, C); yy = allm.y.values[rows]; mm = allm.model.values[rows]
            else:  # agg always in train; CV only over main_v2 problems
                yy_all, g = allm.y.values, allm.base_id.values
                ug = np.unique(g[mv]); s = np.zeros(len(mv)); c = np.zeros(len(mv))
                for r in range(10):
                    perm_ = np.random.default_rng(r).permutation(ug); fold = {p: i % 5 for i, p in enumerate(perm_)}
                    f = np.array([fold[p] for p in g[mv]])
                    for k in range(5):
                        te = mv[f == k]; tr = np.concatenate([np.where(allm.source == "agg")[0], mv[f != k]])
                        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(X[tr], yy_all[tr])
                        s[f == k] += m.decision_function(X[te]); c[f == k] += 1
                s = s / np.maximum(c, 1); yy = allm.y.values[mv]; mm = allm.model.values[mv]
            a, lo, hi = auc_ci(yy, s)
            per = {k.split('__')[-1][:8]: round(roc_auc_score(yy[mm == k], s[mm == k]), 2) for k in np.unique(mm) if len(np.unique(yy[mm == k])) == 2}
            print(f"{tag:42s} {nm:22s} C={C:<5} AUC {a:.2f} [{lo:.2f},{hi:.2f}]  per-model {per}")
