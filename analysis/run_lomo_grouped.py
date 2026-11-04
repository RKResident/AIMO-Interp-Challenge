src_code = open(__file__.replace("run_lomo_grouped.py", "run_experiments.py")).read().split("# ----------------------------------------------------------------------------------------- 2/3")[0]
exec(src_code)
others = {s: load(s)[0] for s in ["Qwen__Qwen3.5-4B", "Skywork__Skywork-OR1-Math-7B", "allenai__Olmo-3-7B-Think"]}
allm = pd.concat([meta.assign(model=DS)] + [m.assign(model=s) for s, m in others.items()], ignore_index=True)
allm["y"] = allm.is_robust.map({True: 1, False: 0, "True": 1, "False": 0}).astype(float)
common = [c for c in tcols + fc + sc if c in allm.columns and allm[c].notna().mean() > 0.8]
Xa = src_standardize(allm[common].astype(float).values, (allm.model + allm.source).values)
mainm = allm[(allm.source == "main_v2") & allm.y.notna()].index.values
print("labeled main_v2 rows", len(mainm), "unique problems", allm.loc[mainm, "base_id"].nunique())
for C in (0.01, 0.1):
    oo, yy, mm = [], [], []
    for i in mainm:
        hold = set(allm.base_id[allm.model == allm.model[i]][allm.source == "main_v2"])  # all problems of the held-out model
        tr = np.array([k for k in mainm if allm.model[k] != allm.model[i] and allm.base_id[k] != allm.base_id[i] and allm.base_id[k] not in hold])
        if len(np.unique(allm.y[tr])) < 2:
            continue
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(Xa[tr], allm.y[tr])
        oo.append(m.decision_function(Xa[[i]])[0]); yy.append(allm.y[i]); mm.append(allm.model[i])
    oo, yy, mm = np.array(oo), np.array(yy), np.array(mm)
    print(f"C={C}: problem-disjoint LOMO pooled AUC n={len(yy)}:", [round(float(v), 2) for v in auc_ci(yy, oo)],
          {k.split('__')[-1][:10]: round(roc_auc_score(yy[mm == k], oo[mm == k]), 2) for k in np.unique(mm) if len(np.unique(yy[mm == k])) == 2})
# also: scalar control using only trace length
Xl = src_standardize(np.log(allm[["n_gen", "n_prompt"]].astype(float).values + 1), (allm.model + allm.source).values)
oo, yy = [], []
for i in mainm:
    hold = set(allm.base_id[(allm.model == allm.model[i]) & (allm.source == "main_v2")])
    tr = np.array([k for k in mainm if allm.model[k] != allm.model[i] and allm.base_id[k] not in hold])
    if len(np.unique(allm.y[tr])) < 2: continue
    oo.append(make_pipeline(StandardScaler(), LogisticRegression()).fit(Xl[tr], allm.y[tr]).decision_function(Xl[[i]])[0]); yy.append(allm.y[i])
print("length-only control, same protocol:", [round(float(v), 2) for v in auc_ci(np.array(yy), np.array(oo))])
