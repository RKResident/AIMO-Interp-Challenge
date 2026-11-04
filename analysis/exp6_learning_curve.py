"""Exp 6: learning curve on agg (DeepSeek, 137 problems). Held-out test = 27 problems (stratified), trained on n in {25,50,75,100,110}; 40 random splits."""
from common import *
from sklearn.decomposition import PCA
allm, feats = load_all()
agg = datasets(allm)["agg"]
y = allm.y.values[agg].astype(int)
Dy = pd.read_pickle(A / "exp4_features.pkl"); T = pd.read_pickle(A / "exp3_features.pkl")
fcols = [c for c in allm.columns if c.startswith(("fa_", "sc_", "agree_")) and allm[c].dtype != object and allm[c].notna().mean() > .8]
tcols = [c for c in allm.columns if c.startswith(("lp_", "ent_", "top1_")) and allm[c].dtype != object]
Z = lengths(allm)
sets = {
    "length only": (Z[agg], "low"),
    "trace stats": (zgroup(allm[tcols].astype(float).values, allm.grp.values)[agg], "low"),
    "forced-answer features": (zgroup(allm[fcols].astype(float).values, allm.grp.values)[agg], "low"),
    "hidden-state drift": (zgroup(Dy[[c for c in Dy if c.startswith("h")]].values, allm.grp.values)[agg], "low"),
    "text markers": (zgroup(T[[c for c in T if c.startswith("t_") and c != "t_words"]].values, allm.grp.values)[agg], "low"),
    "prompt state (layer 24)": (zgroup(np.vstack([feats[DS]["prompt_last"][:, 24]] + [np.zeros((len(allm) - len(feats[DS]["prompt_last"]), 4096))]), allm.grp.values)[agg], "high"),
}
sizes = [25, 50, 75, 100, 110]
rng = np.random.default_rng(0)
res = {k: {n: [] for n in sizes} for k in sets}
for rep in range(40):
    tr_all, te = next(iter(StratifiedKFold(5, shuffle=True, random_state=rep).split(np.zeros(len(y)), y)))
    for n in sizes:
        pick = []
        for cls in (0, 1):
            idx = tr_all[y[tr_all] == cls]; k = max(2, int(round(n * len(idx) / len(tr_all)))); pick += list(rng.choice(idx, min(k, len(idx)), replace=False))
        pick = np.array(pick)
        for nm, (X, kind) in sets.items():
            mdl = make_pipeline(StandardScaler(), PCA(8, random_state=0), StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)) if kind == "high" else \
                  make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
            res[nm][n].append(roc_auc_score(y[te], mdl.fit(X[pick], y[pick]).decision_function(X[te])))
print("=== Exp 6: learning curve on agg (mean held-out AUC over 40 splits; ± is the s.d. across splits, test n=27) ===")
print(f"{'train n':>26s} " + " ".join(f"{n:>11d}" for n in sizes))
for nm in sets:
    print(f"{nm:>26s} " + " ".join(f"{np.mean(res[nm][n]):.3f}±{np.std(res[nm][n]):.2f}" for n in sizes))
print("slope of mean AUC from n=50 to n=110 (per +10 problems):")
for nm in sets:
    print(f"  {nm:26s} {(np.mean(res[nm][110]) - np.mean(res[nm][50])) / 6:+.4f}")
