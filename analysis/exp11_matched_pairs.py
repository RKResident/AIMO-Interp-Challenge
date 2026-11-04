"""Idea 4: agg, length-matched pairs. Each robust problem is paired with the nearest unused non-robust problem in (log prompt length,
log trace length). Then (a) paired Wilcoxon per feature with BH correction, (b) CV on the matched set (pairs kept together in a fold)."""
from common import *
from scipy.stats import wilcoxon
allm, _ = load_all()
agg = datasets(allm)["agg"]; y = allm.y.values
L = np.log(allm[["n_prompt", "n_gen"]].astype(float).values + 1)
Ls = (L - L[agg].mean(0)) / L[agg].std(0)
rob = [i for i in agg if y[i] == 1]; non = [i for i in agg if y[i] == 0]
used, pairs = set(), []
for i in sorted(rob, key=lambda i: Ls[i, 1]):                     # greedy nearest-neighbour without replacement
    cand = [j for j in non if j not in used]
    j = cand[int(np.argmin([np.sum((Ls[i] - Ls[j]) ** 2) for j in cand]))]; used.add(j); pairs.append((i, j))
pi, pj = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
print(f"=== Idea 4: agg length-matched pairs: {len(pairs)} pairs ===")
for k, nm in enumerate(("log prompt len", "log trace len")):
    print(f"  balance {nm}: robust mean {L[pi, k].mean():.2f} vs matched non-robust {L[pj, k].mean():.2f}  (all non-robust {L[non, k].mean():.2f}); "
          f"paired Wilcoxon p={wilcoxon(L[pi, k] - L[pj, k]).pvalue:.2f}")
S8 = pd.read_pickle(A / "exp8_features.pkl"); D4 = pd.read_pickle(A / "exp4_features.pkl"); T3 = pd.read_pickle(A / "exp3_features.pkl")
R7 = pickle.load(open(A / "exp7_features.pkl", "rb")); b1024 = pd.DataFrame([x if x is not None else {} for x in R7[1024]]).add_prefix("b1024:")
base = allm[[c for c in allm.columns if c.startswith(("lp_", "ent_", "top1_", "prob_", "fa_", "sc_", "agree_")) and allm[c].dtype != object]].astype(float)
ALL = pd.concat([base.reset_index(drop=True), S8.add_prefix("ab:"), D4.add_prefix("dyn:"), T3.add_prefix("txt:"), b1024], axis=1)
res = []
for c in ALL.columns:
    x = ALL[c].values.astype(float); d = x[pi] - x[pj]; ok = np.isfinite(d)
    if ok.sum() >= 15 and np.count_nonzero(d[ok]) >= 10:
        res.append((c, np.median(d[ok]), (d[ok] > 0).mean(), wilcoxon(d[ok]).pvalue, int(ok.sum())))
R = pd.DataFrame(res, columns=["feature", "median_diff(rob-non)", "frac_rob_higher", "p", "n"])
o = np.argsort(R.p.values); q = np.empty(len(R)); adj = R.p.values[o] * len(R) / np.arange(1, len(R) + 1); q[o] = np.minimum(1, np.minimum.accumulate(adj[::-1])[::-1])
R["q_BH"] = q; R = R.sort_values("p"); R.to_csv(A / "exp11_matched_pairs.csv", index=False)
print(f"\n  paired tests on {len(R)} features: {(R.q_BH < .1).sum()} with BH q<0.1; smallest p {R.p.min():.4f}")
print(R.head(15).round(4).to_string(index=False))
# CV on the matched set (pairs never split)
rows = np.concatenate([pi, pj]); grp_pair = np.zeros(len(allm), int); grp_pair[pi] = np.arange(len(pairs)); grp_pair[pj] = np.arange(len(pairs))
Lz = lengths(allm)
print("\n  CV on the matched set (each pair stays in one fold), 5-fold x 10:")


def cv(X, yy, reps=10, C=0.05):
    s, c = np.zeros(len(rows)), np.zeros(len(rows))
    for r in range(reps):
        for tr, te in folds(rows, yy, grp_pair, r):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(X[rows[tr]], yy[rows[tr]])
            s[te] += m.decision_function(X[rows[te]]); c[te] += 1
    return s / np.maximum(c, 1)


def shuf(seed):  # swap labels within pairs at random (the right null for a paired design)
    yy = y.copy(); flip = np.random.default_rng(seed).random(len(pairs)) < .5
    yy[pi[flip]], yy[pj[flip]] = 0, 1; return yy


fams = {"length (should be ~0.5 now)": Lz, "trace scalars": base[[c for c in base if c.startswith(("lp_", "ent_", "top1_"))]].values,
        "forced answer": base[[c for c in base if c.startswith(("fa_", "sc_", "agree_"))]].values, "A/B sensitivity": S8[[c for c in S8 if c.endswith(("_mean", "_max", "_bmean"))]].values,
        "dynamics": D4.values, "text markers": T3[[c for c in T3 if c.startswith("t_") and c != "t_words"]].values, "fixed-budget 1024": b1024.values,
        "prompt stats": allm[["prob_lp_mean", "prob_lp_min", "prob_ent_mean"]].astype(float).values}
for nm, X in fams.items():
    Xz = zgroup(X.astype(float), allm.grp.values)
    s = cv(Xz, y); a = auc_ci(y[rows], s); nl = np.mean([roc_auc_score(shuf(i)[rows], cv(Xz, shuf(i), reps=3)) for i in range(6)])
    print(f"    {nm:28s} AUC {a[0]:.2f} [{a[1]:.2f},{a[2]:.2f}]  within-pair-shuffle null {nl:.2f}")
