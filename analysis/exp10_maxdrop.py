"""Idea 3: continuous target max_drop on main_v2 (all rows incl. borderline ones with no binary label). Ridge regression, CV grouped by problem;
score = Spearman(OOF prediction, max_drop) (expect NEGATIVE = 'predicted robust' rows drop less if we predict robustness; here we predict max_drop
directly so expect POSITIVE). Null = max_drop shuffled across problems. Plus single-feature Spearman at the problem level."""
from common import *
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
allm, _ = load_all()
rows = np.where((allm.source == "main_v2") & allm.max_drop.notna())[0]
md = allm.max_drop.astype(float).values
pb = np.unique(allm.base_id.values, return_inverse=True)[1]
grp = allm.grp.values
print(f"=== Idea 3: max_drop on main_v2: {len(rows)} rows ({int(allm.y[rows].isna().sum())} without a binary label), {len(set(allm.base_id.values[rows]))} problems ===")
print("max_drop by label:", allm.iloc[rows].groupby(allm.y[rows].fillna(-1)).max_drop.describe()[["count", "mean", "min", "max"]].round(3).to_dict("index"))
S8 = pd.read_pickle(A / "exp8_features.pkl"); R7 = pickle.load(open(A / "exp7_features.pkl", "rb"))
D4 = pd.read_pickle(A / "exp4_features.pkl"); T3 = pd.read_pickle(A / "exp3_features.pkl")
b2048 = pd.DataFrame([x if x is not None else {} for x in R7[2048]])
fams = {
    "length only": lengths(allm),
    "prompt stats (problem logprob/entropy)": allm[["prob_lp_mean", "prob_lp_min", "prob_ent_mean"]].astype(float).values,
    "A/B suffix sensitivity": S8[[c for c in S8 if c.endswith(("_mean", "_max", "_bmean"))]].values,
    "fixed-budget 2048 (all families)": b2048.values.astype(float),
    "trace scalars (whole trace)": allm[[c for c in allm.columns if c.startswith(("lp_", "ent_", "top1_"))]].astype(float).values,
    "forced-answer conf/consistency": allm[[c for c in allm.columns if c.startswith(("fa_", "sc_", "agree_")) and allm[c].notna().mean() > .8]].astype(float).values,
    "dynamics (exp4)": D4.values.astype(float),
    "text markers": T3[[c for c in T3 if c.startswith("t_") and c != "t_words"]].values,
}


def oof(X, target, reps=10, alpha=30.0):
    s, c = np.zeros(len(rows)), np.zeros(len(rows))
    for r in range(reps):
        for tr, te in folds(rows, np.zeros(len(allm)), pb, r):
            m = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X[rows[tr]], target[rows[tr]])
            s[te] += m.predict(X[rows[te]]); c[te] += 1
    return s / np.maximum(c, 1)


def shuffled(seed):
    t = md.copy(); pr = allm.base_id.values[rows]; up = np.unique(pr)
    val = {p: md[rows][pr == p].mean() for p in up}; pm = dict(zip(up, np.random.default_rng(seed).permutation(up)))
    # keep within-problem spread, move problem means around
    t[rows] = [md[i] - val[p] + val[pm[p]] for i, p in zip(rows, pr)]
    return t


Lz = lengths(allm)
for nm, X in fams.items():
    Xz = zgroup(X, grp)
    s = oof(Xz, md); rho = spearmanr(s, md[rows])[0]
    nl = [spearmanr(oof(Xz, t, reps=3), t[rows])[0] for t in (shuffled(i) for i in range(8))]
    Xr = Xz.copy(); Xr[rows] = Xr[rows] - LinearRegression().fit(Lz[rows], Xr[rows]).predict(Lz[rows])
    rr = spearmanr(oof(Xr, md), md[rows])[0] if nm != "length only" else np.nan
    print(f"  {nm:38s} Spearman(OOF pred, max_drop) {rho:+.2f}   null {np.mean(nl):+.2f} (sd {np.std(nl):.2f})   length-removed {rr:+.2f}", flush=True)
# problem level single features (mean over models), Spearman with problem mean max_drop
pr = allm.base_id.values[rows]; up = np.unique(pr)
ymd = np.array([md[rows][pr == p].mean() for p in up])
print(f"\n  problem-level single-feature Spearman with mean max_drop ({len(up)} problems), top 10 of the length-free sets:")
cand = pd.concat([S8[[c for c in S8 if c.endswith(("_mean", "_bmean"))]], b2048.add_prefix("b2048:")], axis=1)
Cz = zgroup(cand.values.astype(float), grp)
res = []
for j, c in enumerate(cand.columns):
    v = np.array([Cz[rows][pr == p, j].mean() for p in up])
    if np.std(v) > 0: res.append((c, *spearmanr(v, ymd)))
res.sort(key=lambda t: t[2])
for c, rho, p in res[:10]: print(f"    {c:28s} rho {rho:+.2f}  p {p:.3f}")
print(f"    ({len(res)} features scanned; the smallest p expected by chance is ~{1/len(res):.3f})")
