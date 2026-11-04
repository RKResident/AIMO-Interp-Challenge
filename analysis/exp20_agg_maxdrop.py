"""max_drop regression on agg (DeepSeek). Robust <=> max_drop == 0 here, so two targets: (i) all 137 problems, (ii) SEVERITY among the 101
non-robust problems (max_drop 0.5-1.0). Ridge (PCA for high-dim), 5-fold x 8 OOF predictions, Spearman with the target.
Controls: length-only, all lengths regressed out, within-length-bin Spearman, shuffled-target null."""
from common import *
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
P2 = ROOT / "extraction_out_generations_p2" / "extraction_out_generations"
allm, feats = load_all(); n_ds = len(feats[DS]["prompt_last"])
agg = datasets(allm)["agg"]; md = allm.max_drop.astype(float).values; y = allm.y.values
L3 = np.log(allm[["n_prompt", "n_gen", "L_think"]].astype(float).values + 1)
Lthink = np.array([force_json(r.model, r.id)["L"] for r in allm.itertuples()])
D4 = pd.read_pickle(A / "exp4_features.pkl"); T3 = pd.read_pickle(A / "exp3_features.pkl"); S8 = pd.read_pickle(A / "exp8_features.pkl")
PL = np.zeros((len(allm), 4096), np.float32); PL[:n_ds] = feats[DS]["prompt_last"][:, 30]
LP = np.zeros((len(allm), 1152), np.float32); LP[:n_ds] = feats[DS]["lap_force"][:, 3].mean(1)[..., 0].reshape(n_ds, -1)
fcols = [c for c in allm.columns if c.startswith(("fa_", "sc_", "agree_")) and allm[c].dtype != object and allm[c].notna().mean() > .8]
tcols = [c for c in allm.columns if c.startswith(("lp_", "ent_", "top1_"))]
TW = np.full((len(allm), 1152), np.nan, np.float32)
for i in agg:
    z = np.load(P2 / allm.model.values[i] / "toha" / f"{allm.id.values[i]}.npz"); ce = list(z["cut_eff"])
    if 2048 in ce and Lthink[i] > 2048: TW[i] = z["win"][ce.index(2048)][..., 0].reshape(-1)
FAMS = {"length only": (L3, "low"), "prompt probe (layer 30)": (PL, "high"), "anchoring cosine": (D4[[f"h{j}_cos_prob_mean" for j in range(6)]].values, "low"),
        "forced-answer conf/consistency": (allm[fcols].astype(float).values, "low"), "trace token statistics": (allm[tcols].astype(float).values, "low"),
        "Laplacian top-1, forced, end of thinking": (LP, "high"), "text markers": (T3[[c for c in T3 if c.startswith("t_") and c != "t_words"]].values, "low"),
        "A/B suffix sensitivity": (S8[[c for c in S8 if c.endswith(("_mean", "_max", "_bmean"))]].values, "low"),
        "TOHA reasoning window @2048": (TW, "high")}


def oof(X, rows, t, kind, reps=8, seed=0):
    s, c = np.zeros(len(rows)), np.zeros(len(rows))
    for r in range(reps):
        for tr, te in KFold(5, shuffle=True, random_state=seed + r).split(rows):
            mdl = make_pipeline(StandardScaler(), PCA(8, random_state=0), Ridge(alpha=10.0)) if kind == "high" else make_pipeline(StandardScaler(), Ridge(alpha=30.0))
            s[te] += mdl.fit(X[rows[tr]], t[rows[tr]]).predict(X[rows[te]]); c[te] += 1
    return s / c


def wb_spearman(t, s, bins):
    num = den = 0.0
    for b in np.unique(bins):
        m = bins == b
        if m.sum() >= 5 and np.std(t[m]) > 0: num += spearmanr(t[m], s[m])[0] * m.sum(); den += m.sum()
    return num / den if den else np.nan


for tag, base_rows in (("ALL agg problems (n=137; mostly the binary label)", agg), ("SEVERITY among non-robust problems only", agg[y[agg] == 0])):
    print(f"\n=== max_drop, {tag} ===")
    for nm, (X, kind) in FAMS.items():
        rows = base_rows[np.isfinite(X[base_rows]).all(1)]
        Xz = zgroup(np.where(np.isfinite(X), X, np.nan), allm.grp.values) if kind == "low" else np.nan_to_num(X)
        s = oof(Xz, rows, md, kind); rho = spearmanr(s, md[rows])[0]
        bins = length_bins(allm, rows); wb = wb_spearman(md[rows], s, bins)
        nl = []
        for sd in range(5):
            t2 = md.copy(); t2[rows] = np.random.default_rng(sd).permutation(md[rows]); nl.append(spearmanr(oof(Xz, rows, t2, kind, reps=2, seed=30 + sd), t2[rows])[0])
        if nm != "length only":
            Xr = Xz.astype(np.float64).copy(); Xr[rows] -= LinearRegression().fit(L3[rows], Xr[rows]).predict(L3[rows])
            rr = spearmanr(oof(Xr, rows, md, kind), md[rows])[0]
        else:
            rr = np.nan
        print(f"  {nm:42s} n={len(rows):3d}  Spearman {rho:+.2f} (null {np.mean(nl):+.2f} +- {np.std(nl):.2f})  | lengths removed {rr:+.2f}  | within-length-bin {wb:+.2f}", flush=True)
