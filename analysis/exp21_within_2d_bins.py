"""Stricter version of the within-bin check: bins on BOTH thinking length and prompt length (3 x 3 quantile bins), because prompt length
alone predicts robustness on agg (AUC ~0.72) and the 1-D thinking-length bins let it through. Scores only same-bin robust/non-robust pairs;
null = labels shuffled within the same 2-D bins (so a pure length/prompt-length feature scores at its null). agg only (main_v2 has 22 problems)."""
from common import *
from sklearn.decomposition import PCA
P2 = ROOT / "extraction_out_generations_p2" / "extraction_out_generations"
allm, feats = load_all(); n_ds = len(feats[DS]["prompt_last"])
agg = datasets(allm)["agg"]; y = allm.y.values; ids = np.arange(len(allm))
Lthink = np.array([force_json(r.model, r.id)["L"] for r in allm.itertuples()])


def bins2d(rows, nb=3):
    a = np.log(allm.L_think.astype(float).values[rows] + 1); b = np.log(allm.n_prompt.astype(float).values[rows] + 1)
    qa = np.digitize(a, np.quantile(a, np.linspace(0, 1, nb + 1)[1:-1])); qb = np.digitize(b, np.quantile(b, np.linspace(0, 1, nb + 1)[1:-1]))
    return qa * nb + qb


def shuffle_within(rows, bins, seed):
    y2 = y.copy(); rng = np.random.default_rng(seed)
    for b in np.unique(bins):
        idx = rows[bins == b]; y2[idx] = rng.permutation(y2[idx])
    return y2


def cv(X, rows, yy, kind, reps=8, seed=0):
    s, c = np.zeros(len(rows)), np.zeros(len(rows))
    for r in range(reps):
        for tr, te in folds(rows, yy, ids, seed + r):
            mdl = make_pipeline(StandardScaler(), PCA(8, random_state=0), StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)) if kind == "high" \
                else make_pipeline(StandardScaler(), LogisticRegression(C=0.03, max_iter=3000))
            s[te] += mdl.fit(X[rows[tr]], yy[rows[tr]]).decision_function(X[rows[te]]); c[te] += 1
    return s / np.maximum(c, 1)


D4 = pd.read_pickle(A / "exp4_features.pkl"); T3 = pd.read_pickle(A / "exp3_features.pkl")
feat, probe = pickle.load(open(A / "exp19_cache.pkl", "rb"))


def mat(lst, d):
    X = np.full((len(allm), d), np.nan, np.float32)
    for i, x in enumerate(lst):
        if x is not None and len(x) == d: X[i] = x
    return X


PL = np.zeros((len(allm), 4096), np.float32); PL[:n_ds] = feats[DS]["prompt_last"][:, 30]
LP = np.zeros((len(allm), 1152), np.float32); LP[:n_ds] = feats[DS]["lap_force"][:, 3].mean(1)[..., 0].reshape(n_ds, -1)
TW = np.full((len(allm), 1152), np.nan, np.float32)
for i in agg:
    z = np.load(P2 / allm.model.values[i] / "toha" / f"{allm.id.values[i]}.npz"); ce = list(z["cut_eff"])
    if 2048 in ce and Lthink[i] > 2048: TW[i] = z["win"][ce.index(2048)][..., 0].reshape(-1)
L3 = np.log(allm[["n_prompt", "n_gen", "L_think"]].astype(float).values + 1)
fams = [("length + prompt length (control)", L3, "low"), ("prompt probe (layer 30)", PL, "high"),
        ("anchoring cosine", D4[[f"h{j}_cos_prob_mean" for j in range(6)]].values, "low"),
        ("Laplacian top-1, forced, end of thinking", LP, "high"), ("text markers", T3[[c for c in T3 if c.startswith("t_") and c != "t_words"]].values, "low"),
        ("probe: mean of 64 random thinking windows", mat(probe["bag_mean"], 4096), "high"),
        ("probe: mean of a random 1024-token segment", mat(probe["segment_mean"], 4096), "high"),
        ("TOHA reasoning window @2048", TW, "high")]
print("=== agg: AUC over same-bin pairs, bins = 3 thinking-length x 3 prompt-length quantiles ===")
for nm, X, kind in fams:
    rows = agg[np.isfinite(X[agg]).all(1)]
    Xz = zgroup(X, allm.grp.values) if kind == "low" else np.nan_to_num(X)
    b2 = bins2d(rows); s = cv(Xz, rows, y, kind)
    w = within_bin_auc(y[rows], s, b2)
    nl = []
    for sd in range(8):
        y2 = shuffle_within(rows, b2, sd); nl.append(within_bin_auc(y2[rows], cv(Xz, rows, y2, kind, reps=2, seed=50 + sd), b2))
    nl = np.array(nl)
    print(f"  {nm:44s} n={len(rows):3d}  plain AUC {roc_auc_score(y[rows], s):.2f} | 2-D within-bin AUC {w:.2f}  (null {nl.mean():.2f} +- {nl.std():.2f};  z = {(w - nl.mean()) / (nl.std() + 1e-9):+.1f})", flush=True)
