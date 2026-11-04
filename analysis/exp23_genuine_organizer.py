"""Robust (36) vs GENUINELY non-robust (organizer decay permutation that keeps every distinct number of the original) on agg (DeepSeek).
Information-loss-only non-robust problems (52) are left out. Every feature family from the project; per family: plain CV AUC (+ shuffled null),
all-lengths-removed AUC, and AUC over same-bin pairs in 3x3 thinking-length x prompt-length bins (+ within-bin shuffled null, z)."""
import os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
_src = open(HERE / "run_experiments.py").read().split("# ----------------------------------------------------------------------------------------- 2/3")[0]
_g = {"__file__": str(HERE / "run_experiments.py")}; exec(_src, _g)
FAM_DS, n_ds = _g["FAM"], len(_g["meta"])
from common import *
from collections import Counter
from sklearn.decomposition import PCA
P2 = ROOT / "extraction_out_generations_p2" / "extraction_out_generations"
allm, feats = load_all(); agg = datasets(allm)["agg"]
assert list(allm.id.values[:n_ds]) == list(_g["meta"].id.values)
Lthink = np.array([force_json(r.model, r.id)["L"] for r in allm.itertuples()])
L3 = np.log(allm[["n_prompt", "n_gen", "L_think"]].astype(float).values + 1)

# ---------------- groups: from the ORGANIZERS' decay-causing permutations (agg csv, `permutations_causing_decay`)
# genuine  = non-robust problem with >= 1 organizer decay permutation that keeps every DISTINCT number of the original
# info_loss = every organizer decay permutation drops at least one distinct number (most contain no numbers at all)
a = pd.read_csv(ROOT / "augmented-sample-math-agg.csv")
nums = lambda s: set(re.findall(r"\d+(?:\.\d+)?", str(s)))
grp, keep_any = {}, {}
for r in a.itertuples():
    try:
        perms = json.loads(r.permutations_causing_decay)
    except Exception:
        perms = []
    pid = str(r.problem_id)
    if bool(r.model_is_robust) and not perms:
        grp.setdefault(pid, "robust"); continue
    on = nums(r.original_problem)
    k = any(on <= nums(p.get("question", "") if isinstance(p, dict) else p) for p in perms)
    keep_any[pid] = keep_any.get(pid, False) or k
for pid, k in keep_any.items():
    grp[pid] = "genuine" if k else "info_loss"
g = np.array([grp.get(str(p)) if i in set(agg) else None for i, p in enumerate(allm.parent_id.astype(str).values)])
print({k: int((g == k).sum()) for k in ("robust", "genuine", "info_loss")})
rows_all = np.array([i for i in agg if g[i] in ("robust", "genuine")])
Y = np.zeros(len(allm)); Y[rows_all] = (g[rows_all] == "robust").astype(float)
print(f"robust {int(Y[rows_all].sum())} vs genuinely non-robust {int((1 - Y[rows_all]).sum())}")

# ---------------- families
def pad(X):
    X = np.asarray(X, np.float32); out = np.full((len(allm),) + X.shape[1:], np.nan, np.float32); out[:len(X)] = X; return out


FAMS = {}
for k, (X, kind) in FAM_DS.items():
    FAMS[k] = (pad(X), kind)
T3 = pd.read_pickle(A / "exp3_features.pkl"); D4 = pd.read_pickle(A / "exp4_features.pkl"); S8 = pd.read_pickle(A / "exp8_features.pkl")
FAMS["text: markers"] = (T3[[c for c in T3 if c.startswith("t_") and c != "t_words"]].values, "low")
FAMS["text: entropy bursts"] = (T3[[c for c in T3 if c.startswith("e_")]].values, "low")
FAMS["dynamics: hidden drift"] = (D4[[c for c in D4 if c.startswith("h")]].values, "low")
FAMS["dynamics: anchoring cosine"] = (D4[[f"h{j}_cos_prob_mean" for j in range(6)]].values, "low")
FAMS["dynamics: answer commitment"] = (D4[[c for c in D4 if c.startswith("c")]].values, "low")
FAMS["dynamics: Laplacian curves"] = (D4[[c for c in D4 if c.startswith("l_")]].values, "low")
FAMS["A/B suffix sensitivity"] = (S8[[c for c in S8 if c.endswith(("_mean", "_max", "_bmean"))]].values, "low")
R7 = pickle.load(open(A / "exp7_features.pkl", "rb"))
for b in (1024, 2048):
    F7 = pd.DataFrame([x if x is not None else {} for x in R7[b]])
    for nm, pre in (("prefix token stats", "p_"), ("forced answer", "f_"), ("hidden state", "h_"), ("Laplacian", ("lmn", "lmf", "ltn", "ltf")), ("text markers", "t_")):
        cols = [c for c in F7.columns if c.startswith(pre)]
        if cols: FAMS[f"fixed-budget {b}: {nm}"] = (F7[cols].values.astype(float), "low")
for b in (1024, 2048):
    for kind in ("win", "ans"):
        X = np.full((len(allm), 1152), np.nan, np.float32)
        for i in agg:
            z = np.load(P2 / allm.model.values[i] / "toha" / f"{allm.id.values[i]}.npz"); ce = list(z["cut_eff"])
            if b in ce and Lthink[i] > b: X[i] = (z["win"][ce.index(b)] if kind == "win" else z["ans"][ce.index(b), 0])[..., 0].reshape(-1)
        FAMS[f"TOHA {kind} @{b}"] = (X, "high")
R18 = pd.read_pickle(A / "exp18_features.pkl")
for k in (1, 4, 16):
    for b in (2048, "full"):
        cols = [c for c in R18.columns if c.startswith(f"k{k}|{b}|")]
        if cols: FAMS[f"window geometry {16 * k}-tok, {b}"] = (R18[cols].values.astype(float), "low")
feat19, probe19 = pickle.load(open(A / "exp19_cache.pkl", "rb"))
def mat(lst_, dim=None):
    dim = dim or next(len(x) for x in lst_ if x is not None); X = np.full((len(allm), dim), np.nan, np.float32)
    for i, x in enumerate(lst_):
        if x is not None and len(x) == dim: X[i] = x
    return X
for nm in ("bag", "segment", "seg25", "seg50", "seg75"): FAMS[f"subsample: {nm}"] = (mat(feat19[nm]), "low")
FAMS["subsample probe: bag mean state"] = (mat(probe19["bag_mean"], 4096), "high")
FAMS["subsample probe: segment mean state"] = (mat(probe19["segment_mean"], 4096), "high")
print(f"{len(FAMS)} families")


# ---------------- evaluation
def cv(X, rows, yy, kind, reps=8, seed=0):
    s, c = np.zeros(len(rows)), np.zeros(len(rows))
    for r in range(reps):
        for tr, te in folds(rows, yy, np.arange(len(allm)), seed + r):
            if len(np.unique(yy[rows[tr]])) < 2: continue
            m = (make_pipeline(StandardScaler(), PCA(min(8, len(tr) - 2), random_state=0), StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)) if kind == "high"
                 else make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)))
            s[te] += m.fit(X[rows[tr]], yy[rows[tr]]).decision_function(X[rows[te]]); c[te] += 1
    return s / np.maximum(c, 1)


def bins2d(rows):
    a = np.log(allm.L_think.astype(float).values[rows] + 1); b = np.log(allm.n_prompt.astype(float).values[rows] + 1)
    return np.digitize(a, np.quantile(a, [1 / 3, 2 / 3])) * 3 + np.digitize(b, np.quantile(b, [1 / 3, 2 / 3]))


res, oof_store = [], {}
for nm, (X, kind) in FAMS.items():
    rows = rows_all[np.isfinite(X[rows_all]).reshape(len(rows_all), -1).all(1)]
    if len(rows) < 30 or min(Y[rows].sum(), (1 - Y[rows]).sum()) < 8: continue
    X = X.reshape(len(allm), -1)
    Xz = zgroup(X, allm.grp.values) if kind == "low" else np.nan_to_num(X)
    s = cv(Xz, rows, Y, kind); a = auc_ci(Y[rows], s)
    rng = np.random.default_rng(0)
    nl = []
    for sd in range(4):
        y2 = Y.copy(); y2[rows] = rng.permutation(Y[rows]); nl.append(roc_auc_score(y2[rows], cv(Xz, rows, y2, kind, reps=2, seed=40 + sd)))
    Xr = Xz.astype(np.float64).copy(); Xr[rows] -= LinearRegression().fit(L3[rows], Xr[rows]).predict(L3[rows])
    sr = cv(Xr, rows, Y, kind); ar = roc_auc_score(Y[rows], sr)
    b2 = bins2d(rows); w = within_bin_auc(Y[rows], s, b2)
    wn = []
    for sd in range(6):
        y2 = Y.copy()
        for bb in np.unique(b2):
            idx = rows[b2 == bb]; y2[idx] = rng.permutation(Y[idx])
        wn.append(within_bin_auc(y2[rows], cv(Xz, rows, y2, kind, reps=2, seed=70 + sd), b2))
    wn = np.array(wn); z = (w - wn.mean()) / (wn.std() + 1e-9)
    res.append(dict(family=nm, n=len(rows), auc=a[0], lo=a[1], hi=a[2], null=np.mean(nl), auc_len_removed=ar, within2d=w, within_null=wn.mean(), z=z))
    if len(rows) == len(rows_all): oof_store[nm] = (s, sr)
    print(f"  {nm:44s} n={len(rows):2d}  AUC {a[0]:.2f} [{a[1]:.2f},{a[2]:.2f}] null {np.mean(nl):.2f} | lengths removed {ar:.2f} | 2-D within-bin {w:.2f} (null {wn.mean():.2f}, z {z:+.1f})", flush=True)

R = pd.DataFrame(res).sort_values("within2d", ascending=False); R.to_csv(A / "exp23_genuine_organizer.csv", index=False)
# equal-weight ensembles (no label use beyond each family's own OOF scores)
zs = lambda v: (v - v.mean()) / (v.std() + 1e-9)
ctrl = [k for k in oof_store if k.startswith("ctrl:")]
keys = [k for k in oof_store if k not in ctrl]
ens = np.mean([zs(oof_store[k][0]) for k in keys], 0); ens_r = np.mean([zs(oof_store[k][1]) for k in keys], 0)
b2 = bins2d(rows_all)
print(f"\n  ENSEMBLE (mean of {len(keys)} families' OOF z-scores)  AUC {auc_ci(Y[rows_all], ens)[0]:.2f} [{auc_ci(Y[rows_all], ens)[1]:.2f},{auc_ci(Y[rows_all], ens)[2]:.2f}] | "
      f"from length-removed families {roc_auc_score(Y[rows_all], ens_r):.2f} | 2-D within-bin {within_bin_auc(Y[rows_all], ens, b2):.2f}")
print("\nTop 10 by 2-D within-bin AUC:"); print(R.head(10)[["family", "n", "auc", "auc_len_removed", "within2d", "within_null", "z"]].round(2).to_string(index=False))
print(f"\n({len(R)} families tested; with this many, a couple of |z| > 2 are expected by chance)")
