"""TOHA analysis. Per head: minimum-spanning-forest length connecting response tokens to the prompt (1 - attention distances).
'win' = last 128 trace tokens before the cutoff (fixed size -> length-free), 'ans' = forced suffix + forced answer.
Fixed-budget design: only traces still thinking at b, features at exactly b. agg and main_v2 separately; no proxies."""
from common import *
from sklearn.decomposition import PCA
from scipy.stats import rankdata
P2 = ROOT / "extraction_out_generations_p2" / "extraction_out_generations"
allm, _ = load_all()
ds = datasets(allm); y = allm.y.values
L3 = np.log(allm[["n_prompt", "n_gen", "L_think"]].astype(float).values + 1)
BUD = [1024, 2048, 4096]
Lthink = np.array([force_json(r.model, r.id)["L"] for r in allm.itertuples()])
T = {}
for i, r in enumerate(allm.itertuples()):
    p = P2 / r.model / "toha" / f"{r.id}.npz"
    if p.exists(): T[i] = np.load(p)
print(f"TOHA files matched: {len(T)} / {len(allm)}")


def at_budget(i, b, kind):
    z = T.get(i)
    if z is None or Lthink[i] <= b: return None
    ce = list(z["cut_eff"])
    if b not in ce: return None
    k = ce.index(b)
    return z["win"][k] if kind == "win" else z["ans"][k, 0]          # [L, H, 5]


def binned(a):                                                        # [L,H,5] -> model-aligned [6 depth bins x 5 stats x (mean, std over heads)]
    parts = np.array_split(np.arange(a.shape[0]), 6)
    return np.concatenate([np.concatenate([a[p].mean((0, 1)), a[p].std((0, 1))]) for p in parts])


def cv_generic(fitpred, X, rows, yy, groups, reps=8, seed=0):
    s, c = np.zeros(len(rows)), np.zeros(len(rows))
    for r in range(reps):
        for tr, te in folds(rows, yy, groups, seed + r):
            if len(np.unique(yy[rows[tr]])) < 2: continue
            s[te] += fitpred(X[rows[tr]], yy[rows[tr]], X[rows[te]]); c[te] += 1
    return s / np.maximum(c, 1)


def fp_lr(C=0.05):
    return lambda Xtr, ytr, Xte: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(Xtr, ytr).decision_function(Xte)


def fp_pca(k=8):
    return lambda Xtr, ytr, Xte: make_pipeline(StandardScaler(), PCA(min(k, len(Xtr) - 2), random_state=0), StandardScaler(),
                                               LogisticRegression(C=0.3, max_iter=3000)).fit(Xtr, ytr).decision_function(Xte)


def fp_heads(K=10):
    """TOHA-style: pick the K heads whose score best separates the classes ON THE TRAINING FOLD, average their signed z-scores."""
    def f(Xtr, ytr, Xte):
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
        n1 = ytr.sum(); n0 = len(ytr) - n1
        auc = (rankdata(Ztr, axis=0)[ytr == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
        top = np.argsort(-np.abs(auc - .5))[:K]
        return (Zte[:, top] * np.sign(auc[top] - .5)).mean(1)
    return f


def evaluate_set(name, X, rows, groups, fitpred, ds_name, do_resid=True, nnull=4):
    a = auc_ci(y[rows], cv_generic(fitpred, X, rows, y, groups))
    nl = np.mean([roc_auc_score(shuffled_y(allm, rows, ds_name, i)[rows], cv_generic(fitpred, X, rows, shuffled_y(allm, rows, ds_name, i), groups, reps=2)) for i in range(nnull)])
    out = f"  {name:44s} n={len(rows):3d} AUC {a[0]:.2f} [{a[1]:.2f},{a[2]:.2f}] null {nl:.2f}"
    if do_resid:
        Xr = X.astype(np.float64).copy(); Xr[rows] = Xr[rows] - LinearRegression().fit(L3[rows], Xr[rows]).predict(L3[rows])
        ar = auc_ci(y[rows], cv_generic(fitpred, Xr, rows, y, groups))
        ln = roc_auc_score(y[rows], cv_generic(fp_lr(), L3, rows, y, groups))
        out += f" | lengths removed {ar[0]:.2f} [{ar[1]:.2f},{ar[2]:.2f}] | length-only {ln:.2f}"
    print(out, flush=True)


ids = np.arange(len(allm)); pb = np.unique(allm.base_id.values, return_inverse=True)[1]
print("=== TOHA, agg (DeepSeek: 36 layers x 32 heads) ===")
for b in BUD:
    print(f"--- budget {b}: traces still thinking at {b} tokens ---")
    for kind in ("win", "ans"):
        rows = np.array([i for i in ds["agg"] if at_budget(i, b, kind) is not None])
        if len(rows) < 20: continue
        A_ = np.zeros((len(allm),) + at_budget(rows[0], b, kind).shape, np.float32)
        for i in rows: A_[i] = at_budget(i, b, kind)
        tot = A_[..., 0].reshape(len(allm), -1); allst = A_.reshape(len(allm), -1); bn = np.stack([binned(A_[i]) for i in range(len(allm))])
        evaluate_set(f"{kind}: total length, all heads (PCA-8)", tot, rows, ids, fp_pca(), "agg")
        for K in (5, 10, 20):
            evaluate_set(f"{kind}: total length, top-{K} heads chosen in-fold", tot, rows, ids, fp_heads(K), "agg", do_resid=(K == 10))
        evaluate_set(f"{kind}: all 5 stats, all heads (PCA-8)", allst, rows, ids, fp_pca(), "agg")
        evaluate_set(f"{kind}: depth-binned summaries (60)", bn, rows, ids, fp_lr(), "agg")

print("\n=== TOHA, main_v2 (4 models pooled, depth-binned summaries z-scored within model, grouped by problem) ===")
for b in BUD:
    for kind in ("win", "ans"):
        rows = np.array([i for i in ds["main_v2"] if at_budget(i, b, kind) is not None])
        if len(rows) < 12 or len(np.unique(y[rows])) < 2: continue
        bn = np.zeros((len(allm), 60), np.float32)
        for i in rows: bn[i] = binned(at_budget(i, b, kind))
        bn = zgroup(np.where(np.isin(np.arange(len(allm)), rows)[:, None], bn, np.nan), allm.grp.values)
        evaluate_set(f"budget {b}, {kind}: depth-binned summaries", bn, rows, pb, fp_lr(0.03), "main_v2")
        tot_b = np.zeros((len(allm), 6), np.float32)
        for i in rows: a = at_budget(i, b, kind); tot_b[i] = [a[p][..., 0].mean() for p in np.array_split(np.arange(a.shape[0]), 6)]
        tot_b = zgroup(np.where(np.isin(np.arange(len(allm)), rows)[:, None], tot_b, np.nan), allm.grp.values)
        evaluate_set(f"budget {b}, {kind}: mean total length per depth bin (6)", tot_b, rows, pb, fp_lr(0.1), "main_v2")

# length-matched pairs on agg (prompt + trace length), budget 1024 'win' and 'ans'
print("\n=== agg, length-matched pairs (prompt + trace length), budget 1024 ===")
agg = ds["agg"]; Ls = (L3[:, :2] - L3[agg, :2].mean(0)) / L3[agg, :2].std(0)
elig = np.array([i for i in agg if at_budget(i, 1024, "win") is not None])
rob = [i for i in elig if y[i] == 1]; non = [i for i in elig if y[i] == 0]; used, pairs = set(), []
for i in sorted(rob, key=lambda i: Ls[i, 1]):
    cand = [j for j in non if j not in used]
    if not cand: break
    j = cand[int(np.argmin([np.sum((Ls[i] - Ls[j]) ** 2) for j in cand]))]; used.add(j); pairs.append((i, j))
pi, pj = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]); mrows = np.concatenate([pi, pj])
gp = np.zeros(len(allm), int); gp[pi] = np.arange(len(pairs)); gp[pj] = np.arange(len(pairs))
for kind in ("win", "ans"):
    A_ = np.zeros((len(allm), 36, 32, 5), np.float32)
    for i in mrows: A_[i] = at_budget(i, 1024, kind)
    for nm, X, fpp in ((f"{kind}: total length, all heads (PCA-8)", A_[..., 0].reshape(len(allm), -1), fp_pca()),
                       (f"{kind}: top-10 heads chosen in-fold", A_[..., 0].reshape(len(allm), -1), fp_heads(10)),
                       (f"{kind}: depth-binned summaries", np.stack([binned(A_[i]) for i in range(len(allm))]), fp_lr())):
        a = auc_ci(y[mrows], cv_generic(fpp, X, mrows, y, gp))
        nl = []
        for s_ in range(4):
            yy = y.copy(); flip = np.random.default_rng(s_).random(len(pairs)) < .5; yy[pi[flip]], yy[pj[flip]] = 0, 1
            nl.append(roc_auc_score(yy[mrows], cv_generic(fpp, X, mrows, yy, gp, reps=2)))
        print(f"  {nm:44s} pairs={len(pairs)} AUC {a[0]:.2f} [{a[1]:.2f},{a[2]:.2f}] within-pair null {np.mean(nl):.2f}", flush=True)
