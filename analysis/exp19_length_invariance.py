"""Length invariance by construction + a uniform re-score of the best families.
Part A (window states, thinking part only): random fixed-size subsamples (order-free 'bag' and sorted), random contiguous 1024-token segments,
relative-position segments. Part B: every family scored with (i) plain OOF AUC, (ii) WITHIN-LENGTH-BIN AUC, (iii) a length-preserving null
(labels shuffled only within length bins). agg and main_v2 separately; main_v2 grouped by problem; no proxies."""
from common import *
from sklearn.decomposition import PCA
P2 = ROOT / "extraction_out_generations_p2" / "extraction_out_generations"
allm, feats = load_all()
ds = datasets(allm); y = allm.y.values; ids = np.arange(len(allm)); pb = np.unique(allm.base_id.values, return_inverse=True)[1]
Lthink = np.array([force_json(r.model, r.id)["L"] for r in allm.itertuples()])
M, DRAWS, SEG = 64, 20, 64        # 64 windows of 16 tokens = 1024 tokens
rng = np.random.default_rng(0)


def bagfeat(H):                    # order-free distribution features of a set of states [m, d]
    nh = np.linalg.norm(H, axis=1); c = H.mean(0); Hc = H - c
    ev = np.clip(np.linalg.eigvalsh(Hc @ Hc.T), 0, None); Hn = H / (nh[:, None] + 1e-9); S = Hn @ Hn.T; iu = np.triu_indices(len(H), 1)
    d = np.linalg.norm(H[:, None] - H[None], axis=-1); np.fill_diagonal(d, np.inf)
    return [np.linalg.norm(Hc, axis=1).mean() / nh.mean(), ev.sum() ** 2 / (np.sum(ev ** 2) + 1e-12) / len(H), S[iu].mean(), S[iu].std(),
            nh.std() / nh.mean(), d.min(1).mean() / nh.mean(), ev[-1] / (ev.sum() + 1e-12)]


def trajfeat(H):
    nh = np.linalg.norm(H, axis=1).mean() + 1e-9; D = np.diff(H, axis=0); sp = np.linalg.norm(D, axis=1)
    return [sp.mean() / nh, sp.std() / (sp.mean() + 1e-9), ((D[1:] * D[:-1]).sum(1) / (sp[1:] * sp[:-1] + 1e-9)).mean(),
            sp.sum() / (np.linalg.norm(H[-1] - H[0]) + 1e-9), float(H[0] @ H[-1] / (np.linalg.norm(H[0]) * np.linalg.norm(H[-1]) + 1e-9))]


import os
CACHE = A / "exp19_cache.pkl"
feat = {k: [None] * len(allm) for k in ("bag", "sorted", "segment", "seg25", "seg50", "seg75")}
probe = {k: [None] * len(allm) for k in ("bag_mean", "segment_mean")}
for i, r in enumerate(allm.itertuples() if not CACHE.exists() else []):
    p = P2 / r.model / "tf" / f"{r.id}.npz"
    if not p.exists(): continue
    ws = np.load(p)["win_states"].astype(np.float32)
    nw = min(len(ws), max(1, Lthink[i] // 16))                       # thinking part only
    if nw < M: continue
    T = ws[:nw]
    fb, fs, fg, pm, sm = [], [], [], [], []
    for _ in range(DRAWS):
        idx = rng.choice(nw, M, replace=False)
        fb.append(np.concatenate([bagfeat(T[idx, l]) for l in range(T.shape[1])]))
        fs.append(np.concatenate([trajfeat(T[np.sort(idx), l]) for l in range(T.shape[1])]))
        st = rng.integers(0, nw - SEG + 1)
        fg.append(np.concatenate([trajfeat(T[st:st + SEG, l]) + bagfeat(T[st:st + SEG, l]) for l in range(T.shape[1])]))
        pm.append(T[idx, 3].mean(0)); sm.append(T[st:st + SEG, 3].mean(0))
    feat["bag"][i], feat["sorted"][i], feat["segment"][i] = np.mean(fb, 0), np.mean(fs, 0), np.mean(fg, 0)
    if r.model == DS:  # probes live in one model's representation space
        probe["bag_mean"][i], probe["segment_mean"][i] = np.mean(pm, 0), np.mean(sm, 0)
    for q, nm in ((.25, "seg25"), (.5, "seg50"), (.75, "seg75")):
        st = int(np.clip(q * nw - SEG / 2, 0, nw - SEG))
        feat[nm][i] = np.concatenate([trajfeat(T[st:st + SEG, l]) + bagfeat(T[st:st + SEG, l]) for l in range(T.shape[1])])
    if (i + 1) % 60 == 0: print("processed", i + 1, "/", len(allm), flush=True)


if CACHE.exists():
    feat, probe = pickle.load(open(CACHE, "rb"))
else:
    pickle.dump((feat, probe), open(CACHE, "wb"))


def as_matrix(lst):
    d = next(len(x) for x in lst if x is not None)
    X = np.full((len(allm), d), np.nan, np.float32)
    for i, x in enumerate(lst):
        if x is not None: X[i] = x
    return X


def cv(X, rows, yy, groups, kind, reps=8, seed=0):
    s, c = np.zeros(len(rows)), np.zeros(len(rows))
    for r in range(reps):
        for tr, te in folds(rows, yy, groups, seed + r):
            if len(np.unique(yy[rows[tr]])) < 2: continue
            mdl = make_pipeline(StandardScaler(), PCA(8, random_state=0), StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)) if kind == "high" \
                else make_pipeline(StandardScaler(), LogisticRegression(C=0.03, max_iter=3000))
            s[te] += mdl.fit(X[rows[tr]], yy[rows[tr]]).decision_function(X[rows[te]]); c[te] += 1
    return s / np.maximum(c, 1)


def score(name, X, dsn, kind="low", restrict=None):
    rows = ds[dsn] if restrict is None else np.intersect1d(ds[dsn], restrict)
    Xz = zgroup(X, allm.grp.values) if kind == "low" else np.nan_to_num(X)
    groups = ids if dsn == "agg" else pb
    if len(rows) < 12 or len(np.unique(y[rows])) < 2: return
    s = cv(Xz, rows, y, groups, kind); bins = length_bins(allm, rows)
    a = roc_auc_score(y[rows], s); w = within_bin_auc(y[rows], s, bins)
    nl, nw = [], []
    for sd in range(5):
        y2 = shuffled_y_len(allm, rows, dsn, sd); s2 = cv(Xz, rows, y2, groups, kind, reps=2, seed=40 + sd)
        nl.append(roc_auc_score(y2[rows], s2)); nw.append(within_bin_auc(y2[rows], s2, bins))
    sl = cv(lengths(allm), rows, y, groups, "low")
    print(f"  {name:46s} {dsn:7s} n={len(rows):3d}  AUC {a:.2f} (length-preserving null {np.mean(nl):.2f}, length-only {roc_auc_score(y[rows], sl):.2f})"
          f"  | WITHIN-BIN AUC {w:.2f} (null {np.mean(nw):.2f} +- {np.std(nw):.2f}; length-only {within_bin_auc(y[rows], sl, bins):.2f})", flush=True)


SKIP_A = os.environ.get("SKIP_A") == "1"
print("\n=== Part A: subsampling / fixed-size segments of the thinking (window states, 1024 tokens = 64 x 16-token means) ===")
elig = np.array([i for i in range(len(allm)) if feat["bag"][i] is not None])
for nm, desc in (("bag", "random 64 windows, order-free (x20 draws)"), ("sorted", "random 64 windows, in order (x20)"),
                 ("segment", "random contiguous 1024-token segment (x20)"), ("seg25", "segment at 25% of thinking"),
                 ("seg50", "segment at 50% of thinking"), ("seg75", "segment at 75% of thinking")):
    for dsn in ("agg", "main_v2"):
        if not SKIP_A: score(desc, as_matrix(feat[nm]), dsn, "low", elig)
for nm in ("bag_mean", "segment_mean"):
    score(f"probe: mean state of {nm.replace('_mean', '')} (layer slot 3)", as_matrix(probe[nm]), "agg", "high", elig)

print("\n=== Part B: best families re-scored with within-length-bin AUC and a length-preserving null ===")
D4 = pd.read_pickle(A / "exp4_features.pkl"); S8 = pd.read_pickle(A / "exp8_features.pkl"); T3 = pd.read_pickle(A / "exp3_features.pkl")
n_ds = len(feats[DS]["prompt_last"])
PL = np.zeros((len(allm), 4096), np.float32); PL[:n_ds] = feats[DS]["prompt_last"][:, 30]
fcols = [c for c in allm.columns if c.startswith(("fa_", "sc_", "agree_")) and allm[c].dtype != object and allm[c].notna().mean() > .8]
tcols = [c for c in allm.columns if c.startswith(("lp_", "ent_", "top1_"))]
LP = np.zeros((len(allm), 36 * 32), np.float32); LP[:n_ds] = feats[DS]["lap_force"][:, 3].mean(1)[..., 0].reshape(n_ds, -1)
# TOHA reasoning window at 2048 (DeepSeek, all heads)
TW = np.full((len(allm), 36 * 32), np.nan, np.float32)
for i in ds["agg"]:
    z = np.load(P2 / allm.model.values[i] / "toha" / f"{allm.id.values[i]}.npz"); ce = list(z["cut_eff"])
    if 2048 in ce and Lthink[i] > 2048: TW[i] = z["win"][ce.index(2048)][..., 0].reshape(-1)
fams = [("prompt probe, last prompt token layer 30", PL, "high", ["agg"]),
        ("anchoring cosine (6 layers)", D4[[f"h{j}_cos_prob_mean" for j in range(6)]].values, "low", ["agg", "main_v2"]),
        ("forced-answer confidence + consistency", allm[fcols].astype(float).values, "low", ["agg", "main_v2"]),
        ("trace token statistics", allm[tcols].astype(float).values, "low", ["agg", "main_v2"]),
        ("Laplacian top-1 eigenvalue, forced, end of thinking", LP, "high", ["agg"]),
        ("text markers", T3[[c for c in T3 if c.startswith("t_") and c != "t_words"]].values, "low", ["agg", "main_v2"]),
        ("A/B suffix sensitivity", S8[[c for c in S8 if c.endswith(("_mean", "_max", "_bmean"))]].values, "low", ["agg", "main_v2"])]
for nm, X, kind, dl in fams:
    for dsn in dl: score(nm, X, dsn, kind)
rows2048 = np.where(np.isfinite(TW).all(1))[0]
score("TOHA reasoning window @2048, all heads", np.nan_to_num(TW), "agg", "high", rows2048)
