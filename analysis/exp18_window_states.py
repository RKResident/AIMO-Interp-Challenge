"""Exact consecutive-token window means (16-token means saved by --save-window-states --stride 16, 6 layers) -> moving averages over
16 / 64 / 256 consecutive tokens (k = 1 / 4 / 16 windows, evaluated every 16 tokens). Trajectory geometry + probes.
Length-free design: first b tokens of traces still thinking at b. agg and main_v2 separately; no proxies."""
from common import *
from sklearn.decomposition import PCA
P2 = ROOT / "extraction_out_generations_p2" / "extraction_out_generations"
allm, _ = load_all()
ds = datasets(allm); y = allm.y.values
L3 = np.log(allm[["n_prompt", "n_gen", "L_think"]].astype(float).values + 1)
Lthink = np.array([force_json(r.model, r.id)["L"] for r in allm.itertuples()])
KS = [1, 4, 16]; BUD = [2048, 4096, "full"]; W = 16


def geom(H):
    Wn = len(H)
    if Wn < 6: return None
    o = {}; nh = np.linalg.norm(H, axis=1).mean() + 1e-9
    D = np.diff(H, axis=0); sp = np.linalg.norm(D, axis=1)
    o["speed"] = sp.mean() / nh; o["speed_cv"] = sp.std() / (sp.mean() + 1e-9)
    o["speed_slope"] = np.polyfit(np.linspace(0, 1, len(sp)), sp / nh, 1)[0]
    o["turn_cos"] = ((D[1:] * D[:-1]).sum(1) / (sp[1:] * sp[:-1] + 1e-9)).mean()
    o["tortuosity"] = sp.sum() / (np.linalg.norm(H[-1] - H[0]) + 1e-9)
    c = H.mean(0); o["gyration"] = np.linalg.norm(H - c, axis=1).mean() / nh
    o["drift_cos"] = float(H[0] @ H[-1] / (np.linalg.norm(H[0]) * np.linalg.norm(H[-1]) + 1e-9))
    Hc = H - c; ev = np.clip(np.linalg.eigvalsh(Hc @ Hc.T), 0, None); o["eff_dim"] = ev.sum() ** 2 / (np.sum(ev ** 2) + 1e-12) / Wn
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-9); S = Hn @ Hn.T
    gap = max(4, Wn // 16); iu = np.triu_indices(Wn, k=gap); thr = np.quantile(S[np.triu_indices(Wn, k=1)], 0.9)
    o["recurrence"] = float((S[iu] > thr).mean())
    o["late_vs_early_cos"] = float(Hn[: Wn // 4].mean(0) @ Hn[-Wn // 4:].mean(0))
    return o


def smooth(H, k):
    if k == 1: return H
    cs = np.cumsum(np.vstack([np.zeros((1, H.shape[1]), np.float32), H]), 0); return (cs[k:] - cs[:-k]) / k


recs = [dict() for _ in range(len(allm))]
probe = {}                                   # (layer slot, b) -> {row: mean state over first b tokens}   (DeepSeek only, for agg)
for i, r in enumerate(allm.itertuples()):
    p = P2 / r.model / "tf" / f"{r.id}.npz"
    if not p.exists(): continue
    ws = np.load(p)["win_states"].astype(np.float32)          # [n_win, 6, d]
    for li in range(ws.shape[1]):
        H0 = ws[:, li]
        for k in KS:
            Hs = smooth(H0, k)
            for b in BUD:
                if b != "full":
                    if Lthink[i] <= b: continue
                    Hb = Hs[: b // W - (k - 1)]
                else:
                    Hb = Hs
                g = geom(Hb)
                if g is None: continue
                for nm, v in g.items(): recs[i][f"k{k}|{b}|L{li}|{nm}"] = v
        if r.model == DS and li in (2, 3, 4):
            for b in (2048, 4096):
                if Lthink[i] > b:
                    probe.setdefault((li, b, "mean"), {})[i] = H0[: b // W].mean(0)
                    probe.setdefault((li, b, "last256"), {})[i] = H0[b // W - 16: b // W].mean(0)
    if (i + 1) % 60 == 0: print("processed", i + 1, "/", len(allm), flush=True)
R = pd.DataFrame(recs); R.to_pickle(A / "exp18_features.pkl")
print("=== exact consecutive-token moving averages: trajectory geometry (6 layers x 10 features) ===")
for b in BUD:
    for k in KS:
        cols = [c for c in R.columns if c.startswith(f"k{k}|{b}|")]
        if not cols: continue
        X = R[cols].values.astype(float); have = np.isfinite(X).all(1)
        rs = {nm: np.intersect1d(v, np.where(have)[0]) for nm, v in ds.items()}
        evaluate(f"{k * W}-token windows, {'first ' + str(b) if b != 'full' else 'full trace'}", X, allm, C=0.03, rowsets=rs)

print("\n=== probes on the state itself (agg, DeepSeek; PCA-8 + logistic) ===")
agg = ds["agg"]; ids = np.arange(len(allm))
for (li, b, kind), dct in sorted(probe.items()):
    rows = np.array([i for i in agg if i in dct])
    X = np.zeros((len(allm), len(next(iter(dct.values())))), np.float32)
    for i in rows: X[i] = dct[i]

    def cvp(XX, yy, reps=8):
        s, c = np.zeros(len(rows)), np.zeros(len(rows))
        for rr in range(reps):
            for tr, te in folds(rows, yy, ids, rr):
                m = make_pipeline(StandardScaler(), PCA(8, random_state=0), StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)).fit(XX[rows[tr]], yy[rows[tr]])
                s[te] += m.decision_function(XX[rows[te]]); c[te] += 1
        return s / c
    a = auc_ci(y[rows], cvp(X, y))
    Xr = X.astype(np.float64).copy(); Xr[rows] -= LinearRegression().fit(L3[rows], Xr[rows]).predict(L3[rows])
    ar = auc_ci(y[rows], cvp(Xr, y))
    nl = np.mean([roc_auc_score(shuffled_y(allm, rows, "agg", s_)[rows], cvp(X, shuffled_y(allm, rows, "agg", s_), reps=2)) for s_ in range(3)])
    print(f"  layer slot {li}, {kind:7s} state over first {b}: n={len(rows)} AUC {a[0]:.2f} [{a[1]:.2f},{a[2]:.2f}] null {nl:.2f} | lengths removed {ar[0]:.2f} [{ar[1]:.2f},{ar[2]:.2f}]", flush=True)
