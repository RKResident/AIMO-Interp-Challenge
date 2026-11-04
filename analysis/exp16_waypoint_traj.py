"""Hidden-state trajectories from saved_trajectories_{aggregate,v2}_data (state of every 16th generated token, 6 layers).
'Moving average over consecutive tokens' is approximated by averaging k consecutive waypoints (k=4 -> 64-token span, k=16 -> 256).
Geometry features per layer: speed, turning, tortuosity, radius of gyration, drift, effective dimension, recurrence.
Length control by design: fixed budgets (first 2048 / 4096 tokens, only traces that ran at least that long), plus the usual
length control / length-removed numbers. agg (old DeepSeek traces, 137) and main_v2 (34 non-proxy rows, 4 models) evaluated separately."""
import glob
from common import *
from sklearn.decomposition import PCA
ROOTS = {"agg": ROOT / "saved_trajectories_aggregate_data", "main_v2": ROOT / "saved_trajectories_v2_data"}
KS = [1, 4, 16]; BUDS = [2048, 4096, "full"]; STRIDE = 16
recs, metas, probe = [], [], {}
plen = {}
for src, d in ROOTS.items():
    for g in glob.glob(str(d / "*" / "generations.jsonl")):
        for l in open(g):
            r = json.loads(l); plen[r["id"]] = len(r["prompt_ids"])
files = []
for src, d in ROOTS.items():
    for f in sorted(glob.glob(str(d / "*" / "*.npz"))):
        if "gpt_proxy" in f: continue
        files.append((src, f))


def geom(H):
    """H [W, d] float32 trajectory -> dict of scale-free geometry features."""
    W = len(H); out = {}
    if W < 6: return None
    nh = np.linalg.norm(H, axis=1).mean() + 1e-9
    D = np.diff(H, axis=0); sp = np.linalg.norm(D, axis=1)
    out["speed"] = sp.mean() / nh; out["speed_cv"] = sp.std() / (sp.mean() + 1e-9)
    out["speed_slope"] = np.polyfit(np.linspace(0, 1, len(sp)), sp / nh, 1)[0]
    cd = (D[1:] * D[:-1]).sum(1) / (sp[1:] * sp[:-1] + 1e-9); out["turn_cos"] = cd.mean()
    out["tortuosity"] = sp.sum() / (np.linalg.norm(H[-1] - H[0]) + 1e-9)
    c = H.mean(0); out["gyration"] = np.linalg.norm(H - c, axis=1).mean() / nh
    out["drift_cos"] = float(H[0] @ H[-1] / (np.linalg.norm(H[0]) * np.linalg.norm(H[-1]) + 1e-9))
    Hc = H - c; G = Hc @ Hc.T; ev = np.clip(np.linalg.eigvalsh(G), 0, None)
    out["eff_dim"] = ev.sum() ** 2 / (np.sum(ev ** 2) + 1e-12) / W
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-9); S = Hn @ Hn.T
    gap = max(4, W // 16); iu = np.triu_indices(W, k=gap)
    thr = np.quantile(S[np.triu_indices(W, k=1)], 0.9)
    out["recurrence"] = float((S[iu] > thr).mean())
    out["late_vs_early_cos"] = float(Hn[: W // 4].mean(0) @ Hn[-W // 4:].mean(0))
    return out


def smooth(H, k):
    if k == 1: return H
    cs = np.cumsum(np.vstack([np.zeros((1, H.shape[1]), np.float32), H]), 0)
    return (cs[k:] - cs[:-k]) / k


for n, (src, f) in enumerate(files):
    z = np.load(f); mdl = Path(f).parent.name; pid = Path(f).stem
    keys = sorted([k for k in z.files if k.startswith("layer_")], key=lambda s: int(s.split("_")[1]))
    ntok = int(z["tokens_generated"]); y_ = float(z["is_robust"])
    base = pid if src == "agg" else pid.split("_")[-1][:8]
    metas.append(dict(model=mdl, id=pid, source=src, y=y_, base_id=base, n_gen=ntok, L_think=ntok, n_prompt=plen.get(pid, np.nan), stop=str(z["stop_reason"])))
    rec = {}
    for li, key in enumerate(keys):
        H0 = z[key].astype(np.float32)[1:]           # drop the step-1 waypoint so rows are every 16th token
        for k in KS:
            Hs = smooth(H0, k)
            for b in BUDS:
                if b != "full":
                    if ntok < b + 16: continue
                    Hb = Hs[: b // STRIDE - (k - 1)]
                else:
                    Hb = Hs
                g = geom(Hb)
                if g is None: continue
                for nm, v in g.items(): rec[f"k{k}|{b}|L{li}|{nm}"] = v
        if li in (2, 4):                               # mean state over the first 2048 / 4096 tokens for a probe (per-model space)
            for b in (2048, 4096):
                if ntok >= b + 16: probe.setdefault((li, b), {})[n] = H0[: b // STRIDE].mean(0)
    recs.append(rec)
    if (n + 1) % 30 == 0: print("processed", n + 1, "/", len(files), flush=True)
om = pd.DataFrame(metas); om["grp"] = om.model + om.source
om.loc[om.source == "agg", "model"] = DS                # common.datasets() keys agg on the DeepSeek model name
om["n_prompt"] = om.n_prompt.fillna(om.groupby("grp").n_prompt.transform("median")).fillna(100)
R = pd.DataFrame(recs)
pickle.dump({"meta": om, "R": R, "probe": probe}, open(A / "exp16_features.pkl", "wb"))
print("=== Waypoint-trajectory geometry ===")
print(om.groupby(["source", "model"]).agg(n=("y", "size"), robust=("y", "sum"), capped=("stop", lambda s: (s == "cap").sum())).to_string())
for b in BUDS:
    for k in KS:
        cols = [c for c in R.columns if c.startswith(f"k{k}|{b}|")]
        if not cols: continue
        X = R[cols].values.astype(float)
        have = np.isfinite(X).all(1)
        ds = datasets(om); rs = {nm: np.intersect1d(v, np.where(have)[0]) for nm, v in ds.items()}
        span = "single tokens" if k == 1 else f"{k * STRIDE}-token moving average"
        evaluate(f"geometry, {span}, {'first ' + str(b) + ' tok' if b != 'full' else 'full trace'}", X, om, C=0.03, rowsets=rs)
# probe on the mean state over a fixed prefix (agg only: one model, one representation space)
print("\n=== probe on mean hidden state over a fixed prefix (agg, PCA-8 + logistic) ===")
agg = datasets(om)["agg"]
for (li, b), dct in sorted(probe.items()):
    rows = np.array([i for i in agg if i in dct])
    if len(rows) < 30: continue
    X = np.zeros((len(om), len(next(iter(dct.values())))), np.float32)
    for i in rows: X[i] = dct[i]
    s, c = np.zeros(len(rows)), np.zeros(len(rows))
    for r in range(8):
        for tr, te in folds(rows, om.y.values, np.arange(len(om)), r):
            m = make_pipeline(StandardScaler(), PCA(8, random_state=0), StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)).fit(X[rows[tr]], om.y.values[rows[tr]])
            s[te] += m.decision_function(X[rows[te]]); c[te] += 1
    a = auc_ci(om.y.values[rows], s / c)
    print(f"  layer slot {li}, first {b} tokens: n={len(rows)} (robust {int(om.y.values[rows].sum())})  AUC {a[0]:.2f} [{a[1]:.2f},{a[2]:.2f}]")
