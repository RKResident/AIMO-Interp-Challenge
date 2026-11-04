"""exp60: DeepSeek's trajectory spikes (exp59: 88% of each large excursion lies in ONE hidden dimension - a massive-activation channel).
DeepSeek rows: main_v2 10 + olympiad 43 (47 unfinished at 6144); the main_v2 rows also have two T=1 samples (stability check).
1 find the massive channels: per saved layer, the hidden dimensions that carry the largest deviation of the 16-token states, tallied
  over all DeepSeek traces (label-free)
2 spike model: from the top channel's time series at each layer (16-token windows): spike = |value - trace median| > 6 x trace MAD.
  Features: spike rate, rate in the second half minus the first, mean / max spike size (in MADs), mean gap between spikes, share of
  spikes in the last quarter, mean channel value; plus the raw 64-token jump statistics (share of steps > 3x the median jump, largest jump)
3 smoothed model: the same states with the top massive channels (per layer) replaced by the trace median, then a 3-window median filter
  in time; the trajectory features recomputed with the submission's code (kNN global efficiency, TDA H0 spread, segment PC1, late-vs-early
  cosine, PC1 change) and speed variation / step similarity - against the same features on the raw states
Scores: AUC on all DeepSeek rows and on the unfinished ones, permutation p; leave-one-problem-out equal-weight ensembles of each family
(directions from the training fold); for the T=1 rows, Spearman of each feature between greedy and the samples."""
import json, os, re, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "2"
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, "/home/sabyasachi19/PycharmProjects/AimoAGAIN/analysis"); sys.path.insert(0, "/home/sabyasachi19/PycharmProjects/AimoAGAIN/submission_core")
from common import *                                                                    # noqa: E402,F401
from spurious_features import f_global_eff, f_tda_h0_spread, f_seg25_pc1, f_late_vs_early, f_pc1_change   # noqa: E402

M = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"; slug = lambda m: m.replace("/", "__")      # noqa: E731
D = pd.read_csv(A / "exp56_rows.csv"); D = D[D.model_id == M].reset_index(drop=True)
OLR = ROOT / "extra_olympiad_problems_dataset" / "results_olympiad_features" / "out_t1"; MV1 = ROOT / "results_mainv2_t1" / "out_t1"
mv = {json.loads(l)["id"]: json.loads(l) for f in MV1.glob("*/rows_shard*.jsonl") for l in open(f)}
ol = {json.loads(l)["id"]: json.loads(l) for f in OLR.glob("*/rows_shard*.jsonl") for l in open(f)}
MU = np.load(Path("/home/sabyasachi19/PycharmProjects/aimo/getting-started/solutions/spurious-cluster-graph/spurious_artifacts") / f"mu_{slug(M)}.npy")
norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()[:300]                            # noqa: E731
inp = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "t1_run" / "inputs.jsonl")}
D["group"] = [norm(inp[i]["problem"]) for i in D.id]; D["unfinished"] = ~D.finished.astype(bool)


def traces(r):
    z = np.load((MV1 if r.dataset == "main_v2" else OLR) / slug(M) / f"{r.id}.npz"); rec = (mv if r.dataset == "main_v2" else ol)[r.id]
    out = [("greedy", z["t0_ws16"].astype(np.float32), rec["greedy"]["L"])]
    for s in range(2):
        if f"t1s{s}_ws16" in z.files: out.append((f"sample {s + 1}", z[f"t1s{s}_ws16"].astype(np.float32), rec["samples"][s]["L"]))
    return out


# 1 massive channels per layer (label-free): dimension with the largest |x - trace median| per window, tallied
TR = {i: traces(r) for i, r in D.iterrows()}
L6 = TR[0][0][1].shape[1]; tally = [dict() for _ in range(L6)]
for i, T in TR.items():
    ws = T[0][1]; dev = np.abs(ws - np.median(ws, 0))                                       # [n, L, d]
    for l in range(L6):
        for d_ in np.argmax(dev[:, l], axis=1): tally[l][int(d_)] = tally[l].get(int(d_), 0) + 1
TOPDIMS = [sorted(t, key=lambda k: -t[k])[:3] for t in tally]
print("massive channels per saved layer (dimension: share of windows where it carries the largest deviation):")
for l, t in enumerate(tally):
    tot = sum(t.values()); print(f"  layer slot {l}: " + ", ".join(f"{d_}: {t[d_] / tot:.2f}" for d_ in TOPDIMS[l]))


def spike_feats(ws):
    out = {}
    for l in (1, 3):                                                                         # a shallow and a middle saved layer
        v = ws[:, l, TOPDIMS[l][0]]; med = np.median(v); mad = np.median(np.abs(v - med)) + 1e-6; z = np.abs(v - med) / mad
        sp = z > 6; n = len(v); idx = np.where(sp)[0]
        out[f"L{l} spike rate"] = sp.mean(); out[f"L{l} spike rate 2nd half - 1st"] = sp[n // 2:].mean() - sp[: n // 2].mean()
        out[f"L{l} mean spike size (MADs)"] = z[sp].mean() if sp.any() else 0.0; out[f"L{l} max spike size"] = z.max()
        out[f"L{l} mean gap between spikes"] = np.diff(idx).mean() / n if len(idx) > 1 else 1.0
        out[f"L{l} share of spikes in last quarter"] = (idx >= 3 * n // 4).mean() if len(idx) else np.nan
        out[f"L{l} channel mean (signed)"] = float(v.mean())
    X = ws[: len(ws) // 4 * 4, 3].reshape(-1, 4, ws.shape[-1]).mean(1); m = len(X); dj = np.zeros(m)
    for t in range(m):
        nb = [k for k in range(max(0, t - 4), min(m, t + 5)) if k != t]; dj[t] = np.linalg.norm(X[t] - np.median(X[nb], 0))
    dj /= np.median(dj) + 1e-9; out["raw jumps > 3x"] = (dj > 3).mean(); out["raw largest jump"] = dj.max()
    return out


def smooth(ws):
    w = ws.copy()
    for l in range(w.shape[1]):
        for d_ in TOPDIMS[l]: w[:, l, d_] = np.median(w[:, l, d_])
    p = np.pad(w, ((1, 1), (0, 0), (0, 0)), mode="edge")
    return np.median(np.stack([p[:-2], p[1:-1], p[2:]]), axis=0)


def traj_feats(ws, L, mu):
    S64 = ws[: len(ws) // 4 * 4, 3].reshape(-1, 4, ws.shape[-1]).mean(1); dS = np.linalg.norm(np.diff(S64, axis=0), axis=1)
    c = [(S64[k] * S64[k + 1]).sum() / (np.linalg.norm(S64[k]) * np.linalg.norm(S64[k + 1]) + 1e-9) for k in range(len(S64) - 1)]
    return {"kNN global efficiency": f_global_eff(ws[:, 3], mu, L), "TDA H0 spread": f_tda_h0_spread(ws[:, 3], L), "segment PC1 @25%": f_seg25_pc1(ws, L),
            "late-vs-early cosine": f_late_vs_early(ws), "PC1 change 75/25": f_pc1_change(ws), "speed variation (CV)": dS.std() / (dS.mean() + 1e-9),
            "step similarity": float(np.mean(c))}


mu_s = MU.copy(); mu_s[TOPDIMS[3]] = 0.0                                                  # the kNN centring vector without the massive channels
FE = {}                                                                                   # (family, feature) -> [rows x traces]
for i, T in TR.items():
    for k, (nm, ws, L) in enumerate(T):
        vals = {("spike", f): v for f, v in spike_feats(ws).items()}
        vals.update({("raw trajectory", f): v for f, v in traj_feats(ws, L, MU).items()})
        ws_s = smooth(ws); ws_s[:, 3, TOPDIMS[3]] = 0.0
        vals.update({("smoothed trajectory", f): v for f, v in traj_feats(ws_s, L, mu_s).items()})
        for key, v in vals.items(): FE.setdefault(key, np.full((len(D), 3), np.nan))[i, k] = v
y = D.robust.values.astype(int); unf = D.unfinished.values; grp = D.group.values; rng = np.random.default_rng(0)


def auc(x, yy):
    ok = np.isfinite(x); x, yy = x[ok], yy[ok]; p, n = x[yy == 1], x[yy == 0]
    return ((p[:, None] > n[None]).sum() + .5 * (p[:, None] == n[None]).sum()) / (len(p) * len(n)) if len(p) and len(n) else np.nan


def perm_p(x, yy, n=2000):
    a = auc(x, yy); nl = np.array([auc(x, rng.permutation(yy)) for _ in range(n)]); return a, float(np.mean(np.abs(nl - .5) >= abs(a - .5)))


print(f"\nDeepSeek rows {len(D)} ({int(y.sum())} robust); unfinished {int(unf.sum())} ({int(y[unf].sum())} robust)")
print(f"{'family':20s} {'feature':38s} {'AUC all (p)':>14s} {'AUC unfinished (p)':>20s} {'greedy~sample rho':>18s}")
res = []
for (fam, f), V in FE.items():
    g = V[:, 0]; a = perm_p(g, y); b = perm_p(g[unf], y[unf])
    ok = np.isfinite(V).all(1); rho = np.nanmean([spearmanr(V[ok, 0], V[ok, k])[0] for k in (1, 2)]) if ok.sum() >= 6 else np.nan
    res.append((fam, f, a, b, rho))
for fam, f, a, b, rho in sorted(res, key=lambda t: (t[0], -abs(t[3][0] - .5))):
    print(f"{fam:20s} {f:38s} {a[0]:6.2f} ({a[1]:.3f}) {b[0]:12.2f} ({b[1]:.3f}) {rho:14.2f}")


def loo(cols, mask):
    """Equal-weight percentile ensemble of `cols` (greedy values), directions from the training fold; held-out AUC / accuracy on `mask` rows."""
    sc, dec = np.full(len(D), np.nan), np.full(len(D), np.nan)
    X = np.stack([FE[c][:, 0] for c in cols], 1)
    for g in np.unique(grp[mask]):
        tr = (grp != g) & mask; te = np.where((grp == g) & mask)[0]
        sg = [1 if (auc(X[tr, j], y[tr]) >= .5 or np.isnan(auc(X[tr, j], y[tr]))) else -1 for j in range(len(cols))]
        def s(row):
            u = []
            for j in range(len(cols)):
                if not np.isfinite(row[j]): continue
                ref = X[tr, j][np.isfinite(X[tr, j])]; q = np.mean(ref < row[j]) + .5 * np.mean(ref == row[j])
                u.append(q if sg[j] > 0 else 1 - q)
            return np.mean(u) if u else .5
        thr = np.median([s(X[k]) for k in np.where(tr)[0]])
        for k in te: sc[k] = s(X[k]); dec[k] = sc[k] > thr
    m = mask & np.isfinite(sc); d = dec[m].astype(bool); yy = y[m]
    return auc(sc[m], yy), float(np.mean(d == yy)), .5 * (d[yy == 1].mean() + (1 - d[yy == 0]).mean())


print("\nleave-one-problem-out ensembles (DeepSeek only): AUC / accuracy / balanced")
fams = {"spike model (all spike features)": [k for k in FE if k[0] == "spike"],
        "spike model: rate + size + late share (L3)": [("spike", "L3 spike rate"), ("spike", "L3 mean spike size (MADs)"), ("spike", "L3 share of spikes in last quarter")],
        "raw trajectory features": [k for k in FE if k[0] == "raw trajectory"],
        "smoothed trajectory features": [k for k in FE if k[0] == "smoothed trajectory"]}
for nm, cols in fams.items():
    for lab, mask in (("all rows", np.ones(len(D), bool)), ("unfinished", unf)):
        a, acc, bal = loo(cols, mask); print(f"  {nm:46s} {lab:11s} {a:.2f} / {acc:.2f} / {bal:.2f}")
pickle.dump(dict(FE=FE, ids=list(D.id), y=y, unf=unf, topdims=TOPDIMS), open(A / "exp60_deepseek_spikes.pkl", "wb"))
