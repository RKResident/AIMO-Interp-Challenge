"""exp36: the 10 features of our earlier trajectory-dynamics leaderboard submission (its own trajectory_metrics.compute_trajectory_geometry,
imported unchanged), computed on the AVERAGED trajectories, main_v2 only (43 rows, 22 problems).
Layer = the submission's layer for each model (DeepSeek 32, Qwen / Olmo 30, Skywork 25) = the last of our 6 saved trajectory layers;
raw states (no centring / normalising, as the submission). Trajectories:
  sentence  overlap-weighted mean of the exact 16-token window means per sentence           (primary)
  ma64      64-token moving average, evaluated every 16 tokens                                (primary)
  win16     the 16-token window means themselves
  waypoint  the every-16th-token states the submission used (saved_trajectories_v2, 34 rows) (secondary)
Windows: the whole generated trace (as the submission) and the first 4096 tokens (traces still thinking there; length-free).
Scored like exp34/35 (within-model z, problem level, finished fraction + mean log length regressed out, permutation p)."""
import os, sys, glob
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "1"
from scipy.stats import rankdata
src = open("exp32_knn_graphs.py").read().split("def knn_metrics")[0].replace("LAYER, KNN, WINDOWS = 3,", "LAYER, KNN, WINDOWS = 5,")
exec(src)                                                                          # WS[i] = window means at layer index 5, SENT[i] = sentence spans
sys.path.insert(0, "/home/sabyasachi19/PycharmProjects/aimo/getting-started/solutions/trajectory-dynamics")
from trajectory_metrics import compute_trajectory_geometry, FEATURE_NAMES


def sentence_means(i, b):
    W = WS[i]; nw = min(b // 16, len(W)); S = []
    for a0, e0 in SENT[i]:
        if a0 >= b: break
        w = np.arange(a0 // 16, min((e0 - 1) // 16, nw - 1) + 1)
        if len(w) == 0: continue
        ov = np.array([min(e0, 16 * k + 16) - max(a0, 16 * k) for k in w], float)
        S.append((W[w] * ov[:, None]).sum(0) / ov.sum())
    return np.array(S)


def traj(i, kind, b):
    W = WS[i][: max(1, b // 16)]
    if kind == "win16": return W
    if kind == "ma64":
        cs = np.cumsum(np.vstack([np.zeros((1, W.shape[1])), W]), 0); return (cs[4:] - cs[:-4]) / 4
    if kind == "sentence": return sentence_means(i, b)


WP = {}
for i in rows:
    f = glob.glob(str(ROOT / "saved_trajectories_v2_data" / "*" / f"{allm.id.iloc[i]}.npz"))
    if f:
        z = np.load(f[0]); ks = sorted([k for k in z.files if k.startswith("layer_")], key=lambda s: int(s.split("_")[1]))
        WP[i] = z[ks[5]].astype(np.float32)                                         # submission layer (same 6-layer list)

F = {}
for i in rows:
    n_gen = len(WS[i]) * 16
    for win, b in (("full", 10 ** 9), ("4k", 4096)):
        if win == "4k" and Lthink[i] < 4096: continue
        for kind in ("sentence", "ma64", "win16", "waypoint"):
            if kind == "waypoint":
                if i not in WP: continue
                H = WP[i][: (None if win == "full" else 1 + 4096 // 16)]
            else:
                H = traj(i, kind, b)
            if H is None or len(H) < 4: continue
            for k, v in compute_trajectory_geometry(H.astype(np.float32)).items():
                F.setdefault(f"{kind}|{win}|{k}", np.full(len(allm), np.nan))[i] = v
pickle.dump(F, open(A / "exp36_features.pkl", "wb"))

up = np.unique(prob[rows]); yp = np.array([y[rows][prob[rows] == p][0] for p in up]).astype(int)
fp = np.array([fin[rows][prob[rows] == p].mean() for p in up]); lp = np.array([lgn[rows][prob[rows] == p].mean() for p in up])
rng = np.random.default_rng(0)


def auc(x, yy):
    rk = rankdata(x); n1 = yy.sum(); n0 = len(yy) - n1
    return (rk[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def perm_p(x, yy, n=5000):
    a = auc(x, yy); rk = rankdata(x); n1 = yy.sum(); n0 = len(yy) - n1
    YP = rng.permuted(np.tile(yy, (n, 1)), axis=1).astype(float); nl = (YP @ rk - n1 * (n1 + 1) / 2) / (n1 * n0)
    return a, float(np.mean(np.abs(nl - .5) >= abs(a - .5)))


def zmodel(v):
    out = np.full(len(allm), np.nan)
    for m in np.unique(allm.model.values[rows]):
        r = rows[allm.model.values[rows] == m]; x = v[r]
        if np.isfinite(x).sum() >= 2 and np.nanstd(x) > 0: out[r] = (x - np.nanmean(x)) / np.nanstd(x)
    return out


out = []
for nm, v in F.items():
    z = zmodel(v); r = rows[np.isfinite(z[rows])]
    if len(r) < 12 or len(np.unique(z[r])) < 3: continue
    xp = np.array([np.nanmean(z[rows][prob[rows] == p]) if np.isfinite(z[rows][prob[rows] == p]).any() else np.nan for p in up]); ok = np.isfinite(xp)
    if ok.sum() < 8 or len(np.unique(yp[ok])) < 2: continue
    C = np.c_[np.ones(ok.sum()), fp[ok], lp[ok]]; xr = xp[ok] - C @ np.linalg.lstsq(C, xp[ok], rcond=None)[0]
    a_c, p_c = perm_p(xr, yp[ok])
    out.append(dict(feature=nm, rows=len(r), pooled=roc_auc_score(y[r], z[r]), within_finished=within_bin_auc(y[r], z[r], fin[r].astype(int)),
                    controlled=a_c, p_controlled=p_c))
T = pd.DataFrame(out); T[["traj", "window", "metric"]] = T.feature.str.split("|", expand=True)
T.to_csv(A / "exp36_trajdyn_averaged.csv", index=False)
pd.set_option("display.width", 250)
print("AUC > 0.5: higher value -> robust.\n")
for w in ("full", "4k"):
    t = T[T.window == w]
    print(f"=== window {w}: pooled AUC | controlled AUC (p) ===")
    a = t.pivot_table(index="metric", columns="traj", values="pooled").round(2)[["sentence", "ma64", "win16", "waypoint"]]
    c = t.assign(s=t.controlled.round(2).astype(str) + " (" + t.p_controlled.round(2).astype(str) + ")").pivot(index="metric", columns="traj", values="s")[["sentence", "ma64", "win16", "waypoint"]]
    print(a.join(c, rsuffix="_ctrl").to_string(), "\n")
for (tr, w), g in T.groupby(["traj", "window"]):
    print(f"{tr:8s} {w:4s}: controlled p<.05 {int((g.p_controlled < .05).sum())}/{len(g)} (~{.05 * len(g):.1f} by chance)")
