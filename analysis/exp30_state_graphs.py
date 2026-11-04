"""exp30: reasoning traces as graphs over hidden-state clusters (idea from the reasoning-loops paper, papers_to_see/2601.05693).
Hidden states every 16 tokens (saved_trajectories_v2_data, layer ~2/3 depth) -> 64-token steps -> K-means per model (shared across
that model's main_v2 traces, so a node is a reusable "state") -> per trace: label sequence = walk on a graph. Graph metrics on fixed
token windows (length-free), for several K; a metric is only worth anything if its sign and size hold across K.
Scored like exp28/29: within-model z, problem level, (finished fraction, mean log length) regressed out, permutation p."""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "2"
from common import *
import glob
from state_graph import graph_feats
from scipy.stats import rankdata
from sklearn.cluster import KMeans

allm, feats = load_all(); y = allm.y.values
rows = datasets(allm)["main_v2"]; grp = allm.grp.values; prob = allm.base_id.values
fin = (allm.stop.values == "eos").astype(float); lgn = np.log(allm.n_gen.astype(float).values)
CH, KS, WINDOWS = 4, (16, 32, 64), ((0, 4096), (0, 6144), (1024, 4096))
WP = {}
for i in rows:
    f = glob.glob(str(ROOT / "saved_trajectories_v2_data" / "*" / f"{allm.id.iloc[i]}.npz"))
    if f:
        z = np.load(f[0]); ks = sorted([k for k in z.files if k.startswith("layer_")], key=lambda s: int(s.split("_")[1]))
        X = z[ks[3]].astype(np.float32); WP[i] = X[: len(X) // CH * CH].reshape(-1, CH, X.shape[1]).mean(1)


F = {}
for K in KS:
    for mdl in np.unique(allm.model.values[list(WP)]):
        ii = [i for i in WP if allm.model.iloc[i] == mdl]
        mu = np.concatenate([WP[i] for i in ii]).mean(0)
        st = {i: (lambda s: s / (np.linalg.norm(s, axis=1, keepdims=True) + 1e-6))(WP[i] - mu) for i in ii}
        km = KMeans(n_clusters=K, n_init=4, random_state=0).fit(np.concatenate(list(st.values())))
        C = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
        for i in ii:
            lab_all = km.predict(st[i])
            for a0, b0 in WINDOWS:
                s0, s1 = a0 // 64, b0 // 64
                if len(lab_all) < s1: continue
                for k, v in graph_feats(list(lab_all[s0:s1]), C).items():
                    nm = f"K{K}|{a0 // 1024}-{b0 // 1024}k|{k}"
                    F.setdefault(nm, np.full(len(allm), np.nan))[i] = v

# ---------------------------------------------------------------- scoring (as exp29)
up = np.unique(prob[rows]); yp = np.array([y[rows][prob[rows] == p][0] for p in up])
fp = np.array([fin[rows][prob[rows] == p].mean() for p in up]); lp_ = np.array([lgn[rows][prob[rows] == p].mean() for p in up])
rng = np.random.default_rng(0)


def perm_auc(x, yy, n=5000):
    rk = rankdata(x); n1 = yy.sum(); n0 = len(yy) - n1; c = n1 * (n1 + 1) / 2
    a = (rk[yy == 1].sum() - c) / (n1 * n0)
    YP = np.array([rng.permutation(yy) for _ in range(n)]).astype(float); nl = (YP @ rk - c) / (n1 * n0)
    return a, float(np.mean(np.abs(nl - .5) >= abs(a - .5)))


out = []
for nm, v in F.items():
    ok = np.isfinite(v[rows]); r = rows[ok]
    if ok.sum() < 12 or len(np.unique(y[r])) < 2 or len(np.unique(v[r])) < 3: continue
    vz = zgroup(np.where(np.isin(np.arange(len(allm)), r), v, np.nan)[:, None], grp)[:, 0]
    xp = np.array([np.nanmean(np.where(prob[r] == p, vz[r], np.nan)) if (prob[r] == p).any() else np.nan for p in up]); okp = np.isfinite(xp)
    if okp.sum() < 8 or len(np.unique(yp[okp])) < 2: continue
    Cc = np.c_[fp, lp_][okp]; xr = xp[okp] - LinearRegression().fit(Cc, xp[okp]).predict(Cc); apr, ppr = perm_auc(xr, yp[okp])
    K, win, met = nm.split("|")
    out.append(dict(K=K, window=win, metric=met, rows=int(ok.sum()), pooled=roc_auc_score(y[r], vz[r]),
                    within_finished=within_bin_auc(y[r], vz[r], fin[r].astype(int)), problem_controlled=apr, p_controlled=ppr))
T = pd.DataFrame(out); T.to_csv(A / "exp30_state_graphs.csv", index=False)
pd.set_option("display.width", 250)
print("AUC > 0.5: higher value -> robust.  Rows: pooled AUC per K (window 0-4k); stability across K and windows is what matters.\n")
P = T.pivot_table(index="metric", columns=["window", "K"], values="pooled").round(2)
print(P.to_string())
print("\nproblem-level controlled AUC (p) for window 0-4k:")
Q = T[T.window == "0-4k"].assign(s=lambda d: d.problem_controlled.round(2).astype(str) + " (" + d.p_controlled.round(3).astype(str) + ")")
print(Q.pivot(index="metric", columns="K", values="s").to_string())
