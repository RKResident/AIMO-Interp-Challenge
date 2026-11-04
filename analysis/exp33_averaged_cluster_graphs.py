"""exp33: exp29-31's waypoint-only hidden-state measures, recomputed on the AVERAGED trajectory (main_v2, 43 rows).
Nodes as in exp32 (exact 16-token window means -> sentence means or exact 64-token blocks, model mean removed, unit norm), over the
first 2k / 4k / 6k thinking tokens of traces still thinking there. Per model K-means (shared across that model's traces) -> walk on
the cluster graph -> exp30's graph metrics (state_graph.graph_feats); plus exp29's recurrence measures without clustering.
Only computes and saves the features (exp33_features.pkl); scoring and ensembles are in exp34."""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "2"
from sklearn.cluster import KMeans
from state_graph import graph_feats
exec(open("exp32_knn_graphs.py").read().split("def knn_metrics")[0])           # data loading + nodes(i, kind, b) from exp32

KS = {"step64": (16, 32, 64), "sentence": (32, 64, 128)}
F = {}


def put(nm, i, v): F.setdefault(nm, np.full(len(allm), np.nan))[i] = v


NODES = {(i, kind, b): nodes(i, kind, b) for i in rows for kind in KS for b in WINDOWS if Lthink[i] >= b}
for kind in KS:
    for b in WINDOWS:
        for m in np.unique(allm.model.values[rows]):
            ii = [i for i in rows if allm.model.iloc[i] == m and (i, kind, b) in NODES]
            if len(ii) < 3: continue
            allS = np.concatenate([NODES[(i, kind, b)] for i in ii])
            # recurrence without clustering (exp29): far-apart pairs above this model's 95th-percentile similarity
            gap = 8 if kind == "step64" else 30
            sims = np.concatenate([(lambda S: (S @ S.T)[np.triu_indices(len(S), gap)])(NODES[(i, kind, b)]) for i in ii])
            thr = np.quantile(sims, .95)
            for i in ii:
                S = NODES[(i, kind, b)]; T = len(S)
                if T <= gap + 5: continue
                C = S @ S.T; far = np.tril(np.ones((T, T), bool), -gap)
                put(f"{kind}|-|{b // 1024}k|recurrence_rate", i, float((C[far] > thr).mean()))
                put(f"{kind}|-|{b // 1024}k|mean_max_backsim", i, float(np.nanmean(np.where(far, C, np.nan)[gap:].max(1))))
            for K in KS[kind]:
                if len(allS) < 4 * K: continue
                km = KMeans(n_clusters=K, n_init=4, random_state=0).fit(allS)
                Cc = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
                for i in ii:
                    lab = km.predict(NODES[(i, kind, b)])
                    if len(lab) < 10: continue
                    for k, v in graph_feats(list(lab), Cc).items(): put(f"{kind}|K{K}|{b // 1024}k|cluster_{k}", i, v)
        print(f"{kind} {b} done", flush=True)
pickle.dump(F, open(A / "exp33_features.pkl", "wb"))
print(f"saved {len(F)} features -> exp33_features.pkl")
