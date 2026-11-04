"""exp32: kNN ("recurrence network") graphs of the AVERAGED hidden-state trajectory, main_v2 only (43 rows, 22 problems).
Source: exact means of every consecutive 16-token window (extraction_out_generations_p2/.../tf/*.npz 'win_states', 6 layers; the same
generations as extraction_out, so the text lines up). Nodes, over a fixed token window of the thinking (traces still thinking there):
  step64   : exact mean of each consecutive 64-token block
  sentence : mean of the windows overlapping each sentence, weighted by overlap (every sentence covered)
Per trace: cosine kNN graph (k = 5, 10; states centred on the model mean; symmetrised) -> Fiedler value and other Laplacian
eigenvalues, k-edge connectivity, global / local efficiency, diameter, modularity (Louvain), Shannon entropy of the degree
distribution, von Neumann entropy, transitivity, share of kNN edges that join consecutive nodes.
Scored like exp29-31 and checked against phase-free consecutive-step similarity of the same averaged trajectory."""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "2"
from common import *
import json, re
import networkx as nx
from scipy.stats import rankdata
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

P2 = ROOT / "extraction_out_generations_p2" / "extraction_out_generations"
allm, feats = load_all(); y = allm.y.values
rows = datasets(allm)["main_v2"]; grp = allm.grp.values; prob = allm.base_id.values
fin = (allm.stop.values == "eos").astype(float); lgn = np.log(allm.n_gen.astype(float).values)
Lthink = np.full(len(allm), np.nan); Lthink[rows] = [force_json(allm.model.iloc[i], allm.id.iloc[i])["L"] for i in rows]
LAYER, KNN, WINDOWS = 3, (5, 10), (2048, 4096, 6144)
END = re.compile(r"[.!?:]\s*$|\n")

WS, SENT, TOK = {}, {}, {}
for i in rows:
    r = allm.iloc[i]
    WS[i] = np.load(P2 / r.model / "tf" / f"{r.id}.npz")["win_states"][:, LAYER].astype(np.float32)      # [n_win, d], window w = tokens 16w..16w+15
    g = next(json.loads(l) for l in open(P2 / r.model / "generations.jsonl") if json.loads(l)["id"] == r.id)["gen_ids"]
    tok = TOK.setdefault(r.model, Tokenizer.from_file(hf_hub_download(r.model.replace("__", "/", 1), "tokenizer.json", local_files_only=True)))
    ends = [t for t, p in enumerate(tok.decode([x]) for x in g) if END.search(p)]
    SENT[i] = list(zip([0] + [e + 1 for e in ends], [e + 1 for e in ends] + [len(g)]))                # [start, end) token spans
MU = {m: np.concatenate([WS[i] for i in rows if allm.model.iloc[i] == m]).mean(0) for m in np.unique(allm.model.values[rows])}


def nodes(i, kind, b):
    W = WS[i] - MU[allm.model.iloc[i]]; nw = min(b // 16, len(W))
    if kind == "step64":
        S = W[: nw // 4 * 4].reshape(-1, 4, W.shape[1]).mean(1)
    else:
        S = []
        for a0, e0 in SENT[i]:
            if a0 >= b: break
            w = np.arange(a0 // 16, min((e0 - 1) // 16, nw - 1) + 1)
            if len(w) == 0: continue
            ov = np.array([min(e0, 16 * k + 16) - max(a0, 16 * k) for k in w], float)
            S.append((W[w] * ov[:, None]).sum(0) / ov.sum())
        S = np.array(S)
    return S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-6)


def knn_metrics(S, k):
    n = len(S); C = S @ S.T; np.fill_diagonal(C, -np.inf)
    nb = np.argsort(-C, axis=1)[:, :k]; A = np.zeros((n, n))
    A[np.repeat(np.arange(n), k), nb.ravel()] = 1; A = np.maximum(A, A.T)
    G = nx.from_numpy_array(A); deg = A.sum(1); o = {}
    Ln = np.eye(n) - A / np.sqrt(np.outer(deg, deg)); ev = np.sort(np.linalg.eigvalsh(Ln))
    o["fiedler_norm"] = ev[1]; o["lap_eig2to6_mean"] = ev[1:6].mean(); o["lap_eig_max"] = ev[-1]
    Lc = np.diag(deg) - A; evc = np.sort(np.linalg.eigvalsh(Lc)); o["fiedler_comb"] = evc[1] / k
    p = np.clip(evc, 0, None) / evc.sum(); p = p[p > 0]; o["von_neumann_entropy"] = float(-(p * np.log(p)).sum() / np.log(n))
    o["edge_connectivity"] = nx.edge_connectivity(G) / k
    o["global_efficiency"] = nx.global_efficiency(G); o["local_efficiency"] = nx.local_efficiency(G)
    big = G.subgraph(max(nx.connected_components(G), key=len)); o["diameter_rel"] = nx.diameter(big) / n
    o["n_components"] = nx.number_connected_components(G)
    com = nx.community.louvain_communities(G, seed=0); o["modularity"] = nx.community.modularity(G, com); o["n_communities_rel"] = len(com) / n
    h = np.bincount(deg.astype(int)) / n; h = h[h > 0]; o["degree_entropy"] = float(-(h * np.log(h)).sum())
    o["transitivity"] = nx.transitivity(G)
    iu = np.triu_indices(n, 1); o["consecutive_edge_share"] = float(A[np.arange(n - 1), np.arange(1, n)].sum() / A[iu].sum())
    return o


F = {}
for i in rows:
    for b in WINDOWS:
        if Lthink[i] < b: continue                                           # finished thinking before b: leave NaN
        for kind in ("step64", "sentence"):
            S = nodes(i, kind, b)
            for k in KNN:
                if len(S) < 3 * k: continue
                for nm, v in knn_metrics(S, k).items():
                    F.setdefault(f"{kind}|k{k}|{b // 1024}k|{nm}", np.full(len(allm), np.nan))[i] = v
    print(f"row {i} done", flush=True)

# phase-free consecutive-step similarity of the averaged trajectory: 64-token moving averages 64 tokens apart, every 16 tokens
STEP = {b: np.full(len(allm), np.nan) for b in WINDOWS}
for i in rows:
    W = WS[i] - MU[allm.model.iloc[i]]
    for b in WINDOWS:
        if Lthink[i] < b: continue
        H = W[: b // 16]; cs = np.cumsum(np.vstack([np.zeros((1, H.shape[1])), H]), 0); M = (cs[4:] - cs[:-4]) / 4
        M /= np.linalg.norm(M, axis=1, keepdims=True); STEP[b][i] = np.mean(np.sum(M[4:] * M[:-4], 1))

F['reference:consecutive_step_cos'] = STEP[4096]
pickle.dump(F, open(A / "exp32_features.pkl", "wb"))          # feature values, for ensembles (exp34)

# ---------------------------------------------------------------- scoring
up = np.unique(prob[rows]); yp = np.array([y[rows][prob[rows] == p][0] for p in up])
fp = np.array([fin[rows][prob[rows] == p].mean() for p in up]); lp_ = np.array([lgn[rows][prob[rows] == p].mean() for p in up])
rng = np.random.default_rng(0)


def perm_auc(x, yy, n=5000):
    rk = rankdata(x); n1 = yy.sum(); n0 = len(yy) - n1; c = n1 * (n1 + 1) / 2
    a = (rk[yy == 1].sum() - c) / (n1 * n0)
    YP = np.array([rng.permutation(yy) for _ in range(n)]).astype(float); nl = (YP @ rk - c) / (n1 * n0)
    return a, float(np.mean(np.abs(nl - .5) >= abs(a - .5)))


def zm(v, r): return pd.Series(v[r]).groupby(allm.model.values[r]).transform(lambda a: (a - a.mean()) / (a.std() + 1e-9)).values


def score(nm, v, ref):
    ok = np.isfinite(v[rows]) & np.isfinite(ref[rows]); r = rows[ok]
    if ok.sum() < 12 or len(np.unique(y[r])) < 2 or len(np.unique(v[r])) < 3: return None
    vz = zgroup(np.where(np.isin(np.arange(len(allm)), r), v, np.nan)[:, None], grp)[:, 0]
    xp = np.array([np.nanmean(np.where(prob[r] == p, vz[r], np.nan)) if (prob[r] == p).any() else np.nan for p in up]); okp = np.isfinite(xp)
    if okp.sum() < 8 or len(np.unique(yp[okp])) < 2: return None
    Cc = np.c_[fp, lp_][okp]; xr = xp[okp] - LinearRegression().fit(Cc, xp[okp]).predict(Cc); apr, ppr = perm_auc(xr, yp[okp])
    a_, b_ = zm(ref, r), zm(v, r); res = b_ - np.polyval(np.polyfit(a_, b_, 1), a_)
    return dict(feature=nm, rows=int(ok.sum()), problems=int(okp.sum()), pooled=roc_auc_score(y[r], vz[r]), within_finished=within_bin_auc(y[r], vz[r], fin[r].astype(int)),
                problem_controlled=apr, p_controlled=ppr, corr_with_stepsim=np.corrcoef(a_, b_)[0, 1], auc_after_stepsim=roc_auc_score(y[r], res))


out = [s for b in WINDOWS if (s := score(f"reference|-|{b // 1024}k|consecutive_step_cos", STEP[b], STEP[b]))]
for nm, v in F.items():
    s = score(nm, v, STEP[int(nm.split("|")[2][:-1]) * 1024])
    if s: out.append(s)
T = pd.DataFrame(out); T[["node", "k", "window", "metric"]] = T.feature.str.split("|", expand=True)
T.to_csv(A / "exp32_knn_graphs.csv", index=False)
pd.set_option("display.width", 260)
print("\nAUC > 0.5: higher value -> robust.\n")
print(T[T.node == "reference"][["window", "rows", "pooled", "within_finished", "problem_controlled", "p_controlled"]].round(3).to_string(index=False))
for b in ("4k", "2k"):
    t = T[(T.window == b) & (T.node != "reference")]
    print(f"\n=== window 0-{b}: pooled AUC per node type / k   |   AUC after removing consecutive-step similarity ===")
    a = t.pivot_table(index="metric", columns=["node", "k"], values="pooled").round(2)
    r_ = t.pivot_table(index="metric", columns=["node", "k"], values="auc_after_stepsim").round(2); r_.columns = [f"{c[0][:4]}{c[1]}_res" for c in r_.columns]
    a.columns = [f"{c[0][:4]}{c[1]}" for c in a.columns]
    print(a.join(r_).to_string())
print("\nstrongest problem-level controlled:")
print(T[T.node != "reference"].assign(d=(T.problem_controlled - .5).abs()).sort_values("d", ascending=False).head(15)[["feature", "rows", "pooled", "problem_controlled", "p_controlled", "auc_after_stepsim"]].round(3).to_string(index=False))
print(f"\n{(T.node != 'reference').sum()} feature versions; controlled p < 0.05: {int(((T.p_controlled < .05) & (T.node != 'reference')).sum())} (~{0.05 * (T.node != 'reference').sum():.0f} by chance)")
