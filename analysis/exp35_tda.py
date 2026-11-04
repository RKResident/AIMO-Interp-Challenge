"""exp35: topological data analysis of the averaged hidden-state trajectory (idea from "The Shape of Reasoning", arXiv 2510.20665),
main_v2 only (43 rows, 22 problems, 7 non-robust).
Point cloud per trace = the exp32 nodes (sentence-grouped or exact 64-token means of the 16-token window states, layer ~2/3 depth),
model-mean centred ("cen", as exp32) or raw ("raw", as the paper's embeddings); window = first 4096 tokens (traces still thinking
there) or the whole thinking part. Cosine distance -> Vietoris-Rips persistence (ripser, H0 + H1) -> the paper's features:
  diagram   total / mean lifetime, lifetime entropy and skewness, max birth, max finite death, number of H1 bars
  Betti curve (grid of 200 over [0, largest finite death of the trace]) peak, peak location, centroid, spread, FWHM width
  level-0 persistence landscape mean / max / area
Scored like exp34 (within-model z, problem level, finished fraction + mean log length regressed out, permutation p) and checked
against exp32's kNN global efficiency (same nodes). H0 features equal minimum-spanning-tree statistics, so they need no TDA library."""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "2"
from scipy.stats import rankdata, skew
from ripser import ripser
exec(open("exp32_knn_graphs.py").read().split("def knn_metrics")[0])          # data, sentence spans, MU, nodes(i, kind, b)


def nodes_any(i, kind, b, centred):
    if centred:
        return nodes(i, kind, b)
    mu = MU[allm.model.iloc[i]]
    MU[allm.model.iloc[i]] = 0.0 * mu                                           # nodes() centres on MU; raw = centre on zero
    try:
        return nodes(i, kind, b)
    finally:
        MU[allm.model.iloc[i]] = mu


def diagram_feats(dg, eps_max, tag, out):
    fin = dg[np.isfinite(dg[:, 1])]
    life = fin[:, 1] - fin[:, 0]
    life = life[life > 0]
    out[f"{tag}_total_life"] = life.sum()
    out[f"{tag}_mean_life"] = life.mean() if len(life) else 0.0
    p = life / life.sum() if life.sum() > 0 else np.array([])
    out[f"{tag}_entropy"] = float(-(p * np.log(p)).sum()) if len(p) else 0.0
    out[f"{tag}_skewness"] = float(skew(life)) if len(life) > 2 else 0.0
    out[f"{tag}_max_birth"] = float(fin[:, 0].max()) if len(fin) else 0.0
    out[f"{tag}_max_death"] = float(fin[:, 1].max()) if len(fin) else 0.0
    if tag == "H1": out["H1_count"] = float(len(life))
    grid = np.linspace(0, eps_max, 200)
    beta = ((fin[:, :1] <= grid[None]) & (grid[None] < fin[:, 1:2])).sum(0).astype(float) if len(fin) else np.zeros_like(grid)
    if beta.sum() > 0:
        x = grid / eps_max; c = (x * beta).sum() / beta.sum()
        out[f"{tag}_betti_peak"] = beta.max(); out[f"{tag}_betti_location"] = x[beta.argmax()]
        out[f"{tag}_betti_centroid"] = c; out[f"{tag}_betti_spread"] = float(np.sqrt(((x - c) ** 2 * beta).sum() / beta.sum()))
        hm = np.where(beta >= beta.max() / 2)[0]; out[f"{tag}_betti_width"] = x[hm[-1]] - x[hm[0]]
    lam = np.clip(np.minimum(grid[None] - fin[:, :1], fin[:, 1:2] - grid[None]), 0, None).max(0) if len(fin) else np.zeros_like(grid)
    out[f"{tag}_landscape_mean"] = lam.mean(); out[f"{tag}_landscape_max"] = lam.max()
    trap = getattr(np, "trapezoid", None) or np.trapz
    out[f"{tag}_landscape_area"] = float(trap(lam, grid))


F = {}
import time as _t
t0 = _t.time()
for i in rows:
    for kind in ("sentence", "step64"):
        for centred in (True, False):
            for win in ("4k", "think"):
                b = 4096 if win == "4k" else int(Lthink[i])
                if win == "4k" and Lthink[i] < 4096: continue
                S = nodes_any(i, kind, b, centred)
                if len(S) < 10: continue
                D = np.clip(1 - S @ S.T, 0, None); np.fill_diagonal(D, 0)
                dg = ripser(D, distance_matrix=True, maxdim=1)["dgms"]
                eps_max = max(float(np.nanmax(np.where(np.isfinite(d[:, 1]), d[:, 1], np.nan))) if np.isfinite(d[:, 1]).any() else 0 for d in dg) or 1.0
                o = {}
                diagram_feats(dg[0], eps_max, "H0", o); diagram_feats(dg[1], eps_max, "H1", o)
                for k, v in o.items():
                    F.setdefault(f"{kind}|{'cen' if centred else 'raw'}|{win}|{k}", np.full(len(allm), np.nan))[i] = v
    print(f"row {i} done ({_t.time() - t0:.0f}s)", flush=True)
pickle.dump(F, open(A / "exp35_features.pkl", "wb"))

# ---------------------------------------------------------------- scoring (as exp34) + overlap with exp32 global efficiency
up = np.unique(prob[rows]); P = len(up); yp = np.array([y[rows][prob[rows] == p][0] for p in up]).astype(int)
fp = np.array([fin[rows][prob[rows] == p].mean() for p in up]); lp = np.array([lgn[rows][prob[rows] == p].mean() for p in up])
rng = np.random.default_rng(0)
G32 = pickle.load(open(A / "exp32_features.pkl", "rb"))


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


ge = zmodel(G32["step64|k5|4k|global_efficiency"])
out = []
for nm, v in F.items():
    z = zmodel(v); x = z[rows]
    if np.isfinite(x).sum() < 12 or len(np.unique(x[np.isfinite(x)])) < 3: continue
    xp = np.array([np.nanmean(z[rows][prob[rows] == p]) if np.isfinite(z[rows][prob[rows] == p]).any() else np.nan for p in up]); ok = np.isfinite(xp)
    if ok.sum() < 8 or len(np.unique(yp[ok])) < 2: continue
    C = np.c_[np.ones(ok.sum()), fp[ok], lp[ok]]; xr = xp[ok] - C @ np.linalg.lstsq(C, xp[ok], rcond=None)[0]
    a_c, p_c = perm_p(xr, yp[ok])
    r = rows[np.isfinite(z[rows])]
    both = r[np.isfinite(ge[r])]
    out.append(dict(feature=nm, rows=len(r), pooled=roc_auc_score(y[r], z[r]), controlled=a_c, p_controlled=p_c,
                    corr_ge=np.corrcoef(z[both], ge[both])[0, 1] if len(both) > 5 else np.nan))
T = pd.DataFrame(out); T[["node", "centre", "window", "metric"]] = T.feature.str.split("|", expand=True)
T.to_csv(A / "exp35_tda.csv", index=False)
pd.set_option("display.width", 250)
print("\nAUC > 0.5: higher value -> robust.")
for (nd, ce, wi), g in T.groupby(["node", "centre", "window"]):
    print(f"{nd:8s} {ce} {wi:5s}: {len(g)} features, controlled p<.05: {(g.p_controlled < .05).sum()} (~{.05 * len(g):.1f} by chance)")
print("\nthe paper's highlighted features (Betti spread / width), every setting:")
print(T[T.metric.isin(["H0_betti_spread", "H0_betti_width", "H1_betti_width"])].pivot_table(index="metric", columns=["node", "centre", "window"], values="controlled").round(2).to_string())
print("\nstrongest (controlled):")
print(T.assign(d=(T.controlled - .5).abs()).sort_values("d", ascending=False).head(15)[["feature", "rows", "pooled", "controlled", "p_controlled", "corr_ge"]].round(3).to_string(index=False))
print(f"\n{len(T)} features; controlled p<.05: {(T.p_controlled < .05).sum()} (~{.05 * len(T):.0f} by chance)")
