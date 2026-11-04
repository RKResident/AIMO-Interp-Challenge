"""exp63: the user's step-vs-hub hypothesis (robust reasoning is step-like - each segment reads the one before it; spurious reasoning makes
long jumps / leans on a few hub segments) on ALL the reasoning graphs the white-box run saved, not only the sentence graph exp61 scored:
  sentence graph (head mean), step = paragraph graph (head mean), fixed 64-token blocks (length-neutral baseline), 8 sampled layers.
Same graph features as exp61 (per layer third) plus: chain share (share of segments whose strongest earlier-segment edge is the immediately
preceding one), jump entropy (entropy of the attention-weighted distance distribution), problem revisits (share of segments whose prompt mass
exceeds 2x the trace median). Scored as within-model percentile ranks -> AUC on unfinished rows (focus), all rows, main_v2, olympiad;
permutation p (within model) on unfinished; sentence-count-controlled AUC on unfinished (rank residual on the number of sentences within model; unfinished traces all have 6144 tokens)."""
import json, os, pickle, re, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "2"
import numpy as np
import pandas as pd
sys.path.insert(0, "/home/sabyasachi19/PycharmProjects/AimoAGAIN/analysis")


# gini / graph_feats copied verbatim from exp61_whitebox.py (importing it would rerun exp61)
def gini(x):
    x = np.sort(np.abs(np.asarray(x, float))); n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum() + 1e-12)) if n else np.nan


def graph_feats(G, prefix):
    """G [L, n, 2 + n]: attention mass of each reasoning segment (rows) on the prompt numbers, prompt prose and earlier segments."""
    out = {}; L = G.shape[0]; n = G.shape[1]
    for bname, band in (("early", range(0, L // 3)), ("middle", range(L // 3, 2 * L // 3)), ("late", range(2 * L // 3, L))):
        g = G[list(band)].astype(np.float64).mean(0); P = g[:, :2]; Rm = g[:, 2:]
        prev = np.zeros(n); near = np.zeros(n); far = np.zeros(n); jump = np.full(n, np.nan); indeg = np.zeros(n)
        for i in range(1, n):
            w = Rm[i, :i].copy(); s = w.sum()
            if s <= 0: continue
            d = i - np.arange(i); jump[i] = (w * d).sum() / s
            near[i] = w[max(0, i - 2):].sum() / s; far[i] = w[d > 20].sum() / s; indeg[:i] += w / s
        vis = np.arange(n)[::-1].astype(float); indeg_n = indeg / np.maximum(vis, 1)        # per query that could see it
        out[f"{prefix} {bname}: mean jump (segments)"] = float(np.nanmean(jump))
        out[f"{prefix} {bname}: share on previous 2"] = float(near[1:].mean())
        out[f"{prefix} {bname}: share > 20 back"] = float(far[1:].mean())
        out[f"{prefix} {bname}: hub Gini of incoming attention"] = gini(indeg_n[:-1])
        out[f"{prefix} {bname}: top-3 hub share"] = float(np.sort(indeg_n[:-1])[-3:].sum() / (indeg_n[:-1].sum() + 1e-12))
        out[f"{prefix} {bname}: prompt share"] = float(P.sum(1).mean())
        out[f"{prefix} {bname}: numbers / prompt"] = float((P[:, 0] / (P.sum(1) + 1e-12)).mean())
        q = max(1, n // 4); out[f"{prefix} {bname}: prompt share late - early"] = float(P[-q:].sum(1).mean() - P[:q].sum(1).mean())
    return out


ROOT = __import__("pathlib").Path("/home/sabyasachi19/PycharmProjects/AimoAGAIN"); WB = ROOT / "results_whitebox" / "out_whitebox"
W = pickle.load(open(ROOT / "analysis" / "exp61_whitebox.pkl", "rb")); M = W["meta"].reset_index(drop=True)
slug = lambda m: m.replace("/", "__")                                                   # noqa: E731
rid2model = {json.loads(l)["id"]: json.loads(l)["model_id"] for f in WB.glob("*/rows_shard*.jsonl") for l in open(f)}


def extra_feats(G, prefix):
    out = {}; L = G.shape[0]; n = G.shape[1]
    for bname, band in (("early", range(0, L // 3)), ("middle", range(L // 3, 2 * L // 3)), ("late", range(2 * L // 3, L))):
        g = G[list(band)].astype(np.float64).mean(0); P = g[:, :2].sum(1); Rm = g[:, 2:]
        chain, ent = [], []
        for i in range(2, n):
            w = Rm[i, :i]
            if w.sum() <= 0: continue
            chain.append(float(np.argmax(w) == i - 1))
            d = i - np.arange(i); h = np.bincount(np.minimum(d, 64), weights=w, minlength=65); h = h / h.sum()
            ent.append(float(-(h[h > 0] * np.log(h[h > 0])).sum()))
        out[f"{prefix} {bname}: chain share"] = float(np.mean(chain)) if chain else np.nan
        out[f"{prefix} {bname}: jump entropy"] = float(np.mean(ent)) if ent else np.nan
        out[f"{prefix} {bname}: problem revisits"] = float((P > 2 * np.median(P)).mean()) if n > 2 else np.nan
    return out


F = []
for i, r in M.iterrows():
    z = np.load(WB / slug(rid2model[r.id]) / f"{r.id}.npz"); f = {}
    for nm, key in (("sentence", "att_graph_sentence_layer"), ("step", "att_graph_step_layer"), ("block64", "att_graph_block64_layer")):
        G = z[key]
        if G.shape[1] < 4: continue
        f.update(graph_feats(G, nm)); f.update(extra_feats(G, nm))
    f["n steps"] = z["att_graph_step_layer"].shape[1]; f["n sentences"] = z["att_graph_sentence_layer"].shape[1]
    F.append(f)
F = pd.DataFrame(F); y = M.y.values.astype(int); model = M.model.values; unf = M.unf.values.astype(bool); ds = M.dataset.values
nsent = F["n sentences"].values.astype(float)   # unfinished traces all have 6144 tokens, so control for segment count instead
rng = np.random.default_rng(0)


def wrank(x, m):
    out = np.full(len(x), np.nan)
    for g in np.unique(model[m]):
        k = m & (model == g) & ~np.isnan(x)
        if k.sum(): out[k] = pd.Series(x[k]).rank(pct=True).values
    return out


def auc(s, yy):
    ok = ~np.isnan(s); s, yy = s[ok], yy[ok]; p, n = s[yy == 1], s[yy == 0]
    return float(((p[:, None] > n[None]) + .5 * (p[:, None] == n[None])).mean()) if len(p) and len(n) else np.nan


def perm_p(x, m, a, k=1000):
    yy = y[m].copy(); s = wrank(x, m)[m]; mm = model[m]; null = []
    for _ in range(k):
        for g in np.unique(mm): yy[mm == g] = rng.permutation(yy[mm == g])
        null.append(auc(s, yy))
    return float((np.abs(np.array(null) - .5) >= abs(a - .5) - 1e-12).mean())


def ctrl(x, m):                                                            # residual of within-model rank on sentence-count rank
    s = wrank(x, m); l = wrank(nsent, m); ok = m & ~np.isnan(s); r = np.full(len(x), np.nan)
    b = np.polyfit(l[ok], s[ok], 1); r[ok] = s[ok] - np.polyval(b, l[ok]); return r


res = []
for c in F.columns:
    x = F[c].values.astype(float); a_u = auc(wrank(x, unf)[unf], y[unf])
    res.append(dict(feature=c, auc_unf=a_u, p_unf=perm_p(x, unf, a_u), auc_unf_nsentctrl=auc(ctrl(x, unf)[unf], y[unf]),
                    auc_all=auc(wrank(x, np.ones(len(x), bool)), y), auc_mainv2=auc(wrank(x, ds == "main_v2")[ds == "main_v2"], y[ds == "main_v2"]),
                    auc_oly=auc(wrank(x, ds == "olympiad")[ds == "olympiad"], y[ds == "olympiad"]),
                    auc_unf_mainv2=auc(wrank(x, unf & (ds == "main_v2"))[unf & (ds == "main_v2")], y[unf & (ds == "main_v2")]),
                    auc_unf_oly=auc(wrank(x, unf & (ds == "olympiad"))[unf & (ds == "olympiad")], y[unf & (ds == "olympiad")])))
R = pd.DataFrame(res); R["strength"] = (R.auc_unf - .5).abs(); R = R.sort_values("strength", ascending=False)
R["consistent_unf"] = np.sign(R.auc_unf_mainv2 - .5) == np.sign(R.auc_unf_oly - .5)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(f"{len(M)} rows ({unf.sum()} unfinished) | {len(F.columns)} graph features | unfinished p<0.05: {(R.p_unf < .05).sum()} "
      f"(chance ~{.05 * len(F.columns):.1f})\n")
print(R.drop(columns="strength").round(2).to_string(index=False))
R.to_csv(ROOT / "analysis" / "exp63_reasoning_graphs.csv", index=False); F.assign(id=M.id).to_pickle(ROOT / "analysis" / "exp63_graph_features.pkl")
