"""exp38: trajectories ACROSS layers, and how topology changes with depth and over time (main_v2 only, 43 rows, 22 problems).
Data: exact 16-token window means at 6 layers (win_states, e.g. DeepSeek 12/16/20/24/28/32) and the last prompt token at every layer
(prompt_last_all_layers), extraction_out_generations_p2, same traces as everything else.
  A depth path per window: the 6-layer path of each 16-token window -> mean cosine between consecutive layers, depth straightness,
    depth turning; summarised over thinking (mean / std / late quarter minus early quarter)
  B full-depth path of the prompt: last prompt token across all layers -> consecutive-layer cosine, straightness, turning, relative
    depth of the largest change (no generation involved -> length cannot drive it)
  C topology by layer: exp35's candidates (H0 Betti spread, H1 count / Betti peak, H0 total lifetime) on raw 64-token nodes over the
    whole thinking part, at each of the 6 layers -> per-layer values, mean, slope over depth
  D topology over time: the same features in sliding 1024-token windows of thinking (64 window means each, step 512) at the submission
    layer (index 5) and the middle layer (index 3) -> mean, slope over time, late minus early
Scored like exp34/35 (within-model z, problem level, finished fraction + mean log length regressed out, permutation p)."""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "2"
from common import *
from scipy.stats import rankdata
from ripser import ripser

P2 = ROOT / "extraction_out_generations_p2" / "extraction_out_generations"
allm, _ = load_all(); y = allm.y.values; rows = datasets(allm)["main_v2"]; model = allm.model.values; prob = allm.base_id.values
fin = (allm.stop.values == "eos").astype(float); lgn = np.log(allm.n_gen.astype(float).values)
Lthink = {i: force_json(model[i], allm.id.iloc[i])["L"] for i in rows}
F = {}


def put(nm, i, v): F.setdefault(nm, np.full(len(allm), np.nan))[i] = v


def path_geom(H):
    """H [K, d] path -> consecutive cosine, straightness, turning (mean cosine of consecutive steps), relative depth of largest step"""
    Hn = H / (np.linalg.norm(H, axis=-1, keepdims=True) + 1e-9)
    D = np.diff(H, axis=0); sp = np.linalg.norm(D, axis=-1)
    return dict(cons_cos=float((Hn[1:] * Hn[:-1]).sum(-1).mean()), straight=float(np.linalg.norm(H[-1] - H[0]) / (sp.sum() + 1e-9)),
                turn=float(((D[1:] * D[:-1]).sum(-1) / (sp[1:] * sp[:-1] + 1e-9)).mean()) if len(D) > 1 else np.nan,
                argmax_step=float(np.argmax(sp) / max(1, len(sp) - 1)))


def tda(S):
    """raw states -> unit vectors -> cosine Rips (H0, H1) -> exp35 candidate features"""
    S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-9)
    Dm = np.clip(1 - S @ S.T, 0, None); np.fill_diagonal(Dm, 0)
    h0, h1 = ripser(Dm, distance_matrix=True, maxdim=1)["dgms"]
    h0 = h0[np.isfinite(h0[:, 1])]; h1 = h1[h1[:, 1] > h1[:, 0]]
    eps = max(h0[:, 1].max() if len(h0) else 0, h1[:, 1].max() if len(h1) else 0) or 1.0
    grid = np.linspace(0, eps, 200); x = grid / eps
    out = {"H0_total_life": float((h0[:, 1] - h0[:, 0]).sum()), "H1_count": float(len(h1))}
    for tag, dg in (("H0", h0), ("H1", h1)):
        beta = ((dg[:, :1] <= grid[None]) & (grid[None] < dg[:, 1:2])).sum(0).astype(float) if len(dg) else np.zeros_like(grid)
        if beta.sum() > 0:
            c = (x * beta).sum() / beta.sum(); out[f"{tag}_betti_spread"] = float(np.sqrt(((x - c) ** 2 * beta).sum() / beta.sum()))
            out[f"{tag}_betti_peak"] = float(beta.max())
    return out


def trend(v):
    v = np.asarray(v, float); ok = np.isfinite(v)
    return float(np.polyfit(np.linspace(0, 1, len(v))[ok], v[ok], 1)[0]) if ok.sum() >= 3 else np.nan


for i in rows:
    z = np.load(P2 / model[i] / "tf" / f"{allm.id.iloc[i]}.npz")
    ws = z["win_states"].astype(np.float32); nw = min(len(ws), max(1, Lthink[i] // 16)); T = ws[:nw]          # thinking part
    # A: depth path of every window, summarised over thinking
    g = [path_geom(T[t]) for t in range(nw)]
    for k in ("cons_cos", "straight", "turn"):
        v = np.array([d[k] for d in g]); q = max(1, len(v) // 4)
        put(f"A_depth|{k}|mean", i, np.nanmean(v)); put(f"A_depth|{k}|std", i, np.nanstd(v))
        put(f"A_depth|{k}|late_minus_early", i, np.nanmean(v[-q:]) - np.nanmean(v[:q]))
    # B: full-depth path of the last prompt token
    for k, v in path_geom(z["prompt_last_all_layers"].astype(np.float32)).items(): put(f"B_prompt_depth|{k}", i, v)
    # C: topology at each of the 6 layers (raw 64-token nodes, whole thinking)
    per = []
    for l in range(T.shape[1]):
        X = T[: nw // 4 * 4, l].reshape(-1, 4, T.shape[2]).mean(1)
        per.append(tda(X) if len(X) >= 10 else {})
    for k in ("H0_betti_spread", "H1_count", "H1_betti_peak", "H0_total_life"):
        v = [d.get(k, np.nan) for d in per]
        for l, x in enumerate(v): put(f"C_layer|{k}|L{l}", i, x)
        put(f"C_layer|{k}|mean", i, np.nanmean(v)); put(f"C_layer|{k}|depth_slope", i, trend(v))
    # D: topology in sliding 1024-token windows of thinking
    for l in (3, 5):
        starts = range(0, max(1, nw - 64 + 1), 32)
        seq = [tda(T[s:s + 64, l]) for s in starts if s + 64 <= nw]
        if len(seq) < 2: continue
        for k in ("H0_betti_spread", "H1_count", "H1_betti_peak"):
            v = np.array([d.get(k, np.nan) for d in seq]); q = max(1, len(v) // 4)
            put(f"D_time|L{l}|{k}|mean", i, np.nanmean(v)); put(f"D_time|L{l}|{k}|time_slope", i, trend(v))
            put(f"D_time|L{l}|{k}|late_minus_early", i, np.nanmean(v[-q:]) - np.nanmean(v[:q]))
    print(f"row {i} done", flush=True)
pickle.dump(F, open(A / "exp38_features.pkl", "wb"))

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
    for m in np.unique(model[rows]):
        r = rows[model[rows] == m]; x = v[r]
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
    out.append(dict(feature=nm, part=nm.split("|")[0], rows=len(r), pooled=roc_auc_score(y[r], z[r]), controlled=a_c, p_controlled=p_c))
T = pd.DataFrame(out); T.to_csv(A / "exp38_depth_and_tda_dynamics.csv", index=False)
pd.set_option("display.width", 220)
print("\nAUC > 0.5: higher value -> robust.")
for part, g in T.groupby("part"):
    print(f"\n=== {part}: {len(g)} features, controlled p<.05: {(g.p_controlled < .05).sum()} (~{.05 * len(g):.1f} by chance)")
    print(g.assign(d=(g.controlled - .5).abs()).sort_values("d", ascending=False).head(8)[["feature", "rows", "pooled", "controlled", "p_controlled"]].round(3).to_string(index=False))
print(f"\n{len(T)} features; controlled p<.05: {(T.p_controlled < .05).sum()} (~{.05 * len(T):.0f} by chance)")
