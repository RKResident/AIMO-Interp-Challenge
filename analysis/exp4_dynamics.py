"""Exp 4: dynamics over ALL cutoffs (earlier runs used 4): hidden-state drift, answer-commitment curve, entropy trajectory, Laplacian/forcing-shock curves."""
from common import *
from scipy.stats import spearmanr
allm, _ = load_all()
labs_b = ["b512", "b1024", "b2048", "b3072", "b4096", "b6144", "b8192"]
rows = []
for k, r in enumerate(allm.itertuples()):
    fj = force_json(r.model, r.id); tf = np.load(OUT / r.model / "tf" / f"{r.id}.npz")
    L = max(fj["L"], 1); d = {}
    ce = list(tf["cut_eff"]); order = np.argsort(ce); cs = np.array(ce)[order]
    # ---- hidden-state dynamics per sampled layer (6 aligned layers)
    S = tf["cut_last"].astype(np.float32)[order]                      # [n_cut,6,d]
    pm = tf["problem_mean"].astype(np.float32)                         # [6,d]
    for j in range(S.shape[1]):
        X = S[:, j]; nx = np.linalg.norm(X, axis=-1) + 1e-9
        cosc = (X[1:] * X[:-1]).sum(-1) / (nx[1:] * nx[:-1]) if len(X) > 1 else np.array([np.nan])
        step = np.linalg.norm(np.diff(X, axis=0), axis=-1) if len(X) > 1 else np.array([np.nan])
        net = np.linalg.norm(X[-1] - X[0]) + 1e-9
        d[f"h{j}_cos_mean"] = np.nanmean(cosc); d[f"h{j}_cos_min"] = np.nanmin(cosc)
        d[f"h{j}_tortuosity"] = np.nansum(step) / net
        d[f"h{j}_path_per_norm"] = np.nansum(step) / nx.mean()
        cp = (X * pm[j]).sum(-1) / (nx * (np.linalg.norm(pm[j]) + 1e-9))
        d[f"h{j}_cos_prob_mean"] = cp.mean(); d[f"h{j}_cos_prob_slope"] = np.polyfit(np.arange(len(cp)), cp, 1)[0] if len(cp) > 2 else np.nan
        d[f"h{j}_norm_slope"] = np.polyfit(np.arange(len(nx)), np.log(nx), 1)[0] if len(nx) > 2 else np.nan
    # ---- commitment curve on the final answer's first token (variant B) and entropy trajectory
    cuts = sorted(set(fj["labels"].values()))
    for var in "AB":
        res = [fj["results"].get(f"{c}|{var}") for c in cuts]
        fin = res[-1]["top"][0]["ids"][0] if res[-1] and res[-1]["top"] else None
        p = np.array([next((pp for ii, pp in zip(rs["top"][0]["ids"], rs["top"][0]["probs"]) if ii == fin), 0.0) if rs and rs["top"] else np.nan for rs in res])
        ent = np.array([rs["first_entropy"] if rs else np.nan for rs in res])
        top1 = np.array([rs["top"][0]["ids"][0] == fin if rs and rs["top"] else np.nan for rs in res], float)
        cf = np.array(cuts) / L
        d[f"c{var}_p_first"] = p[0]; d[f"c{var}_p_mean"] = np.nanmean(p); d[f"c{var}_p_area"] = np.trapz(np.nan_to_num(p), cf) if len(cf) > 1 else np.nan
        hit = np.where(p > .5)[0]; d[f"c{var}_t50"] = cf[hit[0]] if len(hit) else 1.0
        d[f"c{var}_mono"] = spearmanr(np.arange(len(p)), p)[0] if len(p) > 3 and np.nanstd(p) > 0 else 0.0
        d[f"c{var}_maxdrop"] = np.nanmax(np.maximum.accumulate(p) - p)
        d[f"c{var}_top1_agree"] = np.nanmean(top1)
        d[f"c{var}_ent_first"] = ent[0]; d[f"c{var}_ent_mean"] = np.nanmean(ent); d[f"c{var}_ent_last"] = ent[-1]
        d[f"c{var}_ent_slope"] = np.polyfit(np.arange(len(ent)), np.nan_to_num(ent), 1)[0] if len(ent) > 2 else np.nan
        d[f"c{var}_ent_area"] = np.trapz(np.nan_to_num(ent), cf) if len(cf) > 1 else np.nan
    # ---- Laplacian dynamics + forcing shock
    ap = OUT / r.model / "attn" / f"{r.id}.npz"
    if ap.exists():
        a = np.load(ap); ae = list(a["cut_eff"]); o2 = np.argsort(ae)
        mn = a["mean_noforce"][o2]; mf = a["mean_force"][o2].mean(1)           # [cut,L,H]
        g_no, g_f = np.nanmean(mn, (1, 2)), np.nanmean(mf, (1, 2)); shock = g_f - g_no
        nl = mn.shape[1]; thirds = np.array_split(np.arange(nl), 3)
        for nm, curve in (("no", g_no), ("force", g_f), ("shock", shock)):
            d[f"l_{nm}_first"] = curve[0]; d[f"l_{nm}_last"] = curve[-1]; d[f"l_{nm}_std"] = np.nanstd(curve)
            d[f"l_{nm}_slope"] = np.polyfit(np.arange(len(curve)), curve, 1)[0] if len(curve) > 2 else np.nan
        for ti, th in enumerate(thirds):
            c3 = np.nanmean(mf[:, th] - mn[:, th], (1, 2)); d[f"l_shock_t{ti}_mean"] = np.nanmean(c3)
            d[f"l_no_t{ti}_slope"] = np.polyfit(np.arange(len(c3)), np.nanmean(mn[:, th], (1, 2)), 1)[0] if len(c3) > 2 else np.nan
    rows.append(d)
    if (k + 1) % 60 == 0: print("processed", k + 1, "/", len(allm), flush=True)
Dy = pd.DataFrame(rows); Dy.to_pickle(A / "exp4_features.pkl")
print("=== Exp 4: dynamics ===")
single_auc(Dy, allm, top=12)
H = [c for c in Dy if c.startswith("h")]; C_ = [c for c in Dy if c.startswith("c")]; Lp = [c for c in Dy if c.startswith("l_")]
for nm, cols in (("hidden-state drift (6 layers x 6)", H), ("answer commitment + entropy curves", C_), ("Laplacian dynamics + forcing shock", Lp), ("all dynamics", H + C_ + Lp)):
    evaluate(nm, Dy[cols].values, allm, C=0.03)
