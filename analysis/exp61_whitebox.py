"""exp61: the white-box Vast run (results_whitebox; 118 labelled rows: main_v2 43 + olympiad 75, greedy traces capped at 6144, cut =
end of thinking or 6144). Scalar features per family, scored as usual: within-model AUC on all rows and on the UNFINISHED rows (focus),
after regressing out finished / -log length, permutation p; and the direction on main_v2 vs on the olympiad rows independent of main_v2.
Families:
  counterfactual  the same trace teacher-forced after reworded prompts (friend's 5 rewrites; 3 automatic ones): drop of the clean
                  answer's probability, flips, KL, change of the trace's log-likelihood (all / last quarter), state cosine, MLP neuron
                  overlap at the answer - averaged over the friend's variants and over the automatic ones
  noise           residual-stream noise at the answer: mean probability of the clean answer kept, flip rate (all, and at 50% depth)
  MLP             direct logit attribution (MLP share of the answer logit, late-third share, concentration), MLP write ratio / top-1%
                  activation mass / activation entropy along the reasoning (late layers, late-minus-early), MLP band ablation
  attribution     gradient x input: share on the prompt, on its number tokens (per token, relative to all prompt tokens), on the last
                  10% of the trace, concentration over the trace
  coverage        attention from the reasoning onto the prompt's number tokens vs its prose; the least-read number
  reasoning graph sentence-level attention graph (head mean): attention-weighted jump length, share on the previous 1-2 sentences, share
                  on sentences > 20 back, hub concentration (Gini / top-3 share of incoming attention), prompt share - per layer third
  ablation        the forced answer after deleting reasoning: kept probability for each deletion
  logit lens      the depth at which the final answer becomes top-1 and stays"""
import json, os, pickle, re, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "2"
import numpy as np
import pandas as pd
sys.path.insert(0, "/home/sabyasachi19/PycharmProjects/AimoAGAIN/analysis")
from common import *                                                                    # noqa: E402,F401

WB = ROOT / "results_whitebox" / "out_whitebox"; slug = lambda m: m.replace("/", "__")  # noqa: E731
rows = [json.loads(l) for f in sorted(WB.glob("*/rows_shard*.jsonl")) for l in open(f)]
D56 = pd.read_csv(A / "exp56_rows.csv").set_index("id")
norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()[:300]                            # noqa: E731
inp = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "t1_run" / "inputs.jsonl")}


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


F, meta = [], []
for r in rows:
    z = np.load(WB / slug(r["model_id"]) / f"{r['id']}.npz"); c = r["clean"]; f = {}
    for grp, sel in (("friend", lambda k: k.startswith("friend")), ("auto", lambda k: k.startswith("auto"))):
        cf = [x for x in r["counterfactual"] if sel(x["kind"])]
        if not cf: continue
        f[f"cf {grp}: clean-answer prob kept"] = np.mean([x["p_clean_answer"] / max(c["p1"], 1e-9) for x in cf])
        f[f"cf {grp}: flip rate (-)"] = -np.mean([x["flip"] for x in cf]); f[f"cf {grp}: KL from clean (-)"] = -np.mean([x["kl_from_clean"] for x in cf])
        f[f"cf {grp}: trace log-lik change"] = np.mean([x["dlogprob_mean"] for x in cf]); f[f"cf {grp}: trace log-lik change, last quarter"] = np.mean([x["dlogprob_last_quarter"] for x in cf])
        f[f"cf {grp}: state cosine"] = np.mean([np.mean(x["state_cos_mean"]) for x in cf]); f[f"cf {grp}: state cosine, last quarter"] = np.mean([np.mean(x["state_cos_last_quarter"]) for x in cf])
        jac = [np.array(x["mlp_neuron_jaccard"]) for x in cf if "mlp_neuron_jaccard" in x]
        if jac:
            Lm = len(jac[0]); f[f"cf {grp}: MLP neuron overlap"] = np.mean([j.mean() for j in jac]); f[f"cf {grp}: MLP neuron overlap, late layers"] = np.mean([j[2 * Lm // 3:].mean() for j in jac])
    nz = z["noise"]; f["noise: clean-answer prob kept"] = float(np.mean(nz[:, 3]) / max(c["p1"], 1e-9)); f["noise: flip rate (-)"] = -float(nz[:, 4].mean())
    k = np.isclose(nz[:, 0], .5); f["noise @50% depth: flip rate (-)"] = -float(nz[k, 4].mean())
    da, dm = z["dla_attn"].astype(np.float64), z["dla_mlp"].astype(np.float64); L = len(dm); pos = np.clip(np.r_[da, dm], 0, None)
    f["DLA: MLP share of positive logit"] = float(np.clip(dm, 0, None).sum() / (pos.sum() + 1e-9))
    f["DLA: late-third MLP share (-)"] = -float(np.clip(dm[2 * L // 3:], 0, None).sum() / (pos.sum() + 1e-9))
    f["DLA: concentration (Gini over components)"] = gini(pos)
    for nm, key in (("MLP write ratio", "att_mlp_write_ratio"), ("MLP top-1% activation mass", "att_mlp_top1pct_mass"), ("MLP activation entropy", "att_mlp_act_entropy")):
        Wm = z[key].astype(np.float64); Lw = Wm.shape[0]; late = Wm[2 * Lw // 3:]; q = max(1, Wm.shape[1] // 4)
        f[f"{nm}, late layers"] = float(np.nanmean(late)); f[f"{nm}, late layers: last - first quarter"] = float(np.nanmean(late[:, -q:]) - np.nanmean(late[:, :q]))
    for a in r["mlp_ablation"]:
        f[f"MLP ablation {a['band']} x{a['scale']}: answer kept"] = a["p_clean_answer"] / max(c["p1"], 1e-9)
    att = z["attribution"].astype(np.float64); isn = z["prompt_is_number"].astype(bool); n_p = len(isn); cut = r["attribution_cut"] or r["cut"]
    pa, ta = att[:n_p], att[n_p:n_p + cut]; tot = att.sum() + 1e-12
    f["attribution: prompt share"] = float(pa.sum() / tot)
    f["attribution: numbers vs other prompt tokens (per token)"] = float((pa[isn].mean() if isn.any() else np.nan) / (pa[~isn].mean() + 1e-12))
    f["attribution: last 10% of trace share"] = float(ta[-max(1, len(ta) // 10):].sum() / (ta.sum() + 1e-12)); f["attribution: concentration over trace (Gini)"] = gini(ta)
    cov = z["att_cov_layer"].astype(np.float64).mean(0)
    if isn.any():
        f["coverage: numbers vs prose (per token)"] = float(cov[isn].mean() / (cov[~isn].mean() + 1e-12)); f["coverage: least-read number / mean number"] = float(cov[isn].min() / (cov[isn].mean() + 1e-12))
    f.update(graph_feats(z["att_graph_sentence_layer"], "sentence graph"))
    for nm, a in r["ablation"].items():
        f[f"reasoning ablation '{nm}': answer kept"] = a["p_clean_answer"] / max(c["p1"], 1e-9)
    rk = z["lens_rank"]; Lr = len(rk); stay = next((l for l in range(Lr) if (rk[l:] == 0).all()), Lr)
    f["logit lens: depth where answer becomes top-1 for good (-)"] = -stay / Lr
    F.append(f); d = D56.loc[r["id"]]
    meta.append(dict(id=r["id"], model=r["model_id"], dataset=r["dataset"], y=int(r["robust"]), unf=not bool(d.finished), indep=r["dataset"] == "main_v2" or str(d.indep) == "True",
                     length=-np.log(max(1, r["n_gen"])), group=norm(inp[r["id"]]["problem"])))
X = pd.DataFrame(F); Mt = pd.DataFrame(meta); y = Mt.y.values; mm = Mt.model.values; unf = Mt.unf.values.astype(bool)
rng = np.random.default_rng(0)


def within(x, yy, m_):
    num = den = 0
    for m in np.unique(m_):
        k = (m_ == m) & np.isfinite(x); p, n = x[k & (yy == 1)], x[k & (yy == 0)]
        num += (p[:, None] > n[None]).sum() + .5 * (p[:, None] == n[None]).sum(); den += len(p) * len(n)
    return num / den if den else np.nan


def perm(x, yy, m_, n=1000):
    w = within(x, yy, m_); nl = np.empty(n)
    for t in range(n):
        yp = yy.copy()
        for m in np.unique(m_): k = m_ == m; yp[k] = rng.permutation(yy[k])
        nl[t] = within(x, yp, m_)
    return w, float(np.mean(np.abs(nl - .5) >= abs(w - .5)))


def resid(x, mask):
    out = np.full(len(x), np.nan)
    for m in np.unique(mm[mask]):
        k = mask & (mm == m) & np.isfinite(x)
        if k.sum() < 4: continue
        C = np.c_[np.ones(k.sum()), Mt.length.values[k], (~unf[k]).astype(float)]; out[k] = x[k] - C @ np.linalg.lstsq(C, x[k], rcond=None)[0]
    return out


res = []
for c in X.columns:
    x = X[c].values.astype(float)
    if np.isfinite(x).sum() < 30 or np.nanstd(x) == 0: continue
    a, pa = perm(x, y, mm); u, pu = perm(x[unf], y[unf], mm[unf]); cu, pcu = perm(resid(x, unf)[unf], y[unf], mm[unf])
    mv = Mt.dataset.values == "main_v2"; oi = (Mt.dataset.values == "olympiad") & Mt.indep.values
    res.append(dict(feature=c, auc_all=a, p_all=pa, auc_unf=u, p_unf=pu, auc_unf_ctrl=cu, p_unf_ctrl=pcu,
                    auc_mainv2=within(x[mv], y[mv], mm[mv]), auc_oly_indep=within(x[oi], y[oi], mm[oi])))
R = pd.DataFrame(res); R["strength"] = (R.auc_unf - .5).abs(); R = R.sort_values("strength", ascending=False)
R["consistent"] = np.sign(R.auc_mainv2 - .5) == np.sign(R.auc_oly_indep - .5)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
print(f"{len(X)} rows ({int(unf.sum())} unfinished) | {len(R)} features | p < 0.05 on unfinished: {int((R.p_unf < .05).sum())} (chance ~{.05 * len(R):.1f}) | "
      f"on unfinished after controls: {int((R.p_unf_ctrl < .05).sum())}\n")
print(R[["feature", "auc_unf", "p_unf", "auc_unf_ctrl", "p_unf_ctrl", "auc_all", "p_all", "auc_mainv2", "auc_oly_indep", "consistent"]].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
R.to_csv(A / "exp61_whitebox_features.csv", index=False); pickle.dump(dict(X=X, meta=Mt), open(A / "exp61_whitebox.pkl", "wb"))
