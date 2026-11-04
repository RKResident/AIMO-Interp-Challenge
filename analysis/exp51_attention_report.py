"""exp51: the attention feature class on both labelled sets, same definitions (extract_features.py) and comparable cutoffs.
  lookback (Lookback Lens, teacher forced, per 16 generated tokens, every softmax layer and head): mean ratio, last quarter, late - early,
      prompt attention mass, each over the first 6144 generated tokens (main_v2's 16k files truncated: the ratios are causal, so exact);
  LapEigvals (attention-graph Laplacian, every layer / head): mean and top-1 eigenvalue at the 2048-token cutoff, without and with the
      forced suffix B;
  TOHA (prompt-response spanning forest per head): window and answer totals, share of answer tokens linked straight to the prompt, at
      the first cutoff (1024 tokens in both sets);
  extras at the forced suffix (first cutoff, 1024): attention FFT low-frequency share, KL between adjacent layers / head vs layer mean,
      lookback at the suffix.
Olympiad: all 75 rows, the 49 independent of main_v2, and the 49 after regressing out finished-by-6144 and -log length; main_v2: 43 rows
(TOHA / extras / LapEigvals exist for a subset). Within-model AUC (higher -> robust), label permutation inside each model."""
import glob, json, os, pickle, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "2"
import numpy as np
from pathlib import Path
sys.path.insert(0, "/home/sabyasachi19/PycharmProjects/AimoAGAIN/analysis")
from common import *                                                                    # noqa: E402,F401

OLYA = ROOT / "extra_olympiad_problems_dataset" / "results_olympiad_attention" / "out_attn"; OLYF = ROOT / "extra_olympiad_problems_dataset" / "olympiad-features"
OURS = ROOT / "extra_olympiad_problems_dataset" / "results_olympiad_features" / "out_t1"
MV = {"lookback": ROOT / "extraction_out_generations_pearl_more" / "extraction_out_generations", "extra": ROOT / "extraction_out_generations_pearl_more" / "extraction_out_generations",
      "toha": ROOT / "extraction_out_generations_p2" / "extraction_out_generations", "attn": ROOT / "extraction_out"}
slug = lambda m: m.replace("/", "__")                                                    # noqa: E731
rng = np.random.default_rng(0)


def feats(lookback=None, attn=None, toha=None, extra=None, n_gen=6144):
    out = {}
    if lookback is not None:
        z = np.load(lookback); nw = -(-min(int(z["n_gen"]), n_gen) // int(z["span"])); lb = z["lookback"].astype(np.float32)[:nw]
        pm = z["prompt_mass"].astype(np.float32)[:nw]; q = max(1, len(lb) // 4)
        out.update({"lookback ratio (mean)": lb.mean(), "lookback ratio, last quarter": lb[-q:].mean(), "lookback late - early": lb[-q:].mean() - lb[:q].mean(),
                    "prompt attention mass (mean)": pm.mean(), "prompt mass late - early": pm[-q:].mean() - pm[:q].mean()})
    if attn is not None:
        z = np.load(attn); ce = list(z["cut_eff"]); j = ce.index(2048) if 2048 in ce else int(np.argmin(np.abs(np.array(ce) - 2048)))
        vB = list(z["variant_names"]).index("B")
        out.update({"LapEigvals mean, no forcing @2048": np.nanmean(z["mean_noforce"][j].astype(np.float32)),
                    "LapEigvals mean, forced B @2048": np.nanmean(z["mean_force"][j, vB].astype(np.float32)),
                    "LapEigvals top-1, no forcing @2048 (-)": -np.nanmean(z["topk_noforce"][j, :, :, 0].astype(np.float32))})
    if toha is not None:
        z = np.load(toha, allow_pickle=True); sn = list(z["stat_names"])
        out.update({"TOHA window total @1024 (-)": -np.nanmean(z["win"][0, :, :, sn.index("total")]),
                    "TOHA answer total @1024 (-)": -np.nanmean(z["ans"][0, 0, :, :, sn.index("total")]),
                    "TOHA answer: share linked straight to prompt": np.nanmean(z["ans"][0, 0, :, :, sn.index("frac_direct_to_prompt")])})
    if extra is not None:
        z = np.load(extra, allow_pickle=True); vB = list(z["variant_names"]).index("B"); fft = z["fft"][0, vB].astype(np.float32)
        out.update({"attention FFT low-frequency share @1024": np.nanmean(fft[..., :8].sum(-1) / (fft.sum(-1) + 1e-9)),
                    "KL adjacent layers @1024": np.nanmean(z["kl_adj"][0, vB]), "KL head vs layer mean @1024": np.nanmean(z["kl_headmean"][0, vB]),
                    "lookback at forced suffix @1024": np.nanmean(z["lookback"][0, vB])})
    return out


# olympiad
D = pickle.load(open(A / "exp47_features.pkl", "rb")); oid, oy, om, oind = D["ids"], D["y"], D["model"], D["indep"]
fin, ln = D["F"]["0 shortcuts|finished by 6144"], D["F"]["0 shortcuts|-log trace length"]
O = {}
for i, (rid, m) in enumerate(zip(oid, om)):
    p = lambda d, kind: d / slug(m) / kind / f"{rid}.npz"                              # noqa: E731
    f = feats(lookback=p(OLYA, "lookback"), attn=p(OLYA, "attn"), toha=p(OLYF, "toha") if p(OLYF, "toha").exists() else None,
              extra=p(OLYF, "extra") if p(OLYF, "extra").exists() else None)
    for k, v in f.items(): O.setdefault(k, np.full(len(oid), np.nan))[i] = v
# main_v2
allm, _ = load_all(); rows = datasets(allm)["main_v2"]; my = allm.y.values[rows].astype(int); mm = allm.model.values[rows]
M = {}
for j, i in enumerate(rows):
    m, rid = allm.model.values[i], allm.id.iloc[i]; ng = int(allm.n_gen.values[i])
    p = {k: MV[k] / m / k / f"{rid}.npz" for k in MV}
    f = feats(**{k: (v if v.exists() else None) for k, v in p.items()}, n_gen=min(ng, 6144))   # the submission cap, like the olympiad traces
    for k, v in f.items(): M.setdefault(k, np.full(len(rows), np.nan))[j] = v


def within(x, yy, mm_):
    num = den = 0
    for m in np.unique(mm_):
        k = (mm_ == m) & np.isfinite(x); p, n = x[k & (yy == 1)], x[k & (yy == 0)]
        num += (p[:, None] > n[None]).sum() + .5 * (p[:, None] == n[None]).sum(); den += len(p) * len(n)
    return num / den if den else np.nan


def perm(x, yy, mm_, n=2000):
    w = within(x, yy, mm_); nl = np.empty(n)
    for t in range(n):
        yp = yy.copy()
        for m in np.unique(mm_): k = mm_ == m; yp[k] = rng.permutation(yy[k])
        nl[t] = within(x, yp, mm_)
    return w, float(np.mean(np.abs(nl - .5) >= abs(w - .5)))


def resid(v):
    out = np.full(len(v), np.nan)
    for m in np.unique(om):
        k = (om == m) & np.isfinite(v); C = np.c_[np.ones(k.sum()), fin[k], ln[k]]; out[k] = v[k] - C @ np.linalg.lstsq(C, v[k], rcond=None)[0]
    return out


print(f"{'feature':46s} {'olympiad all':>14s} {'indep 49':>14s} {'controlled':>14s} {'main_v2 (n)':>20s}")
for k in O:
    a = perm(O[k], oy, om); b = perm(O[k][oind], oy[oind], om[oind]); c = perm(resid(O[k])[oind], oy[oind], om[oind])
    mv = M.get(k); d = perm(mv, my, mm) if mv is not None and np.isfinite(mv).sum() >= 10 else (np.nan, np.nan); nmv = int(np.isfinite(mv).sum()) if mv is not None else 0
    print(f"{k:46s} {a[0]:6.2f} ({a[1]:.3f}) {b[0]:6.2f} ({b[1]:.3f}) {c[0]:6.2f} ({c[1]:.3f}) {d[0]:7.2f} ({d[1]:.3f}) n{nmv:2d}")
pickle.dump(dict(O=O, M=M), open(A / "exp51_attention.pkl", "wb"))
