"""exp47b: for exp47's olympiad features, (1) the AUC after removing the shortcuts (finished by 6144, -log length), by regressing each
feature on them inside each model and scoring the residual; (2) the same features recomputed on main_v2 at the same 6144 cap, from our
16k extraction (window states, token entropy, text, lookback files), so each family's direction can be compared across the two sets."""
import json, os, pickle, re, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "2"
import numpy as np
from pathlib import Path
sys.path.insert(0, "/home/sabyasachi19/PycharmProjects/AimoAGAIN/analysis"); sys.path.insert(0, "/home/sabyasachi19/PycharmProjects/AimoAGAIN/submission_core")
from exp47_olympiad_all_ideas import within, perm, MARK, cos                            # noqa: E402  (re-runs exp47 quietly)
from common import *                                                                    # noqa: E402,F401
from spurious_features import f_check_share                                             # noqa: E402

D = pickle.load(open(A / "exp47_features.pkl", "rb")); F, FAM, y, model, indep = D["F"], D["FAM"], D["y"], D["model"], D["indep"]
fin, ln = F["0 shortcuts|finished by 6144"], F["0 shortcuts|-log trace length"]


def resid(v):
    out = np.full(len(v), np.nan)
    for m in np.unique(model):
        k = (model == m) & np.isfinite(v); C = np.c_[np.ones(k.sum()), fin[k], ln[k]]
        out[k] = v[k] - C @ np.linalg.lstsq(C, v[k], rcond=None)[0]
    return out


# main_v2 at 6144: the features that can be rebuilt from our extraction
allm, _ = load_all(); rows = datasets(allm)["main_v2"]; ym = allm.y.values[rows].astype(int); mm = allm.model.values[rows]
P2 = ROOT / "extraction_out_generations_p2" / "extraction_out_generations"; LB = ROOT / "extraction_out_generations_pearl_more" / "extraction_out_generations"
GEN = {}
for f in (ROOT / "extraction_out").glob("*/generations.jsonl"):
    for l in open(f):
        d = json.loads(l); GEN[d["id"]] = d["gen_ids"]
from spurious_inference import load_tokenizer                                           # noqa: E402
TOK = {m: load_tokenizer(HF[m]) for m in MODELS}
M = {}
for j, i in enumerate(rows):
    m, rid = allm.model.values[i], allm.id.iloc[i]; g = GEN[rid][:6144]; text = TOK[m].decode(g, skip_special_tokens=True)
    z = np.load(P2 / m / "tf" / f"{rid}.npz"); ws = z["win_states"][: -(-len(g) // 16)].astype(np.float32); ent = z["gen_entropy"].astype(np.float32)[: len(g)]
    sp = np.linalg.norm(np.diff(ws[:, 3], axis=0), axis=1); words = max(1, len(text.split()))
    vals = {"2 states|speed variation (CV)": sp.std() / (sp.mean() + 1e-9),
            "4 output|entropy bursts: share of tokens > 2 nats (-)": -np.mean(ent > 2), "4 output|token entropy mean (-)": -ent.mean(),
            "6 text|check share": f_check_share(text)}
    for k, pat in MARK.items():
        vals[f"6 text|{k} per 1k words"] = 1000 * len(re.findall(pat, text, flags=re.I)) / words
    p = LB / m / "lookback" / f"{rid}.npz"
    if p.exists():
        zl = np.load(p); lb = zl["lookback"].astype(np.float32)[: -(-len(g) // 16)]; q = max(1, len(lb) // 4)
        vals["3 attention|lookback late - early"] = lb[-q:].mean() - lb[:q].mean(); vals["3 attention|prompt attention mass (mean)"] = zl["prompt_mass"].astype(np.float32)[: len(lb)].mean()
        vals["3 attention|lookback ratio, last quarter"] = lb[-q:].mean()
    for k, v in vals.items():
        M.setdefault(k, np.full(len(rows), np.nan))[j] = v
F28 = pickle.load(open(A / "exp28_features_cache.pkl", "rb")); F44 = pickle.load(open(A / "exp44_features.pkl", "rb")); F32 = pickle.load(open(A / "exp32_features.pkl", "rb"))
M["4 output|forced p1 @6144"] = F28["meta:fa_b6144_B_p1"][rows]; M["4 output|confidence slope over budgets"] = F44["forced|p1 slope over budgets"][rows]
M["4 output|share of saturated budgets"] = F44["forced|share of budgets saturated"][rows]; M["6 graphs+TDA|kNN global efficiency"] = F32["step64|k5|4k|global_efficiency"][rows]
M["2 states|late-vs-early cosine"] = F44["trajectory|late-vs-early cosine (6k)"][rows]; M["4 output|A/B entropy shift (max)"] = F28["ab:dent_max"][rows]
M["0 shortcuts|-log trace length"] = -np.log(np.minimum(allm.n_gen.values[rows], 6144).astype(float))

print(f"{'feature':52s} {'olympiad indep':>14s} {'controlled':>12s} {'main_v2 @6144 (p)':>18s}")
keys = [k for k in F if np.isfinite(F[k]).sum() >= 30]
out = []
for k in keys:
    w, p = perm(F[k][indep], y[indep], model[indep], 1000)
    if p >= .05 and k not in M: continue
    rv = resid(F[k]); wc, pc = perm(rv[indep], y[indep], model[indep], 1000)
    mv = perm(M[k], ym, mm, 1000) if k in M else (np.nan, np.nan)
    out.append((k, w, p, wc, pc, *mv))
for k, w, p, wc, pc, a, b in sorted(out, key=lambda t: -abs(t[1] - .5)):
    print(f"{k:52s} {w:6.2f} ({p:.3f}) {wc:5.2f} ({pc:.3f}) {a:8.2f} ({b:.3f})")
pickle.dump(dict(M=M, y=ym, model=mm), open(A / "exp47b_mainv2_values.pkl", "wb"))
