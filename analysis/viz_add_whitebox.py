"""Adds the white-box run's per-problem results to analysis/viz/explorer_data.json (field 'wb' per row), for the explorer's white-box panels:
  ablation   the clean forced answer's probability (relative to the clean run) after deleting reasoning: no reasoning, first 1024 tokens only,
             middle half deleted, last 1024 tokens only; plus flips
  rewording  per reworded prompt (friend's 5, automatic 3): answer kept / flip / KL, and along the reasoning (64-token steps) the change of
             the trace's log-likelihood and the hidden-state cosine to the original (mean over the 6 saved layers)
  source     gradient x input attribution of the answer: per prompt token (with its text, numbers flagged) and per 64-token reasoning step;
             direct logit attribution per layer (attention, MLP); logit-lens probability of the answer per layer
  noise      flip rate of the answer under residual noise, per depth x strength (mean over seeds); MLP band ablation"""
import json, os, sys
import numpy as np
sys.path.insert(0, "/home/sabyasachi19/PycharmProjects/AimoAGAIN/analysis"); sys.path.insert(0, "/home/sabyasachi19/PycharmProjects/AimoAGAIN/submission_core")
from common import *                                                                    # noqa: E402,F401
from spurious_inference import load_tokenizer                                           # noqa: E402

WB = ROOT / "results_whitebox" / "out_whitebox"; slug = lambda m: m.replace("/", "__")  # noqa: E731
P = A / "viz" / "explorer_data.json"; data = json.load(open(P))
rows = {json.loads(l)["id"]: json.loads(l) for f in WB.glob("*/rows_shard*.jsonl") for l in open(f)}
TOK = {}


def step64(v, agg="mean"):
    v = np.asarray(v, np.float64); n = len(v) // 64 * 64
    if not n: return []
    b = v[:n].reshape(-1, 64); return [round(float(x), 4) for x in (b.mean(1) if agg == "mean" else b.sum(1))]


for rec in data["rows"]:
    r = rows.get(rec["id"])
    if r is None: continue
    m = r["model_id"]
    if m not in TOK: TOK[m] = load_tokenizer(m)
    z = np.load(WB / slug(m) / f"{r['id']}.npz"); c = r["clean"]; p1 = max(c["p1"], 1e-9)
    wb = {"clean_p1": round(c["p1"], 4), "clean_answer": c["answer"][:24],
          "ablation": [{"name": k, "kept": round(v["p_clean_answer"] / p1, 4), "flip": bool(v["flip"]), "answer": v["answer"][:16]} for k, v in r["ablation"].items()]}
    rw = []
    for k, v in enumerate(r["counterfactual"]):
        sc = z[f"cf{k}_state_cos"].astype(np.float64).mean(1)                                # per 16-token window, mean over layers
        nsc = len(sc) // 4 * 4
        rw.append({"kind": v["kind"], "kept": round(v["p_clean_answer"] / p1, 4), "flip": bool(v["flip"]), "kl": round(v["kl_from_clean"], 4),
                   "answer": v["answer"][:16], "dll": step64(z[f"cf{k}_dlogprob"]),
                   "cos": [round(float(x), 5) for x in (sc[:nsc].reshape(-1, 4).mean(1) if nsc else sc)]})
    wb["rewording"] = rw
    att = z["attribution"].astype(np.float64); isn = z["prompt_is_number"].astype(bool); n_p = len(isn); cut = r["attribution_cut"] or r["cut"]
    pa = att[:n_p]; span = z["problem_span"]; lo, hi = (int(span[0]), int(span[1])) if span[0] >= 0 else (0, n_p)
    toks = [TOK[m].decode([int(t)]) for t in z["prompt_ids"][lo:hi]]
    pmax = pa[lo:hi].max() if hi > lo else 1.0
    wb["prompt_attr"] = [[t, round(float(a / (pmax + 1e-12)), 3), bool(n)] for t, a, n in zip(toks, pa[lo:hi], isn[lo:hi])]
    ta = att[n_p:n_p + cut]; s64 = step64(ta, "sum"); tot = sum(s64) or 1.0
    wb["trace_attr"] = [round(v / tot, 4) for v in s64]
    wb["prompt_share"] = round(float(pa.sum() / (att.sum() + 1e-12)), 4)
    wb["dla_attn"] = [round(float(x), 3) for x in z["dla_attn"]]; wb["dla_mlp"] = [round(float(x), 3) for x in z["dla_mlp"]]
    wb["lens_p"] = [round(float(x), 4) for x in z["lens_p"]]
    nz = z["noise"]; grid = []
    for d in (0.25, 0.5, 0.75):
        grid.append([round(float(nz[(np.isclose(nz[:, 0], d)) & (np.isclose(nz[:, 1], a)), 4].mean()), 3) for a in (0.05, 0.1, 0.2, 0.4)])
    wb["noise_flip"] = grid
    wb["mlp_ablation"] = [{"band": a["band"], "scale": a["scale"], "kept": round(a["p_clean_answer"] / p1, 4)} for a in r["mlp_ablation"]]
    rec["wb"] = wb
json.dump(data, open(P, "w"), separators=(",", ":"))
print(sum("wb" in r for r in data["rows"]), "rows with white-box data | file", round(os.path.getsize(P) / 1e6, 2), "MB")
