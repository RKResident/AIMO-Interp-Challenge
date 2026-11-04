"""exp47c: every built submission recipe (current cap-6144 builds), scored on the olympiad rows exactly as the submission scores one
problem per call: score_trace / decide_absolute with the recipe's own main_v2 references and threshold (per-model split inactive).
Feature values: ours from the Vast run (greedy, cap 6144); PEARL from the friend's files (same 5 rates x 4 seeds as the submission);
cluster cycle rank with the recipe's own K-means centroids. Also a few candidate recipes built the same way (main_v2 references at 6144
from exp47b's rebuild), for the suggestions."""
import json, os, pickle, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "2"
import numpy as np
from pathlib import Path
sys.path.insert(0, "/home/sabyasachi19/PycharmProjects/AimoAGAIN/submission_core")
from spurious_features import score_trace, decide_absolute, f_cluster_cycle_rank, f_pearl_area

ROOT = Path("/home/sabyasachi19/PycharmProjects/AimoAGAIN"); SOL = Path("/home/sabyasachi19/PycharmProjects/aimo/getting-started/solutions")
D = pickle.load(open(ROOT / "analysis" / "exp47_features.pkl", "rb")); F, y, model, indep, ids = D["F"], D["y"], D["model"], D["indep"], D["ids"]
OURS = ROOT / "extra_olympiad_problems_dataset" / "results_olympiad_features" / "out_t1"; FR = ROOT / "extra_olympiad_problems_dataset" / "olympiad-features"
rows = {json.loads(l)["id"]: json.loads(l) for f in sorted(OURS.glob("*/rows_shard*.jsonl")) for l in open(f)}
slug = lambda m: m.replace("/", "__")                                                    # noqa: E731
V = {f: np.full(len(ids), np.nan) for f in ("fa", "ge", "tda", "chk", "len", "ab", "seg", "traj", "pcr", "fas", "fsat", "pearl", "cr")}
for i, rid in enumerate(ids):
    g = rows[rid]["greedy"]
    for f in ("fa", "ge", "tda", "chk", "len", "ab", "seg", "traj", "pcr", "fas", "fsat"):
        V[f][i] = g[f] if g.get(f) is not None else np.nan
    p = FR / slug(model[i]) / "pearl" / f"{rid}.npz"
    if p.exists():
        z = np.load(p, allow_pickle=True); k = z["kind"]; ms = z["mean_state"].astype(np.float32)
        V["pearl"][i] = f_pearl_area(np.concatenate([ms[k == "original"], ms[k == "noise"]]))
LB = {"forced-confidence": "0.57", "length": "0.50", "tda": "0.57 (16k)", "top5-knn": "0.35 (16k)", "confidence-cascade": "0.50 single / 0.71 batched",
      "confidence-knn": "0.50", "cluster-graph": "0.28", "shift-robust": "0.43", "abs-forced-confidence": "0.28", "abs-knn": "0.43",
      "no-forcing": "0.43 (16k, 28% cut)", "confidence-knn-tda": "0.43 (16k, 28% cut)", "confidence-checkshare-length": "pending"}


def evaluate(rec, vals):
    pred, score = np.zeros(len(ids), bool), np.full(len(ids), np.nan)
    for i in range(len(ids)):
        v = {f: vals[f][i] for f in rec["features"]}
        if rec.get("abs_feature"):
            pred[i] = decide_absolute(v, rec); score[i] = v[rec["abs_feature"]]
        else:
            score[i] = score_trace(v, rec, model[i]); pred[i] = score[i] > rec["threshold"]
    out = {}
    for nm, k in (("all", np.ones(len(ids), bool)), ("indep", indep)):
        p, yy = pred[k], y[k] == 1
        out[nm] = (np.mean(p == yy), 0.5 * (p[yy].mean() + (~p[~yy]).mean()), int(p.sum()), int(k.sum()))
    return out


print(f"{'recipe':34s} {'features':28s} {'leaderboard':>26s} {'olympiad all: acc / bal':>24s} {'independent 49: acc / bal (robust calls)':>40s}")
for d in sorted(SOL.glob("spurious-*")):
    rp = d / "spurious_artifacts" / "recipe.json"
    if not rp.exists() or "mv2rate" in d.name: continue
    rec = json.load(open(rp)); name = d.name.removeprefix("spurious-"); vals = dict(V)
    if "cr" in rec["features"]:
        vals["cr"] = np.array([f_cluster_cycle_rank(np.load(OURS / slug(model[i]) / f"{ids[i]}.npz")["t0_ws16"][:, 3].astype(np.float32),
                                                    np.load(d / "spurious_artifacts" / f"mu_{slug(model[i])}.npy"),
                                                    np.load(d / "spurious_artifacts" / f"km64_{slug(model[i])}.npy"), rows[ids[i]]["greedy"]["L"])
                               for i in range(len(ids))])
    r = evaluate(rec, vals); a, b = r["all"], r["indep"]
    print(f"{name:34s} {','.join(rec['features']):28s} {LB.get(name, '-'):>26s} {a[0]:10.2f} / {a[1]:.2f} {b[0]:18.2f} / {b[1]:.2f} ({b[2]}/{b[3]})")
pickle.dump(V, open(ROOT / "analysis" / "exp47c_values.pkl", "wb"))
