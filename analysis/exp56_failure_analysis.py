"""exp56: where do our best recipes fail on our own labelled data? The two best leaderboard recipes - confidence + check share + slope
(0.71) and the same + lookback (0.71) - scored exactly as the submission does one problem at a time (their recipe.json: main_v2
per-model references and threshold) on every labelled row: main_v2 43 (in sample) and olympiad 75 (out of sample). For every row:
correct / false robust / false spurious, and what we know about it - label source, the perturbation record behind the label (olympiad:
which of the 5 perturbations failed, by a wrong answer or by hitting the token cap; main_v2: max_drop), gold-answer correctness of the
greedy forced answer, finished or not, and the feature values. Writes analysis/exp56_rows.csv for reading the traces."""
import csv, glob, json, os, pickle, re, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "2"
import numpy as np
import pandas as pd
sys.path.insert(0, "/home/sabyasachi19/PycharmProjects/AimoAGAIN/analysis"); sys.path.insert(0, "/home/sabyasachi19/PycharmProjects/AimoAGAIN/submission_core")
from common import *                                                                    # noqa: E402,F401
from spurious_features import score_trace                                               # noqa: E402

SOL = Path("/home/sabyasachi19/PycharmProjects/aimo/getting-started/solutions")
REC = {k: json.load(open(SOL / f"spurious-{k}" / "spurious_artifacts" / "recipe.json")) for k in ("confidence-checkshare-slope", "confidence-checkshare-slope-lookback")}
D47 = pickle.load(open(A / "exp47_features.pkl", "rb")); MB = pickle.load(open(A / "exp47b_mainv2_values.pkl", "rb")); MA = pickle.load(open(A / "exp51_attention.pkl", "rb"))
OLY = ROOT / "extra_olympiad_problems_dataset"; inp = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "t1_run" / "inputs.jsonl")}
norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()                                    # noqa: E731
gold = {norm(r["problem"]): r["answer"].strip() for r in csv.DictReader(open(ROOT / "olympiad_run_package" / "olympiad" / "selected_problems.csv"))}
oprows = {json.loads(l)["id"]: json.loads(l) for f in sorted((OLY / "results_olympiad_features" / "out_t1").glob("*/rows_shard*.jsonl")) for l in open(f)}
mvrows = {json.loads(l)["id"]: json.loads(l) for f in sorted((ROOT / "results_mainv2_t1" / "out_t1").glob("*/rows_shard*.jsonl")) for l in open(f)}
# olympiad label records: per problem and model, the 5 perturbations (answer correct? hit the cap?)
LAB, PERT = {}, {}
for m in ("deepseek-ai__DeepSeek-R1-0528-Qwen3-8B", "Skywork__Skywork-OR1-Math-7B"):
    for r in csv.DictReader(open(OLY / "olympiad_labels_greedy" / m / "labels.csv")): LAB[(m, r["pid"])] = r
    for f in glob.glob(str(OLY / "olympiad_labels_greedy" / m / "samples_shard*.jsonl")):
        for l in open(f):
            d = json.loads(l)
            if d["ptype"] != "original": PERT.setdefault((m, d["pid"]), []).append(d)
sel = {norm(r["problem"]): r["pid"] for r in csv.DictReader(open(ROOT / "olympiad_run_package" / "olympiad" / "selected_problems.csv"))}
pq = pd.read_parquet(ROOT / "train-main-v2.parquet"); MD = {(r.model_id, norm(r.problem)): r.max_drop for r in pq.itertuples()}


def num(s):
    m = re.search(r"-?\d+(?:\.\d+)?(?:/\d+)?", (s or "").replace(",", "")); return m.group(0) if m else None


allm, _ = load_all(); rows = datasets(allm)["main_v2"]
out = []
mvF = {"fa": MB["M"]["4 output|forced p1 @6144"], "chk": MB["M"]["6 text|check share"], "fas": MB["M"]["4 output|confidence slope over budgets"],
       "lble": MA["M"]["lookback late - early"], "pmle": MA["M"]["prompt mass late - early"]}
for j, i in enumerate(rows):
    rid = allm.id.iloc[i]; m = HF[allm.model.values[i]]; p = inp[rid]["problem"]; g = mvrows.get(rid, {}).get("greedy", {})
    out.append(dict(id=rid, dataset="main_v2", model_id=m, model=m.split("/")[0][:8], robust=int(allm.y.values[i] == 1), **{f: float(mvF[f][j]) for f in mvF},
                    finished=g.get("think_end") is not None, gold=gold.get(norm(p)), forced_answer=num(g.get("forced", {}).get("b6144", {}).get("B", {}).get("continuation")),
                    max_drop=MD.get((m, norm(p))), problem=p[:140]))
olyF = {"fa": D47["F"]["4 output|forced p1 @6144"], "chk": D47["F"]["6 text|check share"], "fas": D47["F"]["4 output|confidence slope over budgets"],
        "lble": MA["O"]["lookback late - early"], "pmle": MA["O"]["prompt mass late - early"]}
for k, rid in enumerate(D47["ids"]):
    m = D47["model"][k]; ms = m.replace("/", "__"); p = inp[rid]["problem"]; pid = sel.get(norm(p)); L = LAB.get((ms, pid), {}); pr = PERT.get((ms, pid), [])
    g = oprows[rid]["greedy"]
    out.append(dict(id=rid, dataset="olympiad", model_id=m, model=m.split("/")[0][:8], robust=int(D47["y"][k]), indep=bool(D47["indep"][k]), **{f: float(olyF[f][k]) for f in olyF},
                    finished=g.get("think_end") is not None, gold=gold.get(norm(p)), forced_answer=num(g["forced"]["b6144"]["B"].get("continuation")),
                    perturbations_correct=L.get("perturbations_correct"), worst=L.get("worst"), capped=L.get("capped"),
                    failed_perturbations=";".join(f"{d['ptype']}:{'CAP' if d.get('finish') == 'length' else 'wrong'}" for d in pr if str(d.get("correct")) != "1"),
                    problem=p[:140]))
D = pd.DataFrame(out)
for k, r in REC.items():
    s = np.array([score_trace({f: row[f] for f in r["features"]}, r, row["model_id"]) for row in out])
    D[f"pred_{k}"] = (s > r["threshold"]).astype(int); D[f"score_{k}"] = s
D["answer_correct"] = [(a == g_) if (g_ and a) else None for a, g_ in zip(D.forced_answer, D.gold)]
D.to_csv(A / "exp56_rows.csv", index=False)
k = "confidence-checkshare-slope"
D["outcome"] = np.where(D[f"pred_{k}"] == D.robust, "correct", np.where(D[f"pred_{k}"] == 1, "FALSE ROBUST", "FALSE SPURIOUS"))
print(D.groupby(["dataset", "outcome"]).size().unstack(fill_value=0).to_string())
print("\nby model (olympiad):"); print(D[D.dataset == "olympiad"].groupby(["model", "outcome"]).size().unstack(fill_value=0).to_string())
o = D[D.dataset == "olympiad"]
print("\nolympiad non-robust rows: why did they get the label? (failed perturbations: CAP = the perturbed trace hit the token cap)")
for oc in ("FALSE ROBUST", "correct"):
    sub = o[(o.robust == 0) & (o.outcome == oc)]
    caps = sub.failed_perturbations.str.count("CAP"); wrong = sub.failed_perturbations.str.count("wrong")
    print(f"  {oc:15s} n={len(sub):2d} | label caused only by cap hits: {int(((caps > 0) & (wrong == 0)).sum())} | by >= 1 wrong answer: {int((wrong > 0).sum())} | "
          f"perturbations failed: median {np.median(5 - sub.perturbations_correct.astype(float)):.0f}")
print("\nolympiad robust rows called spurious vs called robust: finished by 6144 / answer correct / features")
for oc in ("FALSE SPURIOUS", "correct"):
    sub = o[(o.robust == 1) & (o.outcome == oc)]
    print(f"  {oc:15s} n={len(sub):2d} | finished {sub.finished.mean():.2f} | forced answer = gold {sub.answer_correct.astype(float).mean():.2f} | "
          f"fa {sub.fa.median():.3f} chk {sub.chk.median():.3f} fas {sub.fas.median():.3f}")
print("\nmain_v2 failures:"); print(D[(D.dataset == "main_v2") & (D.outcome != "correct")][["model", "robust", "outcome", "max_drop", "finished", "answer_correct", "fa", "chk", "problem"]].to_string(max_colwidth=60))
