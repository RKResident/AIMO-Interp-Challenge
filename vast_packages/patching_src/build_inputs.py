"""Inputs for the patching Vast run (run locally):
  direct_inputs.jsonl  olympiad, per labelled problem and model: the original and 5 perturbed problem texts with the friend's greedy
                       completions (token ids, truncated to 6144) -> forced answers at 2048 / 4096 / 6144 on the box (no generation)
  patch_inputs.jsonl   the white-box run's 118 labelled rows (prompt ids, greedy ids, problem text) for the activation-patching stages"""
import csv, glob, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent.parent
OLY = ROOT / "extra_olympiad_problems_dataset" / "olympiad_labels_greedy"
text = {(r["pid"], "original"): r["problem"] for r in csv.DictReader(open(ROOT / "olympiad" / "selected_problems.csv"))}
text.update({(r["pid"], r["ptype"]): r["text"] for r in csv.DictReader(open(ROOT / "olympiad" / "perturbations.csv"))})
n = 0
with open(HERE / "direct_inputs.jsonl", "w") as f:
    for mdir in sorted(OLY.glob("*__*")):
        mid = mdir.name.replace("__", "/")
        lab = {r["pid"]: r for r in csv.DictReader(open(mdir / "labels.csv"))}
        for pid, r in sorted(lab.items()):
            if r["label"] not in ("robust", "non_robust"): continue
            recs = {}
            for l in open(mdir / "problems" / f"{pid}.jsonl"):
                d = json.loads(l); recs[d["ptype"]] = d
            items = []
            for pt in ("original", "distract", "domain", "rename", "rephrase", "typos"):
                d = recs.get(pt)
                if d is None or (pid, pt) not in text: continue
                items.append(dict(ptype=pt, text=text[(pid, pt)], gen_ids=d["gen_ids"][:6144], n_tokens=int(d["n_tokens"]), finish=d["finish"],
                                  final_answer=str(d["answer"]), correct=int(d["correct"])))
            f.write(json.dumps(dict(model_id=mid, pid=pid, label=r["label"], items=items)) + "\n"); n += 1
wb = [json.loads(l) for l in open(ROOT / "vast_packages" / "whitebox" / "whitebox_inputs.jsonl")]
with open(HERE / "patch_inputs.jsonl", "w") as f:
    for r in wb:
        f.write(json.dumps({k: r[k] for k in ("id", "dataset", "model_id", "robust", "prompt_ids", "greedy_gen_ids", "problem")}) + "\n")
print(n, "olympiad problems for the direct test |", len(wb), "rows for patching")
