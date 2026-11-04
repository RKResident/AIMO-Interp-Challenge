"""Builds the inputs of the recall + reasoning-graph Vast run (run locally, then zip):
  recall_pairs.jsonl  GSM-Symbolic (apple/GSM-Symbolic, main split, CC BY-NC-ND 4.0, research use): the 100 GSM8K test originals and,
                      for each, its first 3 templated variants (other names / numbers, so the original's answer is wrong)
  graph_inputs.jsonl  the 118 labelled rows of the white-box run (prompt ids + greedy ids, as there) plus, for the 43 main_v2 rows, the two
                      T=1 samples' text (re-tokenised on the Vast box with the model's tokenizer)."""
import json, glob, re
from collections import defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent.parent


def gold(ans):
    return ans.split("####")[-1].strip().replace(",", "")


by = defaultdict(list)
for l in open(HERE / "gsm_symbolic_main.jsonl"):
    d = json.loads(l); by[d["original_id"]].append(d)
with open(HERE / "recall_pairs.jsonl", "w") as f:
    for oid, ds in sorted(by.items(), key=lambda x: int(x[0])):
        ds = sorted(ds, key=lambda d: int(d["instance"])); o = ds[0]
        f.write(json.dumps(dict(pid=oid, kind="original", key=f"{oid}_o", question=o["original_question"], gold=gold(o["original_answer"]))) + "\n")
        for d in ds[:3]:
            f.write(json.dumps(dict(pid=oid, kind="variant", key=f"{oid}_v{d['instance']}", question=d["question"], gold=gold(d["answer"]),
                                    original_gold=gold(o["original_answer"]))) + "\n")
wb = [json.loads(l) for l in open(ROOT / "vast_packages" / "whitebox" / "whitebox_inputs.jsonl")]
t1 = {json.loads(l)["id"]: json.loads(l) for f in glob.glob(str(ROOT / "results_mainv2_t1" / "out_t1" / "*" / "rows_shard*.jsonl")) for l in open(f)}
n_s = 0
with open(HERE / "graph_inputs.jsonl", "w") as f:
    for r in wb:
        rec = {k: r[k] for k in ("id", "dataset", "model_id", "robust", "prompt_ids", "greedy_gen_ids")}
        rec["sample_texts"] = [s["text"] for s in t1[r["id"]]["samples"]] if r["id"] in t1 else []
        n_s += len(rec["sample_texts"]); f.write(json.dumps(rec) + "\n")
print(sum(1 for _ in open(HERE / "recall_pairs.jsonl")), "recall prompts |", len(wb), "graph rows,", n_s, "T=1 samples")
