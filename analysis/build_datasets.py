"""Compile our self-made robustness datasets into datasets/ for publication (2026-11-04). Inputs are the synced labelling outputs; outputs
are small, self-describing files (see datasets/README.md):
  datasets/<name>/problems.jsonl        pid, source, problem, answer
  datasets/<name>/perturbations.jsonl   pid, ptype, text            (5 types: rephrase, rename, distract, domain, typos)
  datasets/<name>/samples_<model>.jsonl pid, ptype, cap, samples[{n_gen, finish, answer, correct, forced_answer, correct_any}]
  datasets/<name>/labels_<model>.csv    per problem: accuracy per version, max_drop and label under two scoring rules
Label rule (inferred from train-main-v2, see README): robust if max_drop <= 0.10 or the original is never solved, non_robust if
max_drop >= 0.25, ambiguous otherwise. 'natural' scores a sample right only if it finished and boxed the gold answer; 'forced' also
accepts the answer forced at the cap. No generated text is copied (answers and correctness only).
  python analysis/build_datasets.py"""
import json, shutil
from pathlib import Path
import pandas as pd

A = Path("/home/sabyasachi19/PycharmProjects/AimoAGAIN"); S = A / "vast_packages/screen"; E = A / "ec2_results"; OUT = A / "datasets"
TYPES = ["rephrase", "rename", "distract", "domain", "typos"]
TAG = {"Qwen/Qwen3.5-4B": "Qwen__Qwen3.5-4B", "allenai/Olmo-3-7B-Think": "allenai__Olmo-3-7B-Think",
       "Skywork/Skywork-OR1-Math-7B": "Skywork__Skywork-OR1-Math-7B", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": "deepseek-ai__DeepSeek-R1-0528-Qwen3-8B"}
KEEP = ("n_gen", "finish", "answer", "correct", "forced_answer", "correct_any")
jl = lambda f: [json.loads(l) for l in open(f)]
PROB = {d["pid"]: d for f in ("pool.jsonl", "easy_pool.jsonl", "oly_pool.jsonl") for d in jl(S / f)}


def merge(files):
    """(pid, ptype) -> {'cap', 'samples'}; a (pid, ptype, round) record seen in an earlier file is skipped, as compute_labels.py does"""
    out, seen = {}, set()
    for f in files:
        if not Path(f).exists(): continue
        for r in jl(f):
            if "samples" not in r: continue
            pt = r.get("ptype", "original")                        # solvability-pass records are originals without a ptype field
            k = (r["pid"], pt, r.get("round", "solve" if "ptype" not in r else 1))
            if k in seen: continue
            seen.add(k); e = out.setdefault((r["pid"], pt), {"cap": r.get("cap"), "samples": []})
            e["samples"] += [{x: s.get(x) for x in KEEP} for s in r["samples"]]
    return out


def labels(samp, pids):
    rows = []
    for p in pids:
        if (p, "original") not in samp or any((p, t) not in samp for t in TYPES): continue
        row = {"pid": p, "source": PROB.get(p, {}).get("source"), "n_original": len(samp[(p, "original")]["samples"])}
        for t in TYPES: row[f"n_{t}"] = len(samp[(p, t)]["samples"])
        for rule, key in (("natural", "correct"), ("forced", "correct_any")):
            acc = {v: sum(bool(s[key]) for s in samp[(p, v)]["samples"]) / len(samp[(p, v)]["samples"]) for v in ["original"] + TYPES}
            worst = min(TYPES, key=lambda t: acc[t]); drop = acc["original"] - acc[worst]
            row.update({f"acc_original_{rule}": acc["original"], **{f"acc_{t}_{rule}": acc[t] for t in TYPES}, f"max_drop_{rule}": round(drop, 4),
                        f"worst_{rule}": worst, f"unsolved_{rule}": acc["original"] == 0,
                        f"label_{rule}": "robust" if acc["original"] == 0 or drop <= 0.10 else "non_robust" if drop >= 0.25 else "ambiguous"})
        rows.append(row)
    return pd.DataFrame(rows)


def write(name, perts_file, runs, extra=None):
    d = OUT / name; d.mkdir(parents=True, exist_ok=True); allp = set(); summary = {}
    for model, files in runs.items():
        samp = merge(files); pids = sorted({p for p, _ in samp}); L = labels(samp, pids)
        if L.empty: continue
        L.to_csv(d / f"labels_{TAG[model]}.csv", index=False); keep = set(L.pid); allp |= keep
        with open(d / f"samples_{TAG[model]}.jsonl", "w") as fo:
            for (p, t), e in sorted(samp.items()):
                if p in keep: fo.write(json.dumps({"pid": p, "ptype": t, "model": model, "cap": e["cap"], "samples": e["samples"]}) + "\n")
        summary[model] = {"problems": len(L), **{f"{r}": L[f"label_{r}"].value_counts().to_dict() for r in ("natural", "forced")},
                          "samples_per_version": sorted(set(L.n_original) | set(L.n_rephrase))}
    with open(d / "problems.jsonl", "w") as fo:
        for p in sorted(allp): fo.write(json.dumps({k: PROB[p][k] for k in ("pid", "source", "problem", "answer")}, ensure_ascii=False) + "\n")
    with open(d / "perturbations.jsonl", "w") as fo:
        for r in jl(perts_file):
            if r["pid"] in allp: fo.write(json.dumps({"pid": r["pid"], "ptype": r["ptype"], "text": r["text"]}, ensure_ascii=False) + "\n")
    for src, dst in (extra or {}).items(): shutil.copyfile(src, d / dst)
    print(name, json.dumps(summary)); return summary


SUM = {}
SUM["set100_hard_32k"] = write("set100_hard_32k", S / "perturbations.jsonl", {
    "Qwen/Qwen3.5-4B": [E / "big/out/solve/Qwen__Qwen3.5-4B.jsonl", E / "big/out/label/Qwen__Qwen3.5-4B.jsonl", E / "a10g/out/label/Qwen__Qwen3.5-4B.jsonl"],
    "allenai/Olmo-3-7B-Think": [E / "big/out/label/allenai__Olmo-3-7B-Think.jsonl"]},
    extra={A / "hard_pool/set100_labels_Qwen__Qwen3.5-4B.csv": "labels_as_used_Qwen__Qwen3.5-4B.csv",
           A / "hard_pool/set100_labels_allenai__Olmo-3-7B-Think.csv": "labels_as_used_allenai__Olmo-3-7B-Think.csv"})
SUM["olympiad_32k"] = write("olympiad_32k", S / "oly_perts.jsonl", {
    "Skywork/Skywork-OR1-Math-7B": [E / "big/out/label_oly/Skywork__Skywork-OR1-Math-7B.jsonl"],
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": [E / "big/out/label_oly/deepseek-ai__DeepSeek-R1-0528-Qwen3-8B.jsonl"]})
SUM["confmatched_6144"] = write("confmatched_6144", S / "new_perts.jsonl", {m: [E / f"label_new/{TAG[m]}.jsonl"] for m in
                                ("Qwen/Qwen3.5-4B", "allenai/Olmo-3-7B-Think", "Skywork/Skywork-OR1-Math-7B")})
SUM["confmatched_32k"] = write("confmatched_32k", S / "new_perts.jsonl", {m: [E / f"label_new32k/{TAG[m]}.jsonl"] for m in
                               ("Qwen/Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B")})

# olympiad greedy labels (our earlier one-sample scheme): copied as produced, plus the problem / perturbation texts
d = OUT / "olympiad_greedy_16k"; d.mkdir(parents=True, exist_ok=True); G = A / "extra_olympiad_problems_dataset/olympiad_labels_greedy"; gp = set()
for m in ("Skywork/Skywork-OR1-Math-7B", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"):
    L = pd.read_csv(G / TAG[m] / "labels.csv"); L.to_csv(d / f"labels_{TAG[m]}.csv", index=False); gp |= set(L.pid)
    with open(d / f"samples_{TAG[m]}.jsonl", "w") as fo:
        for f in sorted((G / TAG[m]).glob("samples_shard*.jsonl")):
            for r in jl(f): fo.write(json.dumps({k: r.get(k) for k in ("pid", "ptype", "stage", "answer", "correct", "n_tokens", "finish")}) + "\n")
    print("olympiad_greedy_16k", m, L.label.value_counts().to_dict())
with open(d / "problems.jsonl", "w") as fo:
    for p in sorted(gp): fo.write(json.dumps({k: PROB[p][k] for k in ("pid", "source", "problem", "answer")}, ensure_ascii=False) + "\n")
with open(d / "perturbations.jsonl", "w") as fo:
    for r in jl(S / "oly_perts.jsonl"):
        if r["pid"] in gp: fo.write(json.dumps({"pid": r["pid"], "ptype": r["ptype"], "text": r["text"]}, ensure_ascii=False) + "\n")
json.dump(SUM, open(OUT / "summary.json", "w"), indent=1)
