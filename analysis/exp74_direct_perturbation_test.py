"""exp74: how well would a DIRECT robustness test work inside a submission? i.e. regenerate the answer for perturbed versions of the problem
and call it robust if the answers agree with the model's answer to the original. Local estimate on the friend's olympiad set, which has a
greedy completion for the original and for each of 5 perturbations (distract, domain, rename, rephrase, typos; greedy, up to 16k tokens).
NOTE the olympiad labels are themselves defined from these completions (robust = every perturbation still CORRECT), so at full budget the
test reproduces the label by construction. What the estimate shows is what survives the submission's constraints:
  - a 6144-token cap: a completion counts only if it finished within 6144 tokens; otherwise its answer is unknown
  - no gold answer: agreement with the model's own original answer instead of correctness
  - fewer perturbation types (budget): every subset of the 5 types
The leaderboard labels come from T=1 sampling at up to 100k tokens, so this is an optimistic proxy for the test labels."""
import csv, glob, itertools, json
import numpy as np

OLY = "/home/sabyasachi19/PycharmProjects/AimoAGAIN/extra_olympiad_problems_dataset/olympiad_labels_greedy/"
PT = ["distract", "domain", "rename", "rephrase", "typos"]
rows = []
for mdir in sorted(glob.glob(OLY + "*/")):
    m = mdir.rstrip("/").split("/")[-1].split("__")[1][:8]
    lab = {r["pid"]: r for r in csv.DictReader(open(mdir + "labels.csv"))}
    comp = {}
    for f in glob.glob(mdir + "samples_shard*.jsonl"):
        for l in open(f):
            d = json.loads(l); comp[(d["pid"], d["ptype"])] = d
    for pid, r in lab.items():
        if r["label"] not in ("robust", "non_robust"): continue
        o = comp.get((pid, "original"))
        if o is None: continue
        rows.append(dict(model=m, pid=pid, y=int(r["label"] == "robust"), orig=o, pert={p: comp.get((pid, p)) for p in PT}))
print(f"{len(rows)} labelled problems (original solved): " + ", ".join(f"{m} {sum(r['model'] == m for r in rows)} ({sum(r['y'] for r in rows if r['model'] == m)} robust)" for m in sorted({r['model'] for r in rows})))
y = np.array([r["y"] for r in rows])


def ans(d, cap):
    """answer if the completion finished within `cap` tokens, else None"""
    if d is None: return None
    if cap and (int(d["n_tokens"]) > cap or d["finish"] != "stop"): return None
    return str(d["answer"]).strip()


def auc(s, yy):
    p, n = s[yy == 1], s[yy == 0]
    return float(((p[:, None] > n[None]) + .5 * (p[:, None] == n[None])).mean())


print("\nhow often each completion finishes within 6144 tokens: original %.2f | perturbations %.2f" % (
    np.mean([ans(r["orig"], 6144) is not None for r in rows]), np.mean([ans(r["pert"][p], 6144) is not None for r in rows for p in PT])))
print("\n                                       robust iff ... | accuracy  AUC(agree share)  coverage")
for cap in (None, 6144):
    for unknown in ("disagree", "skip"):
        for types in ([PT] if cap is None else [PT, ["rename", "typos"], ["distract"], ["rephrase", "distract"], ["distract", "domain", "rename"]]):
            pred, share, cov = [], [], []
            for r in rows:
                a0 = ans(r["orig"], cap)
                ref = a0 if a0 is not None else None
                ag = []
                for p in types:
                    a = ans(r["pert"][p], cap)
                    if a is None or ref is None:
                        if unknown == "disagree": ag.append(0)
                    else:
                        ag.append(int(a == ref))
                share.append(np.mean(ag) if ag else .5); pred.append(all(ag) if ag else True); cov.append(len(ag) / len(types) if unknown == "skip" else 1)
            pred, share = np.array(pred), np.array(share)
            tag = f"cap {cap or 'none'} | unknown={unknown:8s} | types {'+'.join(t[:4] for t in types):22s}"
            print(f"   {tag} | {(pred == (y == 1)).mean():.2f}      {auc(share, y):.2f}            {np.mean(cov):.2f}")
            if cap is None: break
