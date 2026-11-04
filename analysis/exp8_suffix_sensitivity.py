"""Idea 1: suffix-framing sensitivity. At every cutoff the answer was forced twice, with suffix A ("limited time ... Final Answer") and B ("The final
answer is"). How much does the FIRST answer-token distribution move between A and B? (top-20 probs stored; tail mass treated as one bucket)."""
from common import *
allm, _ = load_all()
LABS = ["b512", "b1024", "b2048", "b3072", "b4096", "b6144", "b8192", "f25", "f50", "f75", "f90", "f100"]


def dist(t):
    p = dict(zip(t["ids"], t["probs"])); return p, max(0.0, 1 - sum(p.values()))


def cmp(ra, rb):
    if not ra or not rb or not ra["top"] or not rb["top"]:
        return None
    (pa, ta), (pb, tb) = dist(ra["top"][0]), dist(rb["top"][0])
    ids = set(pa) | set(pb)
    tv = 0.5 * (sum(abs(pa.get(i, 0) - pb.get(i, 0)) for i in ids) + abs(ta - tb))
    ov = sum(min(pa.get(i, 0), pb.get(i, 0)) for i in ids)
    # Jensen-Shannon on the union (+ tail bucket)
    va = np.array([pa.get(i, 0) for i in ids] + [ta]) + 1e-12; vb = np.array([pb.get(i, 0) for i in ids] + [tb]) + 1e-12
    va /= va.sum(); vb /= vb.sum(); m = (va + vb) / 2
    js = 0.5 * (va * np.log(va / m)).sum() + 0.5 * (vb * np.log(vb / m)).sum()
    same1 = float(ra["top"][0]["ids"][0] == rb["top"][0]["ids"][0])
    dis = float(nrm(ra["answer"]) != nrm(rb["answer"]))
    return dict(tv=tv, ov=ov, js=js, same1=same1, dis=dis, dent=abs(ra["first_entropy"] - rb["first_entropy"]),
                dp1=ra["top"][0]["probs"][0] - rb["top"][0]["probs"][0])


rows = []
for r in allm.itertuples():
    fj = force_json(r.model, r.id); d = {}
    per = {}
    for l in LABS:
        c = fj["labels"][l]; x = cmp(fj["results"].get(f"{c}|A"), fj["results"].get(f"{c}|B"))
        if x is None: continue
        per[l] = x
        for k, v in x.items(): d[f"{k}@{l}"] = v
    for k in ("tv", "ov", "js", "same1", "dis", "dent", "dp1"):
        vals = [per[l][k] for l in per]
        d[f"{k}_mean"] = np.mean(vals) if vals else np.nan; d[f"{k}_max"] = np.max(vals) if vals else np.nan
        bv = [per[l][k] for l in per if l.startswith("b")]; d[f"{k}_bmean"] = np.mean(bv) if bv else np.nan
    rows.append(d)
S = pd.DataFrame(rows); S.to_pickle(A / "exp8_features.pkl")
print("=== Idea 1: suffix-framing sensitivity (A vs B first-answer-token distribution) ===")
single_auc(S[[c for c in S if "_" in c and "@" not in c]], allm, top=10)
summ = [c for c in S if c.endswith(("_mean", "_max", "_bmean"))]
evaluate("A/B sensitivity, summaries", S[summ].values, allm, C=0.05)
evaluate("A/B sensitivity, per budget", S[[c for c in S if "@b" in c]].values, allm, C=0.02)
# length-free version: only traces still thinking at b, sensitivity measured AT b
ds = datasets(allm)
for b in (1024, 2048, 4096):
    keep = np.array([i for i, r in enumerate(allm.itertuples()) if force_json(r.model, r.id)["L"] > b])
    rs = {k: np.intersect1d(v, keep) for k, v in ds.items()}
    cols = [c for c in S if c.endswith(f"@b{b}")]
    evaluate(f"A/B sensitivity at {b}, still-thinking rows", S[cols].values, allm, C=0.05, rowsets=rs)
