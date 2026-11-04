"""exp49: T=0 vs T=1 on main_v2 (43 rows, 22 problems). Per row: the greedy trace (T=0) and 2 traces sampled at the label settings
(T 1.0, top_k 40, top_p 0.95), all capped at 6144 tokens (Vast run, mainv2_t1.zip). Families:
  agreement   forced answer string (first number of the 12-token continuation after suffix B) of greedy vs samples, at 2048 / 4096 / 6144 /
              end of thinking; all three traces agree; the samples agree with each other
  correctness share of the 3 forced answers @6144 equal to the gold answer (16 problems that are AIME problems with a known answer)
  confidence  greedy forced p1 minus the samples' mean; samples' mean / min p1; spread of p1 over the 3 traces
  divergence  token where the sample departs from greedy; mean / late cosine of layer-3 window states greedy vs sample; slope of that cosine
  averaged    the submission features (fa, chk, fas, fsat, len, ...) averaged over the 3 traces, and over the 2 samples alone
  sampling    samples finished within 6144 (eos); mean log-prob of sampled tokens; token entropy of samples
Score: within-model AUC (higher -> robust), label permutation inside each model, and the same after regressing out finished-by-6144 and
-log length of the greedy trace (the shortcuts). The greedy-only value of each feature is shown for comparison where one exists."""
import csv, glob, json, os, re, difflib
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "2"
import numpy as np
from pathlib import Path

ROOT = Path("/home/sabyasachi19/PycharmProjects/AimoAGAIN"); OUT = ROOT / "results_mainv2_t1" / "out_t1"
rows = [json.loads(l) for f in sorted(OUT.glob("*/rows_shard*.jsonl")) for l in open(f)]
inp = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "t1_run" / "inputs.jsonl")}
N = len(rows); y = np.array([r["robust"] for r in rows]).astype(int); model = np.array([r["model_id"] for r in rows])
norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()                                    # noqa: E731
GOLD = {norm(r["problem"])[:400]: r["answer"].strip() for r in csv.DictReader(open(ROOT / "olympiad_run_package" / "olympiad" / "selected_problems.csv"))}
gold = []
for r in rows:
    p = norm(inp[r["id"]]["problem"])[:400]; k = max(GOLD, key=lambda q: difflib.SequenceMatcher(None, p, q).ratio())
    gold.append(GOLD[k] if norm(inp[r["id"]]["problem"])[:400] == k else None)   # verbatim matches only (variants have other answers)


def num(s):
    m = re.search(r"-?\d+(?:\.\d+)?(?:/\d+)?", (s or "").replace(",", "")); return m.group(0) if m else (s or "").strip()[:8]


F = {}


def put(k, i, v):
    F.setdefault(k, np.full(N, np.nan))[i] = v


for i, r in enumerate(rows):
    g, S = r["greedy"], r["samples"]; T = [g] + S; z = np.load(OUT / r["model_id"].replace("/", "__") / f"{r['id']}.npz")
    for lab in ("b2048", "b4096", "b6144", "f100"):
        a = [num(t["forced"][lab]["B"].get("continuation")) for t in T]
        put(f"agreement|greedy = samples @{lab} (share)", i, np.mean([a[0] == x for x in a[1:]]))
    a = [num(t["forced"]["b6144"]["B"].get("continuation")) for t in T]
    put("agreement|all three agree @6144", i, float(len(set(a)) == 1)); put("agreement|the two samples agree @6144", i, float(a[1] == a[2]))
    put("agreement|distinct answers over budgets & traces (-)", i, -len({num(t["forced"][f"b{b}"]["B"].get("continuation")) for t in T for b in (2048, 3072, 4096, 6144)}))
    if gold[i] is not None:
        put("correctness|share of 3 forced answers @6144 = gold", i, np.mean([x == gold[i] for x in a]))
        put("correctness|greedy forced answer @6144 = gold", i, float(a[0] == gold[i]))
        put("correctness|samples' forced answers @6144 = gold (share)", i, np.mean([x == gold[i] for x in a[1:]]))
    p1 = np.array([t["fa"] for t in T])
    put("confidence|greedy p1 - samples' mean p1", i, p1[0] - p1[1:].mean()); put("confidence|samples' mean p1", i, p1[1:].mean())
    put("confidence|samples' min p1", i, p1[1:].min()); put("confidence|spread of p1 over 3 traces (-)", i, -p1.std())
    put("confidence|greedy p1 (T=0 only)", i, p1[0])
    cs = [z[f"t1s{s}_cos_vs_greedy"].astype(np.float32) for s in range(len(S))]
    put("divergence|departure token (log)", i, np.mean([np.log1p(t["depart_token"]) for t in S]))
    put("divergence|mean cosine to greedy (layer 3)", i, np.mean([c.mean() for c in cs]))
    put("divergence|cosine to greedy, last quarter", i, np.mean([c[-max(1, len(c) // 4):].mean() for c in cs]))
    put("divergence|cosine to greedy, slope over time", i, np.mean([np.polyfit(np.arange(len(c)), c, 1)[0] if len(c) > 2 else np.nan for c in cs]))
    for f in ("fa", "chk", "fas", "fsat", "len", "ge", "tda", "traj", "seg", "pcr", "ab", "mean_tok_entropy", "mean_tok_logprob"):
        v = np.array([t.get(f) if t.get(f) is not None else np.nan for t in T], float)
        sg = -1 if f == "mean_tok_entropy" else 1
        put(f"averaged|{f}: greedy only", i, sg * v[0]); put(f"averaged|{f}: mean of 3 traces", i, sg * np.nanmean(v)); put(f"averaged|{f}: mean of 2 samples", i, sg * np.nanmean(v[1:]))
    put("sampling|samples finished within 6144", i, np.mean([t["stop"] == "eos" for t in S]))
    put("sampling|greedy finished within 6144", i, float(g["think_end"] is not None))

rng = np.random.default_rng(0)
fin = F["sampling|greedy finished within 6144"]; ln = F["averaged|len: greedy only"]


def within(x, yy, mm):
    num_ = den = 0
    for m in np.unique(mm):
        k = (mm == m) & np.isfinite(x); p, n = x[k & (yy == 1)], x[k & (yy == 0)]
        num_ += (p[:, None] > n[None]).sum() + .5 * (p[:, None] == n[None]).sum(); den += len(p) * len(n)
    return num_ / den if den else np.nan


def perm(x, n=2000):
    w = within(x, y, model); nl = np.empty(n)
    for t in range(n):
        yp = y.copy()
        for m in np.unique(model): k = model == m; yp[k] = rng.permutation(y[k])
        nl[t] = within(x, yp, model)
    return w, float(np.mean(np.abs(nl - .5) >= abs(w - .5)))


def resid(v):
    out = np.full(N, np.nan)
    for m in np.unique(model):
        k = (model == m) & np.isfinite(v)
        if k.sum() < 4: continue
        C = np.c_[np.ones(k.sum()), fin[k], ln[k]]; out[k] = v[k] - C @ np.linalg.lstsq(C, v[k], rcond=None)[0]
    return out


print(f"{N} rows ({y.sum()} robust): " + ", ".join(f"{m.split('/')[0][:8]} {np.sum(model == m)}" for m in np.unique(model))
      + f" | rows with a gold answer: {sum(x is not None for x in gold)}")
print(f"{'feature':62s} {'n':>3s} {'within-model (p)':>17s} {'controlled (p)':>15s}")
for k, v in F.items():
    if np.isfinite(v).sum() < 15 or len(np.unique(v[np.isfinite(v)])) < 2: continue
    w, p = perm(v); wc, pc = perm(resid(v))
    print(f"{k:62s} {int(np.isfinite(v).sum()):3d} {w:8.2f} ({p:.3f}) {wc:7.2f} ({pc:.3f}){'  *' if p < .05 else ''}")

# 49b: stability = the same feature scored on each trace separately (greedy, sample 1, sample 2): three independent draws of the
# reasoning for the same (model, problem). A real property of the problem should score alike on all three.
print(f"\n49b stability: within-model AUC per trace{'':16s} greedy  sample1  sample2   | Spearman of the values across traces")
from scipy.stats import spearmanr
for f in ("fa", "chk", "fsat", "fas", "len", "ge", "tda", "traj", "seg", "pcr", "ab", "mean_tok_entropy"):
    V = np.array([[t.get(f) if t.get(f) is not None else np.nan for t in [r["greedy"]] + r["samples"]] for r in rows], float)
    if f == "mean_tok_entropy": V = -V
    a = [within(V[:, j], y, model) for j in range(3)]
    ok = np.isfinite(V).all(1); rho = np.mean([spearmanr(V[ok, i], V[ok, j])[0] for i, j in ((0, 1), (0, 2), (1, 2))])
    print(f"  {f:44s} {a[0]:6.2f} {a[1]:8.2f} {a[2]:8.2f}   | {rho:5.2f}")

# 49c: does agreement add to confidence + check share? within-model ranks, equal weights
from scipy.stats import rankdata


def rk(v):
    out = np.full(N, np.nan)
    for m in np.unique(model):
        k = (model == m) & np.isfinite(v); out[k] = (rankdata(v[k]) - .5) / k.sum()
    return out


ag = F["agreement|greedy = samples @b6144 (share)"]; fa = F["confidence|greedy p1 (T=0 only)"]; chk = F["averaged|chk: greedy only"]
from scipy.stats import spearmanr as sp
print(f"\n49c: agreement vs greedy confidence, Spearman {sp(ag, fa)[0]:.2f}; vs check share {sp(ag, chk)[0]:.2f}")
for nm, parts in (("confidence + check share", [fa, chk]), ("confidence + check share + agreement", [fa, chk, ag]),
                  ("confidence(3-trace mean) + check share(3-trace mean) + agreement", [F["averaged|fa: mean of 3 traces"], F["averaged|chk: mean of 3 traces"], ag])):
    s = np.nanmean(np.stack([rk(p) for p in parts]), 0); w, p = perm(s); wc, pc = perm(resid(s))
    print(f"  {nm:64s} {w:.2f} (p {p:.3f}) | controlled {wc:.2f} (p {pc:.3f})")
