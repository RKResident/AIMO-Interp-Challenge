"""exp46: the submission's features on the olympiad set (friend's greedy labels: original solved at 16k, then 5 perturbations; rows with an
unsolved original are excluded, like the test pairs). Greedy traces capped at 6144, features from the Vast run (olympiad_features.zip).
Question: do the main_v2 signals keep their direction here, or reverse as on the leaderboard (forced confidence / kNN ~0.3 there)?
Per model and pooled within model: AUC (higher value -> robust), label-permutation p inside each model."""
import json, glob, sys
import numpy as np
from pathlib import Path

R = Path("/home/sabyasachi19/PycharmProjects/AimoAGAIN/extra_olympiad_problems_dataset/results_olympiad_features/out_t1")
rows = [json.loads(l) for f in sorted(R.glob("*/rows_shard*.jsonl")) for l in open(f)]
model = np.array([r["model_id"].split("/")[0][:8] for r in rows]); y = np.array([r["robust"] for r in rows]).astype(int)
G = [r["greedy"] for r in rows]
F = {k: np.array([g.get(k) if g.get(k) is not None else np.nan for g in G], float)
     for k in ("fa", "fas", "fsat", "ab", "ge", "tda", "seg", "traj", "pcr", "chk", "len", "mean_tok_entropy", "mean_tok_logprob")}
for b in (512, 1024, 2048, 3072, 4096):
    F[f"fa p1@{b}"] = np.array([g["forced"][f"b{b}"]["B"]["p1"] for g in G])
F["finished by 6144"] = np.array([g["think_end"] is not None for g in G], float)
rng = np.random.default_rng(0)


def within(x, yy, mm):
    num = den = 0
    for m in np.unique(mm):
        k = (mm == m) & np.isfinite(x); p, n = x[k & (yy == 1)], x[k & (yy == 0)]
        num += (p[:, None] > n[None]).sum() + .5 * (p[:, None] == n[None]).sum(); den += len(p) * len(n)
    return num / den if den else np.nan


def perm_p(x, yy, mm, n=4000):
    w = within(x, yy, mm); nl = []
    for _ in range(n):
        yp = yy.copy()
        for m in np.unique(mm): k = mm == m; yp[k] = rng.permutation(yy[k])
        nl.append(within(x, yp, mm))
    return w, np.mean(np.abs(np.array(nl) - .5) >= abs(w - .5))


print(f"{len(rows)} rows: " + ", ".join(f"{m} {(model == m).sum()} ({y[model == m].sum()} robust)" for m in np.unique(model)))
print(f"traces finished within 6144: robust {F['finished by 6144'][y == 1].mean():.2f}, non-robust {F['finished by 6144'][y == 0].mean():.2f}")
MV2 = {"fa": .84, "ge": .83, "fsat": .74, "fas": .67, "pcr": .62, "tda": .63, "ab": .62, "fa p1@2048": .67, "fa p1@4096": .77}
print(f"\n{'feature':22s} {'DeepSeek':>8s} {'Skywork':>8s} {'within (p)':>14s} {'main_v2':>8s}")
for k, v in F.items():
    if np.isfinite(v).sum() < 10: continue
    per = [within(v[model == m], y[model == m], model[model == m]) for m in ("deepseek", "Skywork")]
    w, p = perm_p(v, y, model)
    print(f"{k:22s} {per[0]:8.2f} {per[1]:8.2f} {w:8.2f} ({p:.3f}) {MV2.get(k, float('nan')):8.2f}")

# attention features (friend's extract_features.py phases, run on the same 6144-token traces): scalar summaries per trace
RA = Path("/home/sabyasachi19/PycharmProjects/AimoAGAIN/extra_olympiad_problems_dataset/results_olympiad_attention/out_attn")
A = {k: np.full(len(rows), np.nan) for k in ("lookback ratio (mean)", "lookback ratio, last quarter", "prompt attention mass (mean)",
                                              "Laplacian eig mean, no forcing (last cut)", "Laplacian eig mean, forced B (last cut)",
                                              "Laplacian eig top-1, no forcing (last cut)")}
for i, r in enumerate(rows):
    d = RA / r["model_id"].replace("/", "__")
    lb, at = d / "lookback" / f"{r['id']}.npz", d / "attn" / f"{r['id']}.npz"
    if lb.exists():
        z = np.load(lb); L = z["lookback"].astype(np.float32); q = max(1, len(L) // 4)
        A["lookback ratio (mean)"][i] = L.mean(); A["lookback ratio, last quarter"][i] = L[-q:].mean()
        A["prompt attention mass (mean)"][i] = z["prompt_mass"].astype(np.float32).mean()
    if at.exists():
        z = np.load(at); j = int(np.argmax(z["cut_eff"]))
        A["Laplacian eig mean, no forcing (last cut)"][i] = np.nanmean(z["mean_noforce"][j].astype(np.float32))
        vB = list(z["variant_names"]).index("B")
        A["Laplacian eig mean, forced B (last cut)"][i] = np.nanmean(z["mean_force"][j, vB].astype(np.float32))
        A["Laplacian eig top-1, no forcing (last cut)"][i] = np.nanmean(z["topk_noforce"][j, :, :, 0].astype(np.float32))
print("\nattention (scalar summaries over layers and heads)")
for k, v in A.items():
    per = [within(v[model == m], y[model == m], model[model == m]) for m in ("deepseek", "Skywork")]
    w, p = perm_p(v, y, model)
    print(f"{k:44s} {per[0]:6.2f} {per[1]:6.2f} {w:8.2f} ({p:.3f})")

# equal-weight composites (within-model ranks averaged; signs fixed in advance from main_v2, NOT from this set, except where marked)
from scipy.stats import rankdata


def rk(v):
    out = np.full(len(v), np.nan)
    for m in np.unique(model):
        k = (model == m) & np.isfinite(v); out[k] = (rankdata(v[k]) - .5) / k.sum()
    return out


C = {"fa + chk": [("fa", 1), ("chk", 1)], "fa + fsat + chk": [("fa", 1), ("fsat", 1), ("chk", 1)],
     "fa + chk + length": [("fa", 1), ("chk", 1), ("len", 1)],
     "fa + chk + prompt mass (sign from this set)": [("fa", 1), ("chk", 1), ("pm", 1)]}
V = {**F, "pm": A["prompt attention mass (mean)"]}
print("\ncomposites")
for nm, parts in C.items():
    s = np.nanmean(np.stack([rk(V[k] * sg) for k, sg in parts]), 0)
    w, p = perm_p(s, y, model)
    print(f"{nm:44s} {within(s[model == 'deepseek'], y[model == 'deepseek'], model[model == 'deepseek']):6.2f} "
          f"{within(s[model == 'Skywork'], y[model == 'Skywork'], model[model == 'Skywork']):6.2f} {w:8.2f} ({p:.3f})")

# 46b: (1) the same test WITHOUT the olympiad problems that also appear in main_v2 (verbatim or as a number-swapped variant: 16 of main_v2's
# 22 problems are AIME 2025/2026 problems in this set), so the olympiad check is independent of main_v2; (2) difficulty proxies per set.
import re, difflib
norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()                                       # noqa: E731
MV = [json.loads(l) for l in open("/home/sabyasachi19/PycharmProjects/AimoAGAIN/t1_run/inputs.jsonl") if '"main_v2"' in l]
mvp = {norm(r["problem"])[:400] for r in MV}
inp = {json.loads(l)["id"]: json.loads(l) for l in open("/home/sabyasachi19/PycharmProjects/AimoAGAIN/t1_run/inputs.jsonl")}
overlap = np.array([max(difflib.SequenceMatcher(None, norm(inp[r["id"]]["problem"])[:400], q).ratio() for q in mvp) > .8 for r in rows])
print(f"\n46b: {overlap.sum()} of {len(rows)} olympiad rows are main_v2 problems (or variants); without them: {(~overlap).sum()} rows "
      f"({y[~overlap].sum()} robust)")
keep = ~overlap
for k in ("chk", "fa", "fas", "fsat", "ab", "ge", "tda", "traj", "pcr", "mean_tok_entropy", "len"):
    w, p = perm_p(F[k][keep], y[keep], model[keep]); print(f"  {k:18s} {w:6.2f} ({p:.3f})")
w, p = perm_p(A["prompt attention mass (mean)"][keep], y[keep], model[keep]); print(f"  {'prompt mass':18s} {w:6.2f} ({p:.3f})")
s = np.nanmean(np.stack([rk(F["fa"]), rk(F["chk"])]), 0); w, p = perm_p(s[keep], y[keep], model[keep]); print(f"  {'fa + chk':18s} {w:6.2f} ({p:.3f})")

# difficulty proxies: share of greedy traces still thinking at 6144 tokens, prompt length (tokens)
TEST_PROMPT = [152, 140, 85, 692, 114, 135, 149, 172, 210, 175, 219, 109, 78, 147]        # leaderboard logs (14 problems)
mv_cap = np.mean([len(r["greedy_gen_ids"]) >= 6144 and True for r in MV])
print(f"\nstill generating at 6144 tokens: main_v2 {mv_cap:.2f} (n={len(MV)}) | olympiad labelled {np.mean([g['n_gen'] >= 6144 for g in G]):.2f} "
      f"(n={len(G)}) | test 13/14 = {13 / 14:.2f}")
print(f"prompt tokens, median: main_v2 {np.median([len(r['prompt_ids']) for r in MV]):.0f} | olympiad labelled "
      f"{np.median([len(inp[r['id']]['prompt_ids']) for r in rows]):.0f} | test {np.median(TEST_PROMPT):.0f}")
