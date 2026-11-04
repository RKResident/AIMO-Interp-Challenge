"""exp54: TOHA scored the way the paper does it (Algorithm 1, arXiv 2504.10063): per head the normalised MSF length d = total / |R|
(= the 'mean_edge' stat), heads ranked by separation Delta = mean d(spurious) - mean d(robust) on labelled training data, the top-N heads
averaged (N <= 10, N chosen by training AUROC). Done per model (heads are model-specific), with head and N selection NESTED inside
leave-one-problem-out folds, so the score of every row is out of sample. Response sets: 'win' (last 128 reasoning tokens before the
cut) and 'ans' (forced suffix B + the forced answer), at the first cut (1024 tokens) and the second (2048), both sets.
Olympiad: friend's toha files (his 10240-token traces; the cuts used here lie inside the first 6144 tokens). main_v2: our p2 toha files."""
import json, os, pickle, re, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "2"
import numpy as np
sys.path.insert(0, "/home/sabyasachi19/PycharmProjects/AimoAGAIN/analysis")
from common import *                                                                    # noqa: E402,F401

FR = ROOT / "extra_olympiad_problems_dataset" / "olympiad-features"; P2 = ROOT / "extraction_out_generations_p2" / "extraction_out_generations"
D47 = pickle.load(open(A / "exp47_features.pkl", "rb"))
allm, _ = load_all(); rows = datasets(allm)["main_v2"]
norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()[:300]                            # noqa: E731
inp = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "t1_run" / "inputs.jsonl")}
rng = np.random.default_rng(0)


def head_d(path, kind, ci):
    z = np.load(path, allow_pickle=True); sn = list(z["stat_names"]); j = sn.index("mean_edge")
    if ci >= len(z["cut_eff"]): return None
    a = z["win"][ci, :, :, j] if kind == "win" else z["ans"][ci, list(z["variant_names"]).index("B") if "B" in list(z["variant_names"]) else 0, :, :, j]
    return a.astype(np.float64).ravel()                                                    # [L * H]


def auc(x, y):
    p, n = x[y == 1], x[y == 0]
    return ((p[:, None] > n[None]).sum() + .5 * (p[:, None] == n[None]).sum()) / (len(p) * len(n)) if len(p) and len(n) else np.nan


def toha_fit(X, y, nmax=10):
    """Algorithm 1: heads sorted by Delta = mean(spurious) - mean(robust) (high divergence = hallucination-like = spurious); N by AUROC."""
    delta = np.nanmean(X[y == 0], 0) - np.nanmean(X[y == 1], 0)
    order = np.argsort(-np.nan_to_num(delta, nan=-np.inf))
    best = max(range(1, nmax + 1), key=lambda n: auc(-X[:, order[:n]].mean(1), y))       # robust score = - mean divergence
    return order[:best]


def oof(X, y, groups):
    out = np.full(len(y), np.nan); picks = []
    for p in np.unique(groups):
        te = groups == p; tr = ~te
        if len(np.unique(y[tr])) < 2: continue
        h = toha_fit(X[tr], y[tr]); picks.append(len(h)); out[te] = -X[te][:, h].mean(1)
    return out, picks


def within(scores, y, mm):
    num = den = 0
    for m in np.unique(mm):
        k = (mm == m) & np.isfinite(scores); p, n = scores[k & (y == 1)], scores[k & (y == 0)]
        num += (p[:, None] > n[None]).sum() + .5 * (p[:, None] == n[None]).sum(); den += len(p) * len(n)
    return num / den if den else np.nan


def perm_p(scores, y, mm, n=2000):
    w = within(scores, y, mm); nl = []
    for _ in range(n):
        yp = y.copy()
        for m in np.unique(mm): k = mm == m; yp[k] = rng.permutation(y[k])
        nl.append(within(scores, yp, mm))
    return w, float(np.mean(np.abs(np.array(nl) - .5) >= abs(w - .5)))


print(f"{'set / response / cut':40s} {'all-heads mean (as before)':>27s} {'TOHA top-N heads, nested LOO':>30s}  per model")
for setname in ("olympiad", "main_v2"):
    if setname == "olympiad":
        ids, ys, ms = D47["ids"], D47["y"].astype(int), D47["model"]
        path = lambda i: FR / ms[i].replace("/", "__") / "toha" / f"{ids[i]}.npz"          # noqa: E731
        groups = np.array([norm(inp[i_]["problem"]) for i_ in ids]); keep = np.array([path(i).exists() for i in range(len(ids))])
    else:
        ids = [allm.id.iloc[i] for i in rows]; ys = (allm.y.values[rows] == 1).astype(int); ms = np.array([HF[m] for m in allm.model.values[rows]])
        path = lambda i: P2 / allm.model.values[rows[i]] / "toha" / f"{ids[i]}.npz"         # noqa: E731
        groups = np.array([norm(inp[i_]["problem"]) for i_ in ids]); keep = np.array([path(i).exists() for i in range(len(ids))])
    for kind in ("win", "ans"):
        for ci, cname in ((0, "1024"), (1, "2048")):
            s_all = np.full(len(ids), np.nan); s_toha = np.full(len(ids), np.nan); per = []
            for m in np.unique(ms):
                idx = [i for i in range(len(ids)) if ms[i] == m and keep[i]]
                V = [head_d(path(i), kind, ci) for i in idx]; ok = [v is not None for v in V]
                idx = [i for i, o in zip(idx, ok) if o]; V = [v for v in V if v is not None]
                if len(idx) < 6 or len(set(len(v) for v in V)) > 1: continue
                X = np.stack(V); y_ = ys[idx]
                s_all[idx] = -np.nanmean(X, 1)
                o, picks = oof(X, y_, groups[idx]); s_toha[idx] = o
                per.append(f"{m.split('/')[0][:8]} {auc(o[np.isfinite(o)], y_[np.isfinite(o)]):.2f} (N~{np.median(picks):.0f})")
            a = perm_p(s_all, ys, ms); b = perm_p(s_toha, ys, ms)
            print(f"{setname + ' / ' + kind + ' / ' + cname:40s} {a[0]:15.2f} (p {a[1]:.3f}) {b[0]:18.2f} (p {b[1]:.3f})  " + " | ".join(per))
