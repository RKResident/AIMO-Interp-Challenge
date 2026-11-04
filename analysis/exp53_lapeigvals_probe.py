"""exp53: LapEigvals as the paper uses it (Binkowski et al., spectral attention features): the top-k attention-graph Laplacian eigenvalues
of every layer and head, concatenated, PCA, logistic regression. Per model (feature sizes differ), leave-one-problem-out; olympiad (75 rows,
DeepSeek / Skywork, 6144-token traces), main_v2 (43 rows, 4 models; our 16k extraction, cutoffs at fixed token budgets so the features at
<= 6144 are cap-independent), and cross-dataset (train olympiad -> test main_v2, same model).
Variants: no forcing / forced suffix B; cutoff 2048 tokens and the last cutoff <= 6144; top-8 / top-64 eigenvalues per head; log scale."""
import json, os, pickle, re, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "2"
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, "/home/sabyasachi19/PycharmProjects/AimoAGAIN/analysis")
from common import *                                                                    # noqa: E402,F401

OLYA = ROOT / "extra_olympiad_problems_dataset" / "results_olympiad_attention" / "out_attn"
D = pickle.load(open(A / "exp47_features.pkl", "rb")); oid, oy, om, oind, oprob = D["ids"], D["y"], D["model"], D["indep"], D["problem"]
allm, _ = load_all(); rows = datasets(allm)["main_v2"]
norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()                                    # noqa: E731
slug = lambda m: m.replace("/", "__")                                                    # noqa: E731
rng = np.random.default_rng(0)


def vec(path, variant, cut, k):
    z = np.load(path); ce = list(z["cut_eff"]); budget = [c for c in ce if c <= 6144]
    j = ce.index(cut) if cut in ce else (ce.index(max(budget)) if cut == "last" else int(np.argmin(np.abs(np.array(ce) - cut))))
    a = z["topk_noforce"][j] if variant == "no forcing" else z["topk_force"][j, list(z["variant_names"]).index("B")]
    a = np.nan_to_num(a[..., :k].astype(np.float32), nan=0.0)
    return np.log(np.clip(a, 1e-8, None)).ravel()


def within(x, yy, mm):
    num = den = 0
    for m in np.unique(mm):
        k = (mm == m) & np.isfinite(x); p, n = x[k & (yy == 1)], x[k & (yy == 0)]
        num += (p[:, None] > n[None]).sum() + .5 * (p[:, None] == n[None]).sum(); den += len(p) * len(n)
    return num / den if den else np.nan


def perm(x, yy, mm, n=2000):
    w = within(x, yy, mm); nl = np.empty(n)
    for t in range(n):
        yp = yy.copy()
        for m in np.unique(mm): k = mm == m; yp[k] = rng.permutation(yy[k])
        nl[t] = within(x, yp, mm)
    return w, float(np.mean(np.abs(nl - .5) >= abs(w - .5)))


def probe_loo(X, y, groups, ncomp=5):
    out = np.full(len(y), np.nan)
    for p in np.unique(groups):
        te = groups == p; tr = ~te
        if len(np.unique(y[tr])) < 2: continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; pca = PCA(min(ncomp, tr.sum() - 1)).fit((X[tr] - mu) / sd)
        clf = LogisticRegression(C=0.1, max_iter=3000).fit(pca.transform((X[tr] - mu) / sd), y[tr])
        out[te] = clf.decision_function(pca.transform((X[te] - mu) / sd))
    return out


# main_v2 attention files: which rows have them
mv = []
for i in rows:
    p = ROOT / "extraction_out" / allm.model.values[i] / "attn" / f"{allm.id.iloc[i]}.npz"
    if p.exists(): mv.append((i, p))
my = np.array([int(allm.y.values[i] == 1) for i, _ in mv]); mmod = np.array([HF[allm.model.values[i]] for i, _ in mv])
mprob = np.array([norm(allm.problem.values[i])[:300] if "problem" in allm else allm.base_id.values[i] for i, _ in mv])
print(f"olympiad {len(oid)} rows; main_v2 rows with attention eigenvalues: {len(mv)} of {len(rows)}")
print(f"{'variant':44s} {'olympiad all':>13s} {'indep 49':>13s} {'main_v2 LOO':>13s} {'olympiad -> main_v2':>21s}")
for variant in ("no forcing", "forced B"):
    for cut in (2048, "last"):
        for k in (8, 64):
            # olympiad, per model LOO by problem
            so = np.full(len(oid), np.nan)
            for m in np.unique(om):
                idx = np.where(om == m)[0]; X = np.stack([vec(OLYA / slug(m) / "attn" / f"{oid[i]}.npz", variant, cut, k) for i in idx])
                so[idx] = probe_loo(X, oy[idx], oprob[idx])
            a = perm(so, oy, om); b = perm(so[oind], oy[oind], om[oind])
            # main_v2, per model LOO by problem
            sm = np.full(len(mv), np.nan)
            for m in np.unique(mmod):
                idx = np.where(mmod == m)[0]
                if len(idx) < 6: continue
                X = np.stack([vec(mv[i][1], variant, cut, k) for i in idx]); sm[idx] = probe_loo(X, my[idx], mprob[idx], ncomp=3)
            c = perm(sm, my, mmod)
            # cross-dataset: train on olympiad independent rows, test main_v2 (DeepSeek, Skywork)
            sx = np.full(len(mv), np.nan)
            for m in np.unique(om):
                tr = np.where((om == m) & oind)[0]; te = np.where(mmod == m)[0]
                if len(te) < 4: continue
                Xtr = np.stack([vec(OLYA / slug(m) / "attn" / f"{oid[i]}.npz", variant, cut, k) for i in tr]); Xte = np.stack([vec(mv[i][1], variant, cut, k) for i in te])
                if Xtr.shape[1] != Xte.shape[1]: continue
                mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; pca = PCA(5).fit((Xtr - mu) / sd)
                sx[te] = LogisticRegression(C=0.1, max_iter=3000).fit(pca.transform((Xtr - mu) / sd), oy[tr]).decision_function(pca.transform((Xte - mu) / sd))
            okx = np.isfinite(sx); d = perm(sx[okx], my[okx], mmod[okx]) if okx.sum() >= 8 else (np.nan, np.nan)
            print(f"{variant + ', cut ' + str(cut) + ', top-' + str(k):44s} {a[0]:5.2f} ({a[1]:.3f}) {b[0]:5.2f} ({b[1]:.3f}) {c[0]:5.2f} ({c[1]:.3f}) {d[0]:9.2f} ({d[1]:.3f}) n{int(okx.sum())}")
