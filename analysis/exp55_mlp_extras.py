"""exp55: first look at the MLP layers, from the 'extras' files (extract_features.py): at the last forced-suffix token, for every layer, the
residual stream after the layer (res_last), the attention / token-mixer output (attn_last) and the MLP output (mlp_last); suffix B, at the
first cut (1024 tokens) and the last saved cut. Features:
  MLP share         ||mlp|| / (||mlp|| + ||attn||) per layer, averaged over layer bands (early / middle / late thirds)
  MLP write size    ||mlp|| / ||residual|| per band (how much the MLPs rewrite the stream at the answer position)
  MLP alignment     cosine(mlp output, residual after the layer) per band (does the MLP push along the final answer direction)
  probes            leave-one-problem-out logistic probe (per model) on the MLP outputs vs the attention outputs of the middle third
Olympiad (75 rows, DeepSeek / Skywork) and the 49 rows independent of main_v2; main_v2 rows with an extras file (Qwen, Skywork, Olmo)."""
import json, os, pickle, re, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "2"
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, "/home/sabyasachi19/PycharmProjects/AimoAGAIN/analysis")
from common import *                                                                    # noqa: E402,F401

FR = ROOT / "extra_olympiad_problems_dataset" / "olympiad-features"; MVX = ROOT / "extraction_out_generations_pearl_more" / "extraction_out_generations"
D47 = pickle.load(open(A / "exp47_features.pkl", "rb"))
allm, _ = load_all(); rows = datasets(allm)["main_v2"]
norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()[:300]                            # noqa: E731
inp = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "t1_run" / "inputs.jsonl")}
rng = np.random.default_rng(0)


def load(path, which):
    z = np.load(path, allow_pickle=True); j = 0 if which == "first" else len(z["cut_eff"]) - 1; v = list(z["variant_names"]).index("B")
    return z["res_last"][j, v].astype(np.float32), z["attn_last"][j, v].astype(np.float32), z["mlp_last"][j, v].astype(np.float32)


def feats(res, att, mlp):
    L = len(res); bands = {"early": range(0, L // 3), "middle": range(L // 3, 2 * L // 3), "late": range(2 * L // 3, L)}
    nm, na, nr = np.linalg.norm(mlp, axis=1), np.linalg.norm(att, axis=1), np.linalg.norm(res, axis=1)
    cos = (mlp * res).sum(1) / (nm * nr + 1e-9); out = {}
    for b, r in bands.items():
        r = list(r)
        out[f"MLP share of the update ({b})"] = float(np.mean(nm[r] / (nm[r] + na[r] + 1e-9)))
        out[f"MLP write size / residual ({b})"] = float(np.mean(nm[r] / (nr[r] + 1e-9)))
        out[f"MLP alignment with residual ({b})"] = float(np.mean(cos[r]))
    return out


def within(x, y, mm):
    num = den = 0
    for m in np.unique(mm):
        k = (mm == m) & np.isfinite(x); p, n = x[k & (y == 1)], x[k & (y == 0)]
        num += (p[:, None] > n[None]).sum() + .5 * (p[:, None] == n[None]).sum(); den += len(p) * len(n)
    return num / den if den else np.nan


def perm(x, y, mm, n=2000):
    w = within(x, y, mm); nl = []
    for _ in range(n):
        yp = y.copy()
        for m in np.unique(mm): k = mm == m; yp[k] = rng.permutation(y[k])
        nl.append(within(x, yp, mm))
    return w, float(np.mean(np.abs(np.array(nl) - .5) >= abs(w - .5)))


def probe(X, y, groups, nc=5):
    out = np.full(len(y), np.nan)
    for p in np.unique(groups):
        te = groups == p; tr = ~te
        if len(np.unique(y[tr])) < 2: continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; pca = PCA(min(nc, tr.sum() - 1), svd_solver="full").fit((X[tr] - mu) / sd)
        out[te] = LogisticRegression(C=0.1, max_iter=3000).fit(pca.transform((X[tr] - mu) / sd), y[tr]).decision_function(pca.transform((X[te] - mu) / sd))
    return out


for which in ("first", "last"):
    # olympiad
    ids, y, mm, ind = D47["ids"], D47["y"].astype(int), D47["model"], D47["indep"]; groups = np.array([norm(inp[i]["problem"]) for i in ids])
    F, PR = {}, {"MLP output, middle third": np.full(len(ids), np.nan), "attention output, middle third": np.full(len(ids), np.nan)}
    data = {}
    for i, rid in enumerate(ids):
        p = FR / mm[i].replace("/", "__") / "extra" / f"{rid}.npz"
        if not p.exists(): continue
        res, att, mlp = load(p, which); data[i] = (att, mlp)
        for k, v in feats(res, att, mlp).items(): F.setdefault(k, np.full(len(ids), np.nan))[i] = v
    for m in np.unique(mm):
        idx = np.array([i for i in data if mm[i] == m]); L = data[idx[0]][0].shape[0]; mid = range(L // 3, 2 * L // 3)
        for nm_, j in (("MLP output, middle third", 1), ("attention output, middle third", 0)):
            X = np.stack([data[i][j][list(mid)].ravel() for i in idx]); PR[nm_][idx] = probe(X, y[idx], groups[idx])
    print(f"\n=== cut: {which} ===  {'feature':44s} {'olympiad all':>14s} {'indep 49':>14s}")
    for k, v in {**F, **{f'probe: {a}': b for a, b in PR.items()}}.items():
        a = perm(v, y, mm); b = perm(v[ind], y[ind], mm[ind])
        print(f"{'':26s}{k:44s} {a[0]:6.2f} ({a[1]:.3f}) {b[0]:6.2f} ({b[1]:.3f})")
    # main_v2 (descriptive features only; too few rows per model for probes)
    Mv, my, mmod = {}, [], []
    for i in rows:
        p = MVX / allm.model.values[i] / "extra" / f"{allm.id.iloc[i]}.npz"
        if not p.exists(): continue
        for k, v in feats(*load(p, which)).items(): Mv.setdefault(k, []).append(v)
        my.append(int(allm.y.values[i] == 1)); mmod.append(allm.model.values[i])
    my, mmod = np.array(my), np.array(mmod)
    print(f"{'':26s}main_v2 rows with extras: {len(my)} ({', '.join(f'{m[:8]} {np.sum(mmod == m)}' for m in np.unique(mmod))})")
    for k, v in Mv.items():
        a = perm(np.array(v), my, mmod); print(f"{'':26s}{'  main_v2: ' + k:44s} {a[0]:6.2f} ({a[1]:.3f})")
