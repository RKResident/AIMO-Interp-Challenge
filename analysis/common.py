"""Shared helpers: load the pooled meta table and evaluate feature sets with the agreed protocol.
 - agg (DeepSeek) and main_v2 (4 models pooled) are cross-validated SEPARATELY; main_v2 folds are grouped by problem.
 - every result comes with a shuffled-label null, a length-only control and a length-removed variant.
 - no gpt_proxy rows."""
import pickle, json, re, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "analysis"
OUT = ROOT / "extraction_out"
DS = "deepseek-ai__DeepSeek-R1-0528-Qwen3-8B"
MODELS = [DS, "Qwen__Qwen3.5-4B", "Skywork__Skywork-OR1-Math-7B", "allenai__Olmo-3-7B-Think"]
HF = {DS: "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "Qwen__Qwen3.5-4B": "Qwen/Qwen3.5-4B",
      "Skywork__Skywork-OR1-Math-7B": "Skywork/Skywork-OR1-Math-7B", "allenai__Olmo-3-7B-Think": "allenai/Olmo-3-7B-Think"}


def load_all():
    metas, feats = [], {}
    for m in MODELS:
        d = pickle.load(open(A / f"feats_{m}.pkl", "rb"))
        metas.append(d["meta"].assign(model=m)); feats[m] = d["F"]
    allm = pd.concat(metas, ignore_index=True)
    allm["y"] = allm.is_robust.map({True: 1, False: 0, "True": 1, "False": 0}).astype(float)
    allm["grp"] = allm.model + allm.source
    return allm, feats


def datasets(allm):
    agg = np.where((allm.model == DS) & (allm.source == "agg") & allm.y.notna())[0]
    main = np.where((allm.source == "main_v2") & allm.y.notna())[0]
    return {"agg": agg, "main_v2": main}


def zgroup(X, groups):
    """z-score every column within its (model x dataset) group; NaN -> 0."""
    X = np.asarray(X, dtype=np.float64).copy()
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < 3:
            continue
        mu, sd = np.nanmean(X[m], 0), np.nanstd(X[m], 0)
        X[m] = (X[m] - mu) / np.where(sd > 1e-8, sd, 1.0)
    return np.nan_to_num(X)


def lengths(allm):
    L = np.log(allm[["n_prompt", "n_gen", "L_think"]].astype(float).values + 1)
    return zgroup(L, allm.grp.values)


def folds(rows, y, groups, seed, k=5):
    g = groups[rows]
    if len(np.unique(g)) == len(rows):
        return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(rows, y[rows]))
    ug = np.unique(g)
    order = np.random.default_rng(seed).permutation(ug)
    fo = {p: i % k for i, p in enumerate(order)}
    f = np.array([fo[p] for p in g])
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k) if (f == i).any()]


def oof_scores(X, rows, y, groups, C=0.1, reps=5, seed=0):
    s, c = np.zeros(len(rows)), np.zeros(len(rows))
    for r in range(reps):
        for tr, te in folds(rows, y, groups, seed + r):
            yy = y[rows[tr]]
            if len(np.unique(yy)) < 2:
                continue
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(X[rows[tr]], yy)
            s[te] += m.decision_function(X[rows[te]]); c[te] += 1
    return s / np.maximum(c, 1)


def auc_ci(yt, s, n=1000, seed=0):
    yt = np.asarray(yt)
    if len(np.unique(yt)) < 2:
        return np.nan, np.nan, np.nan
    a = roc_auc_score(yt, s)
    rng = np.random.default_rng(seed); bs = []
    for _ in range(n):
        i = rng.integers(0, len(yt), len(yt))
        if len(np.unique(yt[i])) == 2:
            bs.append(roc_auc_score(yt[i], s[i]))
    return (a, *np.percentile(bs, [2.5, 97.5]))


def shuffled_y(allm, rows, ds, seed):
    y2 = allm.y.values.copy()
    if ds == "agg":
        y2[rows] = np.random.default_rng(seed).permutation(y2[rows])
    else:  # labels agree across models for a problem -> shuffle at the problem level
        pr = allm.base_id.values[rows]; up = np.unique(pr); lab = {p: y2[rows][pr == p][0] for p in up}
        pm = dict(zip(up, np.random.default_rng(seed).permutation(up)))
        y2[rows] = [lab[pm[p]] for p in pr]
    return y2


def evaluate(name, X, allm, C=0.1, reps=5, nulls=4, quiet=False, resid=True, rowsets=None):
    """X: [len(allm), d] raw feature matrix. Returns {dataset: dict}. Prints one block per dataset.
    rowsets: optional {dataset: row indices} to evaluate on a subset (e.g. traces still running at a budget)."""
    ds_rows = rowsets if rowsets is not None else datasets(allm)
    Z = lengths(allm)
    grp = allm.grp.values
    Xz = zgroup(X, grp)
    pb = np.unique(allm.base_id.values, return_inverse=True)[1]
    res = {}
    for ds, rows in ds_rows.items():
        y = allm.y.values
        out = {}
        if len(rows) < 8 or len(np.unique(y[rows])) < 2 or min(np.bincount(y[rows].astype(int))) < 2:
            if not quiet:
                print(f"  {name:34s} {ds:8s} n={len(rows):3d}  too few rows / one class - skipped", flush=True)
            continue
        s = oof_scores(Xz, rows, y, pb, C, reps)
        out["auc"] = auc_ci(y[rows], s)
        nl = []
        for sd in range(nulls):
            y2 = shuffled_y(allm, rows, ds, sd)
            nl.append(roc_auc_score(y2[rows], oof_scores(Xz, rows, y2, pb, C, 2, seed=50 + sd)))
        out["null"] = float(np.mean(nl))
        out["len"] = roc_auc_score(y[rows], oof_scores(Z, rows, y, pb, C, reps))
        if resid:
            Xr = np.nan_to_num(Xz.copy()); Xr[rows] = Xr[rows] - LinearRegression().fit(Z[rows], Xr[rows]).predict(Z[rows])
            out["resid"] = auc_ci(y[rows], oof_scores(Xr, rows, y, pb, C, reps))
        res[ds] = out
        if not quiet:
            r = out.get("resid", (np.nan,) * 3)
            print(f"  {name:34s} {ds:8s} n={len(rows):3d} d={Xz.shape[1]:3d}  AUC {out['auc'][0]:.2f} [{out['auc'][1]:.2f},{out['auc'][2]:.2f}]  "
                  f"null {out['null']:.2f}  length-only {out['len']:.2f}  | length-removed {r[0]:.2f} [{r[1]:.2f},{r[2]:.2f}]", flush=True)
    return res


def single_auc(F, allm, names=None, top=12, min_n=20):
    """Single-column AUCs on each dataset (sign kept: >0.5 means higher -> more robust)."""
    F = pd.DataFrame(F) if not isinstance(F, pd.DataFrame) else F
    for ds, rows in datasets(allm).items():
        yy = allm.y.values[rows]
        rr = []
        for c in F.columns:
            x = F[c].astype(float).values[rows]; ok = np.isfinite(x)
            if ok.sum() >= min_n and len(np.unique(x[ok])) > 1 and len(np.unique(yy[ok])) == 2:
                a = roc_auc_score(yy[ok], x[ok]); rr.append((c, a, int(ok.sum())))
        rr.sort(key=lambda t: -abs(t[1] - .5))
        print(f"  [{ds}] top single-column AUCs (n={len(rows)}): " + "; ".join(f"{c} {a:.2f} (n={n})" for c, a, n in rr[:top]))


def nrm(s):
    if s is None:
        return None
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", str(s)).strip().strip("$").replace(",", "").replace("\\!", "").replace("\\,", "").replace("\\dfrac", "\\frac")
    return re.sub(r"\s+", "", s).rstrip(".")


def force_json(model, pid):
    return json.load(open(OUT / model / "force" / f"{pid}.json"))


# ---------------------------------------------------------------- length-invariance helpers
def length_bins(allm, rows, nbins=5):
    """quantile bins of log thinking length, computed within the rows of one dataset."""
    L = np.log(allm.L_think.astype(float).values[rows] + 1)
    edges = np.quantile(L, np.linspace(0, 1, nbins + 1)[1:-1])
    return np.digitize(L, edges)


def within_bin_auc(y, s, bins):
    """AUC over robust/non-robust pairs that fall in the SAME length bin (pair-count weighted). Length differences cannot contribute."""
    num = den = 0.0
    for b in np.unique(bins):
        m = bins == b
        yb, sb = y[m], s[m]
        n1, n0 = int(yb.sum()), int(len(yb) - yb.sum())
        if n1 and n0:
            num += roc_auc_score(yb, sb) * n1 * n0; den += n1 * n0
    return num / den if den else np.nan


def shuffled_y_len(allm, rows, ds, seed, nbins=5):
    """label shuffle that preserves the length-label relation: permute only within length bins (problem level for main_v2)."""
    y2 = allm.y.values.copy(); rng = np.random.default_rng(seed); bins = length_bins(allm, rows, nbins)
    if ds == "agg":
        for b in np.unique(bins):
            idx = rows[bins == b]; y2[idx] = rng.permutation(y2[idx])
    else:
        pr = allm.base_id.values[rows]; up = np.unique(pr)
        pbin = {p: int(np.round(np.mean(bins[pr == p]))) for p in up}; lab = {p: y2[rows][pr == p][0] for p in up}
        for b in set(pbin.values()):
            ps = [p for p in up if pbin[p] == b]; perm = dict(zip(ps, rng.permutation(ps)))
            for i, p in zip(rows, pr):
                if p in perm: y2[i] = lab[perm[p]]
    return y2
