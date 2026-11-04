"""exp57: tuning / training on the UNFINISHED traces only (thinking not ended within 6144 tokens): 88 rows (main_v2 30 + olympiad 58),
50 problems. Every score is out of sample on unfinished rows: leave-one-PROBLEM-out over both sets together (a problem appearing in both
is one group). Two training pools are compared for every method, always evaluated on the held-out UNFINISHED rows:
  'unfinished only'   train on the unfinished rows of the other problems
  'all rows'          train on all rows (finished + unfinished) of the other problems  <- what we have been doing
Methods:
  ensembles   per-model percentile references, signs from main_v2 / olympiad training direction (+1 for all listed features), threshold =
              median training score (the test prior is 50/50): confidence + check share + slope (the 0.71 recipe); answer-free
              check share + lookback late-early + prompt mass late-early; all five
  probes      per model, standardise -> PCA(3) -> logistic (C 0.1), on: the forced-suffix state @6144, the last-cutoff state (mean of last 64),
              LapEigvals top-8 forced B; TOHA with the paper's head selection (window @2048) for the models that have it
Reported: within-model AUC and accuracy at the training threshold on the held-out unfinished rows, per training pool."""
import json, os, pickle, re, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "2"
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, "/home/sabyasachi19/PycharmProjects/AimoAGAIN/analysis")
from common import *                                                                    # noqa: E402,F401

D = pd.read_csv(A / "exp56_rows.csv")                                                   # fa, chk, fas, lble, pmle, finished, labels (both sets)
norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()[:300]                            # noqa: E731
inp = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "t1_run" / "inputs.jsonl")}
D["group"] = [norm(inp[i]["problem"]) for i in D.id]; D["unfinished"] = ~D.finished.astype(bool)
slug = lambda m: m.replace("/", "__")                                                    # noqa: E731
OLY = ROOT / "extra_olympiad_problems_dataset"; MV1 = ROOT / "results_mainv2_t1" / "out_t1"


def state(row, kind):
    """Hidden-state inputs for the probes (None when unavailable)."""
    if kind in ("forced-suffix state B @6144", "last-cutoff state"):
        p = (MV1 if row.dataset == "main_v2" else OLY / "results_olympiad_features" / "out_t1") / slug(row.model_id) / f"{row.id}.npz"
        if not p.exists(): return None
        z = np.load(p)
        if kind == "last-cutoff state": return z["t0_cut_mean64"].astype(np.float32)[-1].ravel()
        ce = list(z["t0_cut_eff"]); j = ce.index(max(c for c in ce if c <= 6144)); return z["t0_force_hidden"].astype(np.float32)[j, list(z["t0_force_suffixes"]).index("B")].ravel()
    if kind == "LapEigvals top-8 forced B":
        p = (ROOT / "extraction_out" / slug(row.model_id) / "attn" if row.dataset == "main_v2" else OLY / "results_olympiad_attention" / "out_attn" / slug(row.model_id) / "attn") / f"{row.id}.npz"
        if not p.exists(): return None
        z = np.load(p); ce = list(z["cut_eff"]); j = ce.index(max(c for c in ce if c <= 6144))
        return np.log(np.clip(np.nan_to_num(z["topk_force"][j, list(z["variant_names"]).index("B")][..., :8].astype(np.float32)), 1e-8, None)).ravel()
    if kind == "TOHA window @2048 (per head)":
        p = (ROOT / "extraction_out_generations_p2" / "extraction_out_generations" / slug(row.model_id) / "toha" if row.dataset == "main_v2" else OLY / "olympiad-features" / slug(row.model_id) / "toha") / f"{row.id}.npz"
        if not p.exists(): return None
        z = np.load(p, allow_pickle=True)
        if len(z["cut_eff"]) < 2: return None
        return z["win"][1, :, :, list(z["stat_names"]).index("mean_edge")].astype(np.float64).ravel()
    raise KeyError(kind)


def auc(x, y):
    x, y = np.asarray(x, float), np.asarray(y); ok = np.isfinite(x); x, y = x[ok], y[ok]; p, n = x[y == 1], x[y == 0]
    return ((p[:, None] > n[None]).sum() + .5 * (p[:, None] == n[None]).sum()) / (len(p) * len(n)) if len(p) and len(n) else np.nan


def within(score, y, model):
    num = den = 0
    for m in np.unique(model):
        k = (model == m) & np.isfinite(score); p, n = score[k & (y == 1)], score[k & (y == 0)]
        num += (p[:, None] > n[None]).sum() + .5 * (p[:, None] == n[None]).sum(); den += len(p) * len(n)
    return num / den if den else np.nan


def pct(x, ref):
    ref = ref[np.isfinite(ref)]
    return ((ref < x).sum() + .5 * (ref == x).sum()) / len(ref) if np.isfinite(x) and len(ref) else np.nan


def ensemble(feats, pool):
    """Out-of-fold score and decision for every unfinished row."""
    U = D[D.unfinished]; sc, dec = {}, {}
    for g in U.group.unique():
        tr = D[(D.group != g) & (D.unfinished if pool == "unfinished only" else True)]
        def score(row, tr_m):
            u = [pct(row[f], tr_m[f].values) for f in feats]; u = [v for v in u if np.isfinite(v)]; return np.mean(u) - .5 if u else 0.0
        thr = {m: np.median([score(r, tr[tr.model_id == m]) for _, r in tr[tr.model_id == m].iterrows()]) for m in tr.model_id.unique()}
        for i, r in U[U.group == g].iterrows():
            s = score(r, tr[tr.model_id == r.model_id]); sc[i] = s; dec[i] = s > thr.get(r.model_id, 0.0)
    return U.index, np.array([sc[i] for i in U.index]), np.array([dec[i] for i in U.index])


def probe(kind, pool, nc=3, toha=False):
    U = D[D.unfinished]; sc, dec = {}, {}
    X = {i: state(r, kind) for i, r in D.iterrows()}
    for m in D.model_id.unique():
        idx_all = [i for i in D.index if D.model_id[i] == m and X[i] is not None]
        if len(idx_all) < 8 or len({len(X[i]) for i in idx_all}) > 1: continue
        for g in U[U.model_id == m].group.unique():
            tr = [i for i in idx_all if D.group[i] != g and (D.unfinished[i] if pool == "unfinished only" else True)]
            te = [i for i in idx_all if D.group[i] == g and D.unfinished[i]]
            ytr = D.robust[tr].values
            if not te or len(np.unique(ytr)) < 2: continue
            Xtr = np.stack([X[i] for i in tr]); Xte = np.stack([X[i] for i in te])
            if toha:                                                          # paper's Algorithm 1 head selection on the training rows
                delta = np.nanmean(Xtr[ytr == 0], 0) - np.nanmean(Xtr[ytr == 1], 0); order = np.argsort(-np.nan_to_num(delta, nan=-np.inf))
                best = max(range(1, 11), key=lambda n: auc(-Xtr[:, order[:n]].mean(1), ytr)); h = order[:best]
                s_tr, s_te = -Xtr[:, h].mean(1), -Xte[:, h].mean(1)
            else:
                mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; pca = PCA(min(nc, len(tr) - 1), svd_solver="full").fit((Xtr - mu) / sd)
                clf = LogisticRegression(C=0.1, max_iter=3000).fit(pca.transform((Xtr - mu) / sd), ytr)
                s_tr, s_te = clf.decision_function(pca.transform((Xtr - mu) / sd)), clf.decision_function(pca.transform((Xte - mu) / sd))
            thr = np.median(s_tr)
            for i, s in zip(te, s_te): sc[i] = s; dec[i] = s > thr
    idx = [i for i in U.index if i in sc]
    return np.array(idx), np.array([sc[i] for i in idx]), np.array([dec[i] for i in idx])


rows_out = []
print(f"unfinished rows: {int(D.unfinished.sum())} ({D[D.unfinished].group.nunique()} problems; {int(D[D.unfinished].robust.sum())} robust)\n")
print(f"{'method':52s} {'trained on':16s} {'n':>3s} {'AUC (within model)':>19s} {'accuracy':>9s} {'balanced':>9s} {'called robust':>14s}")
methods = [("ensemble: confidence + check share + slope", lambda p: ensemble(["fa", "chk", "fas"], p)),
           ("ensemble: check share + lookback + prompt mass", lambda p: ensemble(["chk", "lble", "pmle"], p)),
           ("ensemble: all five", lambda p: ensemble(["fa", "chk", "fas", "lble", "pmle"], p)),
           ("probe: forced-suffix state B @6144", lambda p: probe("forced-suffix state B @6144", p)),
           ("probe: last-cutoff state", lambda p: probe("last-cutoff state", p)),
           ("probe: LapEigvals top-8 forced B", lambda p: probe("LapEigvals top-8 forced B", p)),
           ("TOHA head selection: window @2048", lambda p: probe("TOHA window @2048 (per head)", p, toha=True))]
for nm, fn in methods:
    for pool in ("unfinished only", "all rows"):
        idx, s, d = fn(pool); y = D.robust[idx].values; mm = D.model_id[idx].values
        bal = .5 * (d[y == 1].mean() + (1 - d[y == 0]).mean())
        print(f"{nm:52s} {pool:16s} {len(idx):3d} {within(s, y, mm):19.2f} {np.mean(d == y):9.2f} {bal:9.2f} {int(d.sum()):6d}/{len(idx)} (true {int(y.sum())})")
        rows_out.append(dict(method=nm, pool=pool, n=len(idx), auc=within(s, y, mm), acc=np.mean(d == y), bal=bal))
pd.DataFrame(rows_out).to_csv(A / "exp57_results.csv", index=False)
