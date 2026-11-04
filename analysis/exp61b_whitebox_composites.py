"""exp61b: do the white-box signals add to the shipped recipe? Equal-weight percentile ensembles (per-model references from the training
fold, directions chosen in the fold, threshold = each model's median training score), leave one PROBLEM out, evaluated on the held-out
UNFINISHED rows, trained on unfinished rows (the calibration we now use). 'deployable' = computable at test time without the friend's
rewrites (automatic perturbations, forcing the answer with no / truncated reasoning, internal noise)."""
import pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/sabyasachi19/PycharmProjects/AimoAGAIN/analysis")
from common import A
W = pickle.load(open(A / "exp61_whitebox.pkl", "rb")); X, Mt = W["X"], W["meta"]
D56 = pd.read_csv(A / "exp56_rows.csv").set_index("id")
for f in ("fa", "chk", "fas"): X[f] = D56.loc[Mt.id, f].values
y, mm, grp, unf = Mt.y.values, Mt.model.values, Mt.group.values, Mt.unf.values.astype(bool)
def auc(x, yy):
    ok = np.isfinite(x); x, yy = x[ok], yy[ok]; p, n = x[yy == 1], x[yy == 0]
    return ((p[:, None] > n[None]).sum() + .5 * (p[:, None] == n[None]).sum()) / (len(p) * len(n)) if len(p) and len(n) else np.nan
def within(s, yy, m_):
    num = den = 0
    for m in np.unique(m_):
        k = (m_ == m) & np.isfinite(s); p, n = s[k & (yy == 1)], s[k & (yy == 0)]
        num += (p[:, None] > n[None]).sum() + .5 * (p[:, None] == n[None]).sum(); den += len(p) * len(n)
    return num / den if den else np.nan
def loo(cols):
    V = X[cols].values.astype(float); sc = np.full(len(X), np.nan); dec = np.full(len(X), np.nan)
    for g in np.unique(grp[unf]):
        tr = (grp != g) & unf; te = np.where((grp == g) & unf)[0]
        sg = []
        for j in range(len(cols)):
            a = within(V[tr, j], y[tr], mm[tr]); sg.append(1 if (np.isnan(a) or a >= .5) else -1)
        def s(i, ref):
            u = []
            for j in range(len(cols)):
                r = V[ref & (mm == mm[i]), j]; r = r[np.isfinite(r)]
                if not np.isfinite(V[i, j]) or not len(r): continue
                q = np.mean(r < V[i, j]) + .5 * np.mean(r == V[i, j]); u.append(q if sg[j] > 0 else 1 - q)
            return np.mean(u) if u else .5
        for m in np.unique(mm[te]):
            ref = tr & (mm == m)
            if ref.sum() < 3: continue
            thr = np.median([s(k, tr) for k in np.where(ref)[0]])
            for i in te[mm[te] == m]: sc[i] = s(i, tr); dec[i] = sc[i] > thr
    k = unf & np.isfinite(sc); d = dec[k].astype(bool); yy = y[k]
    return within(sc[k], yy, mm[k]), float(np.mean(d == yy)), .5 * (d[yy == 1].mean() + (1 - d[yy == 0]).mean()), int(k.sum())
SETS = {"shipped: confidence + check share + slope": ["fa", "chk", "fas"],
        "white-box deployable: no-reasoning kept + first-1024 kept": ["reasoning ablation 'no reasoning': answer kept", "reasoning ablation 'first 1024 only': answer kept"],
        "white-box deployable: + auto rewording (state cosine, log-lik change)": ["reasoning ablation 'no reasoning': answer kept", "reasoning ablation 'first 1024 only': answer kept",
                                                                                   "cf auto: state cosine, last quarter", "cf auto: trace log-lik change"],
        "white-box deployable: + noise flips": ["reasoning ablation 'no reasoning': answer kept", "reasoning ablation 'first 1024 only': answer kept",
                                                "cf auto: state cosine, last quarter", "cf auto: trace log-lik change", "noise: flip rate (-)"],
        "shipped + no-reasoning kept + first-1024 kept": ["fa", "chk", "fas", "reasoning ablation 'no reasoning': answer kept", "reasoning ablation 'first 1024 only': answer kept"],
        "shipped + all deployable white-box": ["fa", "chk", "fas", "reasoning ablation 'no reasoning': answer kept", "reasoning ablation 'first 1024 only': answer kept",
                                               "cf auto: state cosine, last quarter", "cf auto: trace log-lik change", "noise: flip rate (-)"],
        "reference (not deployable): friend rewrites state cosine + log-lik change": ["cf friend: state cosine, last quarter", "cf friend: trace log-lik change, last quarter"]}
print(f"{'ensemble (held-out unfinished rows, LOO by problem)':78s} {'AUC':>5s} {'acc':>5s} {'bal':>5s}   n")
for nm, cols in SETS.items():
    a, acc, bal, n = loo(cols); print(f"{nm:78s} {a:5.2f} {acc:5.2f} {bal:5.2f}  {n}")
