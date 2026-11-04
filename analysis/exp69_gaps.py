"""exp69: gaps in what we have tried, tested on the 88 unfinished greedy rows (leave-one-PROBLEM-out, everything refitted per fold).
  G1 Lookback Lens as the PAPER does it (Chuang et al. 2024): a logistic regression on the per-head lookback ratios (mean over the trace),
     one classifier per model (we only ever used scalar late-minus-early summaries); plus the scalar summaries recalibrated on unfinished rows.
  G2 unused data: the 86 main_v2 T=1 samples as EXTRA reference / calibration rows (same label as their problem; only the unfinished ones;
     the held-out problem's samples are excluded) - matters most for Qwen (10 rows) and Olmo (5).
  G3 honest per-model selection: the recipe for each model chosen INSIDE each fold (best training-row accuracy among the candidates), so the
     held-out number carries the selection; compared with fixed recipes.
  G4 threshold: each model's median training score (shipped) vs the quantile matching that model's robust share in its training rows."""
import glob, json, os, pickle
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "2"
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = __import__("pathlib").Path("/home/sabyasachi19/PycharmProjects/AimoAGAIN"); A = ROOT / "analysis"
D = pd.read_csv(A / "exp56_rows.csv"); W = pickle.load(open(A / "exp61_whitebox.pkl", "rb")); X = W["X"].copy(); X["id"] = W["meta"].id.values
X["group"] = W["meta"].group.values; X["nzf"] = -X["noise: flip rate (-)"]
X = X.rename(columns={"reasoning ablation 'no reasoning': answer kept": "nrk", "reasoning ablation 'first 1024 only': answer kept": "f1k",
                      "cf auto: state cosine, last quarter": "cfc", "cf auto: trace log-lik change": "cfl"})
U = D[~D.finished.astype(bool)].merge(X[["id", "group", "nrk", "f1k", "cfc", "cfl", "nzf"]], on="id").merge(pd.read_csv(A / "exp63_jump.csv"), on="id")
U = U.reset_index(drop=True); y = U.robust.values.astype(int); grp = U.group.values; mdl = U.model.values
SIGN = {"fa": 1, "chk": 1, "fas": 1, "nrk": -1, "f1k": -1, "cfc": -1, "cfl": -1, "nzf": -1, "jmp": 1, "lbm": 1, "lble": 1, "lblq": 1}
MODELS = ["Qwen", "Skywork", "allenai", "deepseek"]
WB = ROOT / "results_whitebox" / "out_whitebox"; slug = lambda m: m.replace("/", "__")                                # noqa: E731
MID = {r["id"]: r["model_id"] for f in WB.glob("*/rows_shard*.jsonl") for r in map(json.loads, open(f))}
rng = np.random.default_rng(0)


def auc(s, yy):
    ok = ~np.isnan(s); s, yy = s[ok], yy[ok]; p, n = s[yy == 1], s[yy == 0]
    return float(((p[:, None] > n[None]) + .5 * (p[:, None] == n[None])).mean()) if len(p) and len(n) else np.nan


def pct(x, ref):
    ref = np.asarray(ref, float); ref = ref[np.isfinite(ref)]
    return ((ref < x).sum() + .5 * (ref == x).sum()) / len(ref) if len(ref) and np.isfinite(x) else .5


# ------------------------------------------------------------------ G1 Lookback Lens
LBH = {}
for i, rid in enumerate(U.id):
    z = np.load(WB / slug(MID[rid]) / f"{rid}.npz"); lb = z["att_lookback"].astype(np.float32)        # [n_span, L, H]
    n = len(lb); q = max(1, n // 4); LBH[rid] = lb.mean(0).reshape(-1)
    U.loc[i, "lbm"] = float(lb.mean()); U.loc[i, "lble"] = float(lb[-q:].mean() - lb[:q].mean()); U.loc[i, "lblq"] = float(lb[-q:].mean())
print("G1 LOOKBACK LENS on unfinished rows (within-model AUC, higher -> robust)")
for c in ("lbm", "lble", "lblq"):
    print(f"   scalar {c:5s} {auc(U.groupby('model')[c].rank(pct=True).values, y):.2f}   per model " +
          "  ".join(f"{m[:4]} {auc(g[c].values, g.robust.values):.2f}" for m, g in U.groupby('model')))
for C in (0.01, 0.1, 1.0):
    s = np.full(len(U), np.nan)
    for m in MODELS:
        ii = np.where(mdl == m)[0]; Xm = np.array([LBH[U.id[i]] for i in ii]); ym = y[ii]; gm = grp[ii]
        for g in np.unique(gm):
            te, tr = gm == g, gm != g
            if ym[tr].min() == ym[tr].max(): continue
            sc = StandardScaler().fit(Xm[tr]); lr = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(Xm[tr]), ym[tr])
            s[ii[te]] = lr.decision_function(sc.transform(Xm[te]))
    r = pd.Series(s).groupby(mdl).rank(pct=True).values
    print(f"   paper-style per-head logistic C={C:<5}: held-out AUC {auc(r, y):.2f} | per model " + "  ".join(f"{m[:4]} {auc(s[mdl == m], y[mdl == m]):.2f}" for m in MODELS))

# ------------------------------------------------------------------ G2 T=1 samples as extra calibration rows (main_v2)
G66 = pd.read_csv(A / "exp66_graphs_crowding.csv")
t1 = {json.loads(l)["id"]: json.loads(l) for f in glob.glob(str(ROOT / "results_mainv2_t1" / "out_t1" / "*" / "rows_shard*.jsonl")) for l in open(f)}
S = []
for _, r in G66[G66.trace != "greedy"].iterrows():
    k = int(r.trace[-1]); s = t1[r.id]["samples"][k]
    if not pd.isna(r.think_end): continue                                                  # unfinished samples only
    base = U[U.id == r.id]
    if base.empty: base = D[D.id == r.id]
    S.append(dict(id=r.id, model=base.model.iloc[0], robust=int(base.robust.iloc[0]), group=X.set_index("id").group.get(r.id), fa=s["fa"], fas=s["fas"], chk=s["chk"], jmp=r.jump))
S = pd.DataFrame(S)
print(f"\nG2 T=1 SAMPLES AS EXTRA CALIBRATION ROWS: {len(S)} unfinished samples (" + ", ".join(f"{m} {(S.model == m).sum()}" for m in MODELS) + ")")


def evaluate(feats, extra=None, q_mode="median"):
    sc, cl = np.full(len(U), np.nan), np.zeros(len(U), bool)
    for g in np.unique(grp):
        te = grp == g; tr = ~te
        T = U[tr][["model", "robust"] + feats]
        if extra is not None: T = pd.concat([T, extra[extra.group != g][["model", "robust"] + feats]], ignore_index=True)
        def score(row):
            return np.mean([(pct(row[f], T.loc[T.model == row["model"], f]) if SIGN[f] > 0 else 1 - pct(row[f], T.loc[T.model == row["model"], f])) - .5 for f in feats])
        st = T.apply(score, axis=1).values
        thr = {}
        for m in MODELS:
            mm = (T.model == m).values
            if not mm.any(): continue
            qq = .5 if q_mode == "median" else 1 - T.robust.values[mm].mean()
            thr[m] = np.quantile(st[mm], qq)
        for i in np.where(te)[0]:
            sc[i] = score(U.iloc[i]); cl[i] = sc[i] > thr.get(mdl[i], 0)
    return sc, cl


def report(nm, sc, cl, rows=None):
    rows = np.ones(len(U), bool) if rows is None else rows
    per = "  ".join(f"{m[:4]} {auc(sc[rows & (mdl == m)], y[rows & (mdl == m)]):.2f}/{(cl[rows & (mdl == m)] == (y[rows & (mdl == m)] == 1)).mean():.2f}" for m in MODELS)
    print(f"   {nm:34s} all {auc(sc[rows], y[rows]):.2f}/{(cl[rows] == (y[rows] == 1)).mean():.2f} | {per}")


mv = (U.dataset == "main_v2").values
for feats in (["chk"], ["jmp"], ["chk", "jmp"], ["chk", "fas"], ["chk", "fas", "jmp"]):
    a = evaluate(feats); b = evaluate(feats, S)
    print(f"  {'+'.join(feats)} (AUC / accuracy; main_v2 rows only, where samples exist)")
    report("greedy references only", *a, mv); report("+ unfinished T=1 samples", *b, mv)

# ------------------------------------------------------------------ G3 nested per-model selection, G4 threshold quantile
CANDS = {"chk": ["chk"], "jmp": ["jmp"], "chk+jmp": ["chk", "jmp"], "chk+fas": ["chk", "fas"], "jmp+cfc": ["jmp", "cfc"],
         "chk+f1k+cfc+cfl": ["chk", "f1k", "cfc", "cfl"], "all8": ["chk", "fas", "nrk", "f1k", "cfc", "cfl", "nzf", "jmp"]}
print("\nG3 NESTED PER-MODEL SELECTION (candidate chosen per model inside each fold by inner leave-one-problem-out accuracy)")
inner = {}
OUT = {nm: evaluate(fs) for nm, fs in CANDS.items()}                                         # outer held-out, every candidate
for nm in CANDS: report(f"fixed {nm}", *OUT[nm])
# inner accuracy per fold: approximate by the outer held-out calls of the OTHER problems (each is itself held out of its own fold)
sel_sc, sel_cl, picks = np.full(len(U), np.nan), np.zeros(len(U), bool), []
for g in np.unique(grp):
    te = grp == g
    for m in set(mdl[te]):
        mm = (mdl == m) & ~te
        best = max(CANDS, key=lambda nm: ((OUT[nm][1][mm] == (y[mm] == 1)).mean(), auc(OUT[nm][0][mm], y[mm])))
        ii = te & (mdl == m); sel_sc[ii] = OUT[best][0][ii]; sel_cl[ii] = OUT[best][1][ii]; picks.append((m, best))
report("per-model choice, nested", sel_sc, sel_cl)
print("   picks per model:", {m: pd.Series([b for mm, b in picks if mm == m]).value_counts().to_dict() for m in MODELS})
print("\nG4 THRESHOLD: per-model median (shipped) vs the quantile matching the model's robust share in its training rows")
for nm in ("chk+jmp", "all8", "chk+fas"):
    report(f"{nm} median", *OUT[nm]); report(f"{nm} prior-matched", *evaluate(CANDS[nm], q_mode="prior"))
