"""exp70: TOHA (paper's Algorithm 1: per-head normalised MSF length on the last 128 reasoning tokens before the 2048 cut, top-N heads by
separation, N <= 10 by training AUROC) on the UNFINISHED rows, per model, head + N selection nested in leave-one-problem-out - the model
where the jump fails (DeepSeek) is the one where TOHA was best on the olympiad rows (0.84, exp54). Also: does TOHA add to the recipes?"""
import os, pickle, re
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "2"
import numpy as np, pandas as pd
src = open("/home/sabyasachi19/PycharmProjects/AimoAGAIN/analysis/exp54_toha_head_selection.py").read()
exec(src[src.index("def head_d("):src.index("def within(")])
ROOT = __import__("pathlib").Path("/home/sabyasachi19/PycharmProjects/AimoAGAIN"); A = ROOT / "analysis"
OLY = ROOT / "extra_olympiad_problems_dataset" / "olympiad-features"; P2 = ROOT / "extraction_out_generations_p2" / "extraction_out_generations"
D = pd.read_csv(A / "exp56_rows.csv"); W = pickle.load(open(A / "exp61_whitebox.pkl", "rb")); G = dict(zip(W["meta"].id, W["meta"].group))
U = D[~D.finished.astype(bool)].reset_index(drop=True); slug = lambda m: m.replace("/", "__")             # noqa: E731
print(f"{len(U)} unfinished rows")
for kind, ci in (("win", 1), ("win", 0), ("ans", 1)):
    print(f"\nTOHA {kind} @ cut {ci} (0 = 1024, 1 = 2048)")
    for m, g in U.groupby("model"):
        Xs, ys, gs = [], [], []
        for _, r in g.iterrows():
            p = (P2 / slug(r.model_id) / "toha" if r.dataset == "main_v2" else OLY / slug(r.model_id) / "toha") / f"{r.id}.npz"
            if not p.exists(): continue
            v = head_d(p, kind, ci)
            if v is None: continue
            Xs.append(v); ys.append(int(r.robust)); gs.append(G.get(r.id, r.id))
        if len(set(ys)) < 2 or len(ys) < 6:
            print(f"   {m:9s} n={len(ys)} - too few"); continue
        X, yy, gg = np.array(Xs), np.array(ys), np.array(gs); s, picks = oof(X, yy, gg)
        print(f"   {m:9s} n={len(yy):2d} ({yy.sum()} robust) nested held-out AUC {auc(s[np.isfinite(s)], yy[np.isfinite(s)]):.2f} | N median {int(np.median(picks)) if picks else '-'}")

# does nested TOHA (win @ 2048) add to DeepSeek's recipes? equal-weight within-model percentiles, leave-one-problem-out
dd = U[U.model == "deepseek"].reset_index(drop=True); Xs, keep = [], []
for i, r in dd.iterrows():
    p = (P2 / slug(r.model_id) / "toha" if r.dataset == "main_v2" else OLY / slug(r.model_id) / "toha") / f"{r.id}.npz"
    v = head_d(p, "win", 1) if p.exists() else None
    if v is not None: Xs.append(v); keep.append(i)
dd = dd.loc[keep].reset_index(drop=True); X = np.array(Xs); yy = dd.robust.values.astype(int); gg = np.array([G.get(i, i) for i in dd.id])
dd["toha"], _ = oof(X, yy, gg)
W2 = W["X"].copy(); W2["id"] = W["meta"].id.values
W2 = W2.rename(columns={"reasoning ablation 'first 1024 only': answer kept": "f1k", "cf auto: state cosine, last quarter": "cfc", "cf auto: trace log-lik change": "cfl"})
dd = dd.merge(W2[["id", "f1k", "cfc", "cfl"]], on="id", how="left")
SG = {"chk": 1, "fas": 1, "toha": 1, "f1k": -1, "cfc": -1, "cfl": -1}
def pc(x, ref): ref = ref[np.isfinite(ref)]; return ((ref < x).sum() + .5 * (ref == x).sum()) / len(ref)
print("\nDeepSeek unfinished, held-out equal-weight combinations (AUC / accuracy with the per-fold median threshold)")
for fs in (["chk", "fas"], ["toha"], ["chk", "fas", "toha"], ["chk", "f1k", "cfc", "cfl"], ["chk", "f1k", "cfc", "cfl", "toha"], ["chk", "toha"]):
    s, c = np.full(len(dd), np.nan), np.zeros(len(dd), bool)
    for g0 in np.unique(gg):
        te = gg == g0; tr = ~te
        sc = lambda rows: np.array([np.mean([(pc(dd.at[i, f], dd.loc[tr, f].values) if SG[f] > 0 else 1 - pc(dd.at[i, f], dd.loc[tr, f].values)) - .5 for f in fs]) for i in rows])  # noqa: E731
        st = sc(np.where(tr)[0]); s[te] = sc(np.where(te)[0]); c[te] = s[te] > np.median(st)
    print(f"   {'+'.join(fs):24s} {auc(s, yy):.2f} / {(c == (yy == 1)).mean():.2f}")
