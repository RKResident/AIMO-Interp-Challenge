"""exp104 (user 2026-10-31): ideas adapted from 'Hyperbolic Geometry of Reasoning' (Raj, ICLR 2026 GRaM tiny paper), label-free, CPU, on the
run_wb2 saved 64-token window means (3 depths: 25 / 50 / 75%) and trace means (every 2nd layer):
  1. Gromov delta-hyperbolicity of each trace's window trajectory (cosine and L2 distances): exact four-point delta w.r.t. 5 base points via the
     (max,min) Gromov-product formula; absolute, relative (2 delta / diameter), first half vs second half.
  2. compression statistics (the paper's last-layer compression): participation ratio, top-1 / top-5 eigenvalue share, mean / CV of window norms,
     ratios deep / shallow depth, and the last / previous stored layer norm ratio of the trace mean.
  3. fixed hyperbolic map: PCA to 5 dims (fitted on the model's whole label-free window pool), tangent vectors scaled so the median norm is 1.5,
     exp_0 onto the Poincare ball (c = 0.5, 1); hyperbolic path length per step, mean pairwise distance, hyperbolic / Euclidean ratio.
Evaluation exactly as exp103: stratified 5-fold x 20, sign by training AUC, threshold = label-free pool median; per family NESTED member choice.
Length check: Spearman of every feature with the trace length, and the unfinished split (all 6144 tokens). Writes exp104_singles.csv, exp104_families.csv."""
import json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

src = open("/home/sabyasachi19/PycharmProjects/AimoAGAIN/analysis/exp99_cv_candidates.py").read()
exec(src[:src.index("out, nested = [], []")])
RNG = np.random.default_rng(4); REP = 20


def delta_hyp(D, bases):
    """four-point delta via Gromov products at base w: delta_w = max_ij (max_k min(G_ik, G_kj) - G_ij); max over the bases"""
    out = 0.0
    for w in bases:
        G = .5 * (D[w][:, None] + D[w][None, :] - D)
        M = np.max(np.minimum(G[:, :, None], G.T[None, :, :]), axis=1)          # (max,min) product G*G
        out = max(out, float((M - G).max()))
    return out


def dists(X, kind):
    if kind == "cos": U = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8); return np.clip(1 - U @ U.T, 0, None)
    s = (X ** 2).sum(1); return np.sqrt(np.clip(s[:, None] + s[None, :] - 2 * X @ X.T, 0, None))


def poincare_d(Z, c):
    n2 = (Z ** 2).sum(1); d2 = np.clip(n2[:, None] + n2[None, :] - 2 * Z @ Z.T, 0, None)
    return np.arccosh(1 + 2 * c * d2 / np.outer(1 - c * n2, 1 - c * n2)) / np.sqrt(c)


def exp0(V, c):
    nv = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12; return np.tanh(np.sqrt(c) * nv) * V / (np.sqrt(c) * nv)   # standard exp_0, inside the ball radius 1/sqrt(c)


rows_s, rows_f = [], []
for m, tag in TAGS.items():
    B = pd.DataFrame([json.loads(l) for l in open(A + f"ec2_results/wb2/{tag}.jsonl")]).drop_duplicates("pid").set_index("pid")[["n_gen", "still_thinking"]]
    Z = {}
    for p in B.index:
        z = np.load(A + f"ec2_results/wb2/{tag}/{p}.npz"); Z[p] = {k: z[k] for k in ("w64_3layers", "trace_mean")}
    pca = {}
    for di in range(3):                                                           # label-free PCA per depth on the model's whole window pool
        allw = np.concatenate([Z[p]["w64_3layers"][di].astype(np.float64) for p in B.index]); mu = allw.mean(0)
        _, _, Vt = np.linalg.svd(allw - mu, full_matrices=False); P5 = (allw - mu) @ Vt[:5].T
        pca[di] = (mu, Vt[:5], 1.5 / np.median(np.linalg.norm(P5, axis=1)))
    F = {}
    for p in B.index:
        f = {}
        for di in range(3):
            W = Z[p]["w64_3layers"][di].astype(np.float64); n = len(W); h = n // 2
            if n < 8: continue
            bases = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
            for kind in ("cos", "l2"):
                D = dists(W, kind); dm = D.max() + 1e-12; dl = delta_hyp(D, bases)
                f[f"delta_{kind}__d{di}"] = dl; f[f"delta_rel_{kind}__d{di}"] = 2 * dl / dm
                d1 = delta_hyp(D[:h, :h], [0, h // 2, h - 1]); d2 = delta_hyp(D[h:, h:], [0, (n - h) // 2, n - h - 1])
                f[f"delta_rel_late_minus_early_{kind}__d{di}"] = 2 * d2 / (D[h:, h:].max() + 1e-12) - 2 * d1 / (D[:h, :h].max() + 1e-12)
            Wc = W - W.mean(0); ev = np.clip(np.linalg.eigvalsh(Wc @ Wc.T / n), 0, None)[::-1]
            f[f"pr__d{di}"] = ev.sum() ** 2 / ((ev ** 2).sum() + 1e-12); f[f"top1_share__d{di}"] = ev[0] / (ev.sum() + 1e-12); f[f"top5_share__d{di}"] = ev[:5].sum() / (ev.sum() + 1e-12)
            nr = np.linalg.norm(W, axis=1); f[f"norm_mean__d{di}"] = nr.mean(); f[f"norm_cv__d{di}"] = nr.std() / nr.mean()
            mu, V5, s = pca[di]; T = (W - mu) @ V5.T * s
            for c in (0.5, 1.0):
                Zb = exp0(T, c); Dp = poincare_d(Zb, c); De = dists(T, "l2")
                f[f"hyp_path__c{c}_d{di}"] = float(np.mean(np.diag(Dp, 1))); f[f"hyp_spread__c{c}_d{di}"] = float(Dp[np.triu_indices(n, 1)].mean())
                f[f"hyp_over_euc__c{c}_d{di}"] = float(Dp[np.triu_indices(n, 1)].mean() / (De[np.triu_indices(n, 1)].mean() + 1e-12))
        for k in ("pr", "top1_share", "norm_mean", "norm_cv"):
            if f"{k}__d0" in f: f[f"{k}_ratio_deep_shallow"] = f[f"{k}__d2"] / (f[f"{k}__d0"] + 1e-12)
        tn = np.linalg.norm(Z[p]["trace_mean"].astype(np.float64), axis=1); f["norm_last_over_prev"] = tn[-1] / tn[-2]; f["norm_slope_layers"] = float(np.polyfit(np.arange(len(tn)), tn / tn.mean(), 1)[0])
        F[p] = f
    X0 = pd.DataFrame.from_dict(F, orient="index"); B = B.join(X0)
    fams = {}
    for c in X0.columns:
        b = c.split("__")[0]
        fam = ("delta " if b.startswith("delta") else "hyperbolic " if b.startswith("hyp") else "compression ") + b
        fams.setdefault(fam, []).append(c)
    feats = list(X0.columns); FI = {f: j for j, f in enumerate(feats)}; POOL = B[feats].astype(float).values; MED = np.nanmedian(POOL, 0)
    LAB = labels(m)
    for rule in ("forced", "pure"):
        lab = pd.Series(LAB[rule]); lab = lab[lab.index.isin(B.index)]; y = lab.values; n = len(y)
        if n < 10 or y.min() == y.max(): continue
        X = B.loc[lab.index, feats].astype(float).values; ng = B.loc[lab.index, "n_gen"].astype(float).values
        nv = np.array([(p in NOV_O) if m == "allenai" else (SRC.get(p, "").split("/")[-1] in NOVSRC) if m == "Qwen" else True for p in lab.index])
        fn = (B.loc[lab.index, "still_thinking"] == False).values  # noqa: E712
        OS = np.full((REP, n, len(feats)), np.nan); OF = {k: np.full((REP, n), np.nan) for k in fams}; picks = {k: {} for k in fams}
        for r in range(REP):
            folds = np.zeros(n, int)
            for cls in (0, 1):
                w = np.where(y == cls)[0]; RNG.shuffle(w); folds[w] = np.arange(len(w)) % 5
            for k_ in range(5):
                te, tr = folds == k_, folds != k_
                a = np.array([auc(X[tr, j], y[tr]) if np.isfinite(X[tr, j]).sum() > 3 else .5 for j in range(len(feats))]); sg = np.where(a >= .5, 1, -1)
                C = sg * (X - MED) > 0; OS[r, te] = C[te]; acc_tr = (C[tr] == y[tr, None]).mean(0)
                for fam, cols in fams.items():
                    js = [FI[c] for c in cols]; j = js[int(np.nanargmax(acc_tr[js]))]; OF[fam][r, te] = C[te, j]; picks[fam][feats[j]] = picks[fam].get(feats[j], 0) + 1

        def summ(Cm, mk): return float(np.mean([(Cm[r, mk] == y[mk]).mean() for r in range(REP)])) if mk.sum() >= 4 else np.nan
        SPL = (("all", np.ones(n, bool)), ("novel", nv), ("seen", ~nv), ("finished", fn), ("unfinished", ~fn))
        for j, f in enumerate(feats):
            row = dict(model=m, feature=f, rule=rule, n=n, auc_all=auc(X[:, j], y), rho_len=pd.Series(POOL[:, j]).corr(B["n_gen"].astype(float).reset_index(drop=True), method="spearman"))
            for sp, mk in SPL:
                if sp == "all" or not mk.all(): row[f"acc_{sp}"] = summ(OS[:, :, j], mk)
            rows_s.append(row)
        for fam, Cm in OF.items():
            row = dict(model=m, family=fam, rule=rule, n=n, top_pick=max(picks[fam], key=picks[fam].get), top_pick_share=max(picks[fam].values()) / (REP * 5))
            for sp, mk in SPL:
                if sp == "all" or not mk.all(): row[f"acc_{sp}"] = summ(Cm, mk)
            rows_f.append(row)
    print(f"{m}: {len(B)} traces, {len(feats)} features", flush=True)
S = pd.DataFrame(rows_s); Fm = pd.DataFrame(rows_f)
S.to_csv(A + "analysis/exp104_singles.csv", index=False); Fm.to_csv(A + "analysis/exp104_families.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
for m in Fm.model.unique():
    X = Fm[Fm.model == m].pivot_table(index="family", columns="rule", values=["acc_all", "acc_novel", "acc_unfinished"], aggfunc="first").round(2)
    X.columns = [f"{a}_{b}" for a, b in X.columns]
    tp = Fm[(Fm.model == m) & (Fm.rule == "forced")].set_index("family")[["top_pick", "top_pick_share"]]
    print(f"\n=== {m}: NESTED family accuracy"); print(X.join(tp).sort_values("acc_all_forced", ascending=False).to_string())
    print(f"--- {m}: best singles (forced), with Spearman vs trace length over the pool")
    print(S[(S.model == m) & (S.rule == "forced")].sort_values("acc_all", ascending=False).head(8).round(2).to_string(index=False))
