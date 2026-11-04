"""exp106 (user 2026-10-31): CKA.
  1. DEPTH CKA (CPU, run_wb2 arrays): linear CKA between the 64-token window trajectories at 25 / 50 / 75% depth (same windows = same rows),
     whole trace, first half, second half, and late - early.
  2. PERTURBATION CKA (GPU, run_cka.py): clean vs perturbed prompt, same teacher-forced 2048 trace tokens, every layer: linear CKA (tokens / 4),
     CKA of 64-token window means, cosine drift. Versions: the 5 labelling perturbations (sonnet_*, upper bound) and 3 code perturbations
     (deployable). Aggregates: mean over sonnet types, over code types; layer means (all, >= 40% depth).
Evaluation as exp103 / exp104: stratified 5-fold x 20, sign by training AUC, threshold = label-free pool median, per family NESTED member choice.
Splits novel / seen, finished / unfinished; forced and pure. Writes exp106_singles.csv, exp106_families.csv."""
import json, os, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

src = open("/home/sabyasachi19/PycharmProjects/AimoAGAIN/analysis/exp99_cv_candidates.py").read()
exec(src[:src.index("out, nested = [], []")])
RNG = np.random.default_rng(6); REP = 20


def cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0); K, L = X @ X.T, Y @ Y.T
    return float((K * L).sum() / (np.linalg.norm(K) * np.linalg.norm(L) + 1e-12))


rows_s, rows_f = [], []
for m, tag in TAGS.items():
    B = pd.DataFrame([json.loads(l) for l in open(A + f"ec2_results/wb2/{tag}.jsonl")]).drop_duplicates("pid").set_index("pid")[["n_gen", "still_thinking"]]
    F = {}
    for p in B.index:
        W = np.load(A + f"ec2_results/wb2/{tag}/{p}.npz")["w64_3layers"].astype(np.float64); n = W.shape[1]; h = n // 2; f = {}
        if n >= 8:
            for i, j in ((0, 1), (1, 2), (0, 2)):
                f[f"depthcka_d{i}d{j}"] = cka(W[i], W[j]); a1, a2 = cka(W[i][:h], W[j][:h]), cka(W[i][h:], W[j][h:])
                f[f"depthcka_early_d{i}d{j}"] = a1; f[f"depthcka_late_d{i}d{j}"] = a2; f[f"depthcka_lme_d{i}d{j}"] = a2 - a1
        F[p] = f
    B = B.join(pd.DataFrame.from_dict(F, orient="index"))
    fams = {"depth CKA (whole)": [c for c in B.columns if c.startswith("depthcka_d")], "depth CKA (early)": [c for c in B.columns if c.startswith("depthcka_early")],
            "depth CKA (late)": [c for c in B.columns if c.startswith("depthcka_late")], "depth CKA (late - early)": [c for c in B.columns if c.startswith("depthcka_lme")]}
    cf = A + f"ec2_results/cka/{tag}.jsonl"
    if os.path.exists(cf):
        Ck = pd.DataFrame([json.loads(l) for l in open(cf)]).drop_duplicates("pid").set_index("pid").drop(columns=["model", "versions", "n_tf"], errors="ignore")
        L = max(int(c.split("__l")[-1]) for c in Ck.columns if "__l" in c) + 1
        for meas in ("cka", "ckaw", "drift"):
            for grp, pre in (("sonnet", f"{meas}_sonnet_"), ("code", f"{meas}_code_")):
                for l in range(L):
                    cols = [c for c in Ck.columns if c.startswith(pre) and c.endswith(f"__l{l}")]
                    if cols: Ck[f"{meas}_{grp}mean__l{l}"] = Ck[cols].mean(1)
            vers = sorted({c.split("__")[0][len(meas) + 1:] for c in Ck.columns if c.startswith(meas + "_") and "__l" in c})
            for v in vers:
                cols = [f"{meas}_{v}__l{l}" for l in range(L) if f"{meas}_{v}__l{l}" in Ck]
                Ck[f"{meas}_{v}__all"] = Ck[cols].mean(1); Ck[f"{meas}_{v}__ge40"] = Ck[cols[int(.4 * L):]].mean(1)
                fams[f"{meas} {v} (layer)"] = cols
            fams[f"{meas} aggregates"] = [f"{meas}_{v}__{s}" for v in vers for s in ("all", "ge40")]
        B = B.join(Ck, how="left")
    feats = sorted({c for v in fams.values() for c in v}); FI = {f: j for j, f in enumerate(feats)}
    POOL = B[feats].astype(float).values; MED = np.nanmedian(POOL, 0); LAB = labels(m)
    for rule in ("forced", "pure"):
        lab = pd.Series(LAB[rule]); lab = lab[lab.index.isin(B.index)]; y = lab.values; n = len(y)
        if n < 10 or y.min() == y.max(): continue
        X = B.loc[lab.index, feats].astype(float).values
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
                Cm = sg * (X - MED) > 0; OS[r, te] = Cm[te]; acc_tr = (Cm[tr] == y[tr, None]).mean(0)
                for fam, cols in fams.items():
                    js = [FI[c] for c in cols if c in FI]
                    if not js: continue
                    j = js[int(np.nanargmax(acc_tr[js]))]; OF[fam][r, te] = Cm[te, j]; picks[fam][feats[j]] = picks[fam].get(feats[j], 0) + 1

        def summ(Cm, mk): return float(np.mean([(Cm[r, mk] == y[mk]).mean() for r in range(REP)])) if mk.sum() >= 4 else np.nan
        SPL = (("all", np.ones(n, bool)), ("novel", nv), ("seen", ~nv), ("finished", fn), ("unfinished", ~fn))
        for j, f in enumerate(feats):
            row = dict(model=m, feature=f, rule=rule, n=n, auc_all=auc(X[:, j], y), rho_len=pd.Series(POOL[:, j]).corr(B["n_gen"].astype(float).reset_index(drop=True), method="spearman"))
            for sp, mk in SPL:
                if sp == "all" or not mk.all(): row[f"acc_{sp}"] = summ(OS[:, :, j], mk)
            rows_s.append(row)
        for fam, Cm in OF.items():
            if not picks[fam]: continue
            row = dict(model=m, family=fam, rule=rule, n=n, top_pick=max(picks[fam], key=picks[fam].get), top_pick_share=max(picks[fam].values()) / (REP * 5))
            for sp, mk in SPL:
                if sp == "all" or not mk.all(): row[f"acc_{sp}"] = summ(Cm, mk)
            rows_f.append(row)
    print(f"{m}: {len(B)} traces, {len(feats)} features ({'with' if os.path.exists(cf) else 'NO'} perturbation CKA)", flush=True)
S = pd.DataFrame(rows_s); Fm = pd.DataFrame(rows_f)
S.to_csv(A + "analysis/exp106_singles.csv", index=False); Fm.to_csv(A + "analysis/exp106_families.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
for m in Fm.model.unique():
    X = Fm[Fm.model == m].pivot_table(index="family", columns="rule", values=["acc_all", "acc_novel", "acc_unfinished"], aggfunc="first").round(2)
    X.columns = [f"{a}_{b}" for a, b in X.columns]
    tp = Fm[(Fm.model == m) & (Fm.rule == "forced")].set_index("family")[["top_pick", "top_pick_share"]]
    print(f"\n=== {m}: NESTED family accuracy"); print(X.join(tp).sort_values("acc_all_forced", ascending=False).head(25).to_string())
    print(f"--- {m}: best singles (forced)")
    print(S[(S.model == m) & (S.rule == "forced")].sort_values("acc_all", ascending=False).head(8).round(2).to_string(index=False))
