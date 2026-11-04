"""exp92 (user 2026-10-31: new ideas on the new data). CPU only.
 (1) CROSS-MODEL: is robustness a property of the problem? Qwen and Olmo are labelled on the same 50 problems: label agreement, max_drop
     correlation, and whether one model's recipe score / features predict the OTHER model's label (a submission judging Olmo could also run
     Qwen on the same problem, and vice versa).
 (3) CONTINUOUS TARGET: Spearman of each recipe score / feature with -max_drop over ALL problems with a defined drop (ambiguous included, not
     only robust / non-robust), per model; is the v7 choice still the best under this denser target?"""
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr

src = open("/home/sabyasachi19/PycharmProjects/AimoAGAIN/analysis/exp91_generalization.py").read()
exec(src[:src.index('print("=" * 110 + "\\nS1')])
L = {m: pd.read_csv(A + f"hard_pool/set100_labels_{TAG[m]}.csv").set_index("pid") for m in ("Qwen", "allenai")}
F = {m: pd.read_csv(A + ("analysis/exp84_rows.csv" if m == "Qwen" else f"analysis/exp84_rows_{TAG[m]}.csv")).set_index("pid") for m in ("Qwen", "allenai")}
shared = sorted(set(L["Qwen"].index) & set(L["allenai"].index))

print("=" * 100 + f"\n(1) CROSS-MODEL on the {len(shared)} problems labelled for both Qwen and Olmo")
for lab in ("forced", "pure"):
    q, o = L["Qwen"].loc[shared, f"label_{lab}"], L["allenai"].loc[shared, f"label_{lab}"]
    ok = q.isin(["robust", "non_robust"]) & o.isin(["robust", "non_robust"])
    agree = (q[ok] == o[ok]).mean(); base = max((q[ok] == "robust").mean(), 1 - (q[ok] == "robust").mean())
    pr = (q[ok] == "robust").mean() * (o[ok] == "robust").mean() + (q[ok] != "robust").mean() * (o[ok] != "robust").mean()
    dq, do = L["Qwen"].loc[shared, f"max_drop_{lab}"], L["allenai"].loc[shared, f"max_drop_{lab}"]; okd = dq.notna() & do.notna()
    print(f"  {lab}: both labelled {ok.sum()} | label agreement {agree:.2f} (chance {pr:.2f}) | Spearman max_drop Qwen vs Olmo {spearmanr(dq[okd], do[okd])[0]:+.2f} (n={okd.sum()})"
          f" | Olmo robust | Qwen robust: {((o[ok] == 'robust') & (q[ok] == 'robust')).sum()}/{(q[ok] == 'robust').sum()}, | Qwen non-robust: {((o[ok] == 'robust') & (q[ok] != 'robust')).sum()}/{(q[ok] != 'robust').sum()}")
# other model's recipe score as a predictor
for tgt, src_m in (("allenai", "Qwen"), ("Qwen", "allenai")):
    ref = ref_of(src_m, REG[src_m]); P = F[src_m].loc[[p for p in shared if p in F[src_m].index]]
    _, s = calls(BUILDS["v7"][src_m], P.assign(mink=[MINK.get(p, np.nan) for p in P.index]), ref)
    S = pd.Series(s, index=P.index)
    for lab in ("forced", "pure"):
        y = L[tgt].loc[S.index, f"label_{lab}"]; k = y.isin(["robust", "non_robust"]); yy = (y[k] == "robust").astype(int).values
        out = f"  {src_m}'s v7 score -> {tgt}'s {lab} label: AUC {auc(S[k].values, yy):.2f} (n={k.sum()})"
        feats = {f: auc(P.loc[S.index[k], f].values.astype(float), yy) for f in FEATS}
        print(out + " | best single features of the other model: " + ", ".join(f"{f} {a:.2f}" for f, a in sorted(feats.items(), key=lambda t: -abs(t[1] - .5))[:3]))

print("=" * 100 + "\n(3) CONTINUOUS TARGET: Spearman(score, -max_drop) over every problem with a defined drop (ambiguous included)")
for m in ("Qwen", "allenai"):
    ref = ref_of(m, REG[m])
    for lab in ("forced", "pure"):
        d = L[m][L[m][f"max_drop_{lab}"].notna() & (L[m][f"acc_orig_{lab}"] > 0)]; d = d[d.index.isin(F[m].index)]
        X_ = F[m].loc[d.index].assign(mink=[MINK.get(p, np.nan) for p in d.index]); t = -d[f"max_drop_{lab}"].values
        res = {}
        for b in ("max", "v5", "v6", "v7"):
            _, s = calls(BUILDS[b][m], X_, ref); res[b] = spearmanr(s, t)[0]
        P, PR = Pm(X_, ref), Pm(ref, ref)
        rows = [(fs, spearmanr(sc(P, fs, {}), t)[0]) for fs in SUBS]; R = pd.DataFrame(rows, columns=["fs", "rho"]).sort_values("rho", ascending=False)
        n_lab = d[f"label_{lab}"].isin(["robust", "non_robust"]).sum()
        print(f"  {m:8s} {lab:6s} n={len(d)} (vs {n_lab} binary) | " + " ".join(f"{b} {r:+.2f}" for b, r in res.items()) +
              f" | best subset {R.iloc[0].fs} {R.iloc[0].rho:+.2f}; subsets above v7: {(R.rho > res['v7']).mean():.0%}")
        nov = np.array([p in NOVP for p in d.index]) if m == "allenai" else np.array([SRC[p].split('/')[-1] in POOLNOV for p in d.index])
        print(f"           single features (all / novel / seen): " + ", ".join(
            f"{f} {spearmanr(X_[f], t)[0]:+.2f}/{spearmanr(X_[f][nov], t[nov])[0]:+.2f}/{spearmanr(X_[f][~nov], t[~nov])[0]:+.2f}" for f in ("fa", "chk", "f1k", "nrk", "jmp", "lble", "pmle", "cfl")))
