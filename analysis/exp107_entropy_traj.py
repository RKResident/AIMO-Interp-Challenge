"""exp107 (user 2026-11-01): why did forced-answer entropy at 4096 work - is the whole ENTROPY TRAJECTORY informative? (run_entropy.py outputs)
  A. forced-answer curve every 256 tokens (24 budgets): entropy / top-1 prob per budget (nested budget choice), 3-point smoothed entropy,
     curve summaries (mean, last-half mean, slope, min, std, budget of the minimum), answer stability (top-1 changes, share agreeing with the
     final, longest stable run, last change).
  B. token trajectory of the trace itself: next-token entropy, surprisal of the generated token, top-1 prob - global stats (mean, quantiles,
     high-entropy shares, bursts), 256-token window series (per window: nested choice; slope, late - early, lag-1 autocorrelation, std),
     low-frequency power share, and the token entropy around 4096.
Evaluation as exp103-106: stratified 5-fold x 20, sign by training AUC, threshold = label-free pool median, per family NESTED member choice.
Splits novel / seen, finished / unfinished; forced and pure. Parity: forced entropy at 4096 vs run_wb2's force_ent__b4096.
Writes exp107_singles.csv, exp107_families.csv."""
import json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

src = open("/home/sabyasachi19/PycharmProjects/AimoAGAIN/analysis/exp99_cv_candidates.py").read()
exec(src[:src.index("out, nested = [], []")])
RNG = np.random.default_rng(7); REP = 20; BUD = list(range(256, 6145, 256))


def feats_one(z):
    f = {}; fc = z["forced"]; b, H, P, top = fc[:, 0].astype(int), fc[:, 1], fc[:, 2], fc[:, 3].astype(int)
    for bi, h, p in zip(b, H, P):
        if bi in BUD: f[f"fc_ent__b{bi}"] = h; f[f"fc_p1__b{bi}"] = p
    Hs = np.convolve(H, np.ones(3) / 3, "same")
    for bi, h in zip(b, Hs):
        if bi in BUD: f[f"fc_entsm__b{bi}"] = h
    k = len(H); hh = H[k // 2:]
    f.update(fc_ent_mean=H.mean(), fc_ent_lasthalf=hh.mean(), fc_ent_slope=np.polyfit(b, H, 1)[0] if k > 2 else np.nan, fc_ent_min=H.min(), fc_ent_std=H.std(),
             fc_ent_argmin_frac=float(np.argmin(H) / max(1, k - 1)), fc_p1_mean=P.mean(), fc_p1_lasthalf=P[k // 2:].mean(), fc_p1_slope=np.polyfit(b, P, 1)[0] if k > 2 else np.nan)
    ch = (top[1:] != top[:-1]); f["fc_top_changes"] = float(ch.sum()); f["fc_share_final"] = float((top == top[-1]).mean())
    run, best = 1, 1
    for c in ch:
        run = 1 if c else run + 1; best = max(best, run)
    f["fc_longest_stable_frac"] = best / k; f["fc_last_change_frac"] = float((np.where(ch)[0].max() + 1) / k) if ch.any() else 0.0
    e = z["tok_ent"].astype(np.float64); s = -z["tok_lp"].astype(np.float64); p1 = z["tok_p1"].astype(np.float64); n = len(e)
    f.update(tok_ent_mean=e.mean(), tok_ent_median=np.median(e), tok_ent_std=e.std(), tok_ent_q90=np.quantile(e, .9), tok_ent_q99=np.quantile(e, .99),
             tok_ent_gt1=(e > 1).mean(), tok_ent_gt2=(e > 2).mean(), tok_surp_mean=s.mean(), tok_surp_std=s.std(), tok_p1_mean=p1.mean(), tok_p1_lt05=(p1 < .5).mean())
    hi = e > 1; runs = np.diff(np.r_[0, hi.astype(int), 0]); st, en = np.where(runs == 1)[0], np.where(runs == -1)[0]
    f["tok_bursts_per_1k"] = float(((en - st) >= 4).sum() / n * 1000)
    nw = n // 256
    if nw >= 4:
        W = e[:nw * 256].reshape(nw, 256).mean(1)
        for i, w in enumerate(W): f[f"tokwin_ent__w{i}"] = w
        q = max(1, nw // 4)
        f.update(tokwin_slope=np.polyfit(np.arange(nw), W, 1)[0], tokwin_lme=W[-q:].mean() - W[:q].mean(), tokwin_std=W.std(), tokwin_max=W.max(),
                 tokwin_ac1=float(np.corrcoef(W[:-1], W[1:])[0, 1]) if nw > 3 else np.nan)
        x = e - e.mean(); F = np.abs(np.fft.rfft(x)) ** 2; f["tok_lowfreq_share"] = float(F[1:max(2, len(F) // 100)].sum() / (F[1:].sum() + 1e-12))
    if n >= 4352: f["tok_ent_near4096"] = e[3840:4352].mean()
    return f


rows_s, rows_f = [], []
for m, tag in TAGS.items():
    try: B = pd.DataFrame([json.loads(l) for l in open(A + f"ec2_results/entropy/{tag}.jsonl")]).drop_duplicates("pid").set_index("pid")
    except FileNotFoundError: print(f"{m}: no entropy output"); continue
    F = {p: feats_one(np.load(A + f"ec2_results/entropy/{tag}/{p}.npz")) for p in B.index}
    B = B.join(pd.DataFrame.from_dict(F, orient="index"))
    try:                                                              # parity with run_wb2 (same suffix, same trace)
        W2 = pd.DataFrame([json.loads(l) for l in open(A + f"ec2_results/wb2/{tag}.jsonl")]).drop_duplicates("pid").set_index("pid")
        j = B.index.intersection(W2.index); d = (B.loc[j, "fc_ent__b4096"] - W2.loc[j, "force_ent__b4096"]).abs()
        print(f"{m}: parity fc_ent@4096 vs wb2 force_ent__b4096 on {d.notna().sum()} traces: max |diff| {d.max():.2e}, median {d.median():.2e}")
    except Exception as e: print(f"{m}: parity check skipped ({e})")
    cols = [c for c in B.columns if c not in ("model", "n", "still_thinking", "n_forced")]
    fams = {}
    for c in cols:
        if "__b" in c: fams.setdefault(c.split("__")[0] + " (budget)", []).append(c)
        elif "__w" in c: fams.setdefault(c.split("__")[0] + " (window)", []).append(c)
    fams["forced-curve summaries"] = [c for c in cols if c.startswith("fc_") and "__" not in c]
    fams["token-trajectory summaries"] = [c for c in cols if c.startswith("tok") and "__" not in c]
    feats = cols; FI = {f: j for j, f in enumerate(feats)}; POOL = B[feats].astype(float).values; MED = np.nanmedian(POOL, 0); LAB = labels(m)
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
                Cm = sg * (X - MED) > 0; OS[r, te] = Cm[te]; acc_tr = np.nanmean(np.where(np.isfinite(X[tr]), Cm[tr] == y[tr, None], np.nan), 0)
                for fam, cs in fams.items():
                    js = [FI[c] for c in cs]; j = js[int(np.nanargmax(acc_tr[js]))]; OF[fam][r, te] = Cm[te, j]; picks[fam][feats[j]] = picks[fam].get(feats[j], 0) + 1

        def summ(Cm, mk): return float(np.mean([(Cm[r, mk] == y[mk]).mean() for r in range(REP)])) if mk.sum() >= 4 else np.nan
        SPL = (("all", np.ones(n, bool)), ("novel", nv), ("seen", ~nv), ("finished", fn), ("unfinished", ~fn))
        for j, f in enumerate(feats):
            row = dict(model=m, feature=f, rule=rule, n=n, auc_all=auc(X[:, j], y), rho_len=pd.Series(POOL[:, j]).corr(B["n"].astype(float).reset_index(drop=True), method="spearman"))
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
S.to_csv(A + "analysis/exp107_singles.csv", index=False); Fm.to_csv(A + "analysis/exp107_families.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
for m in Fm.model.unique():
    X = Fm[Fm.model == m].pivot_table(index="family", columns="rule", values=["acc_all", "acc_novel", "acc_unfinished"], aggfunc="first").round(2)
    X.columns = [f"{a}_{b}" for a, b in X.columns]
    tp = Fm[(Fm.model == m) & (Fm.rule == "forced")].set_index("family")[["top_pick", "top_pick_share"]]
    print(f"\n=== {m}: NESTED family accuracy"); print(X.join(tp).sort_values("acc_all_forced", ascending=False).to_string())
    print(f"--- {m}: best singles (forced)")
    print(S[(S.model == m) & (S.rule == "forced")].sort_values("acc_all", ascending=False).head(10).round(2).to_string(index=False))
    q = S[(S.model == m) & (S.rule == "forced") & S.feature.str.startswith("fc_ent__b")].copy(); q["b"] = q.feature.str.split("__b").str[-1].astype(int)
    print(f"--- {m}: forced-entropy AUC by budget (forced labels): " + " ".join(f"{int(b)}:{a:.2f}" for b, a in q.sort_values("b")[["b", "auc_all"]].values))
