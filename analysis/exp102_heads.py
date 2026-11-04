"""exp102 (user 2026-10-31: old approaches the retries skipped). Coverage-map item 'individual heads as units', CPU on the battery's saved
per-head arrays: lookback / prompt-mass (sampled layers x heads, whole trace and last quarter), TOHA per head (cut 2048 / full), LapEigvals top-1
and sum of top-8 per head. NESTED: in each training fold (stratified 5-fold x 20) pick (a) the best single head by training accuracy (sign by
training AUC, threshold = the head's label-free pool median) and (b) the mean of the top-5 heads' signed pool percentiles, then score held out.
Splits novel / seen (Qwen, Olmo), finished / unfinished; forced and pure labels. Writes analysis/exp102_heads.csv."""
import json, numpy as np, pandas as pd

src = open("/home/sabyasachi19/PycharmProjects/AimoAGAIN/analysis/exp99_cv_candidates.py").read()
exec(src[:src.index("out, nested = [], []")])
RNG = np.random.default_rng(2); REP = 20
FAMS = ["lookback_le_head", "pmass_le_head", "lookback_lastq_head", "pmass_lastq_head", "toha2048_head", "tohacut_head", "lap_top1", "lap_sum8"]
rows = []
for m, tag in TAGS.items():
    B = pd.DataFrame([json.loads(l) for l in open(A + f"ec2_results/wb/{tag}.jsonl")]).drop_duplicates("pid").set_index("pid")
    H = {f: {} for f in FAMS}
    for p in B.index:
        try: z = np.load(A + f"ec2_results/wb/{tag}/{p}.npz")
        except FileNotFoundError: continue
        for f in FAMS[:6]:
            if f in z.files: H[f][p] = z[f].reshape(-1).astype(float)
        if "lap_topk" in z.files: H["lap_top1"][p] = z["lap_topk"][..., 0].reshape(-1).astype(float); H["lap_sum8"][p] = z["lap_topk"].sum(-1).reshape(-1).astype(float)
    LAB = labels(m); fin_s = (B.still_thinking == False)  # noqa: E712
    for fam, d in H.items():
        if not d: continue
        pool = np.array(list(d.values())); med = np.nanmedian(pool, 0)
        PP = np.stack([pct_matrix(pool[:, j], pool[:, j]) for j in range(pool.shape[1])], 1); pidx = {p: i for i, p in enumerate(d)}
        for rule in ("forced", "pure"):
            lab = pd.Series(LAB[rule]); lab = lab[lab.index.isin(list(d))]; y = lab.values; n = len(y)
            if n < 10 or y.min() == y.max(): continue
            X = np.array([d[p] for p in lab.index]); XP = PP[[pidx[p] for p in lab.index]]
            nv = np.array([(p in NOV_O) if m == "allenai" else (SRC.get(p, "").split("/")[-1] in NOVSRC) if m == "Qwen" else True for p in lab.index])
            fn = fin_s.reindex(lab.index).fillna(False).values.astype(bool)
            C1, C5 = np.full((REP, n), np.nan), np.full((REP, n), np.nan); picks = {}
            for r in range(REP):
                folds = np.zeros(n, int)
                for cls in (0, 1):
                    w = np.where(y == cls)[0]; RNG.shuffle(w); folds[w] = np.arange(len(w)) % 5
                for k in range(5):
                    te, tr = folds == k, folds != k
                    a = np.array([auc(X[tr, j], y[tr]) if np.isfinite(X[tr, j]).sum() > 3 else .5 for j in range(X.shape[1])]); sg = np.where(a >= .5, 1, -1)
                    acc = np.array([((sg[j] * (X[tr, j] - med[j]) > 0) == y[tr]).mean() for j in range(X.shape[1])])
                    j = int(np.nanargmax(acc)); C1[r, te] = sg[j] * (X[te, j] - med[j]) > 0; picks[j] = picks.get(j, 0) + 1
                    top = np.argsort(-np.abs(a - .5))[:5]; s5 = np.nanmean(XP[:, top] * sg[top], 1); C5[r, te] = s5[te] > np.nanmedian(np.nanmean(PP[:, top] * sg[top], 1))
            for nm, C in (("best head", C1), ("top-5 heads", C5)):
                row = dict(model=m, family=fam, rule=rule, method=nm, n=n, n_robust=int(y.sum()), n_heads=X.shape[1])
                for sp, mk in (("all", np.ones(n, bool)), ("novel", nv), ("seen", ~nv), ("finished", fn), ("unfinished", ~fn)):
                    if mk.sum() >= 4 and not (sp != "all" and mk.all()):
                        row[f"acc_{sp}"] = float(np.mean([(C[r, mk] == y[mk]).mean() for r in range(REP)]))
                row["sd_all"] = float(np.std([(C[r] == y).mean() for r in range(REP)]))
                row["top_pick_share"] = max(picks.values()) / (REP * 5) if nm == "best head" else np.nan
                rows.append(row)
        print(m, fam, "done", flush=True)
T = pd.DataFrame(rows); T.to_csv(A + "analysis/exp102_heads.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 400)
print(T.drop(columns=["n_heads"]).round(2).to_string(index=False))
