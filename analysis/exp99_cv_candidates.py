"""exp99 (user 2026-10-31): honest held-out estimates for >= 30 candidate recipes per model, repeated 5-fold CV + nested selection.
Candidates (the SAME list for every model, fixed before looking at CV results): 28 single battery features, 10 family / mixed combinations,
and (Qwen / Olmo, where the submission's own features exist on these problems) 7 recipes on those features incl. the Qwen judge for Olmo.
Recipe = equal-weight mean of signed percentiles against the model's label-free pool (every battery trace of that model); per-feature SIGNS
are fitted on the training folds; threshold t1 = median of the pool's scores (label-free), t2 = fitted on the training folds.
CV: stratified 5-fold, 50 repeats -> mean / sd of held-out accuracy and AUC, overall and per split (novel / seen, finished / unfinished) from
out-of-fold predictions. NESTED selection: in each training fold choose (a) the best recipe of all candidates, (b) the best LAYER of 5
per-layer families, by training accuracy, then score it on the held-out fold. Baselines: v8's FIXED recipes (nothing fitted).
Writes analysis/exp99_cv.csv (recipe x model x rule x split) and analysis/exp99_nested.csv."""
import json, numpy as np, pandas as pd

src = open("/home/sabyasachi19/PycharmProjects/AimoAGAIN/analysis/exp97_battery_eval.py").read()
exec(src[:src.index("rows, layer_rows, lb_rows = [], [], []")])
RNG = np.random.default_rng(0); R_REP, K = 50, 5
SINGLES = ["speed_cv__all", "speed_cv__ge40", "straightness__all", "tortuosity__all", "late_vs_early__all", "slowness__all", "rqa10_det__all",
           "rqa10_lam__all", "rqa20_det__all", "h0_total__all", "h0_spread__all", "h1_total__all", "knn5_clust__all", "knn10_geff__all",
           "knn5_fiedler__all", "pc1_25__all", "pcr__ge40", "depth_straight_le__all", "lookback_lastq__all", "lookback_le__all", "pmass_le__all",
           "pmass_lastq__all", "setup__allatt", "jmp__allatt", "lap_sum8__all", "lap_top1__all", "mlp_write__all", "toha2048__mean"]
COMBOS = {"traj3": ["speed_cv__all", "straightness__all", "tortuosity__all"], "recurrence3": ["rqa10_det__all", "rqa10_lam__all", "rqa20_det__all"],
          "tda3": ["h0_total__all", "h0_spread__all", "h1_total__all"], "knn3": ["knn5_clust__all", "knn10_geff__all", "knn5_fiedler__all"],
          "attention3": ["lookback_lastq__all", "pmass_le__all", "setup__allatt"], "attention3+mlp": ["lookback_lastq__all", "pmass_le__all", "setup__allatt", "mlp_write__all"],
          "lap2": ["lap_sum8__all", "lap_top1__all"], "geometry5": ["speed_cv__all", "straightness__all", "rqa10_det__all", "h0_total__all", "knn5_clust__all"],
          "attn+geom": ["lookback_lastq__all", "pmass_le__all", "speed_cv__all", "rqa10_det__all"],
          "wide12": ["speed_cv__all", "straightness__all", "rqa10_det__all", "h0_total__all", "knn5_clust__all", "pc1_25__all", "lookback_lastq__all",
                     "pmass_le__all", "setup__allatt", "lap_sum8__all", "mlp_write__all", "depth_straight_le__all"]}
SUBF = {"sub_v8_qwen": ["lble", "pmle", "jmp", "cfc"], "sub_olmo_familiar": ["chk", "lble", "pmle"], "sub_f1k": ["f1k"], "sub_chk": ["chk"], "sub_fa": ["fa"],
        "qjudge": ["qjudge"], "qjudge+speed_cv": ["qjudge", "speed_cv__all"]}
LAYERFAM = {"speed_cv": "layer_speed_cv", "rqa10_det": "layer_rqa10_det", "pc1_25": "layer_pc1_25", "mlp_write": "mlp_write_layer", "lookback_le": "lookback_le_head"}


def pct_matrix(vals, pool):
    pool = np.sort(pool[np.isfinite(pool)])
    if not len(pool): return np.full(len(vals), np.nan)
    return (np.searchsorted(pool, vals, "left") + np.searchsorted(pool, vals, "right")) / (2 * len(pool)) - .5


def recipe_scores(P, cols, signs):
    with np.errstate(all="ignore"):
        return np.nanmean(np.stack([P[:, c] * s for c, s in zip(cols, signs)], 1), 1)


out, nested = [], []
for m, tag in TAGS.items():
    B = pd.DataFrame([json.loads(l) for l in open(A + f"ec2_results/wb/{tag}.jsonl")]).drop_duplicates("pid").set_index("pid")
    if m in ("Qwen", "allenai"):                                        # the submission's own 11 features on these problems
        F = pd.DataFrame([json.loads(l) for l in open(A + f"ec2_results/a10g/feats/{tag}.jsonl")]).drop_duplicates("pid").set_index("pid")
        for c in ("lble", "pmle", "jmp", "cfc", "chk", "f1k", "fa"): B[c] = F.reindex(B.index)[c]
    if m == "allenai":                                                  # Qwen's v7 margin on Olmo's problems (the v8 judge)
        src91 = open(A + "analysis/exp91_generalization.py").read(); g = {}
        exec(src91[:src91.index('print("=" * 110 + "\\nS1')], g)
        QF = pd.DataFrame([json.loads(l) for l in open(A + "ec2_results/a10g/feats/Qwen__Qwen3.5-4B.jsonl")]).set_index("pid")
        _, qs = g["calls"](g["BUILDS"]["v7"]["Qwen"], QF.assign(mink=np.nan), g["ref_of"]("Qwen", "R2")); B["qjudge"] = pd.Series(qs, index=QF.index).reindex(B.index)
    cand = {f"single:{f}": [f] for f in SINGLES if f in B} | {f"combo:{k}": [f for f in v if f in B] for k, v in COMBOS.items()}
    cand |= {f"sub:{k}": v for k, v in SUBF.items() if all(f in B for f in v)}
    cand = {k: v for k, v in cand.items() if v}
    feats = sorted({f for v in cand.values() for f in v}); FI = {f: j for j, f in enumerate(feats)}
    POOLP = np.stack([pct_matrix(B[f].astype(float).values, B[f].astype(float).values) for f in feats], 1)      # pool-relative percentiles
    # per-layer arrays for nested layer selection
    LAY = {}
    for nm, key in LAYERFAM.items():
        d = {}
        for p in B.index:
            try: z = np.load(A + f"ec2_results/wb/{tag}/{p}.npz")
            except FileNotFoundError: continue
            if key in z.files: v = z[key]; d[p] = v.mean(1) if v.ndim == 2 else v
        if d: LAY[nm] = d
    LAB = labels(m)
    nov = np.array([(p in NOV_O) if m == "allenai" else (SRC.get(p, "").split("/")[-1] in NOVSRC) if m == "Qwen" else True for p in B.index])
    fin = (B.still_thinking == False).values if "still_thinking" in B else np.zeros(len(B), bool)  # noqa: E712
    for rule in ("forced", "pure"):
        lab = pd.Series(LAB[rule]); lab = lab[lab.index.isin(B.index)]
        ix = np.array([B.index.get_loc(p) for p in lab.index]); y = lab.values; n = len(y)
        if n < 10 or y.min() == y.max(): continue
        P = POOLP[ix]; nv, fn = nov[ix], fin[ix]
        oof = {k: np.full((R_REP, n), np.nan) for k in cand}; oof2 = {k: np.full((R_REP, n), np.nan) for k in cand}; oofs = {k: np.full((R_REP, n), np.nan) for k in cand}
        nest_c, nest_l, picks = np.full((R_REP, n), np.nan), np.full((R_REP, n), np.nan), {}
        for r in range(R_REP):
            folds = np.zeros(n, int)
            for cls in (0, 1):
                w = np.where(y == cls)[0]; RNG.shuffle(w); folds[w] = np.arange(len(w)) % K
            for k_ in range(K):
                te, tr = folds == k_, folds != k_
                sg = {f: (1 if auc(P[tr, FI[f]], y[tr]) >= .5 else -1) for f in feats}
                best = (-1, None, None)
                for nm, fs in cand.items():
                    cols = [FI[f] for f in fs]; s = [sg[f] for f in fs]
                    pool_sc = recipe_scores(POOLP, cols, s); t1 = np.nanmedian(pool_sc)
                    sc_tr, sc_te = recipe_scores(P[tr], cols, s), recipe_scores(P[te], cols, s)
                    u = np.unique(sc_tr[np.isfinite(sc_tr)]); cand_t = np.r_[u[0] - 1, (u[1:] + u[:-1]) / 2, u[-1] + 1] if len(u) else np.array([t1])
                    accs = [((sc_tr > c) == y[tr]).mean() for c in cand_t]; t2 = cand_t[int(np.argmax(accs))]
                    oof[nm][r, te] = sc_te > t1; oof2[nm][r, te] = sc_te > t2; oofs[nm][r, te] = sc_te
                    a_tr = ((sc_tr > t1) == y[tr]).mean()
                    if a_tr > best[0]: best = (a_tr, nm, (sc_te > t1))
                nest_c[r, te] = best[2]; picks[best[1]] = picks.get(best[1], 0) + 1
                bl = (-1, None)                                           # nested layer selection over the 5 per-layer families
                for fam, d in LAY.items():
                    pids = lab.index
                    M = np.array([d.get(p, np.full(len(next(iter(d.values()))), np.nan)) for p in pids], float)
                    allp = np.array([d[p] for p in d], float)
                    for j in range(M.shape[1]):
                        a = auc(M[tr, j], y[tr]); s = 1 if a >= .5 else -1; med = np.nanmedian(allp[:, j])
                        a_tr = ((s * (M[tr, j] - med) > 0) == y[tr]).mean()
                        if a_tr > bl[0]: bl = (a_tr, (s * (M[te, j] - med) > 0))
                nest_l[r, te] = bl[1]
        def summ(C, mask=None):
            mk = np.ones(n, bool) if mask is None else mask
            if mk.sum() < 4: return np.nan, np.nan
            a = np.array([(C[r, mk] == y[mk]).mean() for r in range(R_REP)]); return float(a.mean()), float(a.std())
        for nm in cand:
            row = dict(model=m, rule=rule, recipe=nm, n=n)
            for sp, mk in (("all", None), ("novel", nv), ("seen", ~nv), ("finished", fn), ("unfinished", ~fn)):
                if mk is not None and (mk.sum() < 4 or mk.all()): continue
                row[f"acc_t1_{sp}"], row[f"sd_t1_{sp}"] = summ(oof[nm], mk)
            row["acc_t2_all"], row["sd_t2_all"] = summ(oof2[nm])
            row["auc_oof"] = float(np.mean([auc(oofs[nm][r], y) for r in range(R_REP)]))
            out.append(row)
        for nm, C in (("NESTED best recipe", nest_c), ("NESTED best layer", nest_l)):
            row = dict(model=m, rule=rule, recipe=nm, n=n)
            for sp, mk in (("all", None), ("novel", nv), ("seen", ~nv), ("finished", fn), ("unfinished", ~fn)):
                if mk is not None and (mk.sum() < 4 or mk.all()): continue
                row[f"acc_t1_{sp}"], row[f"sd_t1_{sp}"] = summ(C, mk)
            out.append(row)
        top = sorted(picks.items(), key=lambda t: -t[1])[:5]
        nested.append(dict(model=m, rule=rule, n=n, top_picks=", ".join(f"{k} {v}/{R_REP * K}" for k, v in top)))
        print(f"{m} {rule}: n={n} | nested picks: " + ", ".join(f"{k} {v}" for k, v in top), flush=True)
T = pd.DataFrame(out); T.to_csv(A + "analysis/exp99_cv.csv", index=False); pd.DataFrame(nested).to_csv(A + "analysis/exp99_nested.csv", index=False)
for m in T.model.unique():
    for rule in ("forced", "pure"):
        X = T[(T.model == m) & (T.rule == rule)].sort_values("acc_t1_all", ascending=False)
        if not len(X): continue
        print(f"\n=== {m} {rule} (n={X.n.iloc[0]}): held-out accuracy, repeated 5-fold (mean +- sd over 50 repeats); t1 = label-free median threshold, t2 = fitted")
        for _, r in X.iterrows():
            sp = " ".join(f"{s} {r.get(f'acc_t1_{s}'):.2f}" for s in ("novel", "seen", "finished", "unfinished") if pd.notna(r.get(f"acc_t1_{s}")))
            print(f"  {r.recipe:34s} t1 {r.acc_t1_all:.2f}+-{r.sd_t1_all:.2f}" + (f" | t2 {r.acc_t2_all:.2f} | AUC {r.auc_oof:.2f}" if pd.notna(r.get('acc_t2_all')) else "") + f" | {sp}")
