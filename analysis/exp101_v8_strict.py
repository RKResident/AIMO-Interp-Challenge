"""exp101 (user 2026-10-31): put v8's OWN recipes through exp99's strict framework (repeated stratified 5-fold x 50, nested selection),
on the completed labels (DeepSeek's olympiad redo now finished; Skywork / DeepSeek submission features now extracted on the 80 olympiad traces).
Candidate pool = exp99's 45 (28 battery singles, 10 combos, 7 submission-feature recipes) + baseline const:robust +
  v8 as SHIPPED ('fixed': recipe.json signs, router threshold -5.93, nothing fitted) and v8 REFIT (signs fitted in the training folds; Olmo's
  Min-K% router threshold refit in-fold on the novelty indicator of the set100 problems outside the test fold):
  Qwen lble+pmle+jmp+cfc | Skywork fa+chk+f1k | DeepSeek chk+nrk+cfl+pmle | Olmo v7 router (novel -> f1k, familiar -> chk+lble+pmle),
  v8 = mean(router-margin pctl, Qwen-judge pctl) > .5, v8+speed_cv (3-way mean), router+speed_cv.
Percentiles against each model's label-free pool (all its battery traces), threshold = median of pool scores (judge means: 0.5 as shipped).
Every candidate is scored twice: recipe on every row ('t1') and with finished -> robust ('fin'), so the finished rule is itself tested.
Writes analysis/exp101_cv.csv, exp101_nested.csv."""
import json, numpy as np, pandas as pd

src = open("/home/sabyasachi19/PycharmProjects/AimoAGAIN/analysis/exp99_cv_candidates.py").read()
exec(src[:src.index("out, nested = [], []")])
RNG = np.random.default_rng(1)
SIGN8 = {'chk': 1, 'nrk': -1, 'f1k': -1, 'cfc': -1, 'cfl': -1, 'jmp': 1, 'lble': 1, 'pmle': 1, 'fa': 1}
V8 = {"Qwen": ["lble", "pmle", "jmp", "cfc"], "Skywork": ["fa", "chk", "f1k"], "deepseek": ["chk", "nrk", "cfl", "pmle"]}
ROUTER_T = -5.9263951778411865
PP = [json.loads(l) for l in open(A + "ec2_results/a10g/papers2/allenai__Olmo-3-7B-Think/papers2.jsonl")]
MINK = {d["id"].split("_")[-1]: float(d["mink_20"]) for d in PP if d["dataset"] == "set100"}
NOVSET = {l.strip() for l in open(A + "vast_packages/screen/set24_olmo_novel.txt") if l.strip()}


def fit_router(exclude):
    ids = [p for p in MINK if p not in exclude]; r = np.array([-MINK[p] for p in ids]); yv = np.array([p in NOVSET for p in ids])
    u = np.unique(r); cand = np.r_[u[0] - 1, (u[1:] + u[:-1]) / 2, u[-1] + 1]
    return -float(cand[int(np.argmax([((r > t) == yv).mean() for t in cand]))])


out, nested = [], []
for m, tag in TAGS.items():
    B = pd.DataFrame([json.loads(l) for l in open(A + f"ec2_results/wb/{tag}.jsonl")]).drop_duplicates("pid").set_index("pid")
    ff = A + (f"ec2_results/a10g/feats/{tag}.jsonl" if m in ("Qwen", "allenai") else f"ec2_results/big/feats/out_feats_oly/{tag}.jsonl")
    F = pd.DataFrame([json.loads(l) for l in open(ff)]).drop_duplicates("pid").set_index("pid")
    for c in ("lble", "pmle", "jmp", "cfc", "chk", "f1k", "fa", "nrk", "cfl"):
        if c in F: B[c] = F.reindex(B.index)[c]
    print(f"{m}: {len(B)} pool traces, submission features present for {int(B['chk'].notna().sum())}", flush=True)
    if m == "allenai":
        src91 = open(A + "analysis/exp91_generalization.py").read(); g = {}
        exec(src91[:src91.index('print("=" * 110 + "\\nS1')], g)
        QF = pd.DataFrame([json.loads(l) for l in open(A + "ec2_results/a10g/feats/Qwen__Qwen3.5-4B.jsonl")]).set_index("pid")
        _, qs = g["calls"](g["BUILDS"]["v7"]["Qwen"], QF.assign(mink=np.nan), g["ref_of"]("Qwen", "R2")); B["qjudge"] = pd.Series(qs, index=QF.index).reindex(B.index)
        B["mink_20"] = [MINK.get(p, np.nan) for p in B.index]
    cand = {f"single:{f}": [f] for f in SINGLES if f in B} | {f"combo:{k}": [f for f in v if f in B] for k, v in COMBOS.items()}
    cand |= {f"sub:{k}": v for k, v in SUBF.items() if all(f in B for f in v)}
    if m in V8: cand[f"v8:{m}:refit"] = V8[m]
    cand = {k: v for k, v in cand.items() if v}
    fixed = {f"v8:{m}:fixed": V8[m]} if m in V8 else {}
    special = ["v7router:fixed", "v7router:refit", "v8:fixed", "v8:refit", "v8+speed_cv:refit", "router+speed_cv:refit"] if m == "allenai" else []
    feats = sorted({f for v in list(cand.values()) + list(fixed.values()) for f in v} | ({"chk", "lble", "pmle", "f1k", "qjudge", "speed_cv__all"} if m == "allenai" else set()))
    FI = {f: j for j, f in enumerate(feats)}
    POOLP = np.stack([pct_matrix(B[f].astype(float).values, B[f].astype(float).values) for f in feats], 1)
    LAB = labels(m)
    nov = np.array([(p in NOV_O) if m == "allenai" else (SRC.get(p, "").split("/")[-1] in NOVSRC) if m == "Qwen" else True for p in B.index])
    fin_all = (B.still_thinking == False).values  # noqa: E712

    def olmo_scores(kind, sg, rt):
        """(pool scores, threshold) for Olmo's router-based recipes; sg = signs dict, rt = router threshold."""
        novel_b = B["mink_20"].values < rt
        s_nov = POOLP[:, FI["f1k"]] * sg["f1k"]
        s_fam = np.nanmean(np.stack([POOLP[:, FI[f]] * sg[f] for f in ("chk", "lble", "pmle")], 1), 1)
        marg = np.where(novel_b | ~np.isfinite(B["mink_20"].values), s_nov, s_fam)       # default novel, as shipped
        if kind == "v7router": return marg, 0.0
        pr = pct_matrix(marg, marg) + .5; pj = POOLP[:, FI["qjudge"]] * sg["qjudge"] + .5; psp = POOLP[:, FI["speed_cv__all"]] * sg["speed_cv__all"] + .5
        if kind == "v8": return (pr + pj) / 2, .5
        if kind == "v8+speed_cv": return (pr + pj + psp) / 3, .5
        if kind == "router+speed_cv": return (pr + psp) / 2, .5

    for rule in ("forced", "pure"):
        lab = pd.Series(LAB[rule]); lab = lab[lab.index.isin(B.index)]
        ix = np.array([B.index.get_loc(p) for p in lab.index]); y = lab.values; n = len(y)
        if n < 10 or y.min() == y.max(): continue
        P = POOLP[ix]; nv, fn = nov[ix], fin_all[ix]
        names = list(cand) + list(fixed) + special + ["const:robust"]
        oof = {k: np.full((50, n), np.nan) for k in names}; nest_c = np.full((50, n), np.nan); nest_f = np.full((50, n), np.nan); picks = {}
        # fixed recipes: nothing fitted -> same calls every fold
        fixed_calls = {}
        for nm, fs in fixed.items():
            ps = recipe_scores(POOLP, [FI[f] for f in fs], [SIGN8[f] for f in fs]); fixed_calls[nm] = ps[ix] > np.nanmedian(ps)
        if m == "allenai":
            for kind in ("v7router", "v8"):
                sgf = {f: SIGN8.get(f, 1) for f in ("chk", "lble", "pmle")} | {"f1k": 1, "qjudge": 1, "speed_cv__all": 1}   # router branch signs as shipped (f1k +1)
                ps, t = olmo_scores(kind, sgf, ROUTER_T); fixed_calls[f"{kind}:fixed" if kind == "v7router" else "v8:fixed"] = ps[ix] > t
        fixed_calls["const:robust"] = np.ones(n, bool)
        for r in range(50):
            folds = np.zeros(n, int)
            for cls in (0, 1):
                w = np.where(y == cls)[0]; RNG.shuffle(w); folds[w] = np.arange(len(w)) % 5
            for k_ in range(5):
                te, tr = folds == k_, folds != k_
                sg = {f: (1 if auc(P[tr, FI[f]], y[tr]) >= .5 else -1) for f in feats}
                cs = {}
                for nm, fs in cand.items():
                    ps = recipe_scores(POOLP, [FI[f] for f in fs], [sg[f] for f in fs]); cs[nm] = ps[ix] > np.nanmedian(ps)
                if m == "allenai":
                    rt = fit_router(set(lab.index[te]))                      # branch signs fitted on the training rows routed to that branch
                    nb = B["mink_20"].values[ix] < rt; nb |= ~np.isfinite(B["mink_20"].values[ix]); sgb = dict(sg)
                    for f, msk in (("f1k", nb), ("chk", ~nb), ("lble", ~nb), ("pmle", ~nb)):
                        t_ = tr & msk
                        if t_.sum() >= 3 and y[t_].min() != y[t_].max(): sgb[f] = 1 if auc(P[t_, FI[f]], y[t_]) >= .5 else -1
                    for kind in ("v7router", "v8", "v8+speed_cv", "router+speed_cv"):
                        ps, t = olmo_scores(kind, sgb, rt); cs[f"{kind}:refit"] = ps[ix] > t
                cs |= fixed_calls
                for nm, c in cs.items(): oof[nm][r, te] = c[te]
                # nested: pick by training accuracy over the whole pool, with / without finished -> robust
                for store, use_fin in ((nest_c, False), (nest_f, True)):
                    best = (-1, None)
                    for nm, c in cs.items():
                        cc = c | fn if use_fin else c; a = (cc[tr] == y[tr]).mean()
                        if a > best[0]: best = (a, nm)
                    cc = cs[best[1]] | fn if use_fin else cs[best[1]]; store[r, te] = cc[te]
                    if not use_fin: picks[best[1]] = picks.get(best[1], 0) + 1

        def summ(C, mask=None, use_fin=False):
            mk = np.ones(n, bool) if mask is None else mask
            if mk.sum() < 4: return np.nan
            CC = np.where(fn[None, :], 1, C) if use_fin else C
            return float(np.mean([(CC[r, mk] == y[mk]).mean() for r in range(50)]))
        for nm in names + ["NESTED best (no fin rule)", "NESTED best (fin rule)"]:
            C = oof[nm] if nm in oof else (nest_c if "no fin" in nm else nest_f)
            row = dict(model=m, rule=rule, recipe=nm, n=n, n_robust=int(y.sum()), n_finished=int(fn.sum()))
            for sp, mk in (("all", None), ("novel", nv), ("seen", ~nv), ("finished", fn), ("unfinished", ~fn)):
                if mk is not None and (mk.sum() < 4 or mk.all()): continue
                row[f"acc_{sp}"] = summ(C, mk); row[f"accfin_{sp}"] = summ(C, mk, True)
            row["sd_all"] = float(np.std([(C[r] == y).mean() for r in range(50)]))
            out.append(row)
        top = sorted(picks.items(), key=lambda t: -t[1])[:6]
        nested.append(dict(model=m, rule=rule, n=n, top_picks=", ".join(f"{k} {v}/250" for k, v in top)))
        print(f"{m} {rule}: n={n} ({int(y.sum())} R, {int(fn.sum())} finished) | nested picks: " + ", ".join(f"{k} {v}" for k, v in top), flush=True)
T = pd.DataFrame(out); T.to_csv(A + "analysis/exp101_cv.csv", index=False); pd.DataFrame(nested).to_csv(A + "analysis/exp101_nested.csv", index=False)
KEYS = ("v8:", "v7router", "router+", "const:", "NESTED", "single:speed_cv__all", "single:jmp__allatt", "single:pmass_le__all", "combo:attention3+mlp", "sub:")
for m in T.model.unique():
    for rule in ("forced", "pure"):
        X = T[(T.model == m) & (T.rule == rule)].sort_values("acc_all", ascending=False)
        if not len(X): continue
        print(f"\n=== {m} {rule} (n={X.n.iloc[0]}, {X.n_robust.iloc[0]} R, {X.n_finished.iloc[0]} finished) | rank of {len(X) - 2} candidates; acc / acc with finished->robust")
        for rk, (_, r) in enumerate(X.iterrows(), 1):
            if rk > 5 and not r.recipe.startswith(KEYS): continue
            sp = " ".join(f"{s} {r.get(f'acc_{s}'):.2f}/{r.get(f'accfin_{s}'):.2f}" for s in ("novel", "seen", "finished", "unfinished") if pd.notna(r.get(f"acc_{s}")))
            print(f"  #{rk:<3d}{r.recipe:30s} all {r.acc_all:.2f}/{r.accfin_all:.2f} (sd {r.sd_all:.2f}) | {sp}")
