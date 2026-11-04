"""exp80 (user, 2026-10-28): on main_v2, how often does a leave-k-problems-out split score PERFECTLY for permodel-max and permodel-lookback,
and is that more often than chance? Recipe as built: per-model feature subset (fixed), signed percentiles against that model's unfinished
references, per-model median threshold, finished -> robust. In each split the references and thresholds are recomputed WITHOUT the
held-out problems' rows (label-free, as in the builder), and the held-out main_v2 rows are scored. Null: the same with main_v2 labels
shuffled within model (2000 shuffles x the same splits, a random subset of splits per shuffle)."""
src = open("/home/sabyasachi19/PycharmProjects/AimoAGAIN/analysis/exp73_shift_and_permodel_search.py").read()
exec(src[:src.index('print("(1a) UNFINISHED AT 6144")')])
import itertools
_ALLWB = ["chk", "fas", "nrk", "f1k", "cfc", "cfl", "nzf", "jmp"]
REC = {"permodel-max": {"Qwen": ["jmp", "cfc"], "Skywork": ["fa", "chk", "f1k"], "allenai": ["chk", "lble", "pmle"], "deepseek": ["chk", "nrk", "cfl", "pmle"]},
       "permodel-lookback": {"Qwen": ["lble", "pmle", "jmp", "cfc"], "Skywork": _ALLWB, "allenai": ["lble", "pmle", "jmp", "cfc"], "deepseek": ["chk", "f1k", "cfc", "cfl"]}}
MV = ALL[ALL.dataset == "main_v2"].reset_index(drop=True)
prob = MV.problem.values; probs = sorted(set(prob)); ymv = MV.robust.values.astype(int); fin = MV.finished.astype(bool).values
print(f"main_v2: {len(MV)} rows, {len(probs)} problems, {fin.sum()} finished (-> robust), {ymv.mean():.2f} robust")
Uprob = U.problem.values


def preds(rec, held):
    """predictions for the held-out problems' main_v2 rows, refs/thresholds from the unfinished rows of OTHER problems"""
    out = {}
    for m in MODELS:
        fs = rec[m]; Rf = U[(U.model == m) & ~np.isin(Uprob, list(held))]
        st = np.array([score(r, Rf, fs) for _, r in Rf.iterrows()]); th = np.median(st)
        for i in np.where(np.isin(prob, list(held)) & (MV.model.values == m))[0]:
            out[i] = True if fin[i] else score(MV.iloc[i], Rf, fs) > th
    return out


rng = np.random.default_rng(0)
for name, rec in REC.items():
    print(f"\n{name}")
    full = preds(rec, set(probs)) if False else None
    for k in (1, 2, 3, 4, 5):
        combs = list(itertools.combinations(probs, k))
        if len(combs) > 1500: combs = [tuple(rng.choice(probs, k, replace=False)) for _ in range(1500)]
        P = [preds(rec, set(c)) for c in combs]                                   # predictions do not depend on labels -> reuse for the null
        acc = np.array([np.mean([p[i] == (ymv[i] == 1) for i in p]) for p in P]); perf = np.mean(acc == 1)
        nulls = []
        for _ in range(200):
            yy = ymv.copy()
            for m in MODELS:
                kk = MV.model.values == m; yy[kk] = rng.permutation(yy[kk])
            nulls.append(np.mean([all(p[i] == (yy[i] == 1) for i in p) for p in P]))
        nulls = np.array(nulls); allrob = np.mean([all(ymv[i] == 1 for i in p) for p in P])
        print(f"   leave {k} problem(s) out: {len(P)} splits, held-out rows per split ~{np.mean([len(p) for p in P]):.1f} | mean held-out acc "
              f"{acc.mean():.2f} | PERFECT {perf:.0%} of splits | shuffled-label null {nulls.mean():.0%} (95th pct {np.percentile(nulls, 95):.0%}) | "
              f"'always robust' perfect {allrob:.0%}")
