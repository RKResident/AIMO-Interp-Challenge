"""Idea 2: a 'decay direction' learned WITHOUT robustness labels: mean (variant - parent) hidden-state difference over the 10 parents / 29
decay-causing variants (DeepSeek). Project every other DeepSeek trace onto it; score = projection. No fitting on labels -> no CV needed.
The 10 parents are excluded from scoring. Null: 500 random directions built the same way from random agg pairs."""
from common import *
allm, feats = load_all()
F = feats[DS]; meta = allm[allm.model == DS].reset_index(drop=True)
assert len(meta) == len(F["prompt_last"])
agg = np.where((meta.source == "agg") & meta.y.notna())[0]; main = np.where((meta.source == "main_v2") & meta.y.notna())[0]
perm = np.where(meta.source == "perm")[0]
pid = meta.parent_id.astype(str).values
pmap = {pid[i]: i for i in agg}
pairs = [(v, pmap[pid[v]]) for v in perm if pid[v] in pmap]
parents = sorted({p for _, p in pairs})
agg_eval = np.array([i for i in agg if i not in set(parents)])
tf0 = np.load(OUT / DS / "tf" / f"{meta.id[0]}.npz"); sel = list(tf0["sel_layers"])
REPS = {"prompt_last": lambda j: F["prompt_last"][:, sel[j]], "problem_mean": lambda j: F["prompt_mean"][:, j],
        "state@1024": lambda j: F["cut_last"][:, 0, j], "state@4096": lambda j: F["cut_last"][:, 1, j]}
Lz = np.log(meta[["n_prompt", "n_gen", "L_think"]].astype(float).values + 1)
rng = np.random.default_rng(0)


def proj(X, dvec):
    mu = X[agg].mean(0); Xc = X - mu
    return Xc @ dvec / (np.linalg.norm(dvec) + 1e-9)


def direction(X, prs):
    by = {}
    for v, p in prs: by.setdefault(p, []).append(X[v] - X[p])
    return np.mean([np.mean(v, 0) for v in by.values()], 0)          # parent-weighted mean difference


def resid(s, rows):
    return s[rows] - LinearRegression().fit(Lz[rows], s[rows]).predict(Lz[rows])


out = []
print("=== Idea 2: decay direction (variant - parent), DeepSeek ===")
print(f"pairs={len(pairs)} parents={len(parents)}; scored: agg n={len(agg_eval)} (robust {int(meta.y[agg_eval].sum())}), main_v2 n={len(main)} (robust {int(meta.y[main].sum())})")
print("score = projection onto the direction; AUC < 0.5 means 'more like a decayed variant' -> non-robust (the expected sign)")
for rn, getX in REPS.items():
    for j in range(len(sel)):
        X = getX(j).astype(np.float64)
        dvec = direction(X, pairs)
        s = proj(X, dvec)
        a_agg = roc_auc_score(meta.y[agg_eval], s[agg_eval]); a_res = roc_auc_score(meta.y[agg_eval], resid(s, agg_eval))
        a_main = roc_auc_score(meta.y[main], s[main])
        # null: directions from random agg pairs (same count/structure), excluding the parents
        nulls = []
        pool = agg_eval
        for _ in range(300):
            rp = [(rng.choice(pool), q) for q in rng.choice(pool, len(parents), replace=False) for _ in range(3)]
            nulls.append(roc_auc_score(meta.y[agg_eval], proj(X, direction(X, rp))[agg_eval]))
        nulls = np.array(nulls)
        p_two = np.mean(np.abs(nulls - .5) >= abs(a_agg - .5))
        out.append((rn, sel[j], a_agg, a_res, a_main, p_two, np.percentile(nulls, 2.5), np.percentile(nulls, 97.5)))
T = pd.DataFrame(out, columns=["rep", "layer", "agg_auc", "agg_auc_len_removed", "main_v2_auc(n=10)", "p_vs_random_dirs", "null_lo", "null_hi"])
T["q_BH"] = (lambda p: np.minimum(1, np.minimum.accumulate((np.sort(p) * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1])[np.argsort(np.argsort(p))])(T.p_vs_random_dirs.values)
print(T.round(3).to_string(index=False)); T.to_csv(A / "exp9_decay_direction.csv", index=False)
