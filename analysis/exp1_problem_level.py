"""Exp 1: robustness looks like a property of the PROBLEM (shared main_v2 problems had identical labels in every model). Average features across the
models that solved a problem -> one row per problem (22), CV over problems. Compares with row-level performance."""
from common import *
from sklearn.model_selection import RepeatedStratifiedKFold
allm, feats = load_all()
main = datasets(allm)["main_v2"]
# ---- label consistency
g = allm.iloc[main].groupby("base_id").y.agg(["mean", "size"])
multi = g[g["size"] > 1]
print(f"=== Exp 1: problem-level ===\nmain_v2 problems {len(g)} (robust {int((g['mean'] == 1).sum())}, non-robust {int((g['mean'] == 0).sum())}, mixed {int(((g['mean'] > 0) & (g['mean'] < 1)).sum())}); "
      f"problems seen by >1 model: {len(multi)}, label-consistent across models: {int(((multi['mean'] == 0) | (multi['mean'] == 1)).sum())}/{len(multi)}")


# ---- aligned families (same construction as the pooled run)
def aligned(F, n):
    out = {}
    cl, pm = F["cut_last"], F["prompt_mean"]
    for ci, tag in enumerate(("b1024", "b4096", "f50", "f100")):
        c = cl[:, ci]
        nrm_ = np.log(np.linalg.norm(c, axis=-1) + 1e-6)
        cos = (c * pm).sum(-1) / (np.linalg.norm(c, axis=-1) * np.linalg.norm(pm, axis=-1) + 1e-9)
        out[f"hid:norm+cos@{tag}"] = np.hstack([nrm_, cos])
        for nm, T in (("noforce", F["lap_no"][:, ci]), ("force", F["lap_force"][:, ci].mean(1))):
            parts = np.array_split(np.arange(T.shape[1]), 6)
            out[f"lap:topk_{nm}@{tag}"] = np.hstack([np.stack([np.nanmean(T[:, p][..., 0], (1, 2)), np.nanmean(T[:, p], (1, 2, 3)), np.nanstd(T[:, p][..., 0], (1, 2))], 1) for p in parts])
        for nm, T in (("mean_noforce", F["lap_mean_no"][:, ci]), ("mean_force", F["lap_mean_force"][:, ci].mean(1))):
            parts = np.array_split(np.arange(T.shape[1]), 6)
            out[f"lap:{nm}@{tag}"] = np.hstack([np.stack([np.nanmean(T[:, p], (1, 2)), np.nanstd(T[:, p], (1, 2))], 1) for p in parts])
    return out


stk = {}
for m in MODELS:
    for k, v in aligned(feats[m], None).items():
        stk.setdefault(k, []).append(v)
fam = {k: np.vstack(v) for k, v in stk.items()}
tcols = [c for c in allm.columns if c.startswith(("lp_", "ent_", "top1_", "fa_", "sc_", "agree_")) and allm[c].dtype != object
         and all(allm.loc[allm.model == m, c].notna().mean() > .8 for m in MODELS)]
fam["scalars"] = allm[tcols].astype(float).values
D = pd.read_pickle(A / "exp2_digits.pkl")
Fx = pd.read_pickle(A / "exp2_features.pkl")
extra = {"answer_digits": D[["nd_median_B", "nd_max_B", "nd_mean_AB"]].values}
grp = allm.grp.values
Z = {k: zgroup(v, grp) for k, v in {**fam, **extra}.items()}
Lz = lengths(allm)
pid = allm.base_id.values[main]
ups = np.array(sorted(set(pid))); yp = np.array([allm.y.values[main][pid == p][0] for p in ups])


def prob_level(X):
    return np.vstack([X[main][pid == p].mean(0) for p in ups])


P = {k: prob_level(v) for k, v in Z.items()}
Pl = prob_level(Lz)


def cvp(X, y, reps=20, C=0.1, seed=0):
    s, c = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=reps, random_state=seed).split(X, y):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(X[tr], y[tr])
        s[te] += m.decision_function(X[te]); c[te] += 1
    return s / c


def cvp_ens(keys, y, reps=20, C=0.1, seed=0):
    s, c = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=reps, random_state=seed).split(P[keys[0]], y):
        tot = np.zeros(len(te))
        for k in keys:
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(P[k][tr], y[tr])
            sd = m.decision_function(P[k][tr]).std() + 1e-9
            tot += m.decision_function(P[k][te]) / sd
        s[te] += tot / len(keys); c[te] += 1
    return s / c


def nulls(fn, n=6):
    return float(np.mean([roc_auc_score(*(lambda y2: (y2, fn(y2)))(np.random.default_rng(i).permutation(yp))) for i in range(n)]))


print(f"\nproblem-level CV over {len(ups)} problems ({int(yp.sum())} robust), 5-fold x 20, mean over each problem's models:")
rep = lambda nm, s, nl: print(f"  {nm:34s} AUC {auc_ci(yp, s)[0]:.2f} [{auc_ci(yp, s)[1]:.2f},{auc_ci(yp, s)[2]:.2f}]  shuffled-label null {nl:.2f}")
rep("length-only (control)", cvp(Pl, yp), nulls(lambda y2: cvp(Pl, y2, reps=4)))
rep("answer digits (forced, median/max/mean)", cvp(P["answer_digits"], yp), nulls(lambda y2: cvp(P["answer_digits"], y2, reps=4)))
aligned_keys = [k for k in P if k not in ("answer_digits", "scalars")]
rep("scalars", cvp(P["scalars"], yp, C=0.03), nulls(lambda y2: cvp(P["scalars"], y2, reps=4, C=0.03)))
rep(f"ensemble mean of {len(aligned_keys)} aligned families", cvp_ens(aligned_keys, yp), nulls(lambda y2: cvp_ens(aligned_keys, y2, reps=4), n=4))
rep("ensemble of aligned + scalars", cvp_ens(aligned_keys + ["scalars"], yp), nulls(lambda y2: cvp_ens(aligned_keys + ["scalars"], y2, reps=4), n=4))
# length-removed at problem level
def resid(X): return X - LinearRegression().fit(Pl, X).predict(Pl)
Pr = {k: resid(v) for k, v in P.items()}
Psave = P; P = Pr
rep("ensemble aligned, length removed", cvp_ens(aligned_keys, yp), nulls(lambda y2: cvp_ens(aligned_keys, y2, reps=4), n=4))
rep("answer digits, length removed", cvp(P["answer_digits"], yp), nulls(lambda y2: cvp(P["answer_digits"], y2, reps=4)))
P = Psave
# does anything add to the answer digits?
print("\nadded to answer digits (problem level, C=0.1):")
for k in ["scalars"] + aligned_keys[:0]:
    Xc = np.hstack([P["answer_digits"], P[k]])
    s = cvp(Xc, yp, C=0.03); print(f"  digits + {k:10s} AUC {auc_ci(yp, s)[0]:.2f} [{auc_ci(yp, s)[1]:.2f},{auc_ci(yp, s)[2]:.2f}]")
s_dig = cvp(P["answer_digits"], yp); s_ens = cvp_ens(aligned_keys, yp)
zs = lambda v: (v - v.mean()) / v.std()
sc = zs(s_dig) + zs(s_ens); print(f"  digits OOF + aligned-ensemble OOF (equal weight) AUC {auc_ci(yp, sc)[0]:.2f}  [digits alone {roc_auc_score(yp, s_dig):.2f}, ensemble alone {roc_auc_score(yp, s_ens):.2f}]")
