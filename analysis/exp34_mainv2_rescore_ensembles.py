"""exp34: main_v2 only (43 rows, 22 problems, 7 non-robust). (A) every saved feature (exp28 cache + exp29 / exp32 / exp33) scored
the same way: within-model z, problem means, (finished fraction, mean log length) regressed out, permutation p over the problems
that have the feature (exp28 skipped those with missing problems); per-family hit counts vs chance.
(B) ensembles = mean of signed problem-level z of chosen features:
  fixed  : hand-picked best features, signs from main_v2 -> optimistic (chosen on the same data)
  nested : the whole choice (top-k features by training controlled AUC, at most one per family, signs from training) inside
           leave-one-problem-out, scored on the held-out problem; permutation null of the whole procedure -> honest estimate."""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "2"
from common import *
from scipy.stats import rankdata

allm, _ = load_all(); y = allm.y.values; rows = datasets(allm)["main_v2"]
model = allm.model.values; prob = allm.base_id.values
fin = (allm.stop.values == "eos").astype(float); lgn = np.log(allm.n_gen.astype(float).values)
Lthink = np.full(len(allm), np.nan); Lthink[rows] = [force_json(allm.model.iloc[i], allm.id.iloc[i])["L"] for i in rows]
F = {}
for f, pre in [("exp28_features_cache.pkl", ""), ("exp29_features.pkl", "e29:"), ("exp32_features.pkl", "e32:"), ("exp33_features.pkl", "e33:")]:
    if (A / f).exists():
        for k, v in pickle.load(open(A / f, "rb")).items(): F[pre + k] = np.asarray(v, float)
    else: print("MISSING", f)
v = F["meta:fa_b6144_B_p1"].copy(); v[~(Lthink >= 6144)] = np.nan; F["meta:fa_b6144_B_p1|still_thinking"] = v   # no finishing leak


def fam(n):
    for pre, f in [("meta:fa_b", "forced-answer, fixed budget"), ("meta:fa_f", "forced-answer, fraction"), ("meta:sc_", "self-consistency"),
                   ("meta:prob_", "problem log-prob"), ("meta:", "output token stats"), ("ab:", "A/B suffix"), ("dyn:h", "hidden drift"),
                   ("dyn:c", "answer commitment"), ("dyn:l", "Laplacian / forcing shock"), ("anchor:", "anchoring"), ("text:t_", "text markers"),
                   ("text:e_", "entropy bursts"), ("traj:", "trajectory geometry"), ("subsample:", "segments / subsampling"), ("toha:", "TOHA"),
                   ("lookback:", "lookback"), ("pearl:", "PEARL"), ("memlens:", "MemLens"), ("extra:", "extras"),
                   ("e29:text:", "exp29 repetition text"), ("e29:ent:", "exp29 entropy collapse"), ("e29:hid:", "exp29 waypoints (secondary)"),
                   ("e32:reference", "step similarity (averaged)"), ("e32:sentence", "exp32 kNN graph, sentence"), ("e32:step64", "exp32 kNN graph, 64-token"),
                   ("e33:sentence|-", "exp33 recurrence, sentence"), ("e33:step64|-", "exp33 recurrence, 64-token"),
                   ("e33:sentence", "exp33 cluster graph, sentence"), ("e33:step64", "exp33 cluster graph, 64-token")]:
        if n.startswith(pre): return f
    return "other"


up = np.unique(prob[rows]); P = len(up)
yp = np.array([y[rows][prob[rows] == p][0] for p in up]).astype(int)
fp = np.array([fin[rows][prob[rows] == p].mean() for p in up]); lp = np.array([lgn[rows][prob[rows] == p].mean() for p in up])


def zmodel(v):
    out = np.full(len(allm), np.nan)
    for m in np.unique(model[rows]):
        r = rows[model[rows] == m]; x = v[r]
        if np.isfinite(x).sum() >= 2 and np.nanstd(x) > 0: out[r] = (x - np.nanmean(x)) / np.nanstd(x)
    return out


names, Z, X = [], [], []
for n, v in F.items():
    x = v[rows]; ok = np.isfinite(x)
    if ok.sum() < 12 or len(np.unique(x[ok])) < 3: continue
    z = zmodel(v); xp = np.array([np.nanmean(z[rows][prob[rows] == p]) if np.isfinite(z[rows][prob[rows] == p]).any() else np.nan for p in up])
    if np.isfinite(xp).sum() < 8 or len(np.unique(yp[np.isfinite(xp)])) < 2: continue
    names.append(n); Z.append(z); X.append(xp)
Z = np.array(Z).T; X = np.array(X).T                                        # Z [rows(all), feats] within-model z; X [problems, feats]
FAM = np.array([fam(n) for n in names])
rng = np.random.default_rng(0)


def auc(x, yy):
    rk = rankdata(x); n1 = yy.sum(); n0 = len(yy) - n1
    return (rk[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def resid(x, C): return x - C @ np.linalg.lstsq(C, x, rcond=None)[0]


def perm_p(x, yy, n=5000):
    a = auc(x, yy); rk = rankdata(x); n1 = yy.sum(); n0 = len(yy) - n1
    YP = rng.permuted(np.tile(yy, (n, 1)), axis=1).astype(float); nl = (YP @ rk - n1 * (n1 + 1) / 2) / (n1 * n0)
    return a, float(np.mean(np.abs(nl - .5) >= abs(a - .5)))


# ---------------------------------------------------------------- (A) uniform re-score
out = []
for j, n in enumerate(names):
    ok = np.isfinite(X[:, j]); C = np.c_[np.ones(ok.sum()), fp[ok], lp[ok]]
    a_raw, p_raw = perm_p(X[ok, j], yp[ok]); a_c, p_c = perm_p(resid(X[ok, j], C), yp[ok])
    r = rows[np.isfinite(Z[rows, j])]
    out.append(dict(feature=n, family=FAM[j], rows=len(r), problems=int(ok.sum()), pooled_rows=roc_auc_score(y[r], Z[r, j]),
                    problem=a_raw, p_problem=p_raw, controlled=a_c, p_controlled=p_c))
T = pd.DataFrame(out); T["dist"] = (T.controlled - .5).abs()
T.to_csv(A / "exp34_mainv2_all_features.csv", index=False)
old = pd.read_csv(A / "exp28_mainv2_single_features.csv")
gap = set(old.feature[old.p_controlled.isna()])
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
print(f"(A) {len(T)} features scored on main_v2. exp28 features that had NO controlled score before: {len(gap & set(T.feature))}; best of them now:")
print(T[T.feature.isin(gap)].sort_values("dist", ascending=False).head(10)[["feature", "rows", "problems", "pooled_rows", "controlled", "p_controlled"]].round(3).to_string(index=False))
print("\nnew averaged-trajectory measures (exp33) and exp29 text/entropy, strongest:")
print(T[T.feature.str.startswith(("e33:", "e29:text", "e29:ent"))].sort_values("dist", ascending=False).head(10)[["feature", "rows", "pooled_rows", "controlled", "p_controlled"]].round(3).to_string(index=False))
G = T.groupby("family").agg(n=("feature", "size"), best=("controlled", lambda s: s.iloc[np.argmax(np.abs(s.values - .5))]),
                            p_best=("p_controlled", "min"), hits=("p_controlled", lambda s: int((s < .05).sum())))
G["expected"] = (G.n * .05).round(1); G["hits/expected"] = (G.hits / G.expected.clip(lower=.05)).round(2)
print("\nper family (controlled, problem level):"); print(G.sort_values("hits/expected", ascending=False).round(3).to_string())

# ---------------------------------------------------------------- (B) ensembles
col = {n: j for j, n in enumerate(names)}
SIGN = {n: np.sign(T.controlled[T.feature == n].iloc[0] - .5) for n in names}


def ens_eval(feats):
    js = [col[f] for f in feats if f in col]; S = np.array([SIGN[names[j]] for j in js])
    xp = np.nanmean(X[:, js] * S, 1); xp = np.where(np.isfinite(xp), xp, 0.0); zr = np.nanmean(Z[:, js] * S, 1)   # no feature for a problem -> neutral
    r = rows[np.isfinite(zr[rows])]; C = np.c_[np.ones(P), fp, lp]
    a, p = perm_p(xp, yp); ac, pc = perm_p(resid(xp, C), yp)
    return dict(k=len(js), pooled_rows=roc_auc_score(y[r], zr[r]), problem=a, p_problem=p, controlled=ac, p_controlled=pc)


fa, fa_s, ab, seg, traj = "meta:fa_b6144_B_p1", "meta:fa_b6144_B_p1|still_thinking", "ab:dent_max", "subsample:seg25:f11", "traj:k1|full:late_vs_early_cos"
pearl, ge, fied, toha = "pearl:mean_state:area", "e32:step64|k5|4k|global_efficiency", "e32:step64|k5|4k|fiedler_norm", "toha:ans@1024:total"
FIXED = {"forced conf (6k)": [fa], "forced conf + A/B shift": [fa, ab], "forced conf + kNN global eff": [fa, ge],
         "forced conf + global eff + late-vs-early": [fa, ge, traj], "top 5 (conf, A/B, segment, traj, PEARL)": [fa, ab, seg, traj, pearl],
         "top 5 + global eff": [fa, ab, seg, traj, pearl, ge], "top 5 + global eff + TOHA": [fa, ab, seg, traj, pearl, ge, toha],
         "top 5 + global eff, conf on still-thinking only": [fa_s, ab, seg, traj, pearl, ge],
         "no forcing (segment, traj, PEARL, global eff)": [seg, traj, pearl, ge], "graph only (global eff + Fiedler)": [ge, fied]}
E = pd.DataFrame([dict(ensemble=k, **ens_eval(v)) for k, v in FIXED.items()])
print("\n(B1) FIXED ensembles - features and signs picked on main_v2 itself, so these are OPTIMISTIC:")
print(E.round(3).to_string(index=False))

pool = np.array([j for j, n in enumerate(names) if not n.startswith("e29:hid") and not n.endswith("|still_thinking")])


def nested(yy, k):
    oof = np.zeros(P)
    for h in range(P):
        tr = np.arange(P) != h; Xt = X[tr][:, pool]; mu = np.nanmean(Xt, 0); Xt = np.where(np.isfinite(Xt), Xt, mu)
        C = np.c_[np.ones(tr.sum()), fp[tr], lp[tr]]; R = Xt - C @ np.linalg.lstsq(C, Xt, rcond=None)[0]
        rk = rankdata(R, axis=0); n1 = yy[tr].sum(); n0 = tr.sum() - n1
        a = (rk[yy[tr] == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
        chosen, fams = [], set()
        for j in np.argsort(-np.abs(a - .5)):
            if FAM[pool[j]] in fams: continue
            chosen.append(j); fams.add(FAM[pool[j]])
            if len(chosen) == k: break
        sg = np.sign(a[chosen] - .5); sd = Xt[:, chosen].std(0) + 1e-9
        xh = X[h, pool[chosen]]; xh = np.where(np.isfinite(xh), xh, mu[chosen])
        oof[h] = np.mean(sg * (xh - mu[chosen]) / sd)
    return oof


C = np.c_[np.ones(P), fp, lp]
print("\n(B2) NESTED selection (leave-one-problem-out; top-k features by training controlled AUC, one per family) - honest estimate:")
for k in (1, 3, 5, 10):
    o = nested(yp, k); a, ac = auc(o, yp), auc(resid(o, C), yp)
    null = np.array([auc(resid(nested(yy, k), C), yy) for yy in (rng.permutation(yp) for _ in range(300))])
    print(f"  k={k:2d}: held-out problem AUC {a:.2f} | controlled {ac:.2f} | permutation p of controlled {np.mean(np.abs(null - .5) >= abs(ac - .5)):.3f} "
          f"(null mean {null.mean():.2f}, 95% range {np.quantile(null, .025):.2f}-{np.quantile(null, .975):.2f})")
