"""Follow-ups: (1) agg-trained prompt probe -> DeepSeek main_v2 (transfer check, not pooled CV); (2) prompt probe + problem-anchoring cosine on agg
under both length controls; (3) cross-model agreement on the main_v2 problems solved by >1 model."""
from common import *
from sklearn.decomposition import PCA
from itertools import combinations
allm, feats = load_all()
ds = datasets(allm); agg, main = ds["agg"], ds["main_v2"]; y = allm.y.values
n_ds = len(feats[DS]["prompt_last"]); D4 = pd.read_pickle(A / "exp4_features.pkl")
L3 = np.log(allm[["n_prompt", "n_gen", "L_think"]].astype(float).values + 1)
ds_main = np.array([i for i in main if allm.model.values[i] == DS])


def pad(a):
    out = np.full((len(allm),) + a.shape[1:], np.nan, np.float32); out[:n_ds] = a; return np.nan_to_num(out)


def probe(k=8):
    return make_pipeline(StandardScaler(), PCA(k, random_state=0), StandardScaler(), LogisticRegression(C=0.3, max_iter=3000))


def exact_perm_p(yt, s, n=20000, seed=0):
    a = roc_auc_score(yt, s); rng = np.random.default_rng(seed)
    nl = np.array([roc_auc_score(rng.permutation(yt), s) for _ in range(n)]); return a, float(np.mean(nl >= a))


# ---------------- (1) transfer
print("=== (1) prompt probe trained on ALL agg -> DeepSeek main_v2 (n=%d, robust %d) ===" % (len(ds_main), int(y[ds_main].sum())))
for L_ in (24, 30, 35):
    X = pad(feats[DS]["prompt_last"][:, L_])
    for tag, XX in (("raw", X), ("prompt-length removed", None)):
        if XX is None:
            XX = X.astype(np.float64).copy()
            for rows in (agg, ds_main):  # residualise within each dataset separately, label-free
                lp = L3[rows, :1]; XX[rows] = XX[rows] - LinearRegression().fit(lp, XX[rows]).predict(lp)
        m = probe().fit(XX[agg], y[agg]); s = m.decision_function(XX[ds_main])
        a, p = exact_perm_p(y[ds_main], s)
        print(f"  layer {L_:2d} {tag:22s} AUC {a:.2f}  one-sided permutation p {p:.3f}   (21 robust/non-robust pairs; 0.5 = chance)")

# ---------------- (2) prompt probe + anchoring cosine on agg, both length controls
cosf = D4["h5_cos_prob_mean"].values.astype(float)[:, None]
PL = pad(feats[DS]["prompt_last"][:, 30])


def cv_combo(rows, groups, Xp, Xc, use_p=True, use_c=True, reps=10, yy=None):
    yy = y if yy is None else yy
    s, c = np.zeros(len(rows)), np.zeros(len(rows))
    for r in range(reps):
        for tr, te in folds(rows, yy, groups, r):
            parts_tr, parts_te = [], []
            if use_p:
                pca = make_pipeline(StandardScaler(), PCA(8, random_state=0)).fit(Xp[rows[tr]]); parts_tr.append(pca.transform(Xp[rows[tr]])); parts_te.append(pca.transform(Xp[rows[te]]))
            if use_c:
                parts_tr.append(Xc[rows[tr]]); parts_te.append(Xc[rows[te]])
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)).fit(np.hstack(parts_tr), yy[rows[tr]])
            s[te] += m.decision_function(np.hstack(parts_te)); c[te] += 1
    return s / np.maximum(c, 1)


def resid_all(X, rows):
    X = np.nan_to_num(X.astype(np.float64)).copy(); X[rows] = X[rows] - LinearRegression().fit(L3[rows], X[rows]).predict(L3[rows]); return X


print("\n=== (2) agg: prompt probe (layer 30) + problem-anchoring cosine (last sampled layer) ===")
idA = np.arange(len(allm))
Pr, Cr = resid_all(PL, agg), resid_all(cosf, agg)
# matched pairs on prompt AND trace length (same construction as idea 4)
Ls = (L3[:, :2] - L3[agg, :2].mean(0)) / L3[agg, :2].std(0)
rob = [i for i in agg if y[i] == 1]; non = [i for i in agg if y[i] == 0]; used, pairs = set(), []
for i in sorted(rob, key=lambda i: Ls[i, 1]):
    cand = [j for j in non if j not in used]; j = cand[int(np.argmin([np.sum((Ls[i] - Ls[j]) ** 2) for j in cand]))]; used.add(j); pairs.append((i, j))
pi, pj = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]); mrows = np.concatenate([pi, pj])
gp = np.zeros(len(allm), int); gp[pi] = np.arange(len(pairs)); gp[pj] = np.arange(len(pairs))


def pair_null(fn, n=6):
    out = []
    for i in range(n):
        yy = y.copy(); flip = np.random.default_rng(i).random(len(pairs)) < .5; yy[pi[flip]], yy[pj[flip]] = 0, 1
        out.append(roc_auc_score(yy[mrows], fn(yy)))
    return np.mean(out)


for nm, up, uc in (("prompt probe alone", True, False), ("anchoring cosine alone", False, True), ("prompt probe + cosine", True, True)):
    a_raw = auc_ci(y[agg], cv_combo(agg, idA, PL, cosf, up, uc))
    a_res = auc_ci(y[agg], cv_combo(agg, idA, Pr, Cr, up, uc))
    nl_res = np.mean([roc_auc_score(shuffled_y(allm, agg, "agg", i)[agg], cv_combo(agg, idA, Pr, Cr, up, uc, reps=2, yy=shuffled_y(allm, agg, "agg", i))) for i in range(5)])
    a_m = auc_ci(y[mrows], cv_combo(mrows, gp, PL, cosf, up, uc))
    nl_m = pair_null(lambda yy: cv_combo(mrows, gp, PL, cosf, up, uc, reps=2, yy=yy))
    print(f"  {nm:24s} raw {a_raw[0]:.2f} [{a_raw[1]:.2f},{a_raw[2]:.2f}] | all lengths removed {a_res[0]:.2f} [{a_res[1]:.2f},{a_res[2]:.2f}] null {nl_res:.2f} | "
          f"length-matched pairs {a_m[0]:.2f} [{a_m[1]:.2f},{a_m[2]:.2f}] null {nl_m:.2f}", flush=True)

# ---------------- (3) cross-model agreement on shared main_v2 problems
print("\n=== (3) cross-model agreement, main_v2 problems solved by >1 model ===")
labs = ["b1024", "b2048", "b4096", "f50", "f100"]
info = {}
for i in main:
    r = allm.iloc[i]; fj = force_json(r.model, r.id)
    a = {l: nrm(fj["results"].get(f"{fj['labels'][l]}|B", {}).get("answer")) for l in labs}
    fin = nrm(fj.get("final_boxed")) or a["f100"]
    info.setdefault(r.base_id, []).append(dict(model=r.model, y=r.y, fin=fin, finished=r.stop == "eos", ans=a, ent=[fj["results"].get(f"{fj['labels'][l]}|B", {}).get("first_entropy", np.nan) for l in labs]))
rows = []
for p, lst in info.items():
    if len(lst) < 2: continue
    pr = list(combinations(lst, 2))
    d = {"problem": p, "y": lst[0]["y"], "n_models": len(lst),
         "final_agree": np.mean([a["fin"] == b["fin"] and a["fin"] is not None for a, b in pr]),
         "frac_finished": np.mean([x["finished"] for x in lst]),
         "ent_mean": np.nanmean([np.nanmean(x["ent"]) for x in lst])}
    for l in labs: d[f"agree@{l}"] = np.mean([a["ans"][l] == b["ans"][l] and a["ans"][l] is not None for a, b in pr])
    rows.append(d)
Q = pd.DataFrame(rows)
print(f"  {len(Q)} shared problems: robust {int(Q.y.sum())}, non-robust {int((Q.y == 0).sum())}, models per problem {Q.n_models.tolist()}")
print(Q.round(2).sort_values("y").to_string(index=False))
for c in [c for c in Q.columns if c not in ("problem", "y", "n_models")]:
    if Q[c].nunique() > 1 and Q.y.nunique() == 2:
        a, p = exact_perm_p(Q.y.values, Q[c].values)
        print(f"  {c:16s} AUC {a:.2f}  one-sided permutation p {p:.3f}")
