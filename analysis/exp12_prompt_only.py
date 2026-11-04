"""Idea 5: prompt-only features (computed before any trace exists, so trace length cannot leak), with PROMPT length controlled two ways:
residualising on log prompt length, and prompt-length-matched pairs (agg). agg uses DeepSeek's full prompt states; main_v2 uses model-aligned summaries."""
from common import *
from sklearn.decomposition import PCA
from scipy.stats import wilcoxon
allm, feats = load_all()
ds = datasets(allm); agg, main = ds["agg"], ds["main_v2"]; y = allm.y.values; grp = allm.grp.values
lp = np.log(allm.n_prompt.astype(float).values + 1)
pb = np.unique(allm.base_id.values, return_inverse=True)[1]
n_ds = len(feats[DS]["prompt_last"])


def cv(X, rows, yy, groups, kind="low", reps=8, C=0.1, k=8):
    s, c = np.zeros(len(rows)), np.zeros(len(rows))
    for r in range(reps):
        for tr, te in folds(rows, yy, groups, r):
            if len(np.unique(yy[rows[tr]])) < 2: continue
            mdl = make_pipeline(StandardScaler(), PCA(k, random_state=0), StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)) if kind == "high" \
                else make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
            m = mdl.fit(X[rows[tr]], yy[rows[tr]]); s[te] += m.decision_function(X[rows[te]]); c[te] += 1
    return s / np.maximum(c, 1)


def res_on_prompt(X, rows):
    X = np.nan_to_num(X.astype(np.float64)).copy(); X[rows] = X[rows] - LinearRegression().fit(lp[rows, None], X[rows]).predict(lp[rows, None]); return X


def nulls(X, rows, groups, kind, ds_name, n=5):
    return np.mean([roc_auc_score(shuffled_y(allm, rows, ds_name, i)[rows], cv(X, rows, shuffled_y(allm, rows, ds_name, i), groups, kind, reps=2)) for i in range(n)])


# ---------------- agg (DeepSeek full prompt states)
def pad(a):  # DeepSeek arrays cover the first n_ds rows of allm
    out = np.full((len(allm),) + a.shape[1:], np.nan, np.float32); out[:n_ds] = a; return out


famA = {"prompt length only": (lp[:, None], "low"), "problem logprob/entropy": (allm[["prob_lp_mean", "prob_lp_min", "prob_ent_mean"]].astype(float).values, "low")}
for L_ in (8, 13, 19, 24, 30, 35):
    famA[f"last prompt token, layer {L_}"] = (pad(feats[DS]["prompt_last"][:, L_]), "high")
famA["mean problem-token state (6 layers)"] = (pad(feats[DS]["prompt_mean"].reshape(n_ds, -1)), "high")
print("=== Idea 5: prompt-only features, prompt length controlled ===\n[agg] residualised on log prompt length (label-free):")
idA = np.arange(len(allm))
for nm, (X, kind) in famA.items():
    Xr = res_on_prompt(X, agg) if nm != "prompt length only" else np.nan_to_num(X)
    a = auc_ci(y[agg], cv(Xr, agg, y, idA, kind)); a0 = roc_auc_score(y[agg], cv(np.nan_to_num(X), agg, y, idA, kind))
    print(f"  {nm:38s} raw {a0:.2f} | prompt-length removed {a[0]:.2f} [{a[1]:.2f},{a[2]:.2f}]  null {nulls(Xr, agg, idA, kind, 'agg'):.2f}", flush=True)
# prompt-length matched pairs
rob = [i for i in agg if y[i] == 1]; non = [i for i in agg if y[i] == 0]; used, pairs = set(), []
for i in sorted(rob, key=lambda i: lp[i]):
    cand = [j for j in non if j not in used]; j = cand[int(np.argmin([abs(lp[i] - lp[j]) for j in cand]))]; used.add(j); pairs.append((i, j))
pi, pj = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
print(f"\n[agg] prompt-length-matched pairs ({len(pairs)}): mean |log-length gap| {np.mean(np.abs(lp[pi] - lp[pj])):.3f}, prompt tokens robust {np.exp(lp[pi]).mean():.0f} vs non {np.exp(lp[pj]).mean():.0f}")
rows = np.concatenate([pi, pj]); gp = np.zeros(len(allm), int); gp[pi] = np.arange(len(pairs)); gp[pj] = np.arange(len(pairs))
for nm, (X, kind) in famA.items():
    a = auc_ci(y[rows], cv(np.nan_to_num(X), rows, y, gp, kind))
    print(f"  {nm:38s} matched-set CV AUC {a[0]:.2f} [{a[1]:.2f},{a[2]:.2f}]", flush=True)

# ---------------- main_v2 (4 models pooled; model-aligned prompt summaries; grouped by problem)
def aligned_prompt(m):
    F = feats[m]; pl = F["prompt_last"]; pm = F["prompt_mean"]; nl = pl.shape[1]
    sel = sorted({min(nl - 1, max(0, int(round(f * nl)) - 1)) for f in (0.25, 0.4, 0.5, 0.6, 0.75, 1.0)})[:pm.shape[1]]
    norms = np.log(np.linalg.norm(pl[:, sel], axis=-1) + 1e-6)
    cos = (pl[:, sel] * pm).sum(-1) / (np.linalg.norm(pl[:, sel], axis=-1) * np.linalg.norm(pm, axis=-1) + 1e-9)
    pmn = np.log(np.linalg.norm(pm, axis=-1) + 1e-6)
    return np.hstack([norms, cos, pmn])


AP = np.vstack([aligned_prompt(m) for m in MODELS])
famM = {"prompt length only": lp[:, None], "problem logprob/entropy": allm[["prob_lp_mean", "prob_lp_min", "prob_ent_mean"]].astype(float).values,
        "aligned prompt-state summaries": AP, "both": np.hstack([AP, allm[["prob_lp_mean", "prob_lp_min", "prob_ent_mean"]].astype(float).values])}
print(f"\n[main_v2] {len(main)} rows / {len(set(allm.base_id.values[main]))} problems, z-scored within model, grouped by problem:")
for nm, X in famM.items():
    Xz = zgroup(X, grp); Xr = res_on_prompt(Xz, main) if nm != "prompt length only" else Xz
    a0 = roc_auc_score(y[main], cv(Xz, main, y, pb)); a = auc_ci(y[main], cv(Xr, main, y, pb))
    print(f"  {nm:38s} raw {a0:.2f} | prompt-length removed {a[0]:.2f} [{a[1]:.2f},{a[2]:.2f}]  null {nulls(Xr, main, pb, 'low', 'main_v2'):.2f}", flush=True)
