"""exp40: a problem-DIFFICULTY probe from the prompt alone, used as a robustness feature on main_v2 (no robustness labels are used to
train anything; agg contributes only difficulty = its traces' lengths, never its robustness labels).
Input: the last prompt token's hidden state (saved at every layer for every extracted trace). Target: log thinking length
(agg's organiser base accuracy is 1.0 for 133/137 problems and 93% of its traces finish, so length is the usable difficulty signal).
  A  DeepSeek, trained on the 137 agg problems (layer chosen by 5-fold CV R^2 inside agg), applied to DeepSeek's 10 main_v2 traces
  B  every model, self-supervised on main_v2: leave-one-problem-out ridge on that model's other main_v2 traces (layer chosen by the
     leave-one-out length R^2, i.e. by how well difficulty is predicted - not by robustness); DeepSeek also with agg added to training
Feature = predicted log length (prompt-only, pre-generation). Scored like exp34 (within-model z, problem level, finished fraction + mean
log length regressed out, permutation p); 'resid' = predicted minus actual log length (is the model longer / shorter than expected?)."""
import os, json
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "2"
from common import *
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

allm, _ = load_all(); y = allm.y.values; ds = datasets(allm); rows, agg = ds["main_v2"], ds["agg"]
model = allm.model.values; prob = allm.base_id.values
fin = (allm.stop.values == "eos").astype(float); lgn = np.log(allm.n_gen.astype(float).values)
use = np.r_[rows, agg]
Lth = {i: force_json(model[i], allm.id.iloc[i])["L"] for i in use}
logL = np.full(len(allm), np.nan); logL[use] = [np.log(Lth[i]) for i in use]
P = {i: np.load(OUT / model[i] / "tf" / f"{allm.id.iloc[i]}.npz")["prompt_last_all_layers"].astype(np.float32) for i in use}
ALPHAS = np.logspace(-1, 5, 13)


def fit_predict(tr, te, layer):
    X = np.stack([P[i][layer] for i in tr]); mu, sd = X.mean(0), X.std(0) + 1e-6
    r = RidgeCV(alphas=ALPHAS).fit((X - mu) / sd, logL[tr])
    return r.predict((np.stack([P[i][layer] for i in te]) - mu) / sd)


out_feat = {}
# ---------------------------------------------------------------- A: DeepSeek, trained on agg difficulty only
nl = P[agg[0]].shape[0]; kf = KFold(5, shuffle=True, random_state=0); r2 = []
for l in range(nl):
    pred = np.zeros(len(agg))
    for tr, te in kf.split(agg): pred[te] = fit_predict(agg[tr], agg[te], l)
    r2.append(1 - ((pred - logL[agg]) ** 2).sum() / ((logL[agg] - logL[agg].mean()) ** 2).sum())
best = int(np.argmax(r2)); dsr = rows[model[rows] == DS]
pa = np.full(len(allm), np.nan); pa[dsr] = fit_predict(agg, dsr, best)
print(f"A  DeepSeek agg-trained difficulty probe: best layer {best}/{nl} (CV R^2 in agg {r2[best]:.2f}; median over layers {np.median(r2):.2f})")
print(f"   on DeepSeek main_v2 (10 traces): Spearman(predicted, actual log thinking length) {spearmanr(pa[dsr], logL[dsr])[0]:.2f} | "
      f"AUC(predicted difficulty low -> robust) {roc_auc_score(y[dsr], -pa[dsr]):.2f} | AUC(actual length low -> robust) {roc_auc_score(y[dsr], -logL[dsr]):.2f}")
out_feat["A_ds_agg|pred"] = pa

# ---------------------------------------------------------------- B: every model, leave-one-problem-out on main_v2 (+ agg for DeepSeek)
for variant in ("mv2", "mv2+agg"):
    pb = np.full(len(allm), np.nan); info = []
    for m in MODELS:
        if variant == "mv2+agg" and m != DS: continue
        rm = rows[model[rows] == m]; extra = agg if (variant == "mv2+agg") else np.array([], int)
        nlm = P[rm[0]].shape[0]
        best_l, best_r2, best_pred = None, -np.inf, None
        for l in range(0, nlm, 2):                                    # every other layer
            pred = np.zeros(len(rm))
            for k, i in enumerate(rm):
                tr = np.r_[[j for j in rm if prob[j] != prob[i]], extra].astype(int); pred[k] = fit_predict(tr, np.array([i]), l)[0]
            rr = 1 - ((pred - logL[rm]) ** 2).sum() / ((logL[rm] - logL[rm].mean()) ** 2).sum()
            if rr > best_r2: best_l, best_r2, best_pred = l, rr, pred
        pb[rm] = best_pred
        info.append(f"{m.split('__')[1][:10]} layer {best_l} LOO length R^2 {best_r2:.2f} Spearman {spearmanr(best_pred, logL[rm])[0]:.2f}")
    print(f"B  {variant}: " + " | ".join(info))
    out_feat[f"B_{variant}|pred"] = pb
    out_feat[f"B_{variant}|resid"] = logL - pb                       # longer than the prompt predicted -> ?

# ---------------------------------------------------------------- scoring (main_v2)
up = np.unique(prob[rows]); yp = np.array([y[rows][prob[rows] == p][0] for p in up]).astype(int)
fp = np.array([fin[rows][prob[rows] == p].mean() for p in up]); lp = np.array([lgn[rows][prob[rows] == p].mean() for p in up])
rng = np.random.default_rng(0)


def zm(v):
    o = np.full(len(allm), np.nan)
    for m in np.unique(model[rows]):
        r = rows[model[rows] == m]; x = v[r]
        if np.isfinite(x).sum() >= 2 and np.nanstd(x) > 0: o[r] = (x - np.nanmean(x)) / np.nanstd(x)
    return o


def auc(x, yy):
    rk = rankdata(x); n1 = yy.sum(); n0 = len(yy) - n1
    return (rk[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


print("\nAUC > 0.5: higher value -> robust (so for predicted difficulty, < 0.5 means 'predicted harder -> non-robust').")
for nm, v in out_feat.items():
    z = zm(v); r = rows[np.isfinite(z[rows])]
    if len(r) < 12:
        print(f"  {nm:18s} rows {len(r)} (DeepSeek only - see A above)"); continue
    xp = np.array([np.nanmean(z[rows][prob[rows] == p]) if np.isfinite(z[rows][prob[rows] == p]).any() else np.nan for p in up]); ok = np.isfinite(xp)
    C = np.c_[np.ones(ok.sum()), fp[ok], lp[ok]]; xr = xp[ok] - C @ np.linalg.lstsq(C, xp[ok], rcond=None)[0]
    a = auc(xr, yp[ok]); rk = rankdata(xr); y_ = yp[ok]; n1 = y_.sum(); n0 = len(y_) - n1
    YP = rng.permuted(np.tile(y_, (5000, 1)), axis=1).astype(float); nlv = (YP @ rk - n1 * (n1 + 1) / 2) / (n1 * n0)
    print(f"  {nm:18s} rows {len(r)} | pooled {roc_auc_score(y[r], z[r]):.2f} | problem {auc(xp[ok], yp[ok]):.2f} | controlled {a:.2f} "
          f"(p {np.mean(np.abs(nlv - .5) >= abs(a - .5)):.3f})")
pickle.dump(out_feat, open(A / "exp40_features.pkl", "wb"))
