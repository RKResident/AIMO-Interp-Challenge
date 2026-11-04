"""exp39: is there a linear direction separating robust from non-robust traces? (main_v2 only, 43 rows, 22 problems, 7 non-robust)
Per model (state spaces differ, so no pooling of directions) and per layer, a difference-of-means direction
    w = mean(robust states) - mean(non-robust states)      (the most stable estimator with ~10 rows / ~3 non-robust per model)
is estimated leave-one-PROBLEM-out (all rows of the held-out problem excluded), and the held-out row is scored by its projection on w.
Variants (unit-normalised states, model-mean centred on the training rows):
  think_mean   mean of the 16-token window means over the whole thinking part, at each of the 6 trajectory layers
  first2k      the same over the first 2048 tokens
  prompt       the last prompt token, at each of the 6 trajectory layers
Out-of-fold projections are z-scored within model and pooled; scored at the row and problem level (with the usual finished-fraction +
mean-log-length control), with a permutation null of the WHOLE procedure (labels shuffled across problems, directions refitted)."""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "2"
from common import *
from scipy.stats import rankdata

P2 = ROOT / "extraction_out_generations_p2" / "extraction_out_generations"
allm, _ = load_all(); y = allm.y.values; rows = datasets(allm)["main_v2"]; model = allm.model.values; prob = allm.base_id.values
fin = (allm.stop.values == "eos").astype(float); lgn = np.log(allm.n_gen.astype(float).values)
Lthink = {i: force_json(model[i], allm.id.iloc[i])["L"] for i in rows}
X = {}                                                   # (variant, layer slot) -> {row: unit vector}
for i in rows:
    z = np.load(P2 / model[i] / "tf" / f"{allm.id.iloc[i]}.npz")
    ws = z["win_states"].astype(np.float32); tl = list(z["traj_layers"]); nw = min(len(ws), max(1, Lthink[i] // 16))
    pl = z["prompt_last_all_layers"].astype(np.float32)
    for l in range(ws.shape[1]):
        for nm, v in (("think_mean", ws[:nw, l].mean(0)), ("first2k", ws[: min(nw, 128), l].mean(0)), ("prompt", pl[tl[l]])):
            X.setdefault((nm, l), {})[i] = v / (np.linalg.norm(v) + 1e-9)


def oof_scores(key, yy):
    """leave-one-problem-out diff-of-means projections, per model; yy = labels over allm (may be permuted)"""
    s = np.full(len(allm), np.nan)
    for m in np.unique(model[rows]):
        rm = [i for i in rows if model[i] == m]
        for i in rm:
            tr = [j for j in rm if prob[j] != prob[i]]
            pos = [X[key][j] for j in tr if yy[j] == 1]; neg = [X[key][j] for j in tr if yy[j] == 0]
            if not pos or not neg: continue
            mu = np.mean([X[key][j] for j in tr], 0)
            w = np.mean(pos, 0) - np.mean(neg, 0)
            s[i] = float((X[key][i] - mu) @ w / (np.linalg.norm(w) + 1e-9))
    return s


def zmodel(v):
    out = np.full(len(allm), np.nan)
    for m in np.unique(model[rows]):
        r = rows[model[rows] == m]; x = v[r]
        if np.isfinite(x).sum() >= 2 and np.nanstd(x) > 0: out[r] = (x - np.nanmean(x)) / np.nanstd(x)
    return out


up = np.unique(prob[rows])
fp = np.array([fin[rows][prob[rows] == p].mean() for p in up]); lp = np.array([lgn[rows][prob[rows] == p].mean() for p in up])


def auc(x, yy):
    rk = rankdata(x); n1 = yy.sum(); n0 = len(yy) - n1
    return (rk[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def evaluate(s, yy):
    z = zmodel(s); r = rows[np.isfinite(z[rows])]
    ypr = np.array([yy[rows][prob[rows] == p][0] for p in up]).astype(int)
    xp = np.array([np.nanmean(z[rows][prob[rows] == p]) if np.isfinite(z[rows][prob[rows] == p]).any() else np.nan for p in up]); ok = np.isfinite(xp)
    C = np.c_[np.ones(ok.sum()), fp[ok], lp[ok]]; xr = xp[ok] - C @ np.linalg.lstsq(C, xp[ok], rcond=None)[0]
    return roc_auc_score(yy[r], z[r]), auc(xp[ok], ypr[ok]), auc(xr, ypr[ok])


rng = np.random.default_rng(0)
NPERM = 200
res = []
for key in sorted(X):
    a_row, a_prob, a_ctrl = evaluate(oof_scores(key, y), y)
    null = []
    for _ in range(NPERM):                                   # shuffle problem labels (all rows of a problem share a label), refit everything
        perm = dict(zip(up, rng.permutation([y[rows][prob[rows] == p][0] for p in up])))
        yy = y.copy(); yy[rows] = [perm[prob[i]] for i in rows]
        null.append(evaluate(oof_scores(key, yy), yy)[2])
    null = np.array(null)
    res.append(dict(variant=key[0], layer=key[1], row_auc=a_row, problem_auc=a_prob, controlled=a_ctrl,
                    p_perm=float(np.mean(np.abs(null - .5) >= abs(a_ctrl - .5))), null_95=f"{np.quantile(null, .025):.2f}-{np.quantile(null, .975):.2f}"))
    print(res[-1], flush=True)
T = pd.DataFrame(res); T.to_csv(A / "exp39_linear_direction.csv", index=False)
pd.set_option("display.width", 200)
print("\nAUC > 0.5: the held-out projection on the robust-minus-non-robust direction is higher for robust traces.")
print(T.round(3).to_string(index=False))
print(f"\n{len(T)} variant x layer settings; permutation p < .05: {(T.p_perm < .05).sum()} (~{.05 * len(T):.1f} by chance)")
