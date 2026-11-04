"""Combined interpretable model, fully nested CV on agg (DeepSeek), then transfer to DeepSeek main_v2.
Components: P = last-prompt-token state (layer chosen in-fold, PCA), C = problem-anchoring cosines (cos(trace state, problem state) averaged over
cutoffs, 6 sampled layers), F = attention Fourier spectrum at the first extras cutoff (<=1024 tokens; head-mean, PCA).
Everything (component set, layer, PCA size, C) is selected inside the training folds. Three conditions: raw / all lengths removed / length-matched pairs."""
from common import *
from sklearn.decomposition import PCA
from itertools import product
allm, feats = load_all()
EX = ROOT / "extra_deepsek" / "extra"
n_ds = len(feats[DS]["prompt_last"]); y = allm.y.values
agg = datasets(allm)["agg"]; dsm = np.array([i for i in datasets(allm)["main_v2"] if allm.model.values[i] == DS])
D4 = pd.read_pickle(A / "exp4_features.pkl")
L3 = np.log(allm[["n_prompt", "n_gen", "L_think"]].astype(float).values + 1)
PL = {L_: np.zeros((len(allm), 4096), np.float32) for L_ in (24, 30, 35)}
for L_ in PL: PL[L_][:n_ds] = feats[DS]["prompt_last"][:, L_]
COS = np.nan_to_num(D4[[f"h{j}_cos_prob_mean" for j in range(6)]].values.astype(np.float32))
FF = np.zeros((len(allm), 36 * 32), np.float32)
for i in np.concatenate([agg, dsm]):
    e = np.load(EX / f"{allm.id.values[i]}.npz"); FF[i] = np.nan_to_num(np.nanmean(e["fft"][0].mean(0), 1).reshape(-1))
SETS = {"P": ("P",), "C": ("C",), "P+C": ("P", "C"), "P+C+F": ("P", "C", "F")}
GRID = list(product(SETS, (24, 30, 35), (4, 8), (0.1, 0.3, 1.0)))


class Model:
    def __init__(self, cfg): self.set, self.layer, self.k, self.C = cfg

    def _blocks(self, X, fit):
        out = []
        for comp in SETS[self.set]:
            M = X[comp] if comp != "P" else X["P"][self.layer]
            if comp == "C":
                out.append(M); continue
            if fit: setattr(self, f"pca_{comp}", make_pipeline(StandardScaler(), PCA(self.k, random_state=0)).fit(M))
            out.append(getattr(self, f"pca_{comp}").transform(M))
        return np.hstack(out)

    def fit(self, X, yy):
        self.lr = make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=3000)).fit(self._blocks(X, True), yy); return self

    def score(self, X): return self.lr.decision_function(self._blocks(X, False))


def sub(X, idx): return {"P": {L_: v[idx] for L_, v in X["P"].items()}, "C": X["C"][idx], "F": X["F"][idx]}


def inner_select(X, rows, yy, groups, seed):
    best, bcfg = -1, None
    inner = folds(rows, yy, groups, 1000 + seed)
    for cfg in GRID:
        s = np.zeros(len(rows))
        for tr, te in inner:
            s[te] = Model(cfg).fit(sub(X, rows[tr]), yy[rows[tr]]).score(sub(X, rows[te]))
        a = roc_auc_score(yy[rows], s)
        if a > best: best, bcfg = a, cfg
    return bcfg, best


def nested(X, rows, yy, groups, reps=3, seed0=0, verbose=False):
    s, c = np.zeros(len(rows)), np.zeros(len(rows)); chosen = []
    for r in range(reps):
        for tr, te in folds(rows, yy, groups, seed0 + r):
            cfg, _ = inner_select(X, rows[tr], yy, groups, seed0 + r)
            chosen.append(cfg)
            s[te] += Model(cfg).fit(sub(X, rows[tr]), yy[rows[tr]]).score(sub(X, rows[te])); c[te] += 1
    return s / np.maximum(c, 1), chosen


def resid(X, rows_list):
    Xr = {"P": {L_: v.astype(np.float64).copy() for L_, v in X["P"].items()}, "C": X["C"].astype(np.float64).copy(), "F": X["F"].astype(np.float64).copy()}
    for rows in rows_list:  # label-free, within each dataset separately
        Z = L3[rows]
        for L_ in Xr["P"]: Xr["P"][L_][rows] -= LinearRegression().fit(Z, Xr["P"][L_][rows]).predict(Z)
        for k in ("C", "F"): Xr[k][rows] -= LinearRegression().fit(Z, Xr[k][rows]).predict(Z)
    return Xr


X0 = {"P": PL, "C": COS, "F": FF}
Xr = resid(X0, [agg, dsm])
ids = np.arange(len(allm))
# matched pairs on prompt + trace length
Ls = (L3[:, :2] - L3[agg, :2].mean(0)) / L3[agg, :2].std(0)
rob = [i for i in agg if y[i] == 1]; non = [i for i in agg if y[i] == 0]; used, pairs = set(), []
for i in sorted(rob, key=lambda i: Ls[i, 1]):
    cand = [j for j in non if j not in used]; j = cand[int(np.argmin([np.sum((Ls[i] - Ls[j]) ** 2) for j in cand]))]; used.add(j); pairs.append((i, j))
pi, pj = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]); mrows = np.concatenate([pi, pj])
gp = np.zeros(len(allm), int); gp[pi] = np.arange(len(pairs)); gp[pj] = np.arange(len(pairs))
print(f"=== Combined model, nested CV on agg (grid of {len(GRID)} configs selected in-fold) ===", flush=True)
for nm, X, rows, groups in (("raw", X0, agg, ids), ("all lengths removed", Xr, agg, ids), ("length-matched pairs", X0, mrows, gp)):
    s, chosen = nested(X, rows, y, groups)
    a = auc_ci(y[rows], s)
    nl = []
    for i in range(3):
        if nm == "length-matched pairs":
            yy = y.copy(); flip = np.random.default_rng(i).random(len(pairs)) < .5; yy[pi[flip]], yy[pj[flip]] = 0, 1
        else:
            yy = shuffled_y(allm, rows, "agg", i)
        nl.append(roc_auc_score(yy[rows], nested(X, rows, yy, groups, reps=1, seed0=60 + i)[0]))
    sets_ = pd.Series([c[0] for c in chosen]).value_counts().to_dict(); lay = pd.Series([c[1] for c in chosen]).value_counts().to_dict()
    print(f"  {nm:22s} n={len(rows):3d}  AUC {a[0]:.3f} [{a[1]:.3f},{a[2]:.3f}]  shuffled-label null {np.mean(nl):.2f}   chosen sets {sets_}  layers {lay}", flush=True)

print("\n=== Transfer: config selected by CV on ALL agg, fit on all agg, scored on DeepSeek main_v2 (n=%d, robust %d) ===" % (len(dsm), int(y[dsm].sum())))
for nm, X in (("raw", X0), ("all lengths removed", Xr)):
    cfg, inner_auc = inner_select(X, agg, y, ids, 0)
    s = Model(cfg).fit(sub(X, agg), y[agg]).score(sub(X, dsm))
    a = roc_auc_score(y[dsm], s); rng = np.random.default_rng(0)
    p = np.mean([roc_auc_score(rng.permutation(y[dsm]), s) >= a for _ in range(20000)])
    print(f"  {nm:22s} chosen {cfg} (inner AUC on agg {inner_auc:.2f})  ->  main_v2 AUC {a:.2f}, one-sided permutation p {p:.3f}")
    for st in SETS:  # every component set at its agg-selected layer/k/C, for transparency
        sub_grid = [g for g in GRID if g[0] == st]
        best = max(sub_grid, key=lambda g: roc_auc_score(y[agg], np.concatenate([Model(g).fit(sub(X, agg[tr]), y[agg[tr]]).score(sub(X, agg[te])) for tr, te in folds(agg, y, ids, 7)])[np.argsort(np.concatenate([te for _, te in folds(agg, y, ids, 7)]))]))
        s2 = Model(best).fit(sub(X, agg), y[agg]).score(sub(X, dsm))
        print(f"      set {st:6s} {best}  ->  main_v2 AUC {roc_auc_score(y[dsm], s2):.2f}")
