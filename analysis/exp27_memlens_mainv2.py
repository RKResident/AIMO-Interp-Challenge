"""MemLens on main_v2, 4 models pooled (problem-grouped CV). Residual stream at the forced-answer position ('...\\boxed{') at every layer
(extras res_last), logit lens restricted to the 10 digit tokens (with each model's final RMSNorm convention), per-layer digit entropy /
confidence. Model-aligned features: summaries and a trajectory resampled to 12 relative-depth points. CPU capped."""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "3"
from common import *
R3 = ROOT / "extraction_out_generations_pearl_more" / "extraction_out_generations"
EXDIR = {DS: ROOT / "extra_deepsek" / "extra"}
for m in MODELS[1:]: EXDIR[m] = R3 / m / "extra"
allm, _ = load_all(); y = allm.y.values
main = datasets(allm)["main_v2"]
pb = np.unique(allm.base_id.values, return_inverse=True)[1]
L3 = np.log(allm[["n_prompt", "n_gen", "L_think"]].astype(float).values + 1)
HEAD = {m: np.load(A / "memlens" / f"digit_head_{m}.npz") for m in MODELS}


def lens(res, H):
    x = res.astype(np.float32); x = x / np.sqrt((x ** 2).mean(-1, keepdims=True) + float(H["rms_norm_eps"])) * (float(H["norm_offset"]) + H["norm_weight"])
    lg = x @ H["digit_rows"].T; lg -= lg.max(-1, keepdims=True); p = np.exp(lg); return p / p.sum(-1, keepdims=True)


SUM, TRJ, check = {}, {}, {m: [] for m in MODELS}
for i in main:
    m = allm.model.values[i]; pid = allm.id.values[i]; f = EXDIR[m] / f"{pid}.npz"
    if not f.exists(): continue
    e = np.load(f); H = HEAD[m]; did = list(H["digit_ids"]); ce = list(e["cut_eff"])
    P = lens(e["res_last"], H)                                      # [cut, var, L, 10]
    nl = P.shape[2]
    ent = -(P * np.log(P + 1e-12)).sum(-1); mx = P.max(-1)
    stable = P.argmax(-1) == P.argmax(-1)[..., -1:]
    emerg = (nl - np.flip(stable, -1).cumprod(-1).sum(-1)) / nl
    late, mid = slice(int(nl * 2 / 3), nl), slice(int(nl * .5), int(nl * .75))
    s = np.stack([emerg, ent.mean(-1), ent[..., late].mean(-1), mx[..., -1], mx[..., mid].mean(-1), mx[..., -1] - mx[..., mid].mean(-1)], -1).mean(1)   # [cut, 6]
    srt = -np.sort(-P, -1)[..., :3]                                 # answer-agnostic: top-3 digit probabilities
    tr = np.concatenate([srt, ent[..., None], mx[..., None]], -1).mean(1)   # [cut, L, 5]
    grid = np.linspace(0, nl - 1, 12)
    trr = np.stack([np.stack([np.interp(grid, np.arange(nl), tr[c, :, k]) for k in range(tr.shape[-1])], -1) for c in range(tr.shape[0])])  # [cut, 12, 5]
    SUM[i] = (s[0], s[-1]); TRJ[i] = (trr[0].reshape(-1), trr[-1].reshape(-1))
    fj = force_json(m, pid)
    for k, c in enumerate(ce):
        r = fj["results"].get(f"{c}|B")
        if r and r["top"] and r["top"][0]["ids"][0] in did:
            check[m].append(did[int(P[k, list(e["variant_names"]).index("B"), -1].argmax())] == r["top"][0]["ids"][0])
print("logit-lens check (final-layer digit argmax == model's actual first answer digit):")
for m in MODELS: print(f"  {m.split('__')[-1]:28s} {np.mean(check[m]) if check[m] else float('nan'):.3f} of {len(check[m])} cases")
rows = np.array(sorted(SUM)); print(f"main_v2 rows with extras: {len(rows)} (robust {int(y[rows].sum())}) over {len(set(allm.base_id.values[rows]))} problems")


def cv(X, yy, reps=10, seed=0, C=0.1):
    s, c = np.zeros(len(rows)), np.zeros(len(rows))
    for r in range(reps):
        for tr, te in folds(rows, yy, pb, seed + r):
            if len(np.unique(yy[rows[tr]])) < 2: continue
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(X[rows[tr]], yy[rows[tr]])
            s[te] += m.decision_function(X[rows[te]]); c[te] += 1
    return s / np.maximum(c, 1)


def report(name, X, C=0.1):
    Xz = zgroup(X, allm.grp.values)
    s = cv(Xz, y, C=C); a = auc_ci(y[rows], s)
    nl = [roc_auc_score(shuffled_y(allm, rows, "main_v2", k)[rows], cv(Xz, shuffled_y(allm, rows, "main_v2", k), reps=3, seed=50 + k, C=C)) for k in range(6)]
    Xr = Xz.copy(); Xr[rows] -= LinearRegression().fit(L3[rows], Xr[rows]).predict(L3[rows])
    ar = roc_auc_score(y[rows], cv(Xr, y, C=C)); ln = roc_auc_score(y[rows], cv(zgroup(L3, allm.grp.values), y))
    mm = allm.model.values[rows]
    per = {k.split('__')[-1][:8]: round(roc_auc_score(y[rows][mm == k], s[mm == k]), 2) for k in MODELS if len(np.unique(y[rows][mm == k])) == 2}
    print(f"  {name:44s} AUC {a[0]:.2f} [{a[1]:.2f},{a[2]:.2f}] null {np.mean(nl):.2f} +- {np.std(nl):.2f} | lengths removed {ar:.2f} | length-only {ln:.2f} | per model {per}", flush=True)


NAMES = ["emergence depth", "entropy mean", "entropy late", "final max conf", "mid max conf", "final-minus-mid conf"]
for k, tag in ((0, "first cutoff (<=1024 tok)"), (1, "last cutoff (largest budget)")):
    S = np.full((len(allm), 6), np.nan, np.float32); T = np.full((len(allm), 60), np.nan, np.float32)
    for i in rows: S[i] = SUM[i][k]; T[i] = TRJ[i][k]
    print(f"\n=== {tag} ===")
    report("summaries (6)", S)
    report("trajectory, 12 depth points x 5 (top-3 p, entropy, conf)", T, C=0.03)
    Sz = zgroup(S, allm.grp.values); mm = allm.model.values[rows]
    print("  single summaries, pooled (z within model; AUC, robust positive): " + "; ".join(f"{n} {roc_auc_score(y[rows], Sz[rows, j]):.2f}" for j, n in enumerate(NAMES)))
    for m in MODELS:
        sel = rows[mm == m]
        if len(np.unique(y[sel])) == 2:
            print(f"    {m.split('__')[-1]:28s} (n={len(sel)}, robust {int(y[sel].sum())}): " + "; ".join(f"{n} {roc_auc_score(y[sel], S[sel, j]):.2f}" for j, n in enumerate(NAMES[:4])))
