"""exp45: does the test-set reversal show up in main_v2's capped regime? The leaderboard logs imply 13/14 test traces hit the 6144 cap,
with forced confidence at 6144 / kNN / A/B reversed (AUC ~0.3) and forced p1 at 2048 positive (~0.75). Here: main_v2 split by whether
thinking ended inside the first 6144 tokens; per subset the pooled and within-model AUC of the same features, with a label-permutation p."""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "2"
from common import *
allm, _ = load_all(); y = allm.y.values; rows = datasets(allm)["main_v2"]; model = allm.model.values
F28 = pickle.load(open(A / "exp28_features_cache.pkl", "rb")); F32 = pickle.load(open(A / "exp32_features.pkl", "rb"))
F44 = pickle.load(open(A / "exp44_features.pkl", "rb")); F38 = pickle.load(open(A / "exp38_features.pkl", "rb"))
BUD = [512, 1024, 2048, 3072, 4096, 6144]
V = {f"fa p1@{b}": np.full(len(allm), np.nan) for b in BUD}; capped = np.zeros(len(allm), bool)
for i in rows:
    fj = force_json(model[i], allm.id.iloc[i]); te = fj.get("think_end")
    capped[i] = te is None or te >= 6144
    for b in BUD: V[f"fa p1@{b}"][i] = fj["results"][f"{fj['labels'][f'b{b}']}|B"]["top"][0]["probs"][0]
V["fa p1 2048 - 6144"] = V["fa p1@2048"] - V["fa p1@6144"]
V["kNN global eff"] = F32["step64|k5|4k|global_efficiency"]; V["A/B shift (16k)"] = F28["ab:dent_max"]
V["TDA H0 spread (16k)"] = F38["C_layer|H0_betti_spread|L3"]
for k in ("forced|p1 slope over budgets", "forced|share of budgets saturated", "trajectory|PC1 share 75% / 25%"): V[k.split("|")[1]] = F44[k]
rng = np.random.default_rng(0)


def auc(x, yy):
    p, n = x[yy == 1], x[yy == 0]
    return ((p[:, None] > n[None]).sum() + .5 * (p[:, None] == n[None]).sum()) / (len(p) * len(n)) if len(p) and len(n) else np.nan


def within(x, yy, mm):
    num = den = 0
    for m in np.unique(mm):
        k = mm == m; p, n = x[k & (yy == 1)], x[k & (yy == 0)]
        num += (p[:, None] > n[None]).sum() + .5 * (p[:, None] == n[None]).sum(); den += len(p) * len(n)
    return num / den if den else np.nan


for name, sub in (("all main_v2", rows), ("capped (thinking not ended by 6144)", rows[capped[rows]]), ("finished within 6144", rows[~capped[rows]])):
    yy = y[sub].astype(int)
    print(f"\n== {name}: {len(sub)} rows, {yy.sum()} robust | per model " + str({m[:8]: f"{yy[model[sub] == m].sum()}/{(model[sub] == m).sum()}" for m in MODELS}))
    print(f"{'feature':28s} {'pooled':>7s} {'within':>7s} {'p(within)':>9s}")
    for f, v in V.items():
        x = v[sub]; ok = np.isfinite(x)
        if ok.sum() < 6: continue
        xs, ys, ms = x[ok], yy[ok], model[sub][ok]; nl = []        # null: labels shuffled inside each model
        for _ in range(2000):
            yp = ys.copy()
            for m in np.unique(ms): k = ms == m; yp[k] = rng.permutation(ys[k])
            nl.append(within(xs, yp, ms))
        w = within(xs, ys, ms); p = np.mean(np.abs(np.array(nl) - .5) >= abs(w - .5))
        print(f"{f:28s} {auc(xs, ys):7.2f} {w:7.2f} {p:9.3f}")
