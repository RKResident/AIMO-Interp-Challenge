"""Extras (DeepSeek only -> agg), tested properly: all 36 layers, depth trajectories, lookback / KL / Fourier at FIXED budgets.
Length controls: (i) fixed budget = only traces still thinking at b, features at exactly b tokens; (ii) all lengths regressed out; (iii) length-matched pairs."""
from common import *
from sklearn.decomposition import PCA
allm, _ = load_all()
EX = ROOT / "extra_deepsek" / "extra"
agg = datasets(allm)["agg"]; y = allm.y.values
L3 = np.log(allm[["n_prompt", "n_gen", "L_think"]].astype(float).values + 1)
BUD = [1024, 2048, 4096]
store = {b: {} for b in BUD}
for i in agg:
    r = allm.iloc[i]; p = EX / f"{r.id}.npz"
    if not p.exists(): continue
    e = np.load(p); ce = list(e["cut_eff"]); Lt = force_json(r.model, r.id)["L"]
    for b in BUD:
        if b not in ce or Lt <= b: continue
        k = ce.index(b)
        res = e["res_last"][k].astype(np.float32).mean(0); att = e["attn_last"][k].astype(np.float32).mean(0); mlp = e["mlp_last"][k].astype(np.float32).mean(0)  # [36,d]
        rn = np.linalg.norm(res, axis=-1); an = np.linalg.norm(att, axis=-1); mn = np.linalg.norm(mlp, axis=-1)
        cosl = (res[1:] * res[:-1]).sum(-1) / (rn[1:] * rn[:-1] + 1e-9)
        look = e["lookback"][k].mean(0); pm = e["prompt_mass"][k].mean(0); ll = e["lookback_last"][k].mean(0)       # [36,32]
        store[b][i] = dict(res=res.astype(np.float16), traj=np.concatenate([np.log(rn), cosl, np.log(an + 1e-6) - np.log(mn + 1e-6)]),
                           look_layer=look.mean(1), look_head=look.reshape(-1), pmass=pm.mean(1), lookl=ll.mean(1),
                           kl=e["kl_adj"][k].mean(0).reshape(-1), klhm=e["kl_headmean"][k].mean(0).reshape(-1),
                           fft=np.nanmean(e["fft"][k].mean(0), 1).reshape(-1))
print("=== Extras, full test on agg (DeepSeek) ===")
for b in BUD: print(f"  budget {b}: {len(store[b])} agg traces still thinking (robust {int(sum(y[i] for i in store[b]))})")


def cv(X, rows, yy, groups, kind, reps=8, k=8, C=0.05):
    s, c = np.zeros(len(rows)), np.zeros(len(rows))
    for r in range(reps):
        for tr, te in folds(rows, yy, groups, r):
            if len(np.unique(yy[rows[tr]])) < 2: continue
            mdl = make_pipeline(StandardScaler(), PCA(min(k, len(tr) - 2), random_state=0), StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)) if kind == "high" \
                else make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
            m = mdl.fit(X[rows[tr]], yy[rows[tr]]); s[te] += m.decision_function(X[rows[te]]); c[te] += 1
    return s / np.maximum(c, 1)


def run(nm, b, getter, kind, quick=False):
    rows = np.array(sorted(store[b]))
    X = np.zeros((len(allm),) + np.asarray(getter(store[b][rows[0]])).shape, np.float32)
    for i in rows: X[i] = getter(store[b][i])
    X = X.reshape(len(allm), -1); X = np.nan_to_num(X)
    ids = np.arange(len(allm))
    a = auc_ci(y[rows], cv(X, rows, y, ids, kind))
    nl = np.mean([roc_auc_score(shuffled_y(allm, rows, "agg", i)[rows], cv(X, rows, shuffled_y(allm, rows, "agg", i), ids, kind, reps=2)) for i in range(3 if quick else 5)])
    Xr = X.astype(np.float64).copy(); Xr[rows] = Xr[rows] - LinearRegression().fit(L3[rows], Xr[rows]).predict(L3[rows])
    ar = auc_ci(y[rows], cv(Xr, rows, y, ids, kind))
    ln = roc_auc_score(y[rows], cv(L3, rows, y, ids, "low"))
    return a, nl, ar, ln, len(rows)


fams = {"lookback ratio, per layer (head mean)": (lambda d: d["look_layer"], "low"),
        "lookback ratio, per layer x head": (lambda d: d["look_head"], "high"),
        "lookback at last suffix token, per layer": (lambda d: d["lookl"], "low"),
        "prompt attention mass, per layer": (lambda d: d["pmass"], "low"),
        "attention KL adjacent layers (50-80% depth)": (lambda d: d["kl"], "high"),
        "attention KL vs head mean (50-80% depth)": (lambda d: d["klhm"], "high"),
        "Fourier spectrum (head mean, per layer)": (lambda d: d["fft"], "high"),
        "depth trajectory: |resid|, cos(l,l+1), attn/mlp ratio": (lambda d: d["traj"], "low")}
for b in BUD:
    print(f"\n--- budget {b} (features at exactly {b} tokens; only traces still thinking) ---")
    for nm, (g, kind) in fams.items():
        a, nl, ar, ln, n = run(nm, b, g, kind)
        print(f"  {nm:52s} n={n:3d} AUC {a[0]:.2f} [{a[1]:.2f},{a[2]:.2f}] null {nl:.2f} | lengths removed {ar[0]:.2f} [{ar[1]:.2f},{ar[2]:.2f}] | length-only {ln:.2f}", flush=True)
# layer sweep of the residual stream at 1024 and 2048
for b in (1024, 2048):
    print(f"\n--- residual stream, layer sweep at {b} tokens (PCA-8 + logistic) ---")
    line = []
    for l in range(0, 36, 3):
        a, nl, ar, ln, n = run(f"res L{l}", b, (lambda d, l=l: d["res"][l].astype(np.float32)), "high", quick=True)
        line.append(f"L{l}: {a[0]:.2f}/{ar[0]:.2f}")
    print("  (raw / lengths removed)  " + "  ".join(line) + f"   null ~0.5, length-only {ln:.2f}, n={n}")
