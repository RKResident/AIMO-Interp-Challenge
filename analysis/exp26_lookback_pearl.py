"""Round 3: Lookback Lens (teacher-forced, per generated token) and PEARL-style prompt sensitivity.
agg (DeepSeek): robust vs all / vs genuine fragility (organizer decay perm keeps the numbers) / vs information loss.
main_v2: 4 models pooled, features z-scored within model, CV grouped by problem. No proxies. CPU capped (run under taskset)."""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "3"
from common import *
from sklearn.decomposition import PCA
R3 = ROOT / "extraction_out_generations_pearl_more" / "extraction_out_generations"
allm, _ = load_all(); y = allm.y.values
ds = datasets(allm); agg, main = ds["agg"], ds["main_v2"]
L3 = np.log(allm[["n_prompt", "n_gen", "L_think"]].astype(float).values + 1)
pb = np.unique(allm.base_id.values, return_inverse=True)[1]; ids = np.arange(len(allm))
RATES = [0.02, 0.05, 0.1, 0.2, 0.3]

# ---------------- groups on agg (organizer permutations)
a = pd.read_csv(ROOT / "augmented-sample-math-agg.csv"); nums = lambda s: set(re.findall(r"\d+(?:\.\d+)?", str(s)))
grp, keep = {}, {}
for r in a.itertuples():
    try: perms = json.loads(r.permutations_causing_decay)
    except Exception: perms = []
    pid = str(r.problem_id)
    if bool(r.model_is_robust) and not perms: grp.setdefault(pid, "robust"); continue
    keep[pid] = keep.get(pid, False) or any(nums(r.original_problem) <= nums(p.get("question", "") if isinstance(p, dict) else p) for p in perms)
for pid, k in keep.items(): grp[pid] = "genuine" if k else "info_loss"
g = np.array([grp.get(str(p)) if i in set(agg) else None for i, p in enumerate(allm.parent_id.astype(str).values)])

# ---------------- Lookback Lens features
LB = {}
for i, r in enumerate(allm.itertuples()):
    p = R3 / r.model / "lookback" / f"{r.id}.npz"
    if p.exists(): LB[i] = np.load(p)


def lb_feat(i, b=None, aligned=False):
    z = LB[i]; lr = z["lookback"].astype(np.float32); sp = int(z["span"]); te = int(z["think_end"])
    nthink = lr.shape[0] if te < 0 else max(1, -(-te // sp))
    if b is not None:
        if (te if te >= 0 else int(z["n_gen"])) <= b: return None          # only traces still thinking at b
        x = lr[: b // sp]
    else:
        x = lr[:nthink]
    m = x.mean(0)                                                       # [L, H] mean lookback ratio over the spans
    if not aligned: return m.reshape(-1)
    bins = np.array_split(np.arange(m.shape[0]), 6)
    half = x[: len(x) // 2].mean(0), x[len(x) // 2:].mean(0)
    return np.concatenate([[m[bb].mean(), m[bb].std()] for bb in bins] + [[(half[1] - half[0])[bb].mean()] for bb in bins])


# ---------------- PEARL features
PR = {}
for i, r in enumerate(allm.itertuples()):
    p = R3 / r.model / "pearl" / f"{r.id}.npz"
    if p.exists(): PR[i] = np.load(p, allow_pickle=True)


def cosd(A, B): return 1 - (A * B).sum(-1) / (np.linalg.norm(A, axis=-1) * np.linalg.norm(B, axis=-1) + 1e-9)


def pearl_feat(i):
    z = PR[i]; kind, rate = z["kind"], z["rate"]
    o = np.where(kind == "original")[0][0]
    curves = {}
    for nm in ("prefix_kl", "prefix_js"):
        curves[nm] = np.array([z[nm][(kind == "noise") & np.isclose(rate, r)].mean() for r in RATES])
    curves["prefix_flip"] = np.array([1 - z["prefix_top1_same"][(kind == "noise") & np.isclose(rate, r)].mean() for r in RATES])
    curves["prefix_dnll"] = np.array([z["prefix_nll"][(kind == "noise") & np.isclose(rate, r)].mean() - z["prefix_nll"][o] for r in RATES])
    for st in ("last_state", "mean_state", "prefix_state"):
        S = z[st].astype(np.float32)
        curves[st] = np.array([cosd(S[(kind == "noise") & np.isclose(rate, r)], S[o][None]).mean() for r in RATES])   # mean over seeds & 6 layers
    f = []
    for c in curves.values():
        d = np.diff(np.concatenate([[0], c]))
        f += list(c) + [d.max(), np.polyfit(RATES, c, 1)[0], np.trapz(c, RATES)]      # values, PEARL max fall-off, slope, area
    return np.array(f, dtype=np.float32)


def sem_feat(i):  # semantic-variant shifts (only rows that have them)
    z = PR[i]; kind = z["kind"]; o = np.where(kind == "original")[0][0]
    if not any(k in kind for k in ("rephrase", "domain")): return None
    out = []
    for t in ("rephrase", "rename", "distract", "domain", "typos"):
        j = np.where(kind == t)[0]
        if not len(j): return None
        j = j[0]; out += [z["prefix_kl"][j], 1 - z["prefix_top1_same"][j], z["prefix_nll"][j] - z["prefix_nll"][o],
                         cosd(z["prefix_state"][j].astype(np.float32), z["prefix_state"][o].astype(np.float32)).mean()]
    return np.array(out, dtype=np.float32)


def mat(fn, rows, **kw):
    vals = {i: fn(i, **kw) for i in rows}; vals = {i: v for i, v in vals.items() if v is not None}
    if not vals: return None, np.array([], int)
    d = len(next(iter(vals.values()))); X = np.full((len(allm), d), np.nan, np.float32)
    for i, v in vals.items(): X[i] = v
    return X, np.array(sorted(vals))


def cv(X, rows, yy, groups, kind, reps=8, seed=0, C=0.05):
    s, c = np.zeros(len(rows)), np.zeros(len(rows))
    for r in range(reps):
        for tr, te in folds(rows, yy, groups, seed + r):
            if len(np.unique(yy[rows[tr]])) < 2: continue
            m = (make_pipeline(StandardScaler(), PCA(min(8, len(tr) - 2), random_state=0), StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)) if kind == "high"
                 else make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)))
            s[te] += m.fit(X[rows[tr]], yy[rows[tr]]).decision_function(X[rows[te]]); c[te] += 1
    return s / np.maximum(c, 1)


def bins2d(rows):
    a_ = np.log(allm.L_think.astype(float).values[rows] + 1); b_ = np.log(allm.n_prompt.astype(float).values[rows] + 1)
    return np.digitize(a_, np.quantile(a_, [1 / 3, 2 / 3])) * 3 + np.digitize(b_, np.quantile(b_, [1 / 3, 2 / 3]))


def report(name, X, rows, yy, groups, kind, dsname, within=True):
    if X is None or len(rows) < 12 or len(np.unique(yy[rows])) < 2 or min(np.bincount(yy[rows].astype(int))) < 3:
        print(f"    {name:52s} n={len(rows):3d}  (too few rows)"); return
    Xz = zgroup(X, allm.grp.values) if kind == "low" else np.nan_to_num(X)
    s = cv(Xz, rows, yy, groups, kind); a1 = auc_ci(yy[rows], s)
    nl = []
    for k in range(4):
        if dsname == "main_v2": y2 = shuffled_y(allm.assign(y=yy), rows, "main_v2", k)
        else: y2 = yy.copy(); y2[rows] = np.random.default_rng(k).permutation(yy[rows])
        nl.append(roc_auc_score(y2[rows], cv(Xz, rows, y2, groups, kind, reps=2, seed=40 + k)))
    Xr = np.nan_to_num(Xz.astype(np.float64)).copy(); Xr[rows] -= LinearRegression().fit(L3[rows], Xr[rows]).predict(L3[rows])
    ar = roc_auc_score(yy[rows], cv(Xr, rows, yy, groups, kind))
    ln = roc_auc_score(yy[rows], cv(zgroup(L3, allm.grp.values), rows, yy, groups, "low"))
    w = within_bin_auc(yy[rows], s, bins2d(rows)) if within else np.nan
    print(f"    {name:52s} n={len(rows):3d} AUC {a1[0]:.2f} [{a1[1]:.2f},{a1[2]:.2f}] null {np.mean(nl):.2f} | lengths removed {ar:.2f} | "
          f"length-only {ln:.2f}" + (f" | 2-D within-bin {w:.2f}" if within else ""), flush=True)


contrasts = [("robust vs all non-robust", ("robust", "genuine", "info_loss")), ("robust vs genuine fragility", ("robust", "genuine")),
             ("robust vs information loss", ("robust", "info_loss"))]
print("=== Lookback Lens (mean lookback ratio over thinking spans) ===")
for cname, keepset in contrasts:
    rows0 = np.array([i for i in agg if g[i] in keepset and i in LB]); yy = np.zeros(len(allm)); yy[rows0] = (g[rows0] == "robust")
    print(f"  [agg: {cname}]")
    for b in (1024, 2048, 4096, None):
        X, rows = mat(lb_feat, rows0, b=b); rows = np.intersect1d(rows, rows0)
        report(f"all heads, {'whole thinking' if b is None else f'first {b} tokens (still thinking)'} (PCA-8)", X, rows, yy, ids, "high", "agg")
        Xa, rows = mat(lb_feat, rows0, b=b, aligned=True); rows = np.intersect1d(rows, rows0)
        report(f"depth-binned summaries, {'whole thinking' if b is None else f'first {b}'}", Xa, rows, yy, ids, "low", "agg")
print("  [main_v2, 4 models pooled, grouped by problem]")
mrows0 = np.array([i for i in main if i in LB])
for b in (1024, 2048, 4096, None):
    Xa, rows = mat(lb_feat, mrows0, b=b, aligned=True)
    report(f"depth-binned summaries, {'whole thinking' if b is None else f'first {b}'}", Xa, rows, y.astype(float), pb, "low", "main_v2", within=False)

print("\n=== PEARL-style sensitivity (prose noise at 2-30%; reasoning-prefix KL/flip/NLL + hidden-state drift curves) ===")
for cname, keepset in contrasts:
    rows0 = np.array([i for i in agg if g[i] in keepset and i in PR]); yy = np.zeros(len(allm)); yy[rows0] = (g[rows0] == "robust")
    X, rows = mat(pearl_feat, rows0); print(f"  [agg: {cname}]")
    report("noise curves (7 measures x 5 rates + fall-off/slope/area)", X, rows, yy, ids, "low", "agg")
mrows0 = np.array([i for i in main if i in PR])
X, rows = mat(pearl_feat, mrows0)
print("  [main_v2, 4 models pooled, grouped by problem]")
report("noise curves", X, rows, y.astype(float), pb, "low", "main_v2", within=False)
Xs, srows = mat(sem_feat, mrows0)
print(f"  [main_v2 rows with semantic variants: n={len(srows)}, robust {int(y[srows].sum()) if len(srows) else 0}]")
report("semantic-variant shifts (5 types x 4 measures)", Xs, srows, y.astype(float), pb, "low", "main_v2", within=False)
# single-measure view on main_v2: PEARL max fall-off of each measure
if len(rows):
    names = ["prefix_kl", "prefix_js", "prefix_flip", "prefix_dnll", "last_state", "mean_state", "prefix_state"]
    Xz = zgroup(X, allm.grp.values)
    print("  main_v2 single PEARL fall-off AUCs (robust positive; >0.5 = larger fall-off in robust): " +
          "; ".join(f"{n} {roc_auc_score(y[rows], Xz[rows, k * 8 + 5]):.2f}" for k, n in enumerate(names)))
