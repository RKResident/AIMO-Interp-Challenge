"""exp58: every feature and every built recipe we have tried, re-scored and re-calibrated on the UNFINISHED traces only (thinking not ended
within 6144 tokens; 88 rows: main_v2 30 + olympiad 58, 50 problems).
Feature table (one value per row, same definitions on both sets wherever possible):
  run_t1 greedy records (both sets, the submission's own code at 6144): fa fas fsat ab ge tda seg traj pcr chk len token entropy / log-prob
  cluster cycle rank (both sets, from the 16-token states and the shipped K-means centroids); PEARL drift area (both sets)
  attention (exp51; both sets): lookback mean / last quarter / late-early, prompt mass / late-early, LapEigvals scalars, TOHA scalars, KL
  exp47 / exp47b (olympiad for all; main_v2 where it was rebuilt): text markers, entropy bursts, speed variation, step similarity, etc.
For every feature: (1) within-model AUC on the unfinished rows (direction-free: |AUC - .5| ranks), label permutation p; (2) a one-feature
classifier scored out of sample - leave one PROBLEM out over both sets; in each fold the direction comes from the training rows, references
are the training rows' values per model, the threshold is each model's median training score - trained on (a) unfinished rows only,
(b) all rows; accuracy on the held-out unfinished rows. Then every built recipe, re-calibrated the same two ways."""
import json, os, pickle, re, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "2"
import numpy as np
import pandas as pd
sys.path.insert(0, "/home/sabyasachi19/PycharmProjects/AimoAGAIN/analysis"); sys.path.insert(0, "/home/sabyasachi19/PycharmProjects/AimoAGAIN/submission_core")
from common import *                                                                    # noqa: E402,F401
from spurious_features import f_cluster_cycle_rank, f_pearl_area                         # noqa: E402

D = pd.read_csv(A / "exp56_rows.csv"); slug = lambda m: m.replace("/", "__")             # noqa: E731
norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()[:300]                            # noqa: E731
inp = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "t1_run" / "inputs.jsonl")}
D["group"] = [norm(inp[i]["problem"]) for i in D.id]; D["unfinished"] = ~D.finished.astype(bool)
OLY = ROOT / "extra_olympiad_problems_dataset"; MV1 = ROOT / "results_mainv2_t1" / "out_t1"; OLR = OLY / "results_olympiad_features" / "out_t1"
mv = {json.loads(l)["id"]: json.loads(l) for f in MV1.glob("*/rows_shard*.jsonl") for l in open(f)}
ol = {json.loads(l)["id"]: json.loads(l) for f in OLR.glob("*/rows_shard*.jsonl") for l in open(f)}
ART = Path("/home/sabyasachi19/PycharmProjects/aimo/getting-started/solutions/spurious-cluster-graph/spurious_artifacts")
F47 = pickle.load(open(A / "exp47_features.pkl", "rb")); MB = pickle.load(open(A / "exp47b_mainv2_values.pkl", "rb")); MA = pickle.load(open(A / "exp51_attention.pkl", "rb"))
F28 = pickle.load(open(A / "exp28_features_cache.pkl", "rb")); V47c = pickle.load(open(A / "exp47c_values.pkl", "rb"))
allm, _ = load_all(); mvrows = datasets(allm)["main_v2"]; mv_pos = {allm.id.iloc[i]: j for j, i in enumerate(mvrows)}; mv_all = {allm.id.iloc[i]: i for i in mvrows}
ol_pos = {rid: k for k, rid in enumerate(F47["ids"])}

T = {}
def put(f, i, v):
    T.setdefault(f, np.full(len(D), np.nan))[i] = np.nan if v is None else float(v)

for i, r in D.iterrows():
    g = (mv if r.dataset == "main_v2" else ol)[r.id]["greedy"]
    for f, nm in (("fa", "forced confidence @6144"), ("fas", "confidence slope"), ("fsat", "saturated share"), ("ab", "A/B entropy shift"),
                  ("ge", "kNN global efficiency"), ("tda", "TDA H0 spread"), ("seg", "segment PC1 @25%"), ("traj", "late-vs-early cosine"),
                  ("pcr", "PC1 change 75/25"), ("chk", "check share"), ("len", "-log length"), ("mean_tok_entropy", "token entropy mean"),
                  ("mean_tok_logprob", "token log-prob mean")):
        put(nm, i, g.get(f))
    z = np.load((MV1 if r.dataset == "main_v2" else OLR) / slug(r.model_id) / f"{r.id}.npz")
    put("cluster cycle rank (K=64)", i, f_cluster_cycle_rank(z["t0_ws16"][:, 3].astype(np.float32), np.load(ART / f"mu_{slug(r.model_id)}.npy"),
                                                             np.load(ART / f"km64_{slug(r.model_id)}.npy"), g["L"]))
    put("PEARL drift area", i, F28["pearl:mean_state:area"][mv_all[r.id]] if r.dataset == "main_v2" else V47c["pearl"][ol_pos[r.id]])
    src, j = (MA["M"], mv_pos[r.id]) if r.dataset == "main_v2" else (MA["O"], ol_pos[r.id])
    for k, v in src.items(): put("attention: " + k, i, v[j])
    if r.dataset == "olympiad":
        for k, v in F47["F"].items():
            if k.startswith(("0 shortcuts", "probe|")): continue
            put("exp47: " + k.split("|", 1)[1], i, v[ol_pos[r.id]])
    else:
        for k, v in MB["M"].items():
            put("exp47: " + k.split("|", 1)[1], i, v[mv_pos[r.id]])
y, model, group, unf = D.robust.values.astype(int), D.model_id.values, D.group.values, D.unfinished.values
rng = np.random.default_rng(0)


def within(x, yy, mm):
    num = den = 0
    for m in np.unique(mm):
        k = (mm == m) & np.isfinite(x); p, n = x[k & (yy == 1)], x[k & (yy == 0)]
        num += (p[:, None] > n[None]).sum() + .5 * (p[:, None] == n[None]).sum(); den += len(p) * len(n)
    return num / den if den else np.nan


def perm(x, yy, mm, n=1000):
    w = within(x, yy, mm); nl = np.empty(n)
    for t in range(n):
        yp = yy.copy()
        for m in np.unique(mm): k = mm == m; yp[k] = rng.permutation(yy[k])
        nl[t] = within(x, yp, mm)
    return w, float(np.mean(np.abs(nl - .5) >= abs(w - .5)))


def pct(v, ref):
    ref = ref[np.isfinite(ref)]
    return ((ref < v).sum() + .5 * (ref == v).sum()) / len(ref) if np.isfinite(v) and len(ref) else np.nan


def loo(cols, pool, fixed_sign=None):
    """Held-out accuracy on unfinished rows of an equal-weight percentile ensemble of `cols` (directions from the training fold unless fixed)."""
    hit = []
    for g in np.unique(group[unf]):
        trm = (group != g) & (unf if pool == "unfinished" else True)
        te = np.where((group == g) & unf)[0]
        signs = {}
        for c in cols:
            if fixed_sign is not None: signs[c] = fixed_sign.get(c, 1); continue
            a = within(T[c][trm], y[trm], model[trm]); signs[c] = 1 if (np.isnan(a) or a >= .5) else -1
        def score(i, ref_mask):
            u = [pct(T[c][i], T[c][ref_mask & (model == model[i])]) for c in cols]; u = [(v if signs[c] > 0 else 1 - v) for c, v in zip(cols, u) if np.isfinite(v)]
            return np.mean(u) if u else .5
        thr = {}
        for m in np.unique(model[te]):                                          # each model's median training score (refs = its training rows)
            refm = trm & (model == m)
            if refm.sum() >= 3: thr[m] = np.median([score(k, trm) for k in np.where(refm)[0]])
        for i in te:
            if model[i] in thr: hit.append((score(i, trm), thr[model[i]], y[i], model[i]))
    if not hit: return np.nan, np.nan, 0
    s, t, yy, mm = map(np.array, zip(*hit)); d = s > t
    bal = .5 * (d[yy == 1].mean() + (1 - d[yy == 0]).mean()) if (yy == 1).any() and (yy == 0).any() else np.nan
    return float(np.mean(d == yy)), float(bal), len(hit)


rows = []
for c in T:
    x = T[c]; n_u = int((np.isfinite(x) & unf).sum())
    if n_u < 25: continue
    a, p = perm(x[unf], y[unf], model[unf]); a_all = within(x[~unf], y[~unf], model[~unf]) if (np.isfinite(x) & ~unf).sum() > 4 else np.nan
    acc_u, bal_u, n = loo([c], "unfinished"); acc_a, bal_a, _ = loo([c], "all")
    rows.append(dict(feature=c, n_unfinished=n_u, sets="both" if (np.isfinite(x) & unf & (D.dataset == "main_v2").values).sum() >= 5 else "olympiad",
                     auc_unfinished=a, p=p, strength=abs(a - .5), acc_unf_cal=acc_u, bal_unf_cal=bal_u, acc_all_cal=acc_a, bal_all_cal=bal_a))
R = pd.DataFrame(rows).sort_values("strength", ascending=False)
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
print(f"unfinished rows {int(unf.sum())} | features scored {len(R)} | p < 0.05 on unfinished: {int((R.p < .05).sum())} (chance ~{0.05 * len(R):.1f})\n")
print(R[["feature", "sets", "n_unfinished", "auc_unfinished", "p", "bal_unf_cal", "bal_all_cal", "acc_unf_cal", "acc_all_cal"]].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
R.to_csv(A / "exp58_features_unfinished.csv", index=False)

# every built recipe, re-calibrated on unfinished vs all rows (directions fixed to the recipe's own signs)
RECIPES = {"forced-confidence": ["forced confidence @6144"], "check-share": ["check share"], "length": ["-log length"], "tda": ["TDA H0 spread"],
           "cluster-graph": ["cluster cycle rank (K=64)"], "confidence-knn": ["forced confidence @6144", "kNN global efficiency"],
           "confidence-knn-tda": ["forced confidence @6144", "kNN global efficiency", "TDA H0 spread"],
           "no-forcing": ["segment PC1 @25%", "late-vs-early cosine", "PEARL drift area", "kNN global efficiency"],
           "top5-knn": ["forced confidence @6144", "A/B entropy shift", "segment PC1 @25%", "late-vs-early cosine", "PEARL drift area", "kNN global efficiency"],
           "shift-robust": ["confidence slope", "saturated share", "PC1 change 75/25"], "abs-pearl (as ensemble)": ["PEARL drift area"],
           "confidence-checkshare-length": ["forced confidence @6144", "check share", "-log length"],
           "confidence-checkshare-slope": ["forced confidence @6144", "check share", "confidence slope"],
           "confidence-checkshare-slope-lookback": ["forced confidence @6144", "check share", "confidence slope", "attention: lookback late - early", "attention: prompt mass late - early"]}
print("\nbuilt recipes, signs fixed (+1, as shipped), held-out unfinished rows: balanced accuracy / accuracy")
print(f"{'recipe':40s} {'calibrated on unfinished':>26s} {'calibrated on all rows':>24s}")
out = []
for nm, cols in RECIPES.items():
    au, bu, n = loo(cols, "unfinished", fixed_sign={c: 1 for c in cols}); aa, ba, _ = loo(cols, "all", fixed_sign={c: 1 for c in cols})
    print(f"{nm:40s} {bu:14.2f} / {au:.2f} {ba:14.2f} / {aa:.2f}   (n {n})"); out.append(dict(recipe=nm, bal_unf=bu, acc_unf=au, bal_all=ba, acc_all=aa, n=n))
pd.DataFrame(out).to_csv(A / "exp58_recipes_unfinished.csv", index=False)
