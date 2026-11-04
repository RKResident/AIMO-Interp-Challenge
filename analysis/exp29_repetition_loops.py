"""exp29: repetition / looping as a sign of a spurious answer (main_v2 only, 43 rows, 22 problems).
Signals from "reasoning loops" (papers_to_see/2601.05693): statement repetition, numerical loops, reflection-marker
density, entropy / max-prob collapse, and hidden-state recurrence (a continuous version of their reasoning-graph cycles).
Each feature is computed on (a) the whole thinking trace and (b) only the first B tokens, for traces still thinking at B,
so that "hit the 16k cap" (the main_v2 shortcut, 0.88 per problem) cannot drive it. Scored like exp28: no fitting,
problem level with (finished fraction, mean log length) regressed out, permutation p."""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "2"
from common import *
import json, re, zlib, glob
from scipy.stats import rankdata
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans

allm, feats = load_all(); y = allm.y.values
rows = datasets(allm)["main_v2"]; m = allm.iloc[rows]
grp = allm.grp.values; prob = allm.base_id.values
fin = (allm.stop.values == "eos").astype(float); lgn = np.log(allm.n_gen.astype(float).values)
Lthink = np.array([force_json(r.model, r.id)["L"] if i in set(rows) else np.nan for i, r in enumerate(allm.itertuples())])
BUDGETS = [2048, 4096]
F = {}


def put(name, i, val):
    if name not in F: F[name] = np.full(len(allm), np.nan)
    F[name][i] = val


G = {}
for f in glob.glob(str(ROOT / "extraction_out" / "*" / "generations.jsonl")):
    for l in open(f):
        d = json.loads(l)
        if d["id"] in set(m.id): G[d["id"]] = d

SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
WORD = re.compile(r"[a-z]+|\d+|[^\sa-z\d]")
REFL = re.compile(r"^(wait|but|maybe|alternatively|hmm|hold on|actually|however|let me (?:check|verify|double))\b", re.I)
CHECKW = re.compile(r"\b(check|verif|confirm|correct|indeed|consistent|double)", re.I)


def units(text):
    return [s.strip().lower() for s in SPLIT.split(text) if len(s.split()) >= 4]


def near_dup(us, thr=.7):
    """for each unit, max 3-shingle Jaccard with any earlier unit (sparse dot products)"""
    sh = [set(zip(*(w[k:] for k in range(3)))) if len(w := WORD.findall(u)) >= 3 else set() for u in us]
    voc = {}; r_, c_ = [], []
    for i, s in enumerate(sh):
        for g in s: r_.append(i); c_.append(voc.setdefault(g, len(voc)))
    if not voc: return np.zeros(len(us))
    X = csr_matrix((np.ones(len(r_)), (r_, c_)), shape=(len(us), len(voc)))
    inter = (X @ X.T).toarray(); sz = np.asarray(X.sum(1)).ravel()
    jac = inter / np.maximum(sz[:, None] + sz[None, :] - inter, 1)
    jac[np.triu_indices(len(us))] = 0                                    # only earlier units
    return jac.max(1)


def longest_periodic_run(ids, pmax=200):
    best = 0
    for p in range(1, min(pmax, len(ids) // 2) + 1):
        eq = np.r_[False, ids[p:] == ids[:-p], False].astype(np.int8)
        d = np.diff(eq); s, e = np.where(d == 1)[0], np.where(d == -1)[0]
        if len(s): best = max(best, int((e - s).max()) + p)
    return best


def text_feats(i, tag, text, ids):
    us = units(text)
    if len(us) < 10: return
    cnt = pd.Series(us).value_counts()
    nd = near_dup(us)
    put(f"text:{tag}:exact_repeat_frac", i, float(np.mean(pd.Series(us).duplicated())))
    put(f"text:{tag}:max_unit_repeats", i, float(np.log(cnt.iloc[0])))
    put(f"text:{tag}:units_repeated_gt3", i, float((cnt > 3).sum() / len(us) * 1000))
    put(f"text:{tag}:near_dup_frac", i, float(np.mean(nd >= .7)))
    put(f"text:{tag}:near_dup_mean", i, float(nd.mean()))
    ck = np.array([bool(CHECKW.search(u)) for u in us])
    if ck.sum() >= 3: put(f"text:{tag}:check_near_dup_frac", i, float(np.mean(nd[ck] >= .7)))
    put(f"text:{tag}:check_share", i, float(ck.mean()))
    put(f"text:{tag}:reflection_density", i, float(np.mean([bool(REFL.match(u)) for u in us])))
    a = np.asarray(ids)
    put(f"text:{tag}:ngram10_copy_rate", i, float(np.mean(pd.Series([hash(tuple(a[k:k + 10])) for k in range(len(a) - 10)]).duplicated())))
    put(f"text:{tag}:zlib_ratio", i, len(zlib.compress(a.astype(np.int32).tobytes(), 9)) / (4 * len(a)))
    put(f"text:{tag}:log_longest_periodic_run", i, float(np.log1p(longest_periodic_run(a))))


def ent_feats(i, tag, ent, top1):
    if len(ent) < 512: return
    k = 64; re_ = np.convolve(ent, np.ones(k) / k, "valid"); rp = np.convolve(top1, np.ones(k) / k, "valid")
    col = re_ < .05
    d = np.diff(np.r_[0, col.astype(np.int8), 0]); runs = np.where(d == -1)[0] - np.where(d == 1)[0]
    put(f"ent:{tag}:collapsed_frac", i, float(col.mean()))
    put(f"ent:{tag}:log_longest_collapse", i, float(np.log1p(runs.max() if len(runs) else 0)))
    put(f"ent:{tag}:maxprob_gt99_frac", i, float((rp > .99).mean()))
    put(f"ent:{tag}:rolling_ent_min", i, float(re_.min()))
    h = len(ent) // 2; put(f"ent:{tag}:late_minus_early", i, float(ent[h:].mean() - ent[:h].mean()))
    q = np.array_split(re_, 8); put(f"ent:{tag}:ent_window_cv", i, float(np.std([x.mean() for x in q]) / (np.mean(re_) + 1e-6)))


loops = []
for i in rows:
    r = allm.iloc[i]; g = G[r.id]; ids = np.array(g["gen_ids"]); lt = int(Lthink[i]); t = g["text"]
    tt = t.split("</think>")[0]; z = np.load(ROOT / "extraction_out" / r.model.replace("/", "__") / "tf" / f"{r.id}.npz")
    ent = z["gen_entropy"].astype(float); top1 = z["gen_top1"].astype(float)
    text_feats(i, "full", tt, ids[:lt]); ent_feats(i, "full", ent[:lt], top1[:lt])
    for B in BUDGETS:
        if lt < B: continue                                                # finished thinking before B: leave NaN
        text_feats(i, f"b{B}", tt[: int(len(tt) * B / lt)], ids[:B]); ent_feats(i, f"b{B}", ent[:B], top1[:B])
    us = units(tt); cnt = pd.Series(us).value_counts()
    loops.append(dict(id=r.id[:40], robust=bool(y[i]), finished=bool(fin[i]), stmt_loop=bool((cnt > 3).any()),
                      num_loop=longest_periodic_run(ids[:lt]) > 500, top_unit=cnt.index[0][:70], reps=int(cnt.iloc[0])))
L = pd.DataFrame(loops)

# ---- hidden-state recurrence on the separately sampled waypoint traces (16 tokens per waypoint, saved_trajectories_v2)
WP = {}
for i in rows:
    f = glob.glob(str(ROOT / "saved_trajectories_v2_data" / "*" / f"{allm.id.iloc[i]}.npz"))
    if f:
        z = np.load(f[0]); ks = sorted([k for k in z.files if k.startswith("layer_")], key=lambda s: int(s.split("_")[1]))
        WP[i] = (z[ks[3]].astype(np.float32), str(z["stop_reason"]), int(z["tokens_generated"]))   # ~2/3 depth
CH = 4                                                                    # 4 waypoints = 64 tokens per step
for mdl in np.unique(allm.model.values[list(WP)]):
    ii = [i for i in WP if allm.model.iloc[i] == mdl]
    steps = {i: WP[i][0][: len(WP[i][0]) // CH * CH].reshape(-1, CH, WP[i][0].shape[1]).mean(1) for i in ii}
    mu = np.concatenate(list(steps.values())).mean(0)
    for i in ii: s = steps[i] - mu; steps[i] = s / (np.linalg.norm(s, axis=1, keepdims=True) + 1e-6)
    allx = np.concatenate(list(steps.values()))
    km = KMeans(n_clusters=min(50, len(allx) // 10), n_init=4, random_state=0).fit(allx)
    thr = np.quantile(np.concatenate([(steps[i] @ steps[i].T)[np.triu_indices(len(steps[i]), 8)] for i in ii]), .95)
    for i in ii:
        for tag, n in [("full", None)] + [(f"b{B}", B // 64) for B in BUDGETS]:
            if n is not None and len(steps[i]) < n: continue
            s = steps[i][:n]; S = s @ s.T; T_ = len(s); lab = km.predict(s)
            far = np.triu(np.ones((T_, T_), bool), 8).T                      # pairs (i, j) with j <= i - 8 (>= 512 tokens back)
            if far.sum() < 20: continue
            put(f"hid:{tag}:recurrence_rate", i, float((S[far] > thr).mean()))
            put(f"hid:{tag}:mean_max_backsim", i, float(np.nanmean(np.where(far, S, np.nan)[8:].max(1))))
            seen_far = [lab[k] in set(lab[:max(0, k - 8)]) for k in range(8, T_)]
            put(f"hid:{tag}:cluster_revisit_rate", i, float(np.mean(seen_far)))
            put(f"hid:{tag}:distinct_clusters_per_step", i, len(set(lab)) / T_)
            put(f"hid:{tag}:adjacent_sim", i, float(np.mean(np.diag(S, 1))))

pickle.dump(F, open(A / "exp29_features.pkl", "wb"))          # feature values, for ensembles (exp34)

# ---------------------------------------------------------------- scoring (as exp28, but permutation over available problems)
up = np.unique(prob[rows]); yp = np.array([y[rows][prob[rows] == p][0] for p in up])
fp = np.array([fin[rows][prob[rows] == p].mean() for p in up]); lp_ = np.array([lgn[rows][prob[rows] == p].mean() for p in up])
rng = np.random.default_rng(0)


def perm_auc(x, yy, n=5000):
    rk = rankdata(x); n1 = yy.sum(); n0 = len(yy) - n1; c = n1 * (n1 + 1) / 2
    a = (rk[yy == 1].sum() - c) / (n1 * n0)
    YP = np.array([rng.permutation(yy) for _ in range(n)]).astype(float); nl = (YP @ rk - c) / (n1 * n0)
    return a, float(np.mean(np.abs(nl - .5) >= abs(a - .5)))


F["baseline:finished"] = fin; F["baseline:-log_length"] = -lgn
out = []
for nm, v in F.items():
    ok = np.isfinite(v[rows]); r = rows[ok]
    if ok.sum() < 12 or len(np.unique(y[r])) < 2 or len(np.unique(v[r])) < 3 and not nm.startswith("baseline"): continue
    vz = zgroup(np.where(np.isin(np.arange(len(allm)), r), v, np.nan)[:, None], grp)[:, 0]
    a_pool = roc_auc_score(y[r], vz[r]); a_fin = within_bin_auc(y[r], vz[r], fin[r].astype(int))
    xp = np.array([np.nanmean(np.where(prob[r] == p, vz[r], np.nan)) if (prob[r] == p).any() else np.nan for p in up]); okp = np.isfinite(xp)
    if okp.sum() < 8 or len(np.unique(yp[okp])) < 2: continue
    C = np.c_[fp, lp_][okp]; ap, pp_ = perm_auc(xp[okp], yp[okp])
    xr = xp[okp] - LinearRegression().fit(C, xp[okp]).predict(C); apr, ppr = perm_auc(xr, yp[okp])
    out.append(dict(feature=nm, rows=int(ok.sum()), problems=int(okp.sum()), nonrobust_problems=int((yp[okp] == 0).sum()), pooled=a_pool,
                    within_finished=a_fin, problem=ap, p_problem=pp_, problem_controlled=apr, p_controlled=ppr))
T = pd.DataFrame(out)
T.to_csv(A / "exp29_repetition_loops.csv", index=False); L.to_csv(A / "exp29_loop_flags.csv", index=False)
pd.set_option("display.width", 250)
print("AUC > 0.5 means a HIGHER value -> robust.  pooled/within_finished: rows; problem*: problem means (z within model).")
print(T.round(3).to_string(index=False))
print("\n=== paper's binary loop criteria on the text traces (statement unit > 3 repeats; periodic run > 500 tokens) ===")
print(pd.crosstab([L.finished, L.robust], [L.stmt_loop, L.num_loop]))
print("\nwaypoint-run stop reason vs robust:"); print(pd.crosstab(pd.Series([WP[i][1] for i in WP], name="v2_stop"), pd.Series([bool(y[i]) for i in WP], name="robust")))
print("\nmost repeated unit per trace:"); print(L.sort_values("reps", ascending=False).head(15).to_string(index=False))
