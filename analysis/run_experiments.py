"""Experiments on the pooled features. Positive class = robust. See README printed at the end."""
import pickle, sys, warnings, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
DS = "deepseek-ai__DeepSeek-R1-0528-Qwen3-8B"
rng = np.random.default_rng(0)


def load(slug):
    d = pickle.load(open(ROOT / f"feats_{slug}.pkl", "rb"))
    return d["meta"].reset_index(drop=True), d["F"]


meta, F = load(DS)
agg = np.where(meta.source == "agg")[0]
main = np.where(meta.source == "main_v2")[0]
prox = np.where(meta.source == "gpt_proxy")[0]
perm = np.where(meta.source == "perm")[0]
y = meta.is_robust.map({True: 1, False: 0, "True": 1, "False": 0}).values.astype(float)
assert meta.loc[agg, "base_id"].is_unique


# ----------------------------------------------------------------------------------------- features
def cols(prefixes=(), contains=(), exclude=()):
    out = [c for c in meta.columns if (any(c.startswith(p) for p in prefixes) or any(s in c for s in contains)) and c not in exclude]
    return [c for c in out if meta[c].dtype != object]


def scal(cs):
    X = meta[cs].astype(float).values
    return X


def log_(c):
    return np.log(meta[c].astype(float).values + 1)[:, None]


REFI = {"b1024": 0, "b4096": 1, "f50": 2, "f100": 3}
FAM = {}
FAM["ctrl:prompt_len"] = (log_("n_prompt"), "low")
FAM["ctrl:trace_len"] = (np.hstack([log_("n_gen"), log_("L_think")]), "low")
FAM["ctrl:len_both"] = (np.hstack([log_("n_prompt"), log_("n_gen"), log_("L_think")]), "low")
FAM["prompt:problem_logprob"] = (scal(["prob_lp_mean", "prob_lp_min", "prob_ent_mean"]), "low")
FAM["prompt:last_tok_L24"] = (F["prompt_last"][:, 24], "high")
FAM["prompt:last_tok_L12"] = (F["prompt_last"][:, 12], "high")
FAM["prompt:problem_mean_hid"] = (F["prompt_mean"].reshape(len(meta), -1), "high")
tcols = cols(prefixes=("lp_", "ent_", "top1_"))
FAM["traj:scalars"] = (scal(tcols), "low")
FAM["traj:scalars+len"] = (np.hstack([scal(tcols), log_("n_gen")]), "low")
fc = cols(prefixes=("fa_",), contains=("_ent", "_mlp", "_p1"))
fc = [c for c in fc if c.startswith("fa_") and not c.endswith("closed")]
FAM["forced:confidence_all"] = (scal(fc), "low")
for lab in ("b1024", "b4096", "f50", "f100"):
    FAM[f"forced:confidence@{lab}"] = (scal([c for c in fc if f"_{lab}_" in c]), "low")
sc = cols(prefixes=("sc_", "agree_"))
FAM["forced:consistency"] = (scal(sc), "low")
FAM["forced:all"] = (scal(fc + sc), "low")
FAM["forced:all+len"] = (np.hstack([scal(fc + sc), log_("n_gen")]), "low")
for lab, i in REFI.items():
    FAM[f"hidden:cut_last@{lab}"] = (F["cut_last"][:, i].reshape(len(meta), -1), "high")
    FAM[f"lap:noforce@{lab}"] = (F["lap_no"][:, i].reshape(len(meta), -1), "high")
    FAM[f"lap:force@{lab}"] = (F["lap_force"][:, i].mean(1).reshape(len(meta), -1), "high")
    FAM[f"lap:mean_noforce@{lab}"] = (F["lap_mean_no"][:, i].reshape(len(meta), -1), "high")
    FAM[f"lap:mean_force@{lab}"] = (F["lap_mean_force"][:, i].mean(1).reshape(len(meta), -1), "high")
if "x_fft" in F:
    for j, tag in enumerate(("first", "last")):
        FAM[f"extra:fft@{tag}"] = (F["x_fft"][:, j].reshape(len(meta), -1), "high")
        FAM[f"extra:fft_headavg@{tag}"] = (np.nanmean(F["x_fft"][:, j], 2).reshape(len(meta), -1), "high")
        FAM[f"extra:lookback@{tag}"] = (F["x_look"][:, j].reshape(len(meta), -1), "high")
        FAM[f"extra:kl_adj@{tag}"] = (F["x_kl"][:, j].reshape(len(meta), -1), "high")
        FAM[f"extra:kl_headmean@{tag}"] = (F["x_klhm"][:, j].reshape(len(meta), -1), "high")
        FAM[f"extra:resid@{tag}"] = (F["x_res"][:, j].reshape(len(meta), -1), "high")
        FAM[f"extra:attn_out@{tag}"] = (F["x_attn"][:, j].reshape(len(meta), -1), "high")
        FAM[f"extra:mlp_out@{tag}"] = (F["x_mlp"][:, j].reshape(len(meta), -1), "high")
for k, (X, kind) in list(FAM.items()):
    FAM[k] = (np.asarray(X, dtype=np.float32), kind)


def src_standardize(X, sources):
    """z-score each column within its dataset (kills dataset-identity shifts). NaN -> 0 after scaling."""
    X = X.copy()
    for s in np.unique(sources):
        m = sources == s
        if m.sum() < 3:
            continue
        mu, sd = np.nanmean(X[m], 0), np.nanstd(X[m], 0)
        X[m] = (X[m] - mu) / np.where(sd > 1e-8, sd, 1.0)
    return np.nan_to_num(X)


def make_model(kind, k=8):
    if kind == "high":
        return make_pipeline(StandardScaler(), PCA(n_components=k, random_state=0), StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))


def auc_ci(yt, s, n=1000):
    if len(np.unique(yt)) < 2:
        return np.nan, np.nan, np.nan
    a = roc_auc_score(yt, s)
    bs = []
    for _ in range(n):
        i = rng.integers(0, len(yt), len(yt))
        if len(np.unique(yt[i])) == 2:
            bs.append(roc_auc_score(yt[i], s[i]))
    return a, *np.percentile(bs, [2.5, 97.5])


def cv_oof(X, kind, idx, extra_idx=(), reps=5, seed=0):
    """Repeated stratified 5-fold on `idx`; returns mean OOF score per sample of idx, and mean score for rows in extra_idx
    (each predicted only by folds that did NOT train on their parent - caller passes a parent->fold rule via extra_parent)."""
    rs = RepeatedStratifiedKFold(n_splits=5, n_repeats=reps, random_state=seed)
    yy = y[idx]
    oof, cnt = np.zeros(len(idx)), np.zeros(len(idx))
    folds = []
    for tr, te in rs.split(idx, yy):
        m = make_model(kind).fit(X[idx[tr]], yy[tr])
        oof[te] += m.decision_function(X[idx[te]])
        cnt[te] += 1
        folds.append((tr, te, m))
    return oof / cnt, folds


results = []


def record(**kw):
    results.append(kw)


print("=" * 100)
print(f"DeepSeek rows: agg={len(agg)} (robust {int(np.nansum(y[agg]))}), main_v2={len(main)} (robust {int(np.nansum(y[main]))}), "
      f"gpt_proxy={len(prox)}, perm={len(perm)}")
print("=" * 100)

# ----------------------------------------------------------------------------------------- 2/3: agg CV + gate
src = meta.source.values
oof_store, model_store = {}, {}
for name, (X, kind) in FAM.items():
    Xs = src_standardize(X, src)
    oof, folds = cv_oof(Xs, kind, agg)
    a, lo, hi = auc_ci(y[agg], oof)
    fa = np.mean([roc_auc_score(y[agg][te], m.decision_function(Xs[agg[te]])) for tr, te, m in folds])
    oof_store[name] = oof
    # gate: fit on all agg, test on main_v2 (+ proxies); (a) with per-source z-scoring, (b) raw agg-fit scaler only
    gate = {}
    for tag, XX in (("src", Xs), ("raw", np.nan_to_num(X))):
        m = make_model(kind).fit(XX[agg], y[agg])
        s_main = m.decision_function(XX[main])
        gate[f"gate_main_{tag}"] = roc_auc_score(y[main], s_main) if len(np.unique(y[main])) == 2 else np.nan
        mp = np.concatenate([main, prox])
        gate[f"gate_main+proxy_{tag}"] = roc_auc_score(y[mp], m.decision_function(XX[mp]))
    record(family=name, dim=X.shape[1], agg_cv_auc=a, ci_lo=lo, ci_hi=hi, mean_fold_auc=fa, **gate)
    print(f"{name:34s} d={X.shape[1]:6d} agg-CV AUC {a:.3f} [{lo:.3f},{hi:.3f}]  gate(main) src={gate['gate_main_src']:.2f} raw={gate['gate_main_raw']:.2f}  "
          f"gate(main+proxy) src={gate['gate_main+proxy_src']:.2f} raw={gate['gate_main+proxy_raw']:.2f}", flush=True)

res = pd.DataFrame(results).sort_values("agg_cv_auc", ascending=False)
res.to_csv(ROOT / "results_deepseek.csv", index=False)

# ----------------------------------------------------------------------------------------- single-feature AUC curves vs cutoff
print("\n--- single-feature AUC (agg) of forced-answer first-token entropy / mean-logprob vs cutoff (variant A,B) ---")
curve = []
for lab in ["b512", "b1024", "b2048", "b3072", "b4096", "b6144", "b8192", "f25", "f50", "f75", "f90", "f100"]:
    row = {"cutoff": lab}
    for v in "AB":
        for q in ("ent", "mlp", "p1"):
            c = f"fa_{lab}_{v}_{q}"
            if c in meta:
                x = meta.loc[agg, c].astype(float).values
                ok = np.isfinite(x)
                row[f"{q}_{v}"] = roc_auc_score(y[agg][ok], x[ok]) if ok.sum() > 10 and len(np.unique(y[agg][ok])) == 2 else np.nan
    curve.append(row)
cv = pd.DataFrame(curve)
print(cv.round(3).to_string(index=False))
cv.to_csv(ROOT / "curve_forced_single_feature.csv", index=False)

# ----------------------------------------------------------------------------------------- controls
print("\n--- single-column AUC of scalar controls / top scalar features on agg (sign-free: max(AUC, 1-AUC)) ---")
sing = []
for c in cols(prefixes=("lp_", "ent_", "top1_", "prob_", "n_", "L_", "sc_", "agree_", "finished")):
    x = meta.loc[agg, c].astype(float).values
    ok = np.isfinite(x)
    if ok.sum() > 30 and len(np.unique(x[ok])) > 1 and len(np.unique(y[agg][ok])) == 2:
        a = roc_auc_score(y[agg][ok], x[ok])
        sing.append((c, a))
sing = sorted(sing, key=lambda t: -abs(t[1] - .5))
for c, a in sing[:15]:
    print(f"  {c:30s} AUC {a:.3f}")

# ----------------------------------------------------------------------------------------- 4: perm paired analysis
print("\n--- decay-causing permutations: does the agg-trained score drop for variants vs their parent? ---")
par = meta.loc[agg, "parent_id"].astype(str).values
perm_par = meta.loc[perm, "parent_id"].astype(str).values
have = np.array([p in set(par) for p in perm_par])
print(f"perm variants={len(perm)}; with parent in agg={int(have.sum())}; distinct parents={len(set(perm_par[have]))}; "
      f"decay_causing rows={int((meta.loc[perm,'decay_causing'].astype(str)=='True').sum())}")
perm_rows = []
for name in ["forced:all", "forced:confidence_all", "traj:scalars", "ctrl:trace_len", "lap:force@f100", "lap:mean_force@f100", "hidden:cut_last@f50",
             "extra:lookback@last", "extra:fft_headavg@last", "extra:resid@last"]:
    if name not in FAM:
        continue
    X, kind = FAM[name]
    Xs = src_standardize(X, src)
    rs = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1)
    ps, pc = np.zeros(len(perm)), np.zeros(len(perm))
    parent_oof, parent_c = np.zeros(len(agg)), np.zeros(len(agg))
    for tr, te in rs.split(agg, y[agg]):
        m = make_model(kind).fit(Xs[agg[tr]], y[agg][tr])
        test_par = set(par[te])
        sel = np.array([p in test_par for p in perm_par])
        if sel.any():
            ps[sel] += m.decision_function(Xs[perm[sel]])
            pc[sel] += 1
        parent_oof[te] += m.decision_function(Xs[agg[te]])
        parent_c[te] += 1
    ps = np.where(pc > 0, ps / np.maximum(pc, 1), np.nan)
    parent_s = dict(zip(par, parent_oof / parent_c))
    d = np.array([ps[i] - parent_s[perm_par[i]] for i in range(len(perm)) if have[i] and pc[i] > 0])
    pp = pd.DataFrame({"par": perm_par[have & (pc > 0)], "d": d}).groupby("par").d.mean()
    w = wilcoxon(pp.values) if len(pp) > 5 else None
    perm_rows.append((name, d.mean(), (d < 0).mean(), pp.mean(), (pp < 0).mean(), None if w is None else w.pvalue))
    print(f"  {name:28s} mean(score_variant - score_parent)={d.mean():+.3f}  frac variants lower={np.mean(d<0):.2f}  "
          f"parent-level mean={pp.mean():+.3f} lower in {np.mean(pp<0):.2f} of {len(pp)} parents  Wilcoxon p={'n/a' if w is None else round(w.pvalue,3)}")
print("  raw paired differences (variant - parent), parent-level means:")
pm = meta.loc[agg].set_index(meta.loc[agg, "parent_id"].astype(str))
for c in ["n_gen", "ent_all_mean", "lp_all_mean", "fa_f100_B_ent", "fa_b1024_B_ent", "sc_f_B_modal", "finished"]:
    v = meta.loc[perm[have], ["parent_id", c]].assign(parent_id=lambda t: t.parent_id.astype(str)).groupby("parent_id")[c].mean()
    base = pm.loc[v.index, c].groupby(level=0).first()
    dd = (v - base).dropna()
    print(f"    {c:18s} mean diff {dd.mean():+.4f}  lower in {np.mean(dd<0):.2f} of {len(dd)} parents")

# ----------------------------------------------------------------------------------------- cross-model scalar gate
print("\n--- cross-model, scalar features only (z-scored within model x dataset) ---")
others = {}
for sl in ["Qwen__Qwen3.5-4B", "Skywork__Skywork-OR1-Math-7B", "allenai__Olmo-3-7B-Think"]:
    try:
        others[sl] = load(sl)[0]
    except Exception as e:
        print("  missing", sl, e)
allm = pd.concat([meta.assign(model=DS)] + [m.assign(model=s) for s, m in others.items()], ignore_index=True)
allm["y"] = allm.is_robust.map({True: 1, False: 0, "True": 1, "False": 0}).astype(float)
common = [c for c in tcols + fc + sc if c in allm.columns and allm[c].notna().mean() > 0.8]
Xa = allm[common].astype(float).values
Xa = src_standardize(Xa, (allm.model + allm.source).values)
lab = allm[allm.y.notna()].index.values
mainm = allm[(allm.source == "main_v2") & allm.y.notna()].index.values
print(f"  common scalar features={len(common)}; labeled main_v2 rows per model: {allm.loc[mainm].groupby('model').size().to_dict()}")
aggm = allm[allm.source == "agg"].index.values
m = make_model("low").fit(Xa[aggm], allm.y[aggm])
for sl in allm.model.unique():
    t = mainm[allm.model[mainm].values == sl]
    if len(np.unique(allm.y[t])) == 2:
        print(f"  train DeepSeek-agg -> test {sl.split('__')[-1]:24s} main_v2 (n={len(t)}, robust={int(allm.y[t].sum())}) AUC {roc_auc_score(allm.y[t], m.decision_function(Xa[t])):.2f}")
tr_pool = np.concatenate([aggm, mainm])
for sl in allm.model.unique():
    t = mainm[allm.model[mainm].values == sl]
    tr = np.array([i for i in mainm if allm.model[i] != sl])
    if len(np.unique(allm.y[t])) == 2 and len(np.unique(allm.y[tr])) == 2:
        mm = make_model("low").fit(Xa[tr], allm.y[tr])
        print(f"  leave-one-model-out (train: other models' main_v2) -> {sl.split('__')[-1]:24s} AUC {roc_auc_score(allm.y[t], mm.decision_function(Xa[t])):.2f}")
t_all = mainm
oo = np.zeros(len(t_all))
for j, i in enumerate(t_all):
    tr = np.array([k for k in t_all if allm.model[k] != allm.model[i]])
    if len(np.unique(allm.y[tr])) == 2:
        oo[j] = make_model("low").fit(Xa[tr], allm.y[tr]).decision_function(Xa[[i]])[0]
print(f"  pooled LOMO main_v2 AUC (n={len(t_all)}): {auc_ci(allm.y[t_all].values, oo)}")

print("\nTop families by agg-CV AUC:")
print(res.head(15)[["family", "dim", "agg_cv_auc", "ci_lo", "ci_hi", "gate_main_src", "gate_main+proxy_src"]].round(3).to_string(index=False))
