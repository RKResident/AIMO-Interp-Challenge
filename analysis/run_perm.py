"""Parent (agg problem) vs its decay-causing permutations (DeepSeek). Unit of inference = parent (n=10). No gpt_proxy rows."""
import pickle, json, csv, re, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr, mannwhitneyu, binomtest
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
DS = "deepseek-ai__DeepSeek-R1-0528-Qwen3-8B"
d = pickle.load(open(ROOT / f"feats_{DS}.pkl", "rb")); meta, F = d["meta"].reset_index(drop=True), d["F"]
meta["pid"] = meta.parent_id.astype(str)
agg = np.where(meta.source == "agg")[0]; perm = np.where(meta.source == "perm")[0]
y = meta.is_robust.map({True: 1, False: 0, "True": 1, "False": 0}).values
pmap = {meta.pid[i]: i for i in agg}
pairs = [(i, pmap[meta.pid[i]]) for i in perm if meta.pid[i] in pmap]                     # (variant row, parent row)
par_ids = sorted({p for _, p in pairs}, key=lambda p: meta.pid[p])
print(f"variants={len(pairs)} parents={len(par_ids)}  robust parents={int(sum(y[p] == 1 for p in par_ids))}  "
      f"variants/parent={[sum(1 for _, q in pairs if q == p) for p in par_ids]}")
rng = np.random.default_rng(0)
BH = lambda p: (lambda o, n: (lambda adj: np.minimum.accumulate(adj[::-1])[::-1])(np.sort(p)[...] * n / (np.arange(n) + 1)))(None, len(p)) if False else None


def bh(pv):
    pv = np.asarray(pv, float); o = np.argsort(pv); n = len(pv); q = np.empty(n)
    adj = pv[o] * n / (np.arange(n) + 1); q[o] = np.minimum(np.minimum.accumulate(adj[::-1])[::-1], 1); return q


def parent_mean(vals):
    """mean over each parent's variants -> array aligned with par_ids"""
    return np.array([np.mean([vals[v] for v, q in pairs if q == p]) for p in par_ids])


# ---------------------------------------------------------------------------- 1. scalar change profile
print("\n=== 1. change profile: parent-level mean of (variant - parent), sign test across parents ===")
num = [c for c in meta.columns if meta[c].dtype != object and c not in ("max_drop", "perm_idx") and meta.loc[agg, c].notna().mean() > .9
       and meta.loc[perm, c].notna().mean() > .9 and meta.loc[agg, c].std() > 0]
rows = []
for c in num:
    x = meta[c].astype(float).values
    dl = parent_mean({v: x[v] - x[p] for v, p in pairs})
    sd = np.nanstd(x[agg]); z = dl / sd
    pw = wilcoxon(dl).pvalue if np.count_nonzero(dl) > 5 else np.nan
    rows.append((c, np.mean(dl), np.mean(z), int((dl > 0).sum()), int((dl < 0).sum()), pw))
t1 = pd.DataFrame(rows, columns=["feature", "mean_delta", "delta_in_agg_sd", "n_up", "n_down", "wilcoxon_p"]).dropna(subset=["wilcoxon_p"])
t1["q_BH"] = bh(t1.wilcoxon_p.values); t1 = t1.sort_values("wilcoxon_p"); t1.to_csv(ROOT / "perm_change_profile.csv", index=False)
print(t1.head(18).round(4).to_string(index=False)); print(f"{len(t1)} features tested; {(t1.q_BH < .1).sum()} with BH q<0.1")

# ---------------------------------------------------------------------------- variant edit size
vt = {}
for x in csv.DictReader(open(ROOT.parent / "decay_causing_permutations_summary.csv")):
    L = json.loads(x["decay_casuing_permutations"]) if x["decay_casuing_permutations"] else []
    vt[x["problem id"]] = (x["original problem"], L)
# ---------------------------------------------------------------------------- 2/3. parent properties vs change
print("\n=== 2. do parent properties predict the change? (Spearman, n=10 parents; exploratory) ===")
logr = lambda c: parent_mean({v: np.log(meta[c][v] + 1) - np.log(meta[c][p] + 1) for v, p in pairs})
tg = {"d_log_n_gen": logr("n_gen"), "d_ent_all_mean": parent_mean({v: meta.ent_all_mean[v] - meta.ent_all_mean[p] for v, p in pairs}),
      "d_fa_b1024_B_ent": parent_mean({v: meta.fa_b1024_B_ent[v] - meta.fa_b1024_B_ent[p] for v, p in pairs}),
      "d_sc_f_B_modal": parent_mean({v: meta.sc_f_B_modal[v] - meta.sc_f_B_modal[p] for v, p in pairs})}
pf = {c: meta[c][par_ids].astype(float).values for c in ["n_gen", "n_prompt", "ent_all_mean", "fa_b1024_B_ent", "prob_lp_mean", "lp_all_mean"]}
for a, xa in pf.items():
    print("  parent " + a.ljust(16) + " ".join(f"{b}: rho={spearmanr(xa, xb)[0]:+.2f} (p={spearmanr(xa, xb)[1]:.2f})" for b, xb in tg.items()))
print("\n=== 3. parent label vs change (descriptive) ===")
lab = np.array([y[p] for p in par_ids]); print("  robust parents:", int((lab == 1).sum()), " non-robust:", int((lab == 0).sum()))
for b, xb in tg.items():
    print(f"  {b:18s} robust mean {xb[lab == 1].mean():+.3f} | non-robust mean {xb[lab == 0].mean():+.3f}")

# ---------------------------------------------------------------------------- 4. edit size
print("\n=== 4. edit size vs change (variant level; 29 variants, parent-clustered so p-values optimistic) ===")
ed = []
for v, p in pairs:
    orig, L = vt[meta.pid[v]]
    idx = int(meta.perm_idx[v]); txt = L[idx] if idx < len(L) else None
    ed.append((len(txt) - len(orig)) / len(orig) if txt else np.nan)
ed = np.array(ed)
for nm, dv in (("d_log_n_gen", np.array([np.log(meta.n_gen[v] + 1) - np.log(meta.n_gen[p] + 1) for v, p in pairs])),
               ("d_ent_all_mean", np.array([meta.ent_all_mean[v] - meta.ent_all_mean[p] for v, p in pairs])),
               ("d_fa_b1024_B_ent", np.array([meta.fa_b1024_B_ent[v] - meta.fa_b1024_B_ent[p] for v, p in pairs]))):
    ok = np.isfinite(ed) & np.isfinite(dv); r = spearmanr(ed[ok], dv[ok]); print(f"  rel. char growth vs {nm}: rho={r[0]:+.2f} p={r[1]:.3f} (n={ok.sum()})")

# ---------------------------------------------------------------------------- 5/6/8. representation distances vs matched null
print("\n=== 5/6/8. distance parent<->variant vs distance to matched random agg problems (z-scored, centered on agg) ===")
nprm, ngen = np.log(meta.n_prompt.astype(float).values), np.log(meta.n_gen.astype(float).values)


def reps():
    R = {}
    for tag, i in (("b1024", 0), ("b4096", 1), ("f50", 2), ("f100", 3)):
        R[f"hidden:cut_last@{tag}"] = F["cut_last"][:, i].reshape(len(meta), -1)
    R["hidden:prompt_last_L12,24"] = F["prompt_last"][:, [12, 24]].reshape(len(meta), -1)
    R["hidden:problem_mean"] = F["prompt_mean"].reshape(len(meta), -1)
    for tag, i in (("b4096", 1), ("f100", 3)):
        R[f"lap:topk_noforce@{tag}"] = F["lap_no"][:, i].reshape(len(meta), -1)
        R[f"lap:topk_force@{tag}"] = F["lap_force"][:, i].mean(1).reshape(len(meta), -1)
        R[f"lap:mean_force@{tag}"] = F["lap_mean_force"][:, i].mean(1).reshape(len(meta), -1)
    for j, tag in enumerate(("first", "last")):
        R[f"extra:lookback@{tag}"] = F["x_look"][:, j].reshape(len(meta), -1)
        R[f"extra:fft_headavg@{tag}"] = np.nanmean(F["x_fft"][:, j], 2).reshape(len(meta), -1)
        R[f"extra:kl_adj@{tag}"] = F["x_kl"][:, j].reshape(len(meta), -1)
        R[f"extra:resid@{tag}"] = F["x_res"][:, j].reshape(len(meta), -1)
        R[f"extra:mlp_out@{tag}"] = F["x_mlp"][:, j].reshape(len(meta), -1)
    return R


def dist_tests(X, name, match):
    X = np.nan_to_num(X.astype(np.float32)); mu, sd = X[agg].mean(0), X[agg].std(0) + 1e-6
    Z = (X - mu) / sd; k = Z.shape[1]
    D = lambda a, b: float(np.linalg.norm(Z[a] - Z[b]) / np.sqrt(k))
    ratio, below = [], []
    for v, p in pairs:
        dpar = D(v, p)
        cand = [a for a in agg if a != p]
        sc = np.zeros(len(cand))
        for col in match: sc += np.abs(col[cand] - col[v]) if False else np.abs(col[np.array(cand)] - col[v])
        near = np.array(cand)[np.argsort(sc)[:15]]
        nul = np.array([D(v, a) for a in near])
        ratio.append(dpar / nul.mean()); below.append(dpar < np.median(nul))
    # parent-level aggregation
    rp = parent_mean({v: r for (v, _), r in zip(pairs, ratio)}); bp = parent_mean({v: float(b) for (v, _), b in zip(pairs, below)})
    pv = wilcoxon(np.log(rp) ).pvalue
    return dict(rep=name, match="+".join(["n_prompt", "n_gen"][:len(match)]), median_dist_ratio=float(np.median(rp)), parents_closer_than_null=int((rp < 1).sum()),
                frac_variants_below_null_median=float(np.mean(below)), wilcoxon_p=float(pv))


out = []
for name, X in reps().items():
    for match in ([nprm], [nprm, ngen]):
        out.append(dist_tests(X, name, match))
t5 = pd.DataFrame(out); t5["q_BH"] = bh(t5.wilcoxon_p.values); t5.to_csv(ROOT / "perm_distance_vs_null.csv", index=False)
print("(ratio = parent->variant distance / mean distance to 15 matched random agg problems; <1 means variants stay closer to their parent)")
print(t5.round(3).to_string(index=False))

# ---------------------------------------------------------------------------- 7. forced-answer trajectory vs gold
print("\n=== 7. forced-answer accuracy vs cutoff (variant B), parent vs variants ===")
fdir = ROOT.parent / "extraction_out" / DS / "force"


def nrm(s):
    if s is None: return None
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", str(s)).strip().strip("$").replace(",", "").replace("\\!", "").replace("\\,", "").replace("\\dfrac", "\\frac")
    return re.sub(r"\s+", "", s).rstrip(".")


def same(a, g):
    a, g = nrm(a), nrm(g)
    if not a or not g: return False
    try: return abs(float(a) - float(g)) < 1e-6 * max(1, abs(float(g)))
    except Exception: return a == g


labs = ["b512", "b1024", "b2048", "b3072", "b4096", "b6144", "b8192", "f25", "f50", "f75", "f90", "f100"]


def acc_curve(i):
    j = json.load(open(fdir / f"{meta.id[i]}.json")); g = meta.gold[i]
    return np.array([same(j["results"].get(f"{j['labels'][l]}|B", {}).get("answer"), g) for l in labs], float), j["labels"], j["L"]


A = {i: acc_curve(i) for i in set(v for v, _ in pairs) | set(par_ids)}
cur = pd.DataFrame({"parent": [np.mean([A[p][0][k] for p in par_ids]) for k in range(len(labs))],
                    "variant": [np.mean([A[v][0][k] for v, _ in pairs]) for k in range(len(labs))]}, index=labs)
print(cur.round(2).T.to_string())
fc_b = lambda i: next((int(A[i][1][l]) for k, l in enumerate(labs) if A[i][0][k] == 1), np.nan)
d_first = parent_mean({v: (np.log(fc_b(v) + 1) - np.log(fc_b(p) + 1)) if np.isfinite(fc_b(v)) and np.isfinite(fc_b(p)) else np.nan for v, p in pairs})
print(f"  first cutoff (tokens) at which forced answer equals gold: parent median {np.nanmedian([fc_b(p) for p in par_ids]):.0f}, "
      f"variants median {np.nanmedian([fc_b(v) for v, _ in pairs]):.0f}; parent-level log-ratio mean {np.nanmean(d_first):+.2f} (>0 = variants need more tokens), "
      f"later in {np.nanmean(d_first > 0):.2f} of parents")
print("  final-boxed correct: parents", int(np.mean([A[p][0][-1] for p in par_ids]) * len(par_ids)), "/", len(par_ids),
      "; variants", int(sum(A[v][0][-1] for v, _ in pairs)), "/", len(pairs))
