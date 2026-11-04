"""exp37: what a 10k token cap (instead of 16k) does to our submission features on main_v2 (43 rows, 22 problems).
Greedy => a trace capped at 10,240 tokens is the first 10,240 tokens of our 16k trace, so every feature is recomputed on the truncated
trace (loop cuts happen at the same place). Unchanged by construction: forced confidence (cut at min(6144, thinking)), kNN global
efficiency (first 4096 tokens), PEARL (prompt only). Recomputed: length, finished, thinking length, 25% segment, late-vs-early cosine.
A/B shift: its fraction-of-thinking cutoffs move and those forced answers were not saved -> left at its 16k value (flagged).
Scored as exp34 with each cap's own finishing / length controls; recipes rebuilt with that cap's reference values."""
import os, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "2"
from common import *
from scipy.stats import rankdata
sys.path.insert(0, str(ROOT / "submission_core"))
from spurious_features import f_seg25_pc1, f_late_vs_early, ensemble_score

P2 = ROOT / "extraction_out_generations_p2" / "extraction_out_generations"
allm, _ = load_all(); y = allm.y.values; rows = datasets(allm)["main_v2"]; model = allm.model.values; prob = allm.base_id.values
F28 = pickle.load(open(A / "exp28_features_cache.pkl", "rb")); F32 = pickle.load(open(A / "exp32_features.pkl", "rb"))
CAPS = (16384, 10240)
V = {c: {k: np.full(len(allm), np.nan) for k in ("fa", "ab", "seg", "traj", "pearl", "ge", "len", "fin", "lgn")} for c in CAPS}
n_over = 0
for i in rows:
    fj = force_json(model[i], allm.id.iloc[i]); te = fj.get("think_end"); n_gen = int(allm.n_gen.iloc[i])
    ws = np.load(P2 / model[i] / "tf" / f"{allm.id.iloc[i]}.npz")["win_states"]
    n_over += n_gen > 10240
    for c in CAPS:
        g = min(n_gen, c); L = te if (te is not None and te < c) else min(fj["L"], c)
        finished = (allm.stop.iloc[i] == "eos") and n_gen <= c
        W = ws[: -(-g // 16)]
        V[c]["fa"][i] = F28["meta:fa_b6144_B_p1"][i]; V[c]["ab"][i] = F28["ab:dent_max"][i]
        V[c]["pearl"][i] = F28["pearl:mean_state:area"][i]; V[c]["ge"][i] = F32["step64|k5|4k|global_efficiency"][i]
        V[c]["seg"][i] = f_seg25_pc1(W, L); V[c]["traj"][i] = f_late_vs_early(W)
        V[c]["len"][i] = -np.log(g); V[c]["fin"][i] = float(finished); V[c]["lgn"][i] = np.log(g)
# sanity: the 16k recomputation must equal the saved values
for k, saved in (("seg", F28["subsample:seg25:f11"]), ("traj", F28["traj:k1|full:late_vs_early_cos"])):
    print(f"16k recompute of {k} matches saved: max |diff| {np.nanmax(np.abs(V[16384][k][rows] - saved[rows])):.1e}")

up = np.unique(prob[rows]); yp = np.array([y[rows][prob[rows] == p][0] for p in up]).astype(int)
rng = np.random.default_rng(0)


def auc(x, yy):
    rk = rankdata(x); n1 = yy.sum(); n0 = len(yy) - n1
    return (rk[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def perm_p(x, yy, n=5000):
    a = auc(x, yy); rk = rankdata(x); n1 = yy.sum(); n0 = len(yy) - n1
    YP = rng.permuted(np.tile(yy, (n, 1)), axis=1).astype(float); nl = (YP @ rk - n1 * (n1 + 1) / 2) / (n1 * n0)
    return a, float(np.mean(np.abs(nl - .5) >= abs(a - .5)))


def zmodel(v):
    out = np.full(len(allm), np.nan)
    for m in np.unique(model[rows]):
        r = rows[model[rows] == m]; x = v[r]
        if np.isfinite(x).sum() >= 2 and np.nanstd(x) > 0: out[r] = (x - np.nanmean(x)) / np.nanstd(x)
    return out


def score(v, c):
    z = zmodel(v); r = rows[np.isfinite(z[rows])]
    xp = np.array([np.nanmean(z[rows][prob[rows] == p]) if np.isfinite(z[rows][prob[rows] == p]).any() else np.nan for p in up]); ok = np.isfinite(xp)
    fp = np.array([V[c]["fin"][rows][prob[rows] == p].mean() for p in up]); lp = np.array([V[c]["lgn"][rows][prob[rows] == p].mean() for p in up])
    Cm = np.c_[np.ones(ok.sum()), fp[ok], lp[ok]]; xr = xp[ok] - Cm @ np.linalg.lstsq(Cm, xp[ok], rcond=None)[0]
    a_c, p_c = perm_p(xr, yp[ok]); a_p, p_p = perm_p(xp[ok], yp[ok])
    return dict(pooled=roc_auc_score(y[r], z[r]), problem=a_p, controlled=a_c, p_controlled=p_c)


print(f"\n{n_over}/43 main_v2 traces are longer than 10,240 tokens; finished within the cap: "
      f"16k {int(V[16384]['fin'][rows].sum())}/43, 10k {int(V[10240]['fin'][rows].sum())}/43")
R = []
for k in ("fin", "len", "fa", "ge", "pearl", "seg", "traj", "ab"):
    for c in CAPS:
        if k == "fin":
            s = dict(pooled=roc_auc_score(y[rows], V[c]["fin"][rows]), problem=perm_p(np.array([V[c]["fin"][rows][prob[rows] == p].mean() for p in up]), yp)[0], controlled=np.nan, p_controlled=np.nan)
        else:
            s = score(V[c][k], c)
        R.append(dict(feature=k, cap=c, **s))
T = pd.DataFrame(R)
pd.set_option("display.width", 200)
print("\nAUC > 0.5: higher value -> robust ('controlled' uses that cap's own finished fraction / log length):")
print(T.pivot_table(index="feature", columns="cap", values=["pooled", "problem", "controlled", "p_controlled"]).round(3).to_string())

# recipes rebuilt at each cap (reference values from that cap), median cut, in-sample balanced accuracy on main_v2
RECIPES = {"forced-confidence": ["fa"], "confidence-knn": ["fa", "ge"], "top5-knn": ["fa", "ab", "seg", "traj", "pearl", "ge"],
           "no-forcing": ["seg", "traj", "pearl", "ge"], "length": ["len"]}
SIGN = {"fa": 1, "ab": 1, "seg": 1, "traj": 1, "pearl": 1, "ge": 1, "len": 1}
print("\nrecipes, median cut, in-sample on main_v2 (optimistic; compare the two caps, not the absolute values):")
for name, feats in RECIPES.items():
    out = []
    for c in CAPS:
        rec = dict(signs=SIGN, ref={f: {HF[m]: sorted(float(x) for x in V[c][f][rows][model[rows] == m] if np.isfinite(x)) for m in MODELS} for f in feats})
        s = np.array([ensemble_score({f: V[c][f][i] for f in feats}, rec, HF[model[i]]) for i in rows]); t = np.quantile(s, .5)
        yy = y[rows] == 1; ba = 0.5 * ((s[yy] > t).mean() + (s[~yy] <= t).mean())
        out.append(f"{c // 1024}k: AUC {roc_auc_score(yy, s):.2f}, balanced acc {ba:.2f}")
    print(f"  {name:18s} " + " | ".join(out))
