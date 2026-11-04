"""exp50: (a) is exp49's kNN instability an artefact of centring the sampled traces on the GREEDY per-model mean? kNN global efficiency is
recomputed for every trace with the per-model mean of that trace type's own states (greedy mean for greedy, sample mean for samples) and
with no centring at all; (b) internal answer-state features from the hidden states at the forced-answer suffix (6 layers, every cutoff):
  - answer-state stability across budgets: cosine of the suffix state at consecutive budgets (2048 -> 3072 -> 4096 -> 6144), per layer
  - answer-state agreement across traces: cosine of greedy vs sample suffix states @6144 (an internal version of self-consistency)
  - reasoning-state agreement across traces: cosine of greedy vs sample mean-of-last-64 states at the last cutoff
(c) cross-dataset probes: a logistic probe on the forced-suffix state @6144 (or the last-cutoff state) trained on the friend's olympiad
rows that are NOT in main_v2, tested on main_v2 (same model: DeepSeek, Skywork) - an out-of-dataset test of a white-box probe."""
import json, os, pickle, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "2"
import numpy as np
from pathlib import Path
sys.path.insert(0, "/home/sabyasachi19/PycharmProjects/AimoAGAIN/submission_core")
from spurious_features import f_global_eff

ROOT = Path("/home/sabyasachi19/PycharmProjects/AimoAGAIN"); OUT = ROOT / "results_mainv2_t1" / "out_t1"
OLY = ROOT / "extra_olympiad_problems_dataset" / "results_olympiad_features" / "out_t1"
rows = [json.loads(l) for f in sorted(OUT.glob("*/rows_shard*.jsonl")) for l in open(f)]
N = len(rows); y = np.array([r["robust"] for r in rows]).astype(int); model = np.array([r["model_id"] for r in rows])
slug = lambda m: m.replace("/", "__")                                                    # noqa: E731
Z = [np.load(OUT / slug(r["model_id"]) / f"{r['id']}.npz") for r in rows]
TR = ["t0"] + [f"t1s{s}" for s in range(2)]
rng = np.random.default_rng(0)


def within(x, yy, mm):
    num = den = 0
    for m in np.unique(mm):
        k = (mm == m) & np.isfinite(x); p, n = x[k & (yy == 1)], x[k & (yy == 0)]
        num += (p[:, None] > n[None]).sum() + .5 * (p[:, None] == n[None]).sum(); den += len(p) * len(n)
    return num / den if den else np.nan


def perm(x, yy=y, mm=model, n=2000):
    w = within(x, yy, mm); nl = np.empty(n)
    for t in range(n):
        yp = yy.copy()
        for m in np.unique(mm): k = mm == m; yp[k] = rng.permutation(yy[k])
        nl[t] = within(x, yp, mm)
    return w, float(np.mean(np.abs(nl - .5) >= abs(w - .5)))


def cos(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# (a) kNN global efficiency with each trace type's own centring
L = {(i, t): (r["greedy"] if t == "t0" else r["samples"][int(t[-1])])["L"] for i, r in enumerate(rows) for t in TR}
MU = {}
for t in TR:
    for m in np.unique(model):
        MU[(t, m)] = np.concatenate([Z[i][f"{t}_ws16"][:, 3].astype(np.float32) for i in range(N) if model[i] == m]).mean(0)
MU_ALL = {m: np.mean([MU[(t, m)] for t in TR], 0) for m in np.unique(model)}
print("(a) kNN global efficiency, within-model AUC per trace (greedy / sample 1 / sample 2)")
for nm, mu_of in (("centred on the greedy mean (as the submission)", lambda t, m: MU[("t0", m)]),
                  ("centred on its own trace type's mean", lambda t, m: MU[(t, m)]),
                  ("centred on the mean of all 3 trace types", lambda t, m: MU_ALL[m]),
                  ("no centring", lambda t, m: np.zeros_like(MU[("t0", m)]))):
    V = np.array([[f_global_eff(Z[i][f"{t}_ws16"][:, 3].astype(np.float32), mu_of(t, model[i]), L[(i, t)]) for t in TR] for i in range(N)])
    ok = np.isfinite(V).all(1); rho = np.corrcoef(np.argsort(np.argsort(V[ok], 0), 0).T)
    print(f"  {nm:48s} " + " / ".join(f"{within(V[:, j], y, model):.2f}" for j in range(3))
          + f" | rank correlation greedy-sample1 {rho[0, 1]:.2f}, greedy-sample2 {rho[0, 2]:.2f}, sample1-sample2 {rho[1, 2]:.2f}")

# (b) internal answer-state features
F = {}


def put(k, i, v):
    F.setdefault(k, np.full(N, np.nan))[i] = v


for i, r in enumerate(rows):
    z = Z[i]; suf = list(z["t0_force_suffixes"]); jB = suf.index("B")
    for t in TR:
        rec = r["greedy"] if t == "t0" else r["samples"][int(t[-1])]
        ce = list(z[f"{t}_cut_eff"]); fh = z[f"{t}_force_hidden"].astype(np.float32)          # [n_cut, 2, 6, d]
        cuts = [rec["cuts"][f"b{b}"] for b in (2048, 3072, 4096, 6144)]
        st = [np.mean([cos(fh[ce.index(a), jB, l], fh[ce.index(b), jB, l]) for l in range(fh.shape[2])]) for a, b in zip(cuts, cuts[1:])]
        put(f"answer-state stability across budgets ({'greedy' if t == 't0' else 'sample ' + t[-1]})", i, np.mean(st))
        put(f"answer-state A vs B suffix agreement @6144 ({'greedy' if t == 't0' else 'sample ' + t[-1]})", i,
            np.mean([cos(fh[ce.index(cuts[-1]), 0, l], fh[ce.index(cuts[-1]), 1, l]) for l in range(fh.shape[2])]))
    g = z["t0_force_hidden"].astype(np.float32)[list(z["t0_cut_eff"]).index(r["greedy"]["cuts"]["b6144"]), jB]
    sa = [z[f"t1s{s}_force_hidden"].astype(np.float32)[list(z[f"t1s{s}_cut_eff"]).index(r["samples"][s]["cuts"]["b6144"]), jB] for s in range(2)]
    put("answer-state agreement greedy vs samples @6144 (mean over layers)", i, np.mean([cos(g[l], s_[l]) for s_ in sa for l in range(g.shape[0])]))
    put("answer-state agreement, last layer only", i, np.mean([cos(g[-1], s_[-1]) for s_ in sa]))
    put("answer-state agreement, sample 1 vs sample 2", i, np.mean([cos(sa[0][l], sa[1][l]) for l in range(g.shape[0])]))
    cg = z["t0_cut_mean64"].astype(np.float32)[-1]; cs_ = [z[f"t1s{s}_cut_mean64"].astype(np.float32)[-1] for s in range(2)]
    put("reasoning-state agreement greedy vs samples (last cutoff)", i, np.mean([cos(cg[l], c[l]) for c in cs_ for l in range(cg.shape[0])]))
print("\n(b) internal answer-state features: within-model AUC (p)")
for k, v in F.items():
    w, p = perm(v); print(f"  {k:66s} {w:.2f} ({p:.3f}){'  *' if p < .05 else ''}")

# (c) olympiad -> main_v2 probes (same model), trained only on the olympiad rows independent of main_v2
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
D47 = pickle.load(open(ROOT / "analysis" / "exp47_features.pkl", "rb")); oid, oind, oy, om = D47["ids"], D47["indep"], D47["y"], D47["model"]
orow = {json.loads(l)["id"]: json.loads(l) for f in sorted(OLY.glob("*/rows_shard*.jsonl")) for l in open(f)}


def state(z, rec, kind, pre="t0"):
    if kind == "forced-suffix state B @6144":
        return z[f"{pre}_force_hidden"].astype(np.float32)[list(z[f"{pre}_cut_eff"]).index(rec["cuts"]["b6144"]), list(z[f"{pre}_force_suffixes"]).index("B")].ravel()
    if kind == "last-cutoff state (mean of last 64)":
        return z[f"{pre}_cut_mean64"].astype(np.float32)[-1].ravel()
    return z[f"{pre}_ws16"].astype(np.float32).mean(0).ravel()


print("\n(c) probe trained on olympiad (independent rows), tested on main_v2, same model: AUC on main_v2 (n)")
for kind in ("forced-suffix state B @6144", "last-cutoff state (mean of last 64)", "reasoning mean state (6 layers)"):
    res = []
    for m in ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "Skywork/Skywork-OR1-Math-7B"):
        tr = [k for k in range(len(oid)) if om[k] == m and oind[k]]
        Xtr = np.stack([state(np.load(OLY / slug(m) / f"{oid[k]}.npz"), orow[oid[k]]["greedy"], kind) for k in tr]); ytr = oy[tr]
        te = [i for i in range(N) if model[i] == m]
        Xte = np.stack([state(Z[i], rows[i]["greedy"], kind) for i in te]); yte = y[te]
        mu_, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; pca = PCA(5).fit((Xtr - mu_) / sd)
        clf = LogisticRegression(C=0.1, max_iter=2000).fit(pca.transform((Xtr - mu_) / sd), ytr)
        s = clf.decision_function(pca.transform((Xte - mu_) / sd)); res.append((m.split("/")[0][:8], within(s, yte, np.array([m] * len(te))), len(te), int(yte.sum())))
    print(f"  {kind:40s} " + " | ".join(f"{a}: {b:.2f} (n {c}, {d} robust)" for a, b, c, d in res))

# (c2) permutation p for the cross-dataset probes, pooled within model over DeepSeek + Skywork main_v2 rows (labels permuted inside each model)
print("\n(c2) cross-dataset probes, pooled within model on main_v2 (22 rows, 16 robust): AUC (p)")
for kind in ("forced-suffix state B @6144", "last-cutoff state (mean of last 64)"):
    s_all = np.full(N, np.nan)
    for m in ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "Skywork/Skywork-OR1-Math-7B"):
        tr = [k for k in range(len(oid)) if om[k] == m and oind[k]]
        Xtr = np.stack([state(np.load(OLY / slug(m) / f"{oid[k]}.npz"), orow[oid[k]]["greedy"], kind) for k in tr]); ytr = oy[tr]
        te = [i for i in range(N) if model[i] == m]; Xte = np.stack([state(Z[i], rows[i]["greedy"], kind) for i in te])
        mu_, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; pca = PCA(5).fit((Xtr - mu_) / sd)
        clf = LogisticRegression(C=0.1, max_iter=2000).fit(pca.transform((Xtr - mu_) / sd), ytr); s_all[te] = clf.decision_function(pca.transform((Xte - mu_) / sd))
    k = np.isfinite(s_all); w, p = perm(s_all[k], y[k], model[k], 5000); print(f"  {kind:40s} {w:.2f} (p {p:.3f})")
