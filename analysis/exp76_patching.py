"""exp76: analyse the Vast patching run (results_patching.tar.gz -> ../patching_results/).
A. Direct test (olympiad, DeepSeek + Skywork, 75 labelled problems x original + 5 perturbations, friend's greedy traces): force an answer at
   2048/4096/6144 (or at the end of thinking if earlier) and call the problem robust iff the perturbed answers agree with the original's.
   No gold needed. Labels: robust = every perturbation still correct (friend's 16k greedy runs).
B. Patching on our 118 labelled traces (main_v2 + olympiad; all four models), at the 6144 cut or end of thinking:
   knockout of the problem / trace / whole prompt from the answer-suffix queries; per-layer last-position patching from 3 auto rewordings
   (same trace) and from the no-reasoning run. Features scored by AUC against robust, unfinished rows first (finished -> robust anyway)."""
import glob, json
import numpy as np
import pandas as pd

R = "/home/sabyasachi19/PycharmProjects/AimoAGAIN/patching_results/out_patching/"
PT = ["distract", "domain", "rename", "rephrase", "typos"]
rng = np.random.default_rng(0)


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int); ok = np.isfinite(s); s, y = s[ok], y[ok]
    p, n = s[y == 1], s[y == 0]
    if not len(p) or not len(n): return np.nan
    return float(((p[:, None] > n[None]) + .5 * (p[:, None] == n[None])).mean())


def perm_p(s, y, n=2000):
    """two-sided permutation p for |AUC - 0.5|"""
    a = abs(auc(s, y) - .5); y = np.asarray(y)
    return float(np.mean([abs(auc(s, rng.permutation(y)) - .5) >= a - 1e-12 for _ in range(n)]))


# ------------------------------------------------------------------------------------------------------------------- A direct test
print("=" * 110 + "\nA. DIRECT TEST (forced answers on the perturbed problems vs the original's)\n")
for f in sorted(glob.glob(R + "*/direct.jsonl")):
    m = f.split("/")[-2].split("__")[1][:8]
    recs = [json.loads(l) for l in open(f)]; by = {}
    for r in recs: by.setdefault(r["pid"], {})[r["ptype"]] = r
    probs = [v for v in by.values() if "original" in v and all(p in v for p in PT)]
    y = np.array([v["original"]["label"] == "robust" for v in probs], int)
    unf = np.array([v["original"]["think_end"] is None or v["original"]["think_end"] > 6144 for v in probs])
    print(f"{m}: {len(probs)} problems, {y.sum()} robust | original still thinking at 6144: {unf.sum()} ({y[unf].sum()} robust)")
    for cut in ("2048", "4096", "6144"):
        share = np.array([np.mean([v[p]["forced"][cut]["answer"] == v["original"]["forced"][cut]["answer"] for p in PT]) for v in probs])
        # confidence-weighted: perturbed answer agrees AND both confident
        cw = np.array([np.mean([(v[p]["forced"][cut]["answer"] == v["original"]["forced"][cut]["answer"]) * v[p]["forced"][cut]["p1"] for p in PT]) for v in probs])
        # oracle: forced answers correct vs gold (not available in a submission; shows the ceiling of forcing at this cut)
        orac = np.array([np.mean([v[p]["forced"][cut]["answer"] == str(v["original"]["final_answer"]) if v["original"]["correct"] else 0 for p in PT]) for v in probs])
        allag = share == 1
        line = f"   cut {cut}: all agree -> robust: accuracy {(allag == y).mean():.2f} (calls {allag.mean():.0%} robust) | AUC agree share {auc(share, y):.2f} " \
               f"| conf-weighted {auc(cw, y):.2f}"
        if unf.sum() >= 4 and 0 < y[unf].sum() < unf.sum():
            line += f" || unfinished only: acc {(allag[unf] == y[unf]).mean():.2f}, AUC {auc(share[unf], y[unf]):.2f} (p {perm_p(share[unf], y[unf]):.2f}), " \
                    f"conf-w {auc(cw[unf], y[unf]):.2f}"
        print(line)
        if cut == "6144":
            for t in PT:
                s1 = np.array([v[t]["forced"][cut]["answer"] == v["original"]["forced"][cut]["answer"] for v in probs], float)
                print(f"      single type {t:9s}: AUC {auc(s1, y):.2f} | unfinished {auc(s1[unf], y[unf]):.2f}")
            print(f"      (oracle: share of perturbed forced answers equal to the gold answer: AUC {auc(orac, y):.2f}, unfinished {auc(orac[unf], y[unf]):.2f})")
    print()

# ------------------------------------------------------------------------------------------------------------------- B patching
print("=" * 110 + "\nB. KNOCKOUT + PATCHING (our 118 labelled rows)\n")
rows = []
for f in sorted(glob.glob(R + "*/patch.jsonl")):
    m = f.split("/")[-2].split("__")[1][:8]
    for l in open(f):
        r = json.loads(l); p0 = max(r["clean_p"], 1e-9); L = r["n_layers"]; d = dict(id=r["id"], model=m, dataset=r["dataset"], y=int(r["robust"]),
                                                                                    unf=r["think_end"] is None, clean_p=r["clean_p"])
        for k, v in r["knockout"].items():
            d[f"ko_{k}"] = np.log(p0) - np.log(max(v["p_clean"], 1e-9)); d[f"koflip_{k}"] = float(v["flip"])
        rw = [r["patch"][k] for k in r["patch"] if k.startswith("reword")]
        d["rw_src"] = np.mean([np.log(max(v["source_p_clean"], 1e-9)) for v in rw]) - np.log(p0)
        for nm, curves in (("rw", [np.array(v["curve"]) for v in rw]), ("nr", [np.array(r["patch"]["noreason"]["curve"])])):
            c = np.mean([np.log(np.maximum(cv, 1e-9)) for cv in curves], 0) - np.log(p0)       # log-ratio patched / clean, per layer
            d[f"{nm}_mean"] = c.mean(); d[f"{nm}_min"] = c.min(); d[f"{nm}_early"] = c[: L // 3].mean(); d[f"{nm}_mid"] = c[L // 3: 2 * L // 3].mean()
            d[f"{nm}_late"] = c[2 * L // 3:].mean()
            half = np.where(c < np.log(.5))[0]; d[f"{nm}_lastdrop"] = (half.max() + 1) / L if len(half) else 0.     # last layer where patching halves p
        d["nr_src"] = np.log(max(r["patch"]["noreason"]["source_p_clean"], 1e-9)) - np.log(p0)
        rows.append(d)
P = pd.DataFrame(rows)
print(f"{len(P)} rows | unfinished {P.unf.sum()} ({P[P.unf].y.sum()} robust) | per model unfinished: " +
      ", ".join(f"{m} {g.unf.sum()}" for m, g in P.groupby("model")))
FE = [c for c in P.columns if c.startswith(("ko_", "koflip_", "rw_", "nr_"))] + ["clean_p"]
U = P[P.unf].copy()
for c in FE: U[c + "_pct"] = U.groupby("model")[c].rank(pct=True)
print("\nAUC (higher value -> robust), unfinished rows; pooled = within-model percentiles; per model raw:")
print(f"   {'feature':16s} pooled (perm p) |  " + "  ".join(f"{m:8s}" for m in sorted(U.model.unique())) + " | main_v2  olympiad")
res = []
for c in FE:
    a = auc(U[c + "_pct"], U.y); res.append((abs(a - .5), c))
    print(f"   {c:16s} {a:.2f}  ({perm_p(U[c + '_pct'], U.y, 1000):.2f})   |  " + "  ".join(f"{auc(g[c], g.y):.2f}    " for _, g in U.groupby("model")) +
          f" | {auc(U[U.dataset == 'main_v2'][c + '_pct'], U[U.dataset == 'main_v2'].y):.2f}     {auc(U[U.dataset == 'olympiad'][c + '_pct'], U[U.dataset == 'olympiad'].y):.2f}")
# relation to existing features
F = pd.read_csv("exp56_rows.csv")[["id", "fa", "chk", "fas", "lble", "pmle"]]
J = U.merge(F, on="id", how="left")
print("\nSpearman with existing features (unfinished rows, within-model percentiles pooled):")
for _, c in sorted(res, reverse=True)[:6]:
    print(f"   {c:16s} " + "  ".join(f"{e} {J[c].rank().corr(J[e].rank()):+.2f}" for e in ["fa", "chk", "fas", "lble", "pmle"]))
P.to_csv("exp76_patching_features.csv", index=False)

# ------------------------------------------------------------------------------------------------------------------- C added value
print("\n" + "=" * 110 + "\nC. ADDED VALUE over the base ensemble (fa + chk + fas, signed within-model percentiles), unfinished rows")
J["prob"] = J.id.str.split("_").str[-1]
for e, sg in (("fa", 1), ("chk", 1), ("fas", 1)): J[e + "_b"] = J.groupby("model")[e].rank(pct=True) * sg      # signs as in our recipes
sgn = {"fa": 1, "chk": 1, "fas": 1}
print("   base signs check: " + ", ".join(f"{e} AUC {auc(J[e + '_b'], J.y):.2f}" for e in sgn))
J["base"] = J[[e + "_b" for e in sgn]].mean(1)
print(f"   base AUC {auc(J.base, J.y):.2f} | partial: AUC of each feature's residual after the base (rank regression), and base+feature with the sign "
      f"chosen leaving the problem out")
for c in FE:
    x = J[c + "_pct"].values; b = J.base.rank(pct=True).values
    res_ = x - np.polyval(np.polyfit(b, x, 1), b)
    pred = np.zeros(len(J))
    for pr in J.prob.unique():                                                               # nested sign: chosen without this problem
        tr = (J.prob != pr).values; s = 1 if auc(x[tr], J.y.values[tr]) >= .5 else -1
        pred[~tr] = (J.base.values[~tr] * 3 + s * x[~tr]) / 4
    print(f"   {c:18s} residual AUC {auc(res_, J.y):.2f} (p {perm_p(res_, J.y, 1000):.2f}) | base+feature (nested sign) {auc(pred, J.y):.2f}")

# direct test vs/with forced confidence on the same olympiad problems
print("\nD. DIRECT TEST together with forced confidence (olympiad, unfinished originals at 6144)")
D56 = pd.read_csv("exp56_rows.csv")
for f in sorted(glob.glob(R + "*/direct.jsonl")):
    mm = f.split("/")[-2]; by = {}
    for l in open(f):
        r = json.loads(l); by.setdefault(r["pid"], {})[r["ptype"]] = r
    T = pd.DataFrame([dict(id=f"{mm}_{pid}", share=np.mean([v[p]["forced"]["6144"]["answer"] == v["original"]["forced"]["6144"]["answer"] for p in PT]),
                           reph=float(v["rephrase"]["forced"]["6144"]["answer"] == v["original"]["forced"]["6144"]["answer"]),
                           unf=v["original"]["think_end"] is None) for pid, v in by.items()]).merge(D56[["id", "robust", "fa", "chk", "fas"]], on="id")
    T = T[T.unf]
    for c in ("share", "fa", "chk", "fas"): T[c + "_p"] = T[c].rank(pct=True)
    T["base"] = (T.fa_p + T.chk_p + T.fas_p) / 3; T["base_r"] = T.base.rank(pct=True)
    print(f"   {mm.split('__')[1][:8]} n={len(T)}: agree share {auc(T.share, T.robust):.2f} | fa {auc(T.fa, T.robust):.2f} | base {auc(T.base, T.robust):.2f} | "
          f"base + agree share {auc((T.base_r + T.share_p) / 2, T.robust):.2f} | base + 3x share {auc((T.base_r + 3 * T.share_p) / 4, T.robust):.2f} | "
          f"corr(share, fa) {T.share.rank().corr(T.fa.rank()):+.2f}")
