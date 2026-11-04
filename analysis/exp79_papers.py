"""exp79: features from the papers in papers_to_see/ (run_papers.py on the A10G), scored against our labels.
Rows: the 118 labelled greedy traces (robust from main_v2 / olympiad; unfinished rows = still thinking at the cut) and the set100 traces
(Qwen: forced-class labels from hard_pool/set100_labels_*.csv as they complete; Olmo: none yet). Per feature, in the direction the
breakdown predicts (PRED: +1 = higher -> robust, -1 = lower -> robust):
  - per-model AUC and pooled within-model AUC (higher = the predicted direction works), within-model permutation p
  - overlap with fa / chk / fas (Spearman, within-model ranks) and added value over the fa+chk+fas base (AUC of the residual)
on the 88 unfinished original rows; and the plain AUC on the set100 Qwen labels."""
import glob, json, sys
import numpy as np, pandas as pd
A = "/home/sabyasachi19/PycharmProjects/AimoAGAIN/"; R = A + "ec2_results/a10g/papers/"; rng = np.random.default_rng(0)
PRED = {"rhythm_corr0": 1, "rhythm_corrmax": 1, "anchor_fai": 1, "fai_mean": 1, "waad_mean": 1, "waad_spike": 1, "anchor_conc3": 1, "anchor_kurt": 1,
        "sup_top": 1, "sup_top_minus_rnd": 1, "minkpp_10": -1, "minkpp_20": -1, "mink_10": -1, "mink_20": -1, "li_sum": 1, "li_def": -1, "li_mid": 1,
        "ent_def": -1, "lid_mean": -1, "lid_p90": -1, "lid_nt_mean": -1, "lid_nt_p90": -1, "dc_l3": -1, "dc_l2": -1, "dc_l23": -1, "cie_tv": -1, "cie_rho": 1}
SRC = {"rhythm": "Preplan & Anchor rhythm (2510.13554)", "anchor": "Thought Anchors receiver heads (2506.19143)", "sup": "Thought Anchors suppression",
       "mink": "Min-K%++ (2404.02936)", "li_": "Layer-wise info deficiency (2412.10246)", "ent_def": "Layer-wise info deficiency (entropy)",
       "lid": "Local intrinsic dimension (2402.18048)", "dc": "LaRA directional collapse (2605.29888)", "cie": "CIE-style discrepancy (2605.25603)",
       "fai": "Preplan & Anchor rhythm", "waad": "Preplan & Anchor rhythm"}


def auc(s, y):
    ok = np.isfinite(s); s, y = s[ok], y[ok]; p, n = s[y == 1], s[y == 0]
    return float(((p[:, None] > n[None]) + .5 * (p[:, None] == n[None])).mean()) if len(p) and len(n) else np.nan


def within(s, y, g):
    num = den = 0
    for m in np.unique(g):
        k = (g == m) & np.isfinite(s); p, n = s[k & (y == 1)], s[k & (y == 0)]
        num += ((p[:, None] > n[None]) + .5 * (p[:, None] == n[None])).sum(); den += len(p) * len(n)
    return num / den if den else np.nan


def perm(s, y, g, n=2000):
    a = abs(within(s, y, g) - .5); null = []
    for _ in range(n):
        yy = y.copy()
        for m in np.unique(g):
            k = g == m; yy[k] = rng.permutation(yy[k])
        null.append(abs(within(s, yy, g) - .5))
    return float(np.mean(np.array(null) >= a - 1e-12))


rows = []
for f in glob.glob(R + "*/papers.jsonl"):
    m = f.split("/")[-2]
    for l in open(f): r = json.loads(l); r["model"] = m; rows.append(r)
D = pd.DataFrame(rows); FE = [c for c in PRED if c in D.columns]
print(f"{len(D)} traces with paper features: " + ", ".join(f"{m[:8]} {n}" for m, n in D.model.value_counts().items()))
B = pd.read_csv(A + "analysis/exp56_rows.csv")[["id", "fa", "chk", "fas"]]
O = D[D.robust.notna() & D.think_end.isna()].merge(B, on="id", how="left").reset_index(drop=True)
y = O.robust.astype(int).values; g = O.model.values
for e in ("fa", "chk", "fas"): O[e + "_r"] = O.groupby("model")[e].rank(pct=True)
O["base"] = O[["fa_r", "chk_r", "fas_r"]].mean(1).rank(pct=True)
print(f"\nORIGINAL LABELS, unfinished rows: {len(O)} ({y.sum()} robust) | per model " + ", ".join(f"{m[:8]} {(g == m).sum()}" for m in np.unique(g)))
print(f"   base fa+chk+fas pooled AUC {within(O.base.values, y, g):.2f}")
print(f"   {'feature':18s} {'source':38s} pooled  perm p | " + " ".join(f"{m[:8]:>8s}" for m in np.unique(g)) + " | rho fa   chk   fas | resid AUC")
res = []
for c in FE:
    s = PRED[c] * O[c].astype(float).values; a = within(s, y, g); p = perm(s, y, g)
    x = O.groupby("model")[c].rank(pct=True).values * PRED[c]; b = O.base.values; okk = np.isfinite(x)
    resid = np.full(len(x), np.nan); resid[okk] = x[okk] - np.polyval(np.polyfit(b[okk], x[okk], 1), b[okk])
    rho = [pd.Series(x).corr(O[e + "_r"], method="spearman") for e in ("fa", "chk", "fas")]
    src = next(v for k, v in SRC.items() if c.startswith(k))
    res.append((p, c)); print(f"   {c:18s} {src[:38]:38s} {a:6.2f}  {p:6.3f} | " + " ".join(f"{auc(s[g == m], y[g == m]):8.2f}" for m in np.unique(g)) +
                              f" | {rho[0]:+.2f} {rho[1]:+.2f} {rho[2]:+.2f} | {within(resid, y, g):.2f}")
ps = np.array([r[0] for r in res])
print(f"   {len(res)} features | share p<0.05 {np.mean(ps < .05):.2f} (null 0.05) | smallest p {ps.min():.3f}")
# set100 Qwen labels (forced class)
try:
    L = pd.read_csv(A + "hard_pool/set100_labels_Qwen__Qwen3.5-4B.csv")
    S = D[(D.dataset == "set100") & (D.model == "Qwen__Qwen3.5-4B")].copy(); S["pid"] = S.id.str.split("_").str[-1]   # Qwen rows only (pids are shared with Olmo)
    S = S.merge(L[["pid", "label_forced"]], on="pid"); S = S[S.label_forced.isin(["robust", "non_robust"])]
    ys = (S.label_forced == "robust").astype(int).values
    print(f"\nSET100 QWEN (forced class): {len(S)} labelled ({ys.sum()} robust) - AUC in the predicted direction, permutation p")
    out = []
    for c in FE:
        s = PRED[c] * S[c].astype(float).values; a = auc(s, ys)
        pp = float(np.mean([abs(auc(s, rng.permutation(ys)) - .5) >= abs(a - .5) - 1e-12 for _ in range(2000)]))
        out.append((pp, c, a))
    for pp, c, a in sorted(out)[:10]: print(f"   {c:18s} AUC {a:.2f}  p {pp:.3f}")
    print(f"   share p<0.05 {np.mean([o[0] < .05 for o in out]):.2f} (null 0.05)")
except FileNotFoundError:
    pass
