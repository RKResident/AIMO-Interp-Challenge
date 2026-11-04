"""exp88 (user 2026-10-30): Self-Critique-style entropy-trajectory similarity (Tao et al., ICLR 2026, arXiv 2510.09259) on SAVED traces only.
main_v2: per problem the greedy trace and 2 plain T=1 samples (cap 6144), full-vocab per-token entropy from the exp66 graphs run
(results_graphs/out_graphs/*/graphs/<id>__{greedy,t1s0,t1s1}.npz). Score per problem = mean over the 2 samples of sim(E_greedy, E_sample):
  pen   paper eq. 9: cos(zero-padded) * min(len)/max(len)
  blk   the same on 64-token block means (coarse phases of the reasoning)
  rho   Spearman of the 64-token block means over the common length
Higher = the model retraces the same entropy path (the paper's "policy collapse" / memorized). AUC raw (higher -> robust), pooled and with
within-model ranks, all rows and unfinished greedy rows; permutation p. Controls: mean greedy entropy, greedy length."""
import glob, json, numpy as np, pandas as pd
from scipy.stats import spearmanr

A = "/home/sabyasachi19/PycharmProjects/AimoAGAIN/"; rng = np.random.default_rng(0); B = 64


def pen(a, b):
    n = max(len(a), len(b)); A_ = np.zeros(n); B_ = np.zeros(n); A_[:len(a)] = a; B_[:len(b)] = b
    return float(A_ @ B_ / (np.linalg.norm(A_) * np.linalg.norm(B_) + 1e-12) * min(len(a), len(b)) / n)


def blocks(e): n = len(e) // B; return e[:n * B].reshape(n, B).mean(1) if n else e


def auc(s, y):
    ok = np.isfinite(s); s, y = s[ok], y[ok]; p, n = s[y == 1], s[y == 0]
    return float(((p[:, None] > n[None]) + .5 * (p[:, None] == n[None])).mean()) if len(p) and len(n) else np.nan


def pv(s, y, n=4000): a = abs(auc(s, y) - .5); return float(np.mean([abs(auc(s, rng.permutation(y)) - .5) >= a - 1e-12 for _ in range(n)]))


rows = []
for f in glob.glob(A + "results_graphs/out_graphs/*/graph_rows.jsonl"):
    G = {json.loads(l)["key"]: json.loads(l) for l in open(f)}; d = f.rsplit("/", 1)[0] + "/graphs/"
    for k, g in G.items():
        if g["trace"] != "greedy" or g["dataset"] != "main_v2": continue
        e0 = np.load(d + k + ".npz")["entropy"].astype(np.float64); sims = []
        for s in ("t1s0", "t1s1"):
            try: e1 = np.load(d + g["id"] + f"__{s}.npz")["entropy"].astype(np.float64)
            except FileNotFoundError: continue
            b0, b1 = blocks(e0), blocks(e1); m = min(len(b0), len(b1))
            sims.append((pen(e0, e1), pen(b0, b1), spearmanr(b0[:m], b1[:m])[0] if m > 2 else np.nan))
        if not sims: continue
        S = np.nanmean(np.array(sims), 0)
        rows.append(dict(id=g["id"], model=g["id"].split("_")[0], robust=int(g["robust"]), unfinished=g["think_end"] is None, n_samples=len(sims),
                         pen=S[0], blk=S[1], rho=S[2], mean_ent=float(e0.mean()), length=len(e0)))
D = pd.DataFrame(rows)
for c in ("pen", "blk", "rho", "mean_ent", "length"): D[c + "_wr"] = D.groupby("model")[c].rank(pct=True)
print(f"{len(D)} main_v2 problems ({D.robust.sum()} robust), {int(D.unfinished.sum())} unfinished greedy; samples per problem {D.n_samples.value_counts().to_dict()}")
print(f"{'score':9s} | all: raw (p)    within-model (p) | unfinished: raw (p)    within-model (p)")
for c in ("pen", "blk", "rho", "mean_ent", "length"):
    out = []
    for X in (D, D[D.unfinished]):
        y = X.robust.values
        for col in (c, c + "_wr"): s = X[col].values.astype(float); out.append(f"{auc(s, y):.2f} ({pv(s, y):.3f})")
    print(f"{c:9s} | " + "  ".join(out[:2]) + "   | " + "  ".join(out[2:]))
print("\nper model (raw AUC, all rows):", {m: {c: round(auc(X[c].values, X.robust.values), 2) for c in ("pen", "blk", "rho")} for m, X in D.groupby("model")})
print("Spearman with controls (all rows):", {c: {k: round(spearmanr(D[c], D[k])[0], 2) for k in ("mean_ent", "length")} for c in ("pen", "blk", "rho")})
D.to_csv(A + "analysis/exp88_rows.csv", index=False)
