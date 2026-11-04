"""exp66: the graphs_crowding Vast run (results_graphs; 118 greedy traces + 86 main_v2 T=1 samples, cut at end of thinking / 6144).
  0 parity     the jump recomputed from this run's head-mean graphs (middle third of attention layers) vs exp63's value (white-box run)
  1 agreement  does the jump describe the PROBLEM or the trace? main_v2: Spearman (within model) greedy vs each T=1 sample and sample vs
               sample; AUC of each trace type against the label; same for crowding / entropy. (kNN / TDA failed here: rho ~0-0.28.)
  2 variants   on unfinished greedy rows: jump with attention sinks removed; jump per attention-layer depth (each layer alone)
  3 heads      per-head jump, heads selected inside leave-one-PROBLEM-out (top-N by training AUC with the training sign), held-out AUC
  4 crowding   sequence crowding (input / output embeddings), last-quarter crowding, entropy: AUC on unfinished greedy rows
All AUCs: within-model percentile ranks, higher -> robust; permutation p within model where shown."""
import json, os, pickle, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "2"
import numpy as np
import pandas as pd
sys.path.insert(0, "/home/sabyasachi19/PycharmProjects/AimoAGAIN/submission_core")
from spurious_features import f_block_jump                                              # noqa: E402

ROOT = __import__("pathlib").Path("/home/sabyasachi19/PycharmProjects/AimoAGAIN"); OUT = ROOT / "results_graphs" / "out_graphs"
W = pickle.load(open(ROOT / "analysis" / "exp61_whitebox.pkl", "rb")); META = W["meta"].set_index("id")
J63 = pd.read_csv(ROOT / "analysis" / "exp63_jump.csv").set_index("id").jmp
rng = np.random.default_rng(0)


def spear(a, b):
    a, b = pd.Series(np.asarray(a, float)), pd.Series(np.asarray(b, float)); ok = a.notna() & b.notna()
    return float(np.corrcoef(a[ok].rank(), b[ok].rank())[0, 1])


def band(g):
    L = g.shape[0]; return g[L // 3:2 * L // 3].astype(np.float64).mean(0)


rows = []
for f in sorted(OUT.glob("*/graph_rows.jsonl")):
    for l in open(f):
        r = json.loads(l); z = np.load(f.parent / "graphs" / f"{r['key']}.npz"); g = z["graph"]; L = g.shape[0]
        jm, far, setup = f_block_jump(band(g)); jm_ns, _, _ = f_block_jump(band(z["graph_nosink"]))
        hs = z["head_summary"]                                                    # [L, H, 3] jump / far / setup
        ci, co, en = (z[k].astype(np.float32) for k in ("crowd_in", "crowd_out", "entropy")); q = max(1, len(ci) // 4)
        rows.append(dict(key=r["key"], id=r["id"], trace=r["trace"], dataset=r["dataset"], y=int(r["robust"]), cut=r["cut"], think_end=r["think_end"],
                         model=META.model.get(r["id"]), group=META.group.get(r["id"]), jump=jm, far=far, setup=setup, jump_nosink=jm_ns,
                         per_layer=[f_block_jump(g[l].astype(np.float64))[0] for l in range(L)], heads=hs[L // 3:2 * L // 3, :, 0],
                         crowd_in=ci.mean(), crowd_out=co.mean(), entropy=en.mean(), crowd_in_lastq=ci[-q:].mean(), crowd_out_lastq=co[-q:].mean(),
                         n_sinks=int((z["sinks"] >= 0).sum())))
D = pd.DataFrame(rows); D["finished"] = D.think_end.notna()
G = D[D.trace == "greedy"].reset_index(drop=True); U = G[~G.finished].reset_index(drop=True)


def auc(s, y):
    ok = ~np.isnan(s); s, y = s[ok], y[ok]; p, n = s[y == 1], s[y == 0]
    return float(((p[:, None] > n[None]) + .5 * (p[:, None] == n[None])).mean()) if len(p) and len(n) else np.nan


def rk(df, c):
    return df.groupby("model")[c].rank(pct=True).values


def perm(df, s, a, k=2000):
    y = df.y.values.copy(); null = []
    for _ in range(k):
        for m in df.model.unique():
            ii = (df.model == m).values; y[ii] = rng.permutation(y[ii])
        null.append(auc(s, y))
    return float((np.abs(np.array(null) - .5) >= abs(a - .5) - 1e-12).mean())


print(f"{len(D)} traces: {len(G)} greedy ({len(U)} unfinished), {(D.trace != 'greedy').sum()} T=1 samples | mean sinks found per trace {D.n_sinks.mean():.1f}\n")
print("0 PARITY with exp63 (greedy): max |jump - exp63| =", round(float(np.nanmax(np.abs(G.jump.values - J63.reindex(G.id).values))), 5),
      "| Spearman", round(spear(G.jump.values, J63.reindex(G.id).values), 4))

print("\n1 SAME-PROBLEM AGREEMENT (main_v2, 43 problems x greedy + 2 T=1 samples; Spearman of within-model ranks)")
M = D[D.dataset == "main_v2"]
for c in ("jump", "far", "setup", "crowd_in", "crowd_out", "entropy"):
    P = M.pivot_table(index="id", columns="trace", values=c); mdl = META.model.reindex(P.index)
    R_ = P.groupby(mdl.values).rank(pct=True)
    rho = lambda a, b: spear(R_[a], R_[b])                     # noqa: E731
    yv = META.y.reindex(P.index).values.astype(int)
    au = {t: auc(R_[t].values, yv) for t in ("greedy", "t1s0", "t1s1")}
    au["mean of 3"] = auc(R_[["greedy", "t1s0", "t1s1"]].mean(1).values, yv)
    print(f"  {c:9s} rho greedy-s0 {rho('greedy', 't1s0'):+.2f}  greedy-s1 {rho('greedy', 't1s1'):+.2f}  s0-s1 {rho('t1s0', 't1s1'):+.2f} | "
          f"AUC (all main_v2 rows) " + "  ".join(f"{k} {v:.2f}" for k, v in au.items()))
MU = M[~M.finished]
print(f"  unfinished main_v2 traces only ({len(MU)} traces): jump AUC by trace type " +
      "  ".join(f"{t} {auc(rk(MU[MU.trace == t], 'jump'), MU[MU.trace == t].y.values):.2f}" for t in ("greedy", "t1s0", "t1s1")))

print("\n2 UNFINISHED GREEDY ROWS: variants of the jump")
for c in ("jump", "jump_nosink", "far", "setup"):
    a = auc(rk(U, c), U.y.values); print(f"  {c:12s} AUC {a:.2f} (p {perm(U, rk(U, c), a):.3f})")
print("  per attention layer (each layer alone; depth = layer index / number of attention layers):")
for m, g in U.groupby("model"):
    PL = np.array(g.per_layer.tolist()); L = PL.shape[1]
    au = [auc(pd.Series(PL[:, l]).rank(pct=True).values, g.y.values) for l in range(L)]
    print(f"    {m.split('/')[1][:12]:12s} n={len(g):2d} L={L:2d} | " + " ".join(f"{v:.2f}" for v in au))

print("\n3 PER-HEAD JUMP, heads selected inside leave-one-problem-out (middle-third layers)")
Hs = np.array([h.reshape(-1) for h in U.heads]) if U.heads.map(lambda h: h.shape).nunique() == 1 else None
for N in (8, 32, 128):
    s = np.full(len(U), np.nan)
    for m in U.model.unique():
        ii = np.where(U.model == m)[0]; X = np.array([U.heads[i].reshape(-1) for i in ii]); yv = U.y.values[ii]; grp = U.group.values[ii]
        for g0 in np.unique(grp):
            te = grp == g0; tr = ~te
            if yv[tr].min() == yv[tr].max(): continue
            a = np.array([auc(X[tr, k], yv[tr]) for k in range(X.shape[1])]); strength = np.abs(a - .5); top = np.argsort(-strength)[:N]
            Z = (X[:, top] - X[tr][:, top].mean(0)) / (X[tr][:, top].std(0) + 1e-9) * np.sign(a[top] - .5)
            s[ii[te]] = Z[te].mean(1)
    print(f"  top-{N:3d} heads: held-out AUC {auc(rk(U.assign(s=s), 's'), U.y.values):.2f}")

print("\n4 CROWDING (unfinished greedy rows; higher -> robust)")
for c in ("crowd_in", "crowd_out", "crowd_in_lastq", "crowd_out_lastq", "entropy"):
    a = auc(rk(U, c), U.y.values); print(f"  {c:16s} AUC {a:.2f} (p {perm(U, rk(U, c), a):.3f}) | per model " +
                                        "  ".join(f"{m.split('/')[1][:8]} {auc(g[c].values, g.y.values):.2f}" for m, g in U.groupby("model")))
print("  partial: crowding after regressing out entropy (within-model ranks):", end=" ")
for c in ("crowd_in", "crowd_out"):
    s, e = rk(U, c), rk(U, "entropy"); b = np.polyfit(e, s, 1); print(f"{c} {auc(s - np.polyval(b, e), U.y.values):.2f}", end="  ")
print("\n  Spearman crowd_in vs entropy", round(float(pd.Series(rk(U, 'crowd_in')).corr(pd.Series(rk(U, 'entropy')))), 2),
      "| vs jump", round(float(pd.Series(rk(U, 'crowd_in')).corr(pd.Series(rk(U, 'jump')))), 2))
D.drop(columns=["per_layer", "heads"]).to_csv(ROOT / "analysis" / "exp66_graphs_crowding.csv", index=False)
