"""exp64b: is exp64's 'far reads land on the first 4 blocks' an attention sink at the start of the reasoning? Recompute the middle-layer
64-token-block features with key block 0 (which holds the reasoning's first tokens, the usual sink position) removed and each query's
reasoning attention renormalised; also with blocks 0-1 removed. Unfinished rows, within-model ranks, AUC (higher -> robust)."""
import pickle, numpy as np, pandas as pd, json
ROOT = __import__("pathlib").Path("/home/sabyasachi19/PycharmProjects/AimoAGAIN"); WB = ROOT / "results_whitebox" / "out_whitebox"
W = pickle.load(open(ROOT / "analysis" / "exp61_whitebox.pkl", "rb")); M = W["meta"].reset_index(drop=True)
mid = {json.loads(l)["id"]: json.loads(l)["model_id"] for f in WB.glob("*/rows_shard*.jsonl") for l in open(f)}
rows = []
for i, r in M.iterrows():
    z = np.load(WB / mid[r.id].replace("/", "__") / f"{r.id}.npz"); G = z["att_graph_block64_layer"].astype(np.float64); L = G.shape[0]
    g = G[L // 3:2 * L // 3].mean(0)[:, 2:]; n = g.shape[0]; f = dict(id=r.id, model=r.model, y=int(r.y), unf=bool(r.unf))
    for drop in (0, 1, 2):
        J, F, S, B0 = [], [], [], []
        for q in range(drop + 1, n):
            w = g[q, :q].copy(); B0.append(w[0] / max(w.sum(), 1e-12)); w[:drop] = 0; s = w.sum()
            if s <= 0: continue
            d = q - np.arange(q); J.append((w * d).sum() / s); F.append(w[d > 20].sum() / s); S.append(w[:4][d[:4] > 20].sum() / s)
        f[f"jump_drop{drop}"], f[f"far_drop{drop}"], f[f"setup_drop{drop}"] = np.mean(J), np.mean(F), np.mean(S)
        if drop == 0: f["block0_share"] = np.mean(B0)
    rows.append(f)
D = pd.DataFrame(rows); U = D[D.unf].reset_index(drop=True); y = U.y.values
def auc(s, yy):
    p, n = s[yy == 1], s[yy == 0]; return float(((p[:, None] > n[None]) + .5 * (p[:, None] == n[None])).mean())
rk = lambda c: U.groupby("model")[c].rank(pct=True).values
print("unfinished rows", len(U))
for c in ["block0_share"] + [f"{k}_drop{d}" for d in (0, 1, 2) for k in ("jump", "far", "setup")]:
    print(f"  {c:14s} AUC {auc(rk(c), y):.2f} | robust mean {U.loc[U.y == 1, c].mean():.4f} spurious {U.loc[U.y == 0, c].mean():.4f}")
