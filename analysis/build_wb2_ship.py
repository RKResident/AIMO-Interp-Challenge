"""wb2_ship_<model>.npz for the v8-wblog2 logging submission (user 2026-10-31), from our run_wb2 outputs only (the leaderboard is never fitted):
  pca_mu/V/s_d{0,1,2}  exp104's label-free PCA-5 per depth (scale -> median tangent norm 1.5)
  km16_d*, km32_d*     exp103's label-free K-means centroids per depth (same seeds / init as exp103)
  dir_<kind>           difference-of-means directions (mean robust - mean non-robust, FORCED labels, all our rows), every stored layer,
                       kind in prompt_last / trace_mean / seg25 / seg50 / seg75 (exp103 fits these in-fold; here once, on everything)
Writes analysis/wb2_ship/wb2_ship_<tag>.npz (float16 except scalars)."""
import json, numpy as np, pandas as pd
src = open("/home/sabyasachi19/PycharmProjects/AimoAGAIN/analysis/exp103_wb2_eval.py").read()
exec(src[:src.index("singles, fams_out = [], []")])
from pathlib import Path
OUT = Path(A) / "analysis/wb2_ship"; OUT.mkdir(exist_ok=True)
for m, tag in TAGS.items():
    B = pd.DataFrame([json.loads(l) for l in open(A + f"ec2_results/wb2/{tag}.jsonl")]).drop_duplicates("pid").set_index("pid")
    Z = {p: dict(np.load(A + f"ec2_results/wb2/{tag}/{p}.npz")) for p in B.index}; ok = [p for p in B.index if "w64_3layers" in Z[p]]
    sh = {}
    for di in range(3):
        allw = np.concatenate([Z[p]["w64_3layers"][di].astype(np.float64) for p in ok]); mu = allw.mean(0)
        _, _, Vt = np.linalg.svd(allw - mu, full_matrices=False); P5 = (allw - mu) @ Vt[:5].T
        sh[f"pca_mu_d{di}"] = mu.astype(np.float32); sh[f"pca_V_d{di}"] = Vt[:5].astype(np.float32); sh[f"pca_s_d{di}"] = np.float64(1.5 / np.median(np.linalg.norm(P5, axis=1)))
        U = np.concatenate([unitn(Z[p]["w64_3layers"][di].astype(np.float32)) for p in ok])
        for k in (16, 32): sh[f"km{k}_d{di}"] = kmeans(U, k, seed=di).astype(np.float16)
    lab = pd.Series(labels(m)["forced"]); lab = lab[lab.index.isin(ok)]; y = lab.values
    for kind, get in (("prompt_last", lambda z: z["prompt_last"]), ("trace_mean", lambda z: z["trace_mean"]), ("seg25", lambda z: z["seg_means"][:, 0]),
                      ("seg50", lambda z: z["seg_means"][:, 1]), ("seg75", lambda z: z["seg_means"][:, 2])):
        M = np.stack([get(Z[p]).astype(np.float32) for p in lab.index]); v = M[y == 1].mean(0) - M[y == 0].mean(0)
        sh[f"dir_{kind}"] = (v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)).astype(np.float16)
    np.savez_compressed(OUT / f"wb2_ship_{tag}.npz", **sh)
    print(m, {k: v.shape for k, v in sh.items() if hasattr(v, "shape") and v.ndim}, f"| {len(lab)} labelled rows ({int(y.sum())} R)", flush=True)
