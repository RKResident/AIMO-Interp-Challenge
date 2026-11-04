"""Adds the reasoning-graph view (exp63 / exp64) to analysis/viz/explorer_data.json (field 'wbg' per row): the white-box run's
64-token-block attention graph averaged over the middle third of the sampled layers, reasoning part only, each query block's row
normalised to its attention on earlier blocks.
  heat     the graph pooled to 128-token cells (query x key, lower triangle), values 0-255 = round(255 * sqrt(share))
  jump     per query block: attention-weighted mean backward distance (blocks)
  far      per query block: share of its reasoning attention on blocks > 20 back;  setup: share on the first 4 blocks (when > 20 back)
  targets  per query block: the two most-read blocks > 20 back, [block, share]
Run after viz_prepare.py and viz_add_whitebox.py."""
import json, os
import numpy as np

ROOT = __import__("pathlib").Path("/home/sabyasachi19/PycharmProjects/AimoAGAIN"); WB = ROOT / "results_whitebox" / "out_whitebox"
P = ROOT / "analysis" / "viz" / "explorer_data.json"; data = json.load(open(P)); slug = lambda m: m.replace("/", "__")  # noqa: E731
mid = {json.loads(l)["id"]: json.loads(l)["model_id"] for f in WB.glob("*/rows_shard*.jsonl") for l in open(f)}
for rec in data["rows"]:
    if rec["id"] not in mid: continue
    z = np.load(WB / slug(mid[rec["id"]]) / f"{rec['id']}.npz"); G = z["att_graph_block64_layer"].astype(np.float64); L = G.shape[0]
    g = G[L // 3:2 * L // 3].mean(0)[:, 2:]; n = g.shape[0]
    S = np.zeros((n, n)); jump, far, setup, targets = [], [], [], []
    for q in range(n):
        w = g[q, :q]; s = w.sum()
        if q == 0 or s <= 0:
            jump.append(None); far.append(None); setup.append(None); targets.append([]); continue
        S[q, :q] = w / s; d = q - np.arange(q); jump.append(round(float((S[q, :q] * d).sum()), 2))
        fm = d > 20; far.append(round(float(S[q, :q][fm].sum()), 4)); setup.append(round(float(S[q, :min(4, q)][fm[:min(4, q)]].sum()), 4))
        ks = np.where(fm)[0]; top = ks[np.argsort(-S[q, ks])][:2] if len(ks) else []
        targets.append([[int(k), round(float(S[q, k]), 4)] for k in top])
    m = -(-n // 2); H = np.zeros((m, m))
    for a in range(m):
        for b in range(a + 1):
            H[a, b] = S[2 * a:2 * a + 2, 2 * b:2 * b + 2].sum() / max(1, S[2 * a:2 * a + 2].shape[0])
    rec["wbg"] = {"n": n, "heat": [[int(round(255 * np.sqrt(min(1, H[a, b])))) for b in range(a + 1)] for a in range(m)],
                  "jump": jump, "far": far, "setup": setup, "targets": targets}
json.dump(data, open(P, "w"), separators=(",", ":"))
print(sum("wbg" in r for r in data["rows"]), "rows with the reasoning graph | file", round(os.path.getsize(P) / 1e6, 2), "MB")
