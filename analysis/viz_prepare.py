"""Data for the failure-explorer page (analysis/viz/failure_explorer.html): every labelled row of main_v2 (43) and olympiad (75).
Per row: label, finished within 6144, the two recipes' calls (0.71 recipe as submitted; the unfinished-calibrated recipe out of sample from
exp57's scheme is approximated by its shipped thresholds), the recipe and trajectory features; a 2D trajectory (PCA of the layer-3 64-token
mean states, PCA fitted per model on all its traces) for the greedy trace and, on main_v2, the two T=1 samples, with a text snippet per
point; lookback ratio and prompt attention mass over the reasoning (64-token points, early / middle / late layer thirds). Per model: the
TOHA separation map (effect size of spurious minus robust mean MSF edge per layer x head, last 128 tokens at the 2048 cut)."""
import json, os, pickle, re, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "2"
import numpy as np
import pandas as pd
sys.path.insert(0, "/home/sabyasachi19/PycharmProjects/AimoAGAIN/analysis"); sys.path.insert(0, "/home/sabyasachi19/PycharmProjects/AimoAGAIN/submission_core")
from common import *                                                                    # noqa: E402,F401
from spurious_inference import load_tokenizer                                           # noqa: E402
from spurious_features import score_trace                                               # noqa: E402

D = pd.read_csv(A / "exp56_rows.csv"); slug = lambda m: m.replace("/", "__")             # noqa: E731
OLY = ROOT / "extra_olympiad_problems_dataset"; MV1 = ROOT / "results_mainv2_t1" / "out_t1"
P2 = ROOT / "extraction_out_generations_p2" / "extraction_out_generations"; LBMV = ROOT / "extraction_out_generations_pearl_more" / "extraction_out_generations"
inp = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "t1_run" / "inputs.jsonl")}
F47 = pickle.load(open(A / "exp47_features.pkl", "rb")); MB = pickle.load(open(A / "exp47b_mainv2_values.pkl", "rb"))
REC = json.load(open("/home/sabyasachi19/PycharmProjects/aimo/getting-started/solutions/spurious-unfinished-calibrated/spurious_artifacts/recipe.json"))
REC71 = json.load(open("/home/sabyasachi19/PycharmProjects/aimo/getting-started/solutions/spurious-confidence-checkshare-slope/spurious_artifacts/recipe.json"))
TOK = {}


def blocks(ws3):                                                                         # 16-token windows -> 64-token means
    n = len(ws3) // 4 * 4
    return ws3[:n].reshape(-1, 4, ws3.shape[-1]).mean(1) if n else ws3


def traces(row):
    """[(name, layer-3 16-token states, token ids, per-token entropy)] for this row: greedy (+ the 2 T=1 samples on main_v2)."""
    out = []
    z = np.load((MV1 if row.dataset == "main_v2" else OLY / "results_olympiad_features" / "out_t1") / slug(row.model_id) / f"{row.id}.npz")
    for t, nm in (("t0", "greedy"), ("t1s0", "sample 1"), ("t1s1", "sample 2")):
        if f"{t}_ws16" not in z.files: continue
        ids = z[f"{t}_gen_ids"] if f"{t}_gen_ids" in z.files else np.array(inp[row.id]["greedy_gen_ids"][:6144])
        ent = z[f"{t}_tok_entropy"].astype(np.float32) if f"{t}_tok_entropy" in z.files else None
        out.append((nm, z[f"{t}_ws16"][:, 3].astype(np.float32), ids, ent))
    return out


def attention(row):
    p = (LBMV / slug(row.model_id) / "lookback" if row.dataset == "main_v2" else OLY / "results_olympiad_attention" / "out_attn" / slug(row.model_id) / "lookback") / f"{row.id}.npz"
    if not p.exists(): return None
    z = np.load(p); nw = min(384, len(z["lookback"])); lb, pm = z["lookback"][:nw].astype(np.float32), z["prompt_mass"][:nw].astype(np.float32)
    L = lb.shape[1]; bands = [range(0, L // 3), range(L // 3, 2 * L // 3), range(2 * L // 3, L)]
    def curve(a, b):
        x = a[:, list(b)].mean((1, 2)); n = len(x) // 4 * 4
        return [round(float(v), 4) for v in (x[:n].reshape(-1, 4).mean(1) if n else x)]
    return {"lookback": [curve(lb, b) for b in bands], "prompt_mass": [curve(pm, b) for b in bands]}


rows, models = [], {}
for _, r in D.iterrows():
    T = traces(r); models.setdefault(r.model_id, []).append((len(rows), T)); rows.append(r)
# Projections, fitted jointly per model over all its traces (greedy + T=1 samples), so paths of different problems share one map.
# Massive-activation channels (exp59-60: they carry the spikes and no robustness signal) are neutralised first: per model, the 3 hidden
# dimensions that most often carry a window's largest deviation are set to the model-wide median.
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
import umap
PROJ, MASSIVE, EVR = {}, {}, {}
for m, lst in models.items():
    blks = [(ri, k, blocks(ws)) for ri, T in lst for k, (_, ws, _, _) in enumerate(T)]
    X = np.concatenate([b for _, _, b in blks])
    dev = np.abs(X - np.median(X, 0)); top = np.bincount(np.argmax(dev, axis=1), minlength=X.shape[1]).argsort()[::-1][:3]
    MASSIVE[m] = [int(d) for d in top]; X = X.copy(); X[:, top] = np.median(X[:, top], 0)
    Z = PCA(50, svd_solver="full", random_state=0).fit_transform((X - X.mean(0)) / (X.std(0) + 1e-6))
    p2 = PCA(2, svd_solver="full").fit(X - X.mean(0)); EVR[m] = [round(float(v), 3) for v in p2.explained_variance_ratio_]
    emb = {"pca": p2.transform(X - X.mean(0)),
           "tsne": TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(Z),
           "umap": umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(Z),
           "isomap": Isomap(n_neighbors=12, n_components=2).fit_transform(Z)}
    pos = 0
    for ri, k, b in blks:
        PROJ[(ri, k)] = {meth: e[pos:pos + len(b)] for meth, e in emb.items()}; pos += len(b)
    print(f"{m.split('/')[1]}: {len(X)} points embedded, massive channels neutralised {MASSIVE[m]}", flush=True)
out = []
for i, r in enumerate(rows):
    m = r.model_id
    if m not in TOK: TOK[m] = load_tokenizer(m)
    trs = []
    for k, (nm, ws, ids, ent) in enumerate(traces(r)):
        P = PROJ[(i, k)]; n = len(P["pca"])
        snip = []
        if nm == "greedy" and ids is not None:
            for s in range(n):
                t = TOK[m].decode([int(x) for x in ids[64 * s: 64 * s + 64]], skip_special_tokens=True)
                snip.append(re.sub(r"\s+", " ", t).strip()[:110])
        e64 = [round(float(ent[64 * s: 64 * s + 64].mean()), 3) for s in range(n)] if ent is not None and len(ent) else None
        trs.append({"name": nm, "proj": {meth: [[round(float(a), 2), round(float(b), 2)] for a, b in v] for meth, v in P.items()}, "text": snip, "entropy": e64})
    fin = bool(r.finished)
    s71 = score_trace({f: r[f] for f in REC71["features"]}, REC71, m)
    su = score_trace({f: r[f] for f in REC["features"]}, REC, m)
    rec = dict(id=r.id, dataset=r.dataset, model=m.split("/")[1], robust=int(r.robust), finished=fin,
               indep=r.dataset == "main_v2" or str(r.indep) == "True",              # olympiad rows whose problem is not in main_v2
               problem=inp[r.id]["problem"][:400], gold=None if pd.isna(r.gold) else str(r.gold), forced_answer=None if pd.isna(r.forced_answer) else str(r.forced_answer),
               call71=int(s71 > REC71["threshold"]), score71=round(float(s71), 4),
               call_unf=1 if fin else int(su > REC["threshold_by_model"][m]), score_unf=round(float(su), 4), thr_unf=round(REC["threshold_by_model"][m], 4),
               feats={f: (None if pd.isna(r[f]) else round(float(r[f]), 5)) for f in ("fa", "chk", "fas", "lble", "pmle")},
               failed_perturbations=None if pd.isna(r.get("failed_perturbations", np.nan)) else r.failed_perturbations,
               traces=trs, attention=attention(r))
    out.append(rec)
# trajectory features per row (to show their instability): kNN efficiency, TDA, segment PC1, late-vs-early cosine (greedy / samples)
mv ={json.loads(l)["id"]: json.loads(l) for f in MV1.glob("*/rows_shard*.jsonl") for l in open(f)}
ol = {json.loads(l)["id"]: json.loads(l) for f in (OLY / "results_olympiad_features" / "out_t1").glob("*/rows_shard*.jsonl") for l in open(f)}
for rec in out:
    src = mv if rec["dataset"] == "main_v2" else ol; g = src[rec["id"]]
    T = [g["greedy"]] + g.get("samples", [])
    rec["traj_feats"] = {f: [None if t.get(f) is None or (isinstance(t.get(f), float) and np.isnan(t[f])) else round(float(t[f]), 4) for t in T] for f in ("ge", "tda", "seg", "traj")}
# TOHA separation maps per model (olympiad where available, else main_v2): (mean spurious - mean robust) / pooled sd of the MSF mean edge
toha = {}
for m in sorted(D.model_id.unique()):
    sub = D[D.model_id == m]; X, y = [], []
    for _, r in sub.iterrows():
        p = (P2 / slug(m) / "toha" if r.dataset == "main_v2" else OLY / "olympiad-features" / slug(m) / "toha") / f"{r.id}.npz"
        if not p.exists(): continue
        z = np.load(p, allow_pickle=True)
        if len(z["cut_eff"]) < 2: continue
        X.append(z["win"][1, :, :, list(z["stat_names"]).index("mean_edge")].astype(np.float64)); y.append(int(r.robust))
    if len(X) < 6: continue
    X, y = np.stack(X), np.array(y); d = (X[y == 0].mean(0) - X[y == 1].mean(0)) / (X.std(0) + 1e-9)
    toha[m.split("/")[1]] = {"effect": [[round(float(v), 2) for v in row] for row in d], "n": int(len(y)), "robust": int(y.sum())}
meta = {"pca_explained": {m.split("/")[1]: EVR[m] for m in EVR}, "massive_channels": {m.split("/")[1]: v for m, v in MASSIVE.items()}, "toha": toha,
        "refs": {f: {m.split("/")[1]: REC["ref"][f][m] for m in REC["threshold_by_model"]} for f in REC["features"]},
        "thresholds_unfinished": {m.split("/")[1]: v for m, v in REC["threshold_by_model"].items()}}
os.makedirs(A / "viz", exist_ok=True)
json.dump({"rows": out, "meta": meta}, open(A / "viz" / "explorer_data.json", "w"), separators=(",", ":"))
print(len(out), "rows | file", round(os.path.getsize(A / "viz" / "explorer_data.json") / 1e6, 2), "MB | TOHA maps", list(toha))
