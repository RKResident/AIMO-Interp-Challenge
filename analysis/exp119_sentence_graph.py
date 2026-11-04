"""exp119 (user 2026-11-02): semantic loops + graph / tree structure of the reasoning, from SENTENCE embeddings of the output text
(all-MiniLM-L6-v2, local GPU) - never tried before (exp29 = surface text repetition; exp30 = K-means on the model's own hidden states).
Traces: greedy screening traces with token ids (cap 6144): main-v2 (ec2_results/mv2, all 4 models, official labels), the old labelled pool
(ec2_results/big/out/screen, Qwen / Olmo, our forced labels), and the confidence-matched relabel problems (ec2_results/easy; labels later, exp117).
Per trace: thinking text -> sentences -> unit embeddings, then
  loops   ret85 / ret90 (share of sentences whose best match >= 5 sentences back has cos >= .85 / .90), late_ret90 (last third),
          maxprev_mean (mean best earlier match), novelty90 (share with no earlier match >= .90)
  motion  speed_mean / speed_cv (1 - cos of consecutive sentences), wspeed_mean / wspeed_cv (8-sentence windows), drift (first vs last 10%),
          conv_slope (slope of cos to the final 10% centroid over position)
  walk    per-model K-means (K = 16, 32) over all sentences of that model; on the cluster walk: self_loop, returns, cycle_rank / step,
          entropy_rate, spectral_gap, hub_share, coverage
  tree    MST over the embeddings: mst_mean (H0 persistence / n), mst_max, chain (MST diameter / n), branch (share of nodes of degree >= 3);
          threshold graph (cos >= .8): components / n
Evaluation: AUC (higher value -> robust) on main-v2 official labels (all rows, unfinished rows; per model + pooled within-model ranks) and on
the old Qwen / Olmo labels, plus a length control (residual on log n_sent within model) and, for the old labels, a difficulty control
(residual on original accuracy). Writes analysis/exp119_feats.csv, exp119_auc.csv; prints the tables.
  python analysis/exp119_sentence_graph.py"""
import json, os, re, sys, numpy as np, pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components, shortest_path
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score
A = "/home/sabyasachi19/PycharmProjects/AimoAGAIN/"; sys.path.insert(0, A + "submission_core")
from spurious_inference import load_tokenizer  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

TAGS = {"Qwen": "Qwen__Qwen3.5-4B", "Skywork": "Skywork__Skywork-OR1-Math-7B", "allenai": "allenai__Olmo-3-7B-Think", "deepseek": "deepseek-ai__DeepSeek-R1-0528-Qwen3-8B"}
SRC = {"mv2": "ec2_results/mv2/scr_{t}.jsonl", "old": "ec2_results/big/out/screen/{t}.jsonl", "easy": "ec2_results/easy/scr_{t}.jsonl"}
emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")


def sentences(text):
    s = [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", text)]
    return [x for x in s if len(x) >= 15][:1500]


def feats_trace(E):
    n = len(E); f = dict(n_sent=n)
    if n < 20: return f
    S = E @ E.T
    best = np.array([S[i, :i - 4].max() if i >= 5 else np.nan for i in range(n)])
    b = best[10:]; f.update(ret85=np.mean(b >= .85), ret90=np.mean(b >= .90), maxprev_mean=np.nanmean(b), novelty90=np.mean(b < .90),
                            late_ret90=np.mean(best[2 * n // 3:] >= .90))
    sp = 1 - np.einsum("ij,ij->i", E[1:], E[:-1]); f.update(speed_mean=sp.mean(), speed_cv=sp.std() / (sp.mean() + 1e-9))
    W = np.array([E[i:i + 8].mean(0) for i in range(0, n - 7, 8)]); W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    if len(W) >= 3: ws = 1 - np.einsum("ij,ij->i", W[1:], W[:-1]); f.update(wspeed_mean=ws.mean(), wspeed_cv=ws.std() / (ws.mean() + 1e-9))
    k = max(2, n // 10); a = E[:k].mean(0); z = E[-k:].mean(0)
    f["drift"] = 1 - a @ z / (np.linalg.norm(a) * np.linalg.norm(z) + 1e-9)
    c = E @ (z / (np.linalg.norm(z) + 1e-9)); f["conv_slope"] = np.polyfit(np.linspace(0, 1, n), c, 1)[0]
    D = np.sqrt(np.clip(2 - 2 * S, 0, None)); np.fill_diagonal(D, 0)
    T = minimum_spanning_tree(D + 1e-9 * (D == 0)).toarray(); T = np.maximum(T, T.T); w = T[T > 0]
    f.update(mst_mean=w.sum() / n, mst_max=w.max())
    deg = (T > 0).sum(1); f["branch"] = np.mean(deg >= 3)
    hop = shortest_path((T > 0).astype(float), unweighted=True, directed=False); f["chain"] = np.nanmax(hop[np.isfinite(hop)]) / n
    f["components80"] = connected_components((S >= .8).astype(int), directed=False)[0] / n
    return f


def walk_feats(lab, K, pre):
    f = {}; x = np.asarray(lab); n = len(x)
    if n < 20: return f
    f[pre + "self_loop"] = np.mean(x[1:] == x[:-1])
    seen, ret = set(), 0
    for i in range(1, n):
        if x[i] != x[i - 1] and x[i] in seen: ret += 1
        seen.add(x[i - 1])
    f[pre + "returns"] = ret / (n - 1)
    edges = {(min(a, b), max(a, b)) for a, b in zip(x[:-1], x[1:]) if a != b}; V = len(set(x))
    f[pre + "cycle_rank"] = (len(edges) - V + 1) / n; f[pre + "coverage"] = V / K
    M = np.zeros((K, K))
    for a, b in zip(x[:-1], x[1:]): M[a, b] += 1
    rs = M.sum(1, keepdims=True); P = np.divide(M, rs, out=np.zeros_like(M), where=rs > 0); pi = rs[:, 0] / rs.sum()
    with np.errstate(divide="ignore", invalid="ignore"): H = -np.nansum(np.where(P > 0, P * np.log(P), 0), 1)
    f[pre + "entropy_rate"] = float((pi * H).sum())
    act = rs[:, 0] > 0; ev = np.sort(np.abs(np.linalg.eigvals(P[np.ix_(act, act)])))[::-1]
    f[pre + "spectral_gap"] = float(1 - ev[1]) if len(ev) > 1 else np.nan
    dg = np.bincount([a for e in edges for a in e], minlength=K); f[pre + "hub_share"] = dg.max() / max(1, 2 * len(edges))
    return f


rows, cache = [], {}
for m, tag in TAGS.items():
    tok = load_tokenizer(tag.replace("__", "/"), local_files_only=False)
    for ds, pat in SRC.items():
        fp = A + pat.format(t=tag)
        if not os.path.exists(fp): continue
        for l in open(fp):
            r = json.loads(l); g = r["gen_ids"]; te = None
            txt = tok.decode(g, skip_special_tokens=False); k = txt.find("</think>"); txt = txt[:k] if k >= 0 else txt
            ss = sentences(txt)
            if len(ss) < 20: continue
            E = emb.encode(ss, batch_size=256, normalize_embeddings=True, convert_to_numpy=True)
            st = r.get("still_thinking"); st = st if isinstance(st, bool) else str(st) == "True"
            cache[(m, ds, r["pid"])] = E; rows.append(dict(model=m, ds=ds, pid=r["pid"], still_thinking=st, **feats_trace(E)))
    print(f"{m}: {sum(1 for k in cache if k[0] == m)} traces embedded", flush=True)
F = pd.DataFrame(rows)
for m in TAGS:                                                          # per-model K-means over all of that model's sentences
    keys = [k for k in cache if k[0] == m]
    if not keys: continue
    X = np.vstack([cache[k] for k in keys])
    for K in (16, 32):
        km = MiniBatchKMeans(K, random_state=0, n_init=3, batch_size=4096).fit(X)
        for k in keys:
            for c, v in walk_feats(km.predict(cache[k]), K, f"k{K}_").items(): F.loc[(F.model == m) & (F.ds == k[1]) & (F.pid == k[2]), c] = v
F.to_csv(A + "analysis/exp119_feats.csv", index=False)

# ---------------------------------------------------------------- labels
MV2 = {(json.loads(l)["model"], json.loads(l)["pid"]): json.loads(l) for l in open(A + "vast_packages/mv2/mv2_pool.jsonl")}
HF = {m: t.replace("__", "/") for m, t in TAGS.items()}
F["y"] = np.nan; F["acc_orig"] = np.nan
for i, r in F.iterrows():
    if r.ds == "mv2":
        lab = MV2.get((HF[r.model], r.pid), {}).get("is_robust")
        if lab is not None: F.at[i, "y"] = int(lab)
for m in ("Qwen", "allenai"):
    L = pd.read_csv(A + f"hard_pool/set100_labels_{TAGS[m]}.csv").set_index("pid")
    for i, r in F[(F.model == m) & (F.ds == "old")].iterrows():
        if r.pid in L.index and L.loc[r.pid, "label_forced"] in ("robust", "non_robust"):
            F.at[i, "y"] = int(L.loc[r.pid, "label_forced"] == "robust"); F.at[i, "acc_orig"] = L.loc[r.pid, "acc_orig_forced"]
FEATS = [c for c in F.columns if c not in ("model", "ds", "pid", "still_thinking", "y", "acc_orig", "n_sent")]


def resid(d, col, cov):
    out = pd.Series(np.nan, index=d.index)
    for m, g in d.groupby("model"):
        ok = g[col].notna() & g[cov].notna()
        if ok.sum() >= 5 and g.loc[ok, cov].std() > 0:
            b = np.polyfit(g.loc[ok, cov], g.loc[ok, col], 1); out[g.index[ok]] = g.loc[ok, col] - np.polyval(b, g.loc[ok, cov])
    return out


def a(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return roc_auc_score(y[ok], x[ok]) if ok.sum() >= 6 and 0 < y[ok].sum() < ok.sum() else np.nan


F["log_n"] = np.log(F.n_sent); res = []
for f in FEATS:
    r = dict(feature=f)
    for name, d in (("mv2_all", F[(F.ds == "mv2") & F.y.notna()]), ("mv2_unfinished", F[(F.ds == "mv2") & F.y.notna() & F.still_thinking]),
                    ("old", F[(F.ds == "old") & F.y.notna()])):
        if not len(d): continue
        rk = d.groupby("model")[f].rank(pct=True); r[f"{name}_pooled"] = a(rk.values, d.y.values)
        r[f"{name}_lenctl"] = a(resid(d, f, "log_n").values, d.y.values)
        if name == "mv2_all":
            for m, g in d.groupby("model"): r[f"mv2_{m}"] = a(g[f].values, g.y.values)
        if name == "old": r["old_diffctl"] = a(resid(d, f, "acc_orig").values, d.y.values)
    res.append(r)
R = pd.DataFrame(res).set_index("feature"); R.to_csv(A + "analysis/exp119_auc.csv")
pd.set_option("display.width", 260); pd.set_option("display.max_rows", 200)
print(f"\nlabelled traces: main-v2 {int(((F.ds == 'mv2') & F.y.notna()).sum())} ({int(((F.ds == 'mv2') & (F.y == 0)).sum())} NR), "
      f"old {int(((F.ds == 'old') & F.y.notna()).sum())} ({int(((F.ds == 'old') & (F.y == 0)).sum())} NR); relabel-problem traces {int((F.ds == 'easy').sum())}")
R["strength"] = (R.mv2_all_pooled - .5).abs()
print(R.sort_values("strength", ascending=False).drop(columns="strength").round(2).to_string())
print("\nnull: with", len(FEATS), "features, |AUC-.5| >= .25 on main-v2 (n ~43) is expected for roughly", round(len(FEATS) * 0.04, 1), "features by chance")
