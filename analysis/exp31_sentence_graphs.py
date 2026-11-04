"""exp31: the reasoning-loops paper's per-SENTENCE construction, approximated from the saved waypoints.
The waypoint run saved its exact token ids (saved_trajectories_v2_data/<model>/generations.jsonl), and waypoint j (j >= 1) is the
hidden state at generated token 16j - 2 (forward step 16j; the last one at the final step). So every waypoint falls in a known
sentence. A sentence's state = mean of the waypoints inside it (0-2 single-token samples for most sentences: an approximation of
the paper's full sentence mean); sentences with no waypoint are skipped. Variant "tokcorr": each waypoint minus this model's mean
state for its token class (digit / punctuation / newline / latex / word / other), since a single token's state depends heavily on
what token it is. Then K-means per model and the exp30 graph metrics, plus sentence-to-sentence cosine (the paper's cross-step
similarity). Fixed token windows; scored like exp29/30 and checked against consecutive-step similarity."""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "2"
from common import *
import json, re, glob
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
from state_graph import graph_feats

allm, feats = load_all(); y = allm.y.values
rows = datasets(allm)["main_v2"]; grp = allm.grp.values; prob = allm.base_id.values
fin = (allm.stop.values == "eos").astype(float); lgn = np.log(allm.n_gen.astype(float).values)
KS, WINDOWS = (32, 64, 128), ((0, 4096), (0, 6144))
TD = ROOT / "saved_trajectories_v2_data"
GEN = {json.loads(l)["id"]: json.loads(l) for f in glob.glob(str(TD / "*" / "generations.jsonl")) for l in open(f)}
END = re.compile(r"[.!?:]\s*$|\n")


def tclass(p):
    if "\n" in p: return 0
    if re.search(r"\d", p): return 1
    if "\\" in p or "$" in p: return 2
    if re.fullmatch(r"\s*[^\w\s]+\s*", p): return 3
    if p.startswith((" ", "Ġ")): return 4
    return 5


TRACE = {}                       # i -> (waypoint states [W, d], waypoint token index [W], sentence id per waypoint, token class per waypoint, n sentences)
cover = []; TOK = {}
for i in rows:
    rid = allm.id.iloc[i]
    f = TD / allm.model.iloc[i] / f"{rid}.npz"
    if not f.exists() or rid not in GEN: continue
    z = np.load(f); ks = sorted([k for k in z.files if k.startswith("layer_")], key=lambda s: int(s.split("_")[1]))
    X = z[ks[3]].astype(np.float32); g = GEN[rid]["gen_ids"]; n = len(g)
    tok = TOK.setdefault(allm.model.iloc[i], Tokenizer.from_file(hf_hub_download(allm.model.iloc[i].replace("__", "/", 1), "tokenizer.json", local_files_only=True)))
    pieces = [tok.decode([t]) for t in g]
    sid = np.cumsum([0] + [bool(END.search(p)) for p in pieces[:-1]])          # sentence index per generated token
    steps = [16 * j for j in range(1, len(X))]; steps[-1] = int(z["forward_steps"]) if int(z["forward_steps"]) % 16 else steps[-1]
    pos = np.array([s - 2 for s in steps]); keep = (pos >= 0) & (pos < n)
    W = X[1:][keep]; pos = pos[keep]
    TRACE[i] = (W, pos, sid[pos], np.array([tclass(pieces[p]) for p in pos]), int(sid[-1]) + 1)
    cover.append((len(set(sid[pos])) / (sid[-1] + 1), n / (sid[-1] + 1)))
print(f"{len(TRACE)} traces | sentences with >= 1 waypoint: {np.mean([c[0] for c in cover]):.0%} | mean tokens per sentence {np.mean([c[1] for c in cover]):.1f}")


def sentence_states(i, mu, cls_mu):
    W, pos, ss, tc, _ = TRACE[i]
    V = W - mu if cls_mu is None else W - cls_mu[tc]
    us = np.unique(ss); S = np.stack([V[ss == u].mean(0) for u in us]); first_tok = np.array([pos[ss == u].min() for u in us])
    return S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-6), first_tok


F = {}
for variant in ("raw", "tokcorr"):
    for mdl in np.unique(allm.model.values[list(TRACE)]):
        ii = [i for i in TRACE if allm.model.iloc[i] == mdl]
        allW = np.concatenate([TRACE[i][0] for i in ii]); allc = np.concatenate([TRACE[i][3] for i in ii]); mu = allW.mean(0)
        cls_mu = None if variant == "raw" else np.stack([allW[allc == c].mean(0) if (allc == c).any() else mu for c in range(6)])
        SS = {i: sentence_states(i, mu, cls_mu) for i in ii}
        for i in ii:   # sentence-to-sentence cosine (paper's cross-step similarity), no clustering
            S, ft = SS[i]
            for a0, b0 in WINDOWS:
                if TRACE[i][1].max() < b0 - 64: continue
                m = (ft >= a0) & (ft < b0); s = S[m]
                if len(s) < 10: continue
                F.setdefault(f"{variant}|-|{b0 // 1024}k|adjacent_sentence_cos", np.full(len(allm), np.nan))[i] = np.mean(np.sum(s[1:] * s[:-1], 1))
        for K in KS:
            allS = np.concatenate([SS[i][0] for i in ii])
            if len(allS) < 4 * K: continue
            km = KMeans(n_clusters=K, n_init=4, random_state=0).fit(allS)
            C = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
            for i in ii:
                S, ft = SS[i]; lab = km.predict(S)
                for a0, b0 in WINDOWS:
                    if TRACE[i][1].max() < b0 - 64: continue
                    m = (ft >= a0) & (ft < b0)
                    if m.sum() < 10: continue
                    for k, v in graph_feats(list(lab[m]), C).items():
                        F.setdefault(f"{variant}|K{K}|{b0 // 1024}k|{k}", np.full(len(allm), np.nan))[i] = v

# consecutive 64-token step similarity on the same traces (exp29's signal), for the redundancy check
adj = np.full(len(allm), np.nan)
for mdl in np.unique(allm.model.values[list(TRACE)]):
    ii = [i for i in TRACE if allm.model.iloc[i] == mdl]
    st = {i: TRACE[i][0][: 256 // 4 * 4][:252].reshape(-1, 4, TRACE[i][0].shape[1]).mean(1) for i in ii}
    mu = np.concatenate(list(st.values())).mean(0)
    for i in ii: s = st[i] - mu; s /= np.linalg.norm(s, axis=1, keepdims=True); adj[i] = np.mean(np.sum(s[1:] * s[:-1], 1))

# ---------------------------------------------------------------- scoring
up = np.unique(prob[rows]); yp = np.array([y[rows][prob[rows] == p][0] for p in up])
fp = np.array([fin[rows][prob[rows] == p].mean() for p in up]); lp_ = np.array([lgn[rows][prob[rows] == p].mean() for p in up])
rng = np.random.default_rng(0)


def perm_auc(x, yy, n=5000):
    rk = rankdata(x); n1 = yy.sum(); n0 = len(yy) - n1; c = n1 * (n1 + 1) / 2
    a = (rk[yy == 1].sum() - c) / (n1 * n0)
    YP = np.array([rng.permutation(yy) for _ in range(n)]).astype(float); nl = (YP @ rk - c) / (n1 * n0)
    return a, float(np.mean(np.abs(nl - .5) >= abs(a - .5)))


def zm(v, r): return pd.Series(v[r]).groupby(allm.model.values[r]).transform(lambda a: (a - a.mean()) / (a.std() + 1e-9)).values


out = []
for nm, v in F.items():
    ok = np.isfinite(v[rows]) & np.isfinite(adj[rows]); r = rows[ok]
    if ok.sum() < 12 or len(np.unique(y[r])) < 2 or len(np.unique(v[r])) < 3: continue
    vz = zgroup(np.where(np.isin(np.arange(len(allm)), r), v, np.nan)[:, None], grp)[:, 0]
    xp = np.array([np.nanmean(np.where(prob[r] == p, vz[r], np.nan)) if (prob[r] == p).any() else np.nan for p in up]); okp = np.isfinite(xp)
    if okp.sum() < 8 or len(np.unique(yp[okp])) < 2: continue
    Cc = np.c_[fp, lp_][okp]; xr = xp[okp] - LinearRegression().fit(Cc, xp[okp]).predict(Cc); apr, ppr = perm_auc(xr, yp[okp])
    a_, b_ = zm(adj, r), zm(v, r); res = b_ - np.polyval(np.polyfit(a_, b_, 1), a_)
    variant, K, win, met = nm.split("|")
    out.append(dict(variant=variant, K=K, window=win, metric=met, rows=int(ok.sum()), pooled=roc_auc_score(y[r], vz[r]),
                    problem_controlled=apr, p_controlled=ppr, corr_with_stepsim=np.corrcoef(a_, b_)[0, 1], auc_after_stepsim=roc_auc_score(y[r], res)))
T = pd.DataFrame(out); T.to_csv(A / "exp31_sentence_graphs.csv", index=False)
pd.set_option("display.width", 250)
r34 = np.array(sorted(TRACE)); print(f"reference: consecutive 64-token step similarity (0-4k) on these rows, pooled AUC {roc_auc_score(y[r34], zm(adj, r34)):.2f}\n")
for variant in ("raw", "tokcorr"):
    t = T[(T.variant == variant) & (T.window == "4k")]
    print(f"=== {variant}, window 0-4k: pooled AUC by K | AUC after removing step similarity (by K) ===")
    a = t.pivot_table(index="metric", columns="K", values="pooled").round(2); b = t.pivot_table(index="metric", columns="K", values="auc_after_stepsim").round(2)
    print(a.join(b, rsuffix="_resid").to_string(), "\n")
print("strongest problem-level controlled (any variant/K/window):")
print(T.assign(d=(T.problem_controlled - .5).abs()).sort_values("d", ascending=False).head(12)[["variant", "K", "window", "metric", "rows", "pooled", "problem_controlled", "p_controlled", "auc_after_stepsim"]].round(3).to_string(index=False))
print(f"\n{len(T)} feature versions; controlled p < 0.05: {int((T.p_controlled < .05).sum())} (~{0.05 * len(T):.0f} expected by chance)")
