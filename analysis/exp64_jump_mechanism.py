"""exp64: what is the 64-token-block backward-jump signal (exp63: middle-layer mean jump 0.70 on unfinished rows, longer -> robust)?
Two local checks on the white-box run's own traces (whitebox_inputs.jsonl greedy_gen_ids, cut at end of thinking / 6144):
  1 looping control  repetition of the trace (share of 8-grams seen earlier; distinct-4-gram ratio; share of blocks whose 8-grams are
                     mostly repeats). AUC of repetition itself; jump AUC after regressing out repetition within model; jump AUC among the
                     less repetitive half; correlation of jump with repetition.
  2 jump targets     for every query block, its middle-layer attention on blocks > 20 back split by what those blocks contain:
                     (a) text copy - the key block shares >= 25% of its 8-grams with the query block (re-reading / looping onto the same
                     text), (b) math - key block with high digit / operator density, (c) prose; and the share of far mass of each kind.
                     Also the share of far mass that lands on the first 4 blocks (problem restatement / setup)."""
import json, os, pickle, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "2"
import numpy as np
import pandas as pd
sys.path.insert(0, "/home/sabyasachi19/PycharmProjects/AimoAGAIN/submission_core")
from spurious_inference import load_tokenizer, find_think_end                          # noqa: E402

ROOT = __import__("pathlib").Path("/home/sabyasachi19/PycharmProjects/AimoAGAIN"); WB = ROOT / "results_whitebox" / "out_whitebox"
INP = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "vast_packages" / "whitebox" / "whitebox_inputs.jsonl")}
W = pickle.load(open(ROOT / "analysis" / "exp61_whitebox.pkl", "rb")); M = W["meta"].reset_index(drop=True)
slug = lambda m: m.replace("/", "__")                                                   # noqa: E731
MATHCH = set("0123456789=+-*/^()[]{}<>\\_|")
TOK = {}


def ngrams(ids, n):
    return [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]


rows = []
for i, r in M.iterrows():
    inp = INP[r.id]; mid = inp["model_id"]
    if mid not in TOK:
        TOK[mid] = load_tokenizer(mid); TOK[mid + "#tid"] = TOK[mid].convert_tokens_to_ids("</think>")
    tok = TOK[mid]; gen = inp["greedy_gen_ids"][:6144]
    te = find_think_end(tok, gen, TOK[mid + "#tid"]); c = min(max(1, te if te is not None else len(gen)), 6144); tr = gen[:c]
    seen, rep = set(), []
    for g in ngrams(tr, 8):
        rep.append(g in seen); seen.add(g)
    rep = np.array(rep, float); d4 = ngrams(tr, 4)
    nb = -(-len(tr) // 64); blocks = [tr[64 * b:64 * b + 64] for b in range(nb)]
    brep = [rep[64 * b:64 * b + 64].mean() if 64 * b < len(rep) else np.nan for b in range(nb)]
    txt = [tok.decode(b) for b in blocks]
    mathd = np.array([sum(ch in MATHCH for ch in t) / max(1, len(t.replace(" ", ""))) for t in txt])
    g8 = [set(ngrams(b, 8)) for b in blocks]
    z = np.load(WB / slug(mid) / f"{r.id}.npz"); G = z["att_graph_block64_layer"].astype(np.float64); L = G.shape[0]
    g = G[L // 3:2 * L // 3].mean(0)[:, 2:]; n = min(g.shape[0], nb)
    mhi = np.median(mathd[:n]) if n else 0
    far = copy = math = prose = setup = 0.0; jumps = []
    for q in range(1, n):
        w = g[q, :q]; s = w.sum()
        if s <= 0: continue
        dist = q - np.arange(q); jumps.append((w * dist).sum() / s)
        for k in np.where(dist > 20)[0]:
            m = w[k] / s; far += m
            ov = len(g8[q] & g8[k]) / max(1, len(g8[k]))
            if ov >= 0.25: copy += m
            elif mathd[k] > mhi: math += m
            else: prose += m
            if k < 4: setup += m
    nq = max(1, n - 1)
    rows.append(dict(id=r.id, model=r.model, dataset=r.dataset, y=int(r.y), unf=bool(r.unf), jump=float(np.mean(jumps)) if jumps else np.nan,
                     rep8=float(rep.mean()), distinct4=len(set(d4)) / max(1, len(d4)), loop_blocks=float(np.nanmean(np.array(brep) > 0.5)),
                     far=far / nq, far_copy=copy / nq, far_math=math / nq, far_prose=prose / nq, far_setup=setup / nq,
                     copy_share=copy / far if far else np.nan, math_share=math / far if far else np.nan, setup_share=setup / far if far else np.nan))
D = pd.DataFrame(rows); U = D[D.unf].reset_index(drop=True); y = U.y.values
rng = np.random.default_rng(0)


def rk(c, df=U): return df.groupby("model")[c].rank(pct=True).values


def auc(s, yy):
    ok = ~np.isnan(s); s, yy = s[ok], yy[ok]; p, n = s[yy == 1], s[yy == 0]
    return float(((p[:, None] > n[None]) + .5 * (p[:, None] == n[None])).mean())


def perm_p(s, a, k=2000):
    null = []
    for _ in range(k):
        yy = y.copy()
        for g in U.model.unique(): m = (U.model == g).values; yy[m] = rng.permutation(yy[m])
        null.append(auc(s, yy))
    return float((np.abs(np.array(null) - .5) >= abs(a - .5) - 1e-12).mean())


print(f"{len(U)} unfinished rows | AUC: higher -> robust (within-model ranks), permutation p within model\n")
print("1 LOOPING CONTROL")
for c in ("jump", "rep8", "distinct4", "loop_blocks"):
    a = auc(rk(c), y); print(f"  {c:12s} AUC {a:.2f} (p {perm_p(rk(c), a):.3f}) | Spearman with jump {np.corrcoef(rk(c), rk('jump'))[0, 1]:+.2f}")
for c in ("rep8", "distinct4", "loop_blocks"):
    s, l = rk("jump"), rk(c); b = np.polyfit(l, s, 1); res = s - np.polyval(b, l); a = auc(res, y)
    print(f"  jump after regressing out {c:12s}: AUC {a:.2f} (p {perm_p(res, a):.3f})")
lo = U.rep8 <= U.groupby("model").rep8.transform("median")
print(f"  jump among the less repetitive half (n={lo.sum()}): AUC {auc(rk('jump', U[lo]), y[lo.values]):.2f} | more repetitive half: "
      f"{auc(rk('jump', U[~lo]), y[~lo.values]):.2f}")
print("\n2 WHERE THE FAR READS LAND (mean per query block; shares of far mass)")
for c in ("far", "far_copy", "far_math", "far_prose", "far_setup", "copy_share", "math_share", "setup_share"):
    a = auc(rk(c), y)
    print(f"  {c:12s} robust mean {U.loc[U.y == 1, c].mean():.4f} | spurious {U.loc[U.y == 0, c].mean():.4f} | AUC {a:.2f} (p {perm_p(rk(c), a):.3f})")
print("\nper model (unfinished): jump AUC / rep8 AUC / far_math AUC / copy_share AUC")
for m, g in U.groupby("model"):
    yy = g.y.values; print(f"  {m:9s} n={len(g):2d}  " + "  ".join(f"{c} {auc(g[c].values.astype(float), yy):.2f}" for c in ("jump", "rep8", "far_math", "copy_share")))
D.to_csv(ROOT / "analysis" / "exp64_jump_mechanism.csv", index=False)
