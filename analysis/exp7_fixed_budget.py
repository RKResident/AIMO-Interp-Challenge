"""Exp 7: remove the length leak by construction.
 (a) budget b in {1024,2048,4096,8192}: keep only traces still THINKING at b, compute every feature from the first b tokens only.
 (b) capped traces only (all 16384 tokens, never finished): features from the whole trace.
agg and main_v2 are evaluated separately; main_v2 grouped by problem. No answer-digit features."""
import os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from common import *
sys.path.insert(0, str(ROOT))
import extract_features as ef
allm, _ = load_all()
MARK = {"wait": r"\bwait\b", "alternatively": r"\balternatively\b", "hmm": r"\bhmm+\b", "but": r"\bbut\b", "actually": r"\bactually\b",
        "maybe": r"\b(maybe|perhaps)\b", "check": r"\b(check|verify|double-check|recheck|confirm)\w*", "mistake": r"\b(mistake|error|wrong|incorrect|oops)\b",
        "restart": r"(start over|from scratch|different approach|another approach|let me try|try again|reconsider|rethink)",
        "therefore": r"\b(therefore|thus|hence)\b", "assume": r"\b(assume|suppose|let's say|let us)\b",
        "answer_is": r"(the answer is|final answer|answer would be|so the answer)", "equals": r"=", "question": r"\?", "guess": r"(guess|probably|likely|seems|looks like)"}
gens = {}
for m in MODELS:
    for l in open(OUT / m / "generations.jsonl"):
        r = json.loads(l); gens[r["id"]] = r["gen_ids"]
toks = {}
BUD = [1024, 2048, 4096, 8192, "cap"]
recs = {b: [] for b in BUD}
for k, r in enumerate(allm.itertuples()):
    fj = force_json(r.model, r.id); tf = np.load(OUT / r.model / "tf" / f"{r.id}.npz")
    ap = OUT / r.model / "attn" / f"{r.id}.npz"; at = np.load(ap) if ap.exists() else None
    if r.model not in toks: toks[r.model] = ef.load_tokenizer(HF[r.model])
    L = fj["L"]; gen = gens[r.id]; capped = r.stop == "cap"
    ce = list(tf["cut_eff"]); ae = list(at["cut_eff"]) if at is not None else []
    for b in BUD:
        bb = len(gen) if b == "cap" else b
        ok = capped if b == "cap" else (L > b)
        if not ok:
            recs[b].append(None); continue
        d = {}
        lp, en, t1 = (tf[x][:bb].astype(np.float32) for x in ("gen_logprob", "gen_entropy", "gen_top1"))
        q = np.array_split(en, 4)
        d.update(p_lp_mean=lp.mean(), p_lp_std=lp.std(), p_ent_mean=en.mean(), p_ent_std=en.std(), p_ent_q90=np.quantile(en, .9),
                 p_top1_mean=t1.mean(), p_ent_gt2=(en > 2).mean(), p_ent_lt01=(en < .1).mean(), p_ent_q1=q[0].mean(), p_ent_q4=q[3].mean(),
                 p_ent_slope=np.polyfit(np.linspace(0, 1, len(en)), en, 1)[0], p_lp_q4=np.array_split(lp, 4)[3].mean())
        c = max(ce) if b == "cap" else b                                   # effective cutoff for the forced / hidden / attention features
        for v in "AB":
            rs = fj["results"].get(f"{c}|{v}")
            if rs:
                d[f"f_{v}_ent"] = rs["first_entropy"]; d[f"f_{v}_mlp"] = rs["mean_logprob"] if rs["mean_logprob"] is not None else np.nan
                d[f"f_{v}_p1"] = rs["top"][0]["probs"][0] if rs["top"] else np.nan; d[f"f_{v}_closed"] = float(rs["closed"])
        d["f_AB_agree"] = float(nrm(fj["results"].get(f"{c}|A", {}).get("answer")) == nrm(fj["results"].get(f"{c}|B", {}).get("answer")))
        if c in ce:
            S = tf["cut_last"][ce.index(c)].astype(np.float32); pm = tf["problem_mean"].astype(np.float32)
            nx = np.linalg.norm(S, axis=-1); cs = (S * pm).sum(-1) / (nx * np.linalg.norm(pm, axis=-1) + 1e-9)
            for j in range(len(nx)): d[f"h_norm{j}"] = np.log(nx[j]); d[f"h_cos{j}"] = cs[j]
        if at is not None and c in ae:
            i = ae.index(c)
            for nm, T in (("lmn", at["mean_noforce"][i]), ("lmf", at["mean_force"][i].mean(0))):
                for bi, p in enumerate(np.array_split(np.arange(T.shape[0]), 6)):
                    d[f"{nm}{bi}_m"] = np.nanmean(T[p]); d[f"{nm}{bi}_s"] = np.nanstd(T[p])
            for nm, T in (("ltn", at["topk_noforce"][i]), ("ltf", at["topk_force"][i].mean(0))):
                for bi, p in enumerate(np.array_split(np.arange(T.shape[0]), 6)):
                    d[f"{nm}{bi}_top1"] = np.nanmean(T[p][..., 0]); d[f"{nm}{bi}_top8"] = np.nanmean(T[p][..., :8])
        text = toks[r.model].decode(gen[:bb], skip_special_tokens=True).lower(); w = max(len(text.split()), 1)
        for nm, pat in MARK.items(): d[f"t_{nm}"] = len(re.findall(pat, text)) / w * 1000
        recs[b].append(d)
    if (k + 1) % 60 == 0: print("processed", k + 1, "/", len(allm), flush=True)
pickle.dump(recs, open(A / "exp7_features.pkl", "wb"))

ds = datasets(allm)
FAMS = {"prefix token stats": "p_", "forced answer at b": "f_", "hidden state at b": "h_", "Laplacian at b": ("lmn", "lmf", "ltn", "ltf"), "text markers in prefix": "t_"}
print("\n=== Exp 7: fixed-budget (still-running) subsets and capped-only ===")
for b in BUD:
    idx = np.array([i for i, x in enumerate(recs[b]) if x is not None])
    F = pd.DataFrame([x if x is not None else {} for x in recs[b]])
    rs = {nm: np.intersect1d(rows, idx) for nm, rows in ds.items()}
    desc = "capped traces only (16384 tokens)" if b == "cap" else f"budget {b}: traces still thinking at {b} tokens, features from first {b}"
    print(f"\n--- {desc}:  " + ", ".join(f"{nm} n={len(v)} (robust {int(allm.y.values[v].sum())}, problems {allm.base_id.values[v].size and len(set(allm.base_id.values[v]))})" for nm, v in rs.items()))
    single_auc(F.loc[:, [c for c in F.columns]].reindex(range(len(allm))), allm.assign(), top=8) if False else None
    for nm, pre in FAMS.items():
        cols = [c for c in F.columns if c.startswith(pre)]
        if not cols: continue
        evaluate(nm, F[cols].values.astype(float), allm, C=0.05, rowsets=rs)
    allc = [c for c in F.columns if c.startswith(("p_", "f_", "h_", "t_", "lmn", "lmf", "ltn", "ltf"))]
    evaluate("ALL families (one model)", F[allc].values.astype(float), allm, C=0.02, rowsets=rs)
    # single-column scan inside the subset
    for nmd, rows in rs.items():
        yy = allm.y.values[rows]
        if len(rows) < 8 or len(np.unique(yy)) < 2: continue
        sc = []
        for c in allc:
            x = F[c].values.astype(float)[rows]; okm = np.isfinite(x)
            if okm.sum() >= 8 and len(np.unique(x[okm])) > 1 and len(np.unique(yy[okm])) == 2: sc.append((c, roc_auc_score(yy[okm], x[okm])))
        sc.sort(key=lambda t: -abs(t[1] - .5))
        print(f"    [{nmd}] top single columns: " + "; ".join(f"{c} {a:.2f}" for c, a in sc[:6]) + f"   (of {len(sc)} scanned - expect extremes by chance)")
