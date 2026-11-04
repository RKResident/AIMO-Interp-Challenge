"""exp117 (user 2026-11-02): exp110 re-run on the 32k RELABEL (ec2_results/label_new32k, both boxes merged): label rules
  natural = official style (natural answers at 32k) | forced32k = natural-or-forced at 32k | at16k = natural-or-forced at 16k (from the same samples).
exp110 (user 2026-11-01): the confidence-matched new labels (out/label_new; 8 original + 8 per perturbation, 5 types, cap 6144) -> official
rule (max_drop <= 0.10 robust, >= 0.25 non-robust, original accuracy 0 -> robust; forced = correct_any, pure = correct). Candidates scored
exactly as on the leaderboard / train-main-v2 (exp109): signs from OUR original labels, thresholds = our pool medians; v8's recipe (Olmo: both
router branches), the shipped overrides (ent4096 for Qwen, segment-25% direction for Olmo = nomination candidate 2), all-robust, and the
exp105 pre-registered leads. Inputs (sync first): ec2_results/label_new, feats_new, wb2_new, wb_new, entropy_easy, entropy_pool.
  python analysis/exp110_newlabels.py"""
import json, os, numpy as np, pandas as pd
src = open("/home/sabyasachi19/PycharmProjects/AimoAGAIN/analysis/exp109_mainv2.py").read()
exec(src[:src.index("rows, corr = [], []")])


def new_labels(tag):
    by = {}
    for f in (A + f"ec2_results/label_new32k/{tag}.jsonl", A + f"ec2_results/label_new32k_a10g/{tag}.jsonl"):
        if not os.path.exists(f): continue
        for l in open(f): r = json.loads(l); by.setdefault(r["pid"], {})[r["ptype"]] = r["samples"]
    if not by: return {}
    out = {}
    for p, v in by.items():
        if "original" not in v or len(v) < 6: continue
        row = {}
        G = {"natural": lambda s: s.get("correct"), "forced32k": lambda s: s.get("correct_any"), "at16k": lambda s: s.get("at", {}).get("16384", {}).get("correct_any")}
        for nm, g in G.items():
            o = np.mean([bool(g(s)) for s in v["original"]]); md = o - min(np.mean([bool(g(s)) for s in v[t]]) for t in v if t != "original")
            row[nm] = 1 if o == 0 else 1 if md <= .10 else 0 if md >= .25 else None
            row[f"acc_{nm}"] = o
        out[p] = row
    return out


rows = []
for m, tag in TAGS.items():
    NL = new_labels(tag)
    if not NL: print(f"{m}: no new labels synced"); continue
    ship = dict(np.load(A + f"analysis/wb2_ship/wb2_ship_{tag}.npz"))
    P = table(tag, "wb", "wb2", "entropy", ship)
    ent_dir = "entropy_easy"; M = {}
    for wb, wb2 in (("wb_new", "wb2_new"),):
        if os.path.exists(A + f"ec2_results/{wb2}/{tag}.jsonl"):
            M = table(tag, wb if os.path.exists(A + f"ec2_results/{wb}/{tag}.jsonl") else "wb2_new", wb2, ent_dir, ship)
    for p in list(M):                                                  # pool pids: entropy features from entropy_pool
        if not any(k.startswith("fc_") for k in M[p]):
            try: M[p].update(feats_one(np.load(A + f"ec2_results/entropy_pool/{tag}/{p}.npz")))
            except FileNotFoundError: pass
    F = {json.loads(l)["pid"]: json.loads(l) for l in open(A + f"ec2_results/feats_new/{tag}.jsonl")} if os.path.exists(A + f"ec2_results/feats_new/{tag}.jsonl") else {}
    lab = pd.Series(labels(m)["forced"]); lab = lab[lab.index.isin(P)]; y = lab.values
    pm = REC["per_model"][HF[m]]; rec = {**REC, **pm}; thr = REC["threshold_by_model"][HF[m]]
    SEG = json.load(open("/home/sabyasachi19/PycharmProjects/aimo/getting-started/solutions/spurious-permodel-v8-ent4096-olmoseg25/spurious_artifacts/recipe.json"))["per_model"]
    for rule in ("natural", "forced32k", "at16k"):
        ids = [p for p in NL if NL[p][rule] is not None and p in M]
        if len(ids) < 4: print(f"{m} {rule}: only {len(ids)} usable rows"); continue
        yt = np.array([NL[p][rule] for p in ids]); fin = np.array([not M[p].get("still_thinking", True) for p in ids])
        easy = np.array([NL[p]["acc_natural"] >= .875 for p in ids])               # difficulty control: originals solved >= 7/8 (natural, 32k)
        def ev(name, calls):
            c = np.where(fin, 1, np.asarray(calls, int))
            bal = float(.5 * ((c[yt == 1] == 1).mean() + (c[yt == 0] == 0).mean())) if 0 < yt.sum() < len(yt) else np.nan
            rows.append(dict(model=m, rule=rule, candidate=name, n=len(ids), n_robust=int(yt.sum()), acc=float((c == yt).mean()), bal=bal,
                             acc_easy=float((c[easy] == yt[easy]).mean()) if easy.any() else np.nan, n_easy=int(easy.sum()),
                             acc_hard=float((c[~easy] == yt[~easy]).mean()) if (~easy).any() else np.nan))
        ev("all robust", np.ones(len(ids)))
        if F and "router" in pm:
            for br in ("novel", "familiar"):
                b = pm["router"][br]; s = [score_trace({f: F.get(p, {}).get(f, np.nan) for f in b["features"]}, {**rec, **b}, HF[m]) for p in ids]; ev(f"v8 {br} branch", np.array(s) > b["threshold"])
        elif F:
            s = [score_trace({f: F.get(p, {}).get(f, np.nan) for f in pm["features"]}, rec, HF[m]) for p in ids]; ev("v8 recipe", np.array(s) > thr)
        for nm, O in (("override (ent4096)", pm.get("override")), ("override (olmoseg25, nomination 2)", SEG.get(HF[m], {}).get("override") if m == "allenai" else None)):
            if O:
                v = np.array([M[p].get(O["feature"], np.nan) for p in ids], float); ev(nm, np.where(np.isfinite(v), O["sign"] * (v - O["threshold"]) > 0, 1))
        for f in PRE[m]:
            x = np.array([P[p].get(f, np.nan) if isinstance(P[p].get(f), (int, float)) else np.nan for p in P], float)
            xl = np.array([P[p].get(f, np.nan) if isinstance(P[p].get(f), (int, float)) else np.nan for p in lab.index], float); v = np.array([M[p].get(f, np.nan) for p in ids], float)
            if np.isfinite(xl).sum() < 8 or not np.isfinite(v).all(): continue
            sg = 1 if auc(xl[np.isfinite(xl)], y[np.isfinite(xl)]) >= .5 else -1; ev(f, sg * (v - np.nanmedian(x)) > 0)
    acc = [NL[p]["acc_natural"] for p in NL]
    for rule in ("natural", "forced32k", "at16k"):                                  # how much of each label is just difficulty?
        q = [p for p in NL if NL[p][rule] is not None]
        if len(q) > 3 and 0 < sum(NL[p][rule] for p in q) < len(q):
            print(f"  {m} {rule}: AUC(original natural accuracy -> robust) = {auc(np.array([NL[p]['acc_natural'] for p in q]), np.array([NL[p][rule] for p in q])):.2f} (n {len(q)}; 0.5 = labels independent of difficulty)")
    print(f"{m}: {len(NL)} newly labelled problems | original accuracy median {np.median(acc):.2f} | labels R/NR/amb "
          f"{sum(NL[p]['natural'] == 1 for p in NL)}/{sum(NL[p]['natural'] == 0 for p in NL)}/{sum(NL[p]['natural'] is None for p in NL)} (natural)")
T = pd.DataFrame(rows); T.to_csv(A + "analysis/exp117_relabel32k.csv", index=False)
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
print(T.pivot_table(index=["model", "candidate"], columns="rule", values="acc", aggfunc="first").round(2).to_string())
print("\n=== balanced accuracy (natural rule)"); print(T[T.rule == "natural"].set_index(["model", "candidate"])[["n", "n_robust", "acc", "bal"]].round(2).to_string())
print("\n=== difficulty-controlled (natural rule): accuracy on originals solved >= 7/8 vs the rest")
print(T[T.rule == "natural"].set_index(["model", "candidate"])[["n_easy", "acc_easy", "acc_hard"]].round(2).to_string())
print(T.groupby(["model", "rule"])[["n", "n_robust"]].first().to_string())
