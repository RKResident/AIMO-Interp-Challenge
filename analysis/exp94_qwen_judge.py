"""exp94 (user 2026-10-31, idea 1b): Qwen as a JUDGE of other models' problems. exp92: Qwen's v7 score predicts Olmo's labels (AUC 0.90 / 0.83).
Here: Qwen's greedy 6144 traces + submission features on the problems Skywork / DeepSeek are labelled on (old main_v2 + olympiad rows, and the
olympiad redo), scored with Qwen's v7 recipe (Qwen's own refs / threshold); AUC vs THAT model's labels, next to the model's own recipe.
Labels: old rows (U, robust flag) and the olympiad redo (ec2_results/big/out/label_oly: our pipeline, round 1; damaged problems only) with
the friend's greedy labels kept for undamaged olympiad problems."""
import json, re, numpy as np, pandas as pd

src = open("/home/sabyasachi19/PycharmProjects/AimoAGAIN/analysis/exp91_generalization.py").read()
exec(src[:src.index('print("=" * 110 + "\\nS1')])
norm = lambda s: re.sub(r"\s+", " ", str(s)).strip()
JP = {json.loads(l)["pid"]: json.loads(l) for l in open(A + "vast_packages/screen/judge_pool.jsonl")}
BYTXT = {norm(d["problem"]): p for p, d in JP.items()}
JF = pd.DataFrame([json.loads(l) for l in open(A + "ec2_results/a10g/judge/feats_Qwen__Qwen3.5-4B.jsonl")]).set_index("pid")
refQ = ref_of("Qwen", "R2"); PQ = Pm(JF, refQ); _, sQ = calls(BUILDS["v7"]["Qwen"], JF, refQ); JF["qscore"] = sQ
print(f"Qwen judge traces: {len(JF)} | still thinking at 6144: {JF.still_thinking.mean():.0%}")


def redo_labels(tag):
    """our pipeline's labels for the redone olympiad problems (round 1), both scoring rules"""
    R = [json.loads(l) for l in open(A + f"ec2_results/big/out/label_oly/{tag}.jsonl")]; by = {}
    for r in R: by.setdefault(r["pid"], {})[r["ptype"]] = r["samples"]
    out = {}
    for p, d in by.items():
        if "original" not in d or len(d) < 6: continue
        row = {}
        for k, nm in (("correct", "pure"), ("correct_any", "forced")):
            o = np.mean([s[k] for s in d["original"]]); md = o - min(np.mean([s[k] for s in d[t]]) for t in d if t != "original")
            row[nm] = "unsolved" if o == 0 else "robust" if md <= .10 else "non_robust" if md >= .25 else "ambiguous"
        out[p] = row
    return out


for m, tag in (("Skywork", "Skywork__Skywork-OR1-Math-7B"), ("deepseek", "deepseek-ai__DeepSeek-R1-0528-Qwen3-8B")):
    O = OLD[m].copy(); O["jpid"] = [i.split("_")[-1] if i.split("_")[-1].startswith("oly") else BYTXT.get(norm(t)) for i, t in zip(O.id, O.problem)]
    refm = ref_of(m, "R1"); _, so = calls(BUILDS["v7"][m], O, refm); O["own"] = so
    O = O[O.jpid.isin(JF.index)]; y = O.y.values; q = JF.loc[O.jpid, "qscore"].values
    print(f"\n=== {m}: OLD labelled rows with a Qwen judge trace: {len(O)} ({y.sum()} robust)")
    print(f"   own v7 recipe AUC {auc(O.own.values, y):.2f} (in-sample: these rows screened it) | Qwen judge AUC {auc(q, y):.2f} | "
          f"mean of the two percentiles {auc(pd.Series(O.own.values).rank(pct=True).values + pd.Series(q).rank(pct=True).values, y):.2f}")
    for f in ("jmp", "lble", "pmle", "cfc", "fa", "chk"): print(f"      Qwen's {f}: AUC {auc(JF.loc[O.jpid, f].values.astype(float), y):.2f}")
    # the olympiad rows among them, against their CLEAN labels (redo where the 16k cap did damage): own recipe vs Qwen judge
    FR0 = pd.read_csv(A + f"extra_olympiad_problems_dataset/olympiad_labels_greedy/{tag}/labels.csv").set_index("pid"); RD0 = redo_labels(tag)
    Oo = O[O.jpid.isin(FR0.index)].copy(); Oo["clean"] = [RD0[p]["forced"] if p in RD0 else FR0.loc[p, "label"] for p in Oo.jpid]
    k = Oo.clean.isin(["robust", "non_robust"]).values; yc = (Oo.clean[k] == "robust").astype(int).values
    if k.sum() >= 6 and 0 < yc.sum() < k.sum():
        print(f"   olympiad rows with own features: {len(Oo)}, labels changed by the redo {int((Oo.clean != np.where(Oo.y == 1, 'robust', 'non_robust')).sum())} | vs CLEAN labels (n={k.sum()}, {yc.sum()} robust): "
              f"own v7 AUC {auc(Oo.own.values[k], yc):.2f} acc {((Oo.own.values[k] > 0) == yc).mean():.2f} | Qwen judge AUC {auc(JF.loc[Oo.jpid[k], 'qscore'].values, yc):.2f}")
    # olympiad redo (our pipeline) + the friend's labels where the 16k cap did no damage
    FR = pd.read_csv(A + f"extra_olympiad_problems_dataset/olympiad_labels_greedy/{tag}/labels.csv").set_index("pid")
    RD = redo_labels(tag)
    for lab in ("forced", "pure"):
        lab_ = {}
        for p in FR.index:
            if p in RD: lab_[p] = RD[p][lab]
            elif FR.loc[p, "label"] == "robust" or (FR.loc[p, "label"] == "non_robust" and FR.loc[p, "capped"] == 0): lab_[p] = FR.loc[p, "label"]
        S = pd.Series(lab_); S = S[S.isin(["robust", "non_robust"]) & S.index.isin(JF.index)]; yy = (S == "robust").astype(int).values
        redo_only = np.array([p in RD for p in S.index])
        print(f"   olympiad ({lab}; redo labels for {redo_only.sum()} problems, friend's clean labels for {(~redo_only).sum()}): n={len(S)} ({yy.sum()} robust) | "
              f"Qwen judge AUC {auc(JF.loc[S.index, 'qscore'].values, yy):.2f}" + (f" | redo rows only AUC {auc(JF.loc[S.index[redo_only], 'qscore'].values, yy[redo_only]):.2f} (n={redo_only.sum()})" if redo_only.sum() >= 6 and 0 < yy[redo_only].sum() < redo_only.sum() else ""))
