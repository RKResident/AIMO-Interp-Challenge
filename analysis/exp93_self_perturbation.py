"""exp93 (user 2026-10-31, new idea 2): self-perturbation test. For each labelled set100 problem, does the model's answer (greedy, 6144 cap,
forced if unfinished) survive perturbations made by CODE (typos / rename / distractor; deployable) or by Sonnet (labelling perturbations;
upper bound)? Feature = share of versions whose answer equals the original's. AUC vs the new labels (forced / pure, novel / seen), the
"all agree -> robust" rule, and the combination with the v7 score (equal-weight mean of the two within-model percentiles: nothing fitted).
  python exp93_self_perturbation.py <direct_results.jsonl> <model short: Qwen | allenai>"""
import json, sys, numpy as np, pandas as pd

src = open("/home/sabyasachi19/PycharmProjects/AimoAGAIN/analysis/exp91_generalization.py").read()
exec(src[:src.index('print("=" * 110 + "\\nS1')])
path, m = sys.argv[1], sys.argv[2]
R = pd.DataFrame([json.loads(l) for l in open(path)]).set_index("pid")
CODE = ["code_typos", "code_rename", "code_distract"]; SON = ["sonnet_typos", "sonnet_rename", "sonnet_distract"]
for k, vs in (("agree_code", CODE), ("agree_sonnet", SON), ("agree_all", CODE + SON)):
    R[k] = [np.mean([a[v] for v in vs if v in a]) if any(v in a for v in vs) else np.nan for a in R.agree]
for v in CODE + SON: R[v] = [float(a.get(v, np.nan)) for a in R.agree]
R["orig_correct"] = [d["versions"]["original"]["answer"] == d["gold"] for _, d in R.iterrows()]
rng = np.random.default_rng(0)


def pv(s, y, n=3000): a = abs(auc(s, y) - .5); return float(np.mean([abs(auc(s, rng.permutation(y)) - .5) >= a - 1e-12 for _ in range(n)]))


print(f"{m}: {len(R)} problems | original answer correct at 6144: {R.orig_correct.mean():.0%} | mean agreement code {R.agree_code.mean():.2f}, sonnet {R.agree_sonnet.mean():.2f}")
ref = ref_of(m, REG[m])
for lab in ("forced", "pure"):
    d = new_df(m, lab); d = d[d.pid.isin(R.index)].set_index("pid"); y = d.y.values; nov = d.novel.values
    print(f"\n  {lab}: {len(d)} labelled ({y.sum()} robust; {nov.sum()} novel)")
    for k in ["agree_code", "agree_sonnet", "agree_all"] + CODE + SON:
        s = R.loc[d.index, k].values.astype(float)
        print(f"     {k:16s} AUC {auc(s, y):.2f} (p {pv(s, y):.3f}) | novel {auc(s[nov], y[nov]):.2f} seen {auc(s[~nov], y[~nov]):.2f}")
    for k in ("agree_code", "agree_sonnet"):
        c = R.loc[d.index, k].values >= 1.0
        print(f"     rule '{k} = 1 -> robust': accuracy {(c == y).mean():.2f} (calls {c.mean():.0%} robust)")
    _, sv = calls(BUILDS["v7"][m], d.reset_index().assign(mink=[MINK.get(p, np.nan) for p in d.index]), ref)
    cv = sv > 0; print(f"     v7 alone: accuracy {(cv == y).mean():.2f}, AUC {auc(sv, y):.2f}")
    for k in ("agree_code", "agree_sonnet"):
        a = R.loc[d.index, k].values.astype(float)
        comb = (pd.Series(sv).rank(pct=True).values + pd.Series(a).rank(pct=True).values) / 2
        cc = comb > np.median(comb)
        print(f"     v7 + {k} (equal-weight percentiles, median cut): AUC {auc(comb, y):.2f}, accuracy {(cc == y).mean():.2f}")
