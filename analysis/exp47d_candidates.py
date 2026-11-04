"""exp47d: candidate recipes, built the way build_submission_artifacts.py builds them (per-model percentile references = main_v2 values at
6144, equal-weight mean of signed percentiles, threshold = median of the main_v2 scores) and scored one problem at a time on the olympiad
rows. Signs come from main_v2 (all features below point the same way on main_v2 and olympiad)."""
import pickle, sys
import numpy as np
sys.path.insert(0, "/home/sabyasachi19/PycharmProjects/AimoAGAIN/submission_core")
from spurious_features import ensemble_score
A = "/home/sabyasachi19/PycharmProjects/AimoAGAIN/analysis/"
MV = pickle.load(open(A + "exp47b_mainv2_values.pkl", "rb")); D = pickle.load(open(A + "exp47_features.pkl", "rb"))
ym, mm = MV["y"], MV["model"]; HFM = {"deepseek-ai__DeepSeek-R1-0528-Qwen3-8B": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "Skywork__Skywork-OR1-Math-7B": "Skywork/Skywork-OR1-Math-7B",
                                      "Qwen__Qwen3.5-4B": "Qwen/Qwen3.5-4B", "allenai__Olmo-3-7B-Think": "allenai/Olmo-3-7B-Think"}
mm = np.array([HFM[m] for m in mm]); y, model, indep = D["y"], D["model"], D["indep"]
KEY = {"fa": "4 output|forced p1 @6144", "chk": "6 text|check share", "len": "0 shortcuts|-log trace length", "fas": "4 output|confidence slope over budgets",
       "fsat": "4 output|share of saturated budgets", "ver": "6 text|verification per 1k words"}
C = {"check share": ["chk"], "confidence + check share": ["fa", "chk"], "confidence + check share + length": ["fa", "chk", "len"],
     "confidence + check share + slope": ["fa", "chk", "fas"], "confidence + check share + verification": ["fa", "chk", "ver"],
     "confidence + slope + saturated + check share + verification": ["fa", "fas", "fsat", "chk", "ver"]}
print(f"{'candidate':58s} {'main_v2 in-sample bal':>21s} {'olympiad all acc/bal':>21s} {'independent 49 acc/bal (calls)':>32s}")
for nm, fs in C.items():
    rec = dict(features=fs, signs={f: 1 for f in fs}, ref={f: {m: sorted(v for v in MV["M"][KEY[f]][mm == m] if np.isfinite(v)) for m in set(mm)} for f in fs})
    S = np.array([ensemble_score({f: MV["M"][KEY[f]][i] for f in fs}, rec, mm[i]) for i in range(len(mm))]); thr = float(np.quantile(S, .5))
    pm = S > thr; bm = 0.5 * (pm[ym == 1].mean() + (~pm[ym == 0]).mean())
    so = np.array([ensemble_score({f: D["F"][KEY[f]][i] for f in fs}, rec, model[i]) for i in range(len(y))]); po = so > thr
    res = []
    for k in (np.ones(len(y), bool), indep):
        p, yy = po[k], y[k] == 1; res.append((np.mean(p == yy), 0.5 * (p[yy].mean() + (~p[~yy]).mean()), int(p.sum()), int(k.sum())))
    print(f"{nm:58s} {bm:21.2f} {res[0][0]:13.2f} / {res[0][1]:.2f} {res[1][0]:18.2f} / {res[1][1]:.2f} ({res[1][2]}/{res[1][3]})")
