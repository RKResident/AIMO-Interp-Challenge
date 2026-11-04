"""exp51b: candidate recipes with the lookback (attention) features, built like build_submission_artifacts.py: per-model percentile
references = main_v2 values at 6144, signs +1 (all point to robust on main_v2), threshold = median of the main_v2 scores; scored one
problem at a time on the olympiad rows (main_v2 references and threshold, nothing fitted on olympiad)."""
import pickle, sys
import numpy as np
sys.path.insert(0, "/home/sabyasachi19/PycharmProjects/AimoAGAIN/submission_core")
from spurious_features import ensemble_score
A = "/home/sabyasachi19/PycharmProjects/AimoAGAIN/analysis/"
MVb = pickle.load(open(A + "exp47b_mainv2_values.pkl", "rb")); MVa = pickle.load(open(A + "exp51_attention.pkl", "rb")); D = pickle.load(open(A + "exp47_features.pkl", "rb"))
HFM = {"deepseek-ai__DeepSeek-R1-0528-Qwen3-8B": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "Skywork__Skywork-OR1-Math-7B": "Skywork/Skywork-OR1-Math-7B",
       "Qwen__Qwen3.5-4B": "Qwen/Qwen3.5-4B", "allenai__Olmo-3-7B-Think": "allenai/Olmo-3-7B-Think"}
mm = np.array([HFM[m] for m in MVb["model"]]); ym = MVb["y"]; y, model, indep = D["y"], D["model"], D["indep"]
MV = {"fa": MVb["M"]["4 output|forced p1 @6144"], "chk": MVb["M"]["6 text|check share"], "fas": MVb["M"]["4 output|confidence slope over budgets"],
      "lble": MVa["M"]["lookback late - early"], "pmle": MVa["M"]["prompt mass late - early"], "lbq": MVa["M"]["lookback ratio, last quarter"]}
OL = {"fa": D["F"]["4 output|forced p1 @6144"], "chk": D["F"]["6 text|check share"], "fas": D["F"]["4 output|confidence slope over budgets"],
      "lble": MVa["O"]["lookback late - early"], "pmle": MVa["O"]["prompt mass late - early"], "lbq": MVa["O"]["lookback ratio, last quarter"]}
C = {"confidence + check share + slope (0.71 on the leaderboard)": ["fa", "chk", "fas"],
     "+ lookback late - early": ["fa", "chk", "fas", "lble"], "+ prompt mass late - early": ["fa", "chk", "fas", "pmle"],
     "+ lookback last quarter": ["fa", "chk", "fas", "lbq"], "+ lookback late - early + prompt mass late - early": ["fa", "chk", "fas", "lble", "pmle"],
     "attention only: lookback late - early + prompt mass late - early": ["lble", "pmle"],
     "attention only: all three lookback features": ["lble", "pmle", "lbq"]}
print(f"{'candidate':66s} {'main_v2 in-sample bal':>21s} {'olympiad all acc/bal':>21s} {'independent 49 acc/bal (calls)':>32s}")
for nm, fs in C.items():
    rec = dict(features=fs, signs={f: 1 for f in fs}, ref={f: {m: sorted(v for v in MV[f][mm == m] if np.isfinite(v)) for m in set(mm)} for f in fs})
    S = np.array([ensemble_score({f: MV[f][i] for f in fs}, rec, mm[i]) for i in range(len(mm))]); thr = float(np.quantile(S, .5))
    pm = S > thr; bm = 0.5 * (pm[ym == 1].mean() + (~pm[ym == 0]).mean())
    so = np.array([ensemble_score({f: OL[f][i] for f in fs}, rec, model[i]) for i in range(len(y))]); po = so > thr
    res = []
    for k in (np.ones(len(y), bool), indep):
        p, yy = po[k], y[k] == 1; res.append((np.mean(p == yy), 0.5 * (p[yy].mean() + (~p[~yy]).mean()), int(p.sum()), int(k.sum())))
    print(f"{nm:66s} {bm:21.2f} {res[0][0]:13.2f} / {res[0][1]:.2f} {res[1][0]:18.2f} / {res[1][1]:.2f} ({res[1][2]}/{res[1][3]})")
