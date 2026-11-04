"""Artifacts for the leaderboard submissions, from main_v2 only (43 rows, 4 models; no gpt proxies).
Each recipe gets the smallest token cap its features need (exp37):
  fa  forced confidence     reads up to min(6144, thinking)      -> 6144
  ge  kNN global efficiency reads the first 4096 tokens            -> 4096
  pearl                     prompt only                            -> 0
  seg / traj / ab / len     read the whole trace                   -> 16384 (the training cap)
Greedy => a trace capped at c is the first c tokens of our 16k trace, so every feature is computed on the main_v2 traces truncated at
the recipe's cap (identical to the saved values for fa / ge / pearl; recomputed for seg, traj, len). Exception: ab's fraction-of-thinking
cutoffs move under a cap and those forced answers were not saved, so ab keeps its 16k values (approximation, flagged in the recipe).
For each recipe: features, signs (direction of the main_v2 controlled AUC, exp34), per model the sorted main_v2 values of every feature
at that cap (the submission scores a value by its percentile among them), and one threshold = the q-quantile of the pooled training
score (q = 0.5: the test set is a 50/50 split; 'mv2rate': main_v2's own 28% non-robust). The kNN feature also needs each model's mean
layer state (exp32 centres states on it) -> mu_<model>.npy. Writes into each solution directory of the getting-started repo."""
import copy, json, shutil, sys
from common import *

SOL = Path("/home/sabyasachi19/PycharmProjects/aimo/getting-started/solutions")
CORE = ROOT / "submission_core"          # solution.py, spurious_inference.py, spurious_features.py
CODE = ["solution.py", "spurious_inference.py", "spurious_features.py", "spurious_papers.py", "wb_battery.py", "wb2_log.py"]
P2 = ROOT / "extraction_out_generations_p2" / "extraction_out_generations"
allm, _ = load_all(); y = allm.y.values; rows = datasets(allm)["main_v2"]; model = allm.model.values
F28 = pickle.load(open(A / "exp28_features_cache.pkl", "rb")); F32 = pickle.load(open(A / "exp32_features.pkl", "rb"))
F38 = pickle.load(open(A / "exp38_features.pkl", "rb"))
F44 = pickle.load(open(A / "exp44_features.pkl", "rb"))
F51 = pickle.load(open(A / "exp51_attention.pkl", "rb"))["M"]
F51O = pickle.load(open(A / "exp51_attention.pkl", "rb"))["O"]                            # the same attention features on the olympiad rows
_W61 = pickle.load(open(A / "exp61_whitebox.pkl", "rb"))                                   # white-box run features, main_v2 + olympiad rows
F33 =pickle.load(open(A / "exp33_features.pkl", "rb")); F29 = pickle.load(open(A / "exp29_features.pkl", "rb"))
NAMES = {"fa": "meta:fa_b6144_B_p1", "ab": "ab:dent_max", "seg": "subsample:seg25:f11", "traj": "traj:k1|full:late_vs_early_cos",
         "pearl": "pearl:mean_state:area", "ge": "e32:step64|k5|4k|global_efficiency", "len": "len",
         "tda": "exp38:C_layer|H0_betti_spread|L3", "cr": "exp33:step64|K64|4k|cluster_cycle_rank_per_step",
         "chk": "exp29:text:full:check_share", "fas": "exp44:forced|p1 slope over budgets",
         "fsat": "exp44:forced|share of budgets saturated", "pcr": "exp44:trajectory|PC1 share 75% / 25%",
         "lble": "exp51:lookback late - early", "pmle": "exp51:prompt mass late - early",
         "nrk": "exp61:reasoning ablation 'no reasoning': answer kept", "f1k": "exp61:reasoning ablation 'first 1024 only': answer kept",
         "cfc": "exp61:cf auto: state cosine, last quarter", "cfl": "exp61:cf auto: trace log-lik change", "nzf": "exp61:noise flip rate",
         "jmp": "exp63:block64 middle: mean jump (segments)", "tha": "exp70:TOHA win@2048, selected heads (DeepSeek)",
         "pp_rhythm": "exp81:rhythm WAAD->FAI peak co-occurrence excess (2510.13554)", "pp_anchor": "exp81:Thought-Anchors suppression log-KL, top - random (2506.19143)",
         "pp_minkpp": "exp81:Min-K%++ k=20 on the problem (2404.02936)", "pp_lid": "exp81:LID vs reference set (2402.18048)",
         "pp_li": "exp81:layer-wise usable information LI (2412.10246)", "pp_lara": "exp81:LaRA directional collapse, middle layers (2605.29888)",
         "pp_cie": "exp81:CIE-style entropic GW discrepancy (2605.25603)", "pp_recall": "exp81:competition-recall direction, last token, trace mean (exp78)"}
T34 = pd.read_csv(A / "exp34_mainv2_all_features.csv").set_index("feature")
SIGN = {k: (1 if (T34.controlled[v] if v in T34.index else .76) > .5 else -1) for k, v in NAMES.items()}   # len: higher -log n_gen -> robust
SIGN["cr"] = SIGN["chk"] = 1                                        # exp33 / exp29: higher -> robust (0.86 / 0.81 controlled)
SIGN["nrk"] = SIGN["f1k"] = SIGN["cfc"] = SIGN["cfl"] = SIGN["nzf"] = -1   # exp61, unfinished traces: all point to spurious (0.29-0.35; flips 0.65 for fewer)
SIGN["lble"] = SIGN["pmle"] = 1                                   # exp51: later spans attend more to the prompt -> robust (main_v2 0.59 / 0.57, olympiad 0.75 / 0.86)
SIGN["fas"] = SIGN["fsat"] = SIGN["pcr"] = 1                     # exp44: all higher -> robust (0.77 / 0.70 / 0.71 controlled)
SIGN["tha"] = 1                                                     # exp70: score = - mean MST edge of the selected heads (short forest = anchored -> robust)
SIGN["jmp"] = 1                                                     # exp63 / exp64: longer backward reads (64-token blocks, middle layers) -> robust (0.70 unfinished)
SIGN["tda"] = 1                                                     # exp35 / exp38: higher H0 Betti spread -> robust (0.77 controlled at this layer)
NAMES_WB = {"nrk": "reasoning ablation 'no reasoning': answer kept", "f1k": "reasoning ablation 'first 1024 only': answer kept",
            "cfc": "cf auto: state cosine, last quarter", "cfl": "cf auto: trace log-lik change", "nzf": "noise flip rate"}
_W61X = _W61["X"].copy(); _W61X["noise flip rate"] = -_W61X["noise: flip rate (-)"]
WB61 = {rid: {k: float(_W61X[NAMES_WB[k]].iloc[j]) for k in NAMES_WB} for j, rid in enumerate(_W61["meta"].id)}   # by row id, both sets
_G63 = pd.read_csv(A / "exp63_jump.csv")                            # exp63 'block64 middle: mean jump (segments)' of the same white-box run, by row id
for rid, v in zip(_G63.id, _G63.jmp):
    WB61.setdefault(rid, {})["jmp"] = float(v)
# papers_to_see/ methods (exp81: faithful versions from submission_core/spurious_papers.py on the A10G; LID leave-self-out against the
# per-model reference set that ships in spurious_artifacts/papers_<model>.npz), by row id; signs from our unfinished rows (exp81)
PP = ["pp_rhythm", "pp_anchor", "pp_minkpp", "pp_lid", "pp_li", "pp_lara", "pp_cie", "pp_recall"]
# exp84/85: the set100 screening traces' submission features (run_feats.py; all still thinking at 6144) as extra LABEL-FREE reference and
# threshold rows for a model (CALIB[...]["extra_models"]): more problems at leaderboard difficulty, most of them novel
EXTRA_ROWS = {}
for _f, _m in ((ROOT / "ec2_results/a10g/feats/Qwen__Qwen3.5-4B.jsonl", "Qwen/Qwen3.5-4B"), (ROOT / "ec2_results/a10g/feats/allenai__Olmo-3-7B-Think.jsonl", "allenai/Olmo-3-7B-Think")):
    if _f.exists(): EXTRA_ROWS[_m] = [json.loads(_l) for _l in open(_f)]
if (A / "exp81_pp_rows.csv").exists():
    _PPR = pd.read_csv(A / "exp81_pp_rows.csv")
    for _r in _PPR.itertuples():
        for _f in PP: WB61.setdefault(_r.id, {})[_f] = float(getattr(_r, _f))
    PP_SIGN = json.load(open(A / "exp81_pp_signs.json"))
else:
    PP_SIGN = {}
SIGN.update({f: int(v) for f, v in PP_SIGN.items()})                # exp81: direction on our unfinished rows (pooled within-model AUC)
# TOHA for DeepSeek (exp54 / exp70): the paper's head selection (Algorithm 1) on our 47 unfinished DeepSeek rows; feature = - mean of the
# selected heads' mean MST edge, window = the last 128 reasoning tokens before the second budget cut (2048 for every unfinished trace)
_src54 = open(A / "exp54_toha_head_selection.py").read(); exec(_src54[_src54.index("def head_d("):_src54.index("def oof(")])
_D56 = pd.read_csv(A / "exp56_rows.csv"); _DSID = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
_TP = {r.id: (ROOT / "extraction_out_generations_p2" / "extraction_out_generations" / "deepseek-ai__DeepSeek-R1-0528-Qwen3-8B" / "toha" / f"{r.id}.npz"
              if r.dataset == "main_v2" else ROOT / "extra_olympiad_problems_dataset" / "olympiad-features" / "deepseek-ai__DeepSeek-R1-0528-Qwen3-8B" / "toha" / f"{r.id}.npz")
       for r in _D56[_D56.model_id == _DSID].itertuples()}
_TV = {rid: head_d(p, "win", 1) for rid, p in _TP.items() if p.exists()}
_TU = [r for r in _D56[(_D56.model_id == _DSID) & ~_D56.finished.astype(bool)].itertuples() if _TV.get(r.id) is not None]
TOHA_HEADS = [int(h) for h in toha_fit(np.array([_TV[r.id] for r in _TU]), np.array([int(r.robust) for r in _TU]))]
for rid, v in _TV.items():
    if v is not None: WB61.setdefault(rid, {})["tha"] = -float(np.mean(v[TOHA_HEADS]))
print(f"TOHA: {len(TOHA_HEADS)} DeepSeek heads selected on {len(_TU)} unfinished rows: {TOHA_HEADS}")
# T=1 samples (exp69 G2): extra reference / calibration rows for models whose recipe only uses features the samples have
_G66 = pd.read_csv(A / "exp66_graphs_crowding.csv")
T1ROWS = {}
for r in _G66[(_G66.trace != "greedy") & _G66.think_end.isna()].itertuples():
    T1ROWS.setdefault(r.id.rsplit("_", 1)[0].replace("__", "/"), []).append(dict(jmp=float(r.jump), robust=int(r.y)))   # id = org__name_hash
CAP = 6144                                                        # every future submission generates at most 6144 tokens
CAP_NEED = {"fa": 6144, "ge": 4096, "pearl": 0, "seg": 16384, "traj": 16384, "ab": 16384, "len": 16384, "tda": 16384, "cr": 4096, "chk": 16384, "fas": 6144, "fsat": 6144, "pcr": 6144, "lble": 6144, "pmle": 6144, "nrk": 6144, "f1k": 6144, "cfc": 6144, "cfl": 6144, "nzf": 6144, "jmp": 6144, "tha": 6144, **{f: 6144 for f in ["pp_rhythm", "pp_anchor", "pp_minkpp", "pp_lid", "pp_li", "pp_lara", "pp_cie", "pp_recall"]}}
RECIPES = {
    "spurious-forced-confidence": ["fa"],
    "spurious-confidence-knn": ["fa", "ge"],
    "spurious-top5-knn": ["fa", "ab", "seg", "traj", "pearl", "ge"],
    "spurious-no-forcing": ["seg", "traj", "pearl", "ge"],
    "spurious-length": ["len"],
    "spurious-tda": ["tda"],
    "spurious-confidence-knn-tda": ["fa", "ge", "tda"],          # three most independent strong signals (pairwise |rho| <= 0.19)
    "spurious-confidence-cascade": ["fa", "ge"],                  # forced confidence decides; kNN orders its near-1.0 ties; half split per model
    "spurious-cluster-graph": ["cr"],                              # exp33 cycle rank over 64 K-means clusters, first 4096 tokens
    "spurious-confidence-cascade-tda": ["fa", "ge", "tda"],       # the 0.71 cascade with kNN AND TDA ordering confidence's ties
    "spurious-check-share": ["chk"],                               # exp29 text signal
    "spurious-shift-robust": ["fas", "fsat", "pcr"],
    "spurious-confidence-checkshare-length": ["fa", "chk", "len"],    # exp46: the signals that transfer to the olympiad set (0.82 there)
    "spurious-confidence-checkshare-slope": ["fa", "chk", "fas"],     # exp47: forced confidence + check share + confidence slope (strong in both sets)
    "spurious-confidence-checkshare-slope-lookback": ["fa", "chk", "fas", "lble", "pmle"],   # exp51: + Lookback-Lens late - early (attention)
    "spurious-lapeigvals-probe": ["fa", "chk", "fas"],            # exp53: LapEigvals probe decides; these three are the fallback ensemble
    "spurious-slope-oly-refs": ["fa", "chk", "fas"],              # the 0.71 recipe, references + threshold from main_v2 AND the olympiad rows
    "spurious-slope-capped-refs": ["fa", "chk", "fas"],           # ... from the traces still thinking at 6144 only (main_v2 + olympiad)
    "spurious-unfinished-calibrated": ["fa", "chk", "fas"],       # exp57: finished -> robust; unfinished -> per-model refs + thresholds from unfinished rows
    "spurious-shift-robust-unfinished": ["fas", "fsat", "pcr"],    # exp58: the within-trace trio, same regime split and unfinished calibration (0.71)
    "spurious-whitebox": ["fa", "chk", "fas", "nrk", "f1k", "cfc", "cfl", "nzf"],   # exp61: shipped trio + deployable white-box signals             # exp44 within-trace features, pooled references, one global cut
    "spurious-whitebox-jump": ["chk", "fas", "nrk", "f1k", "cfc", "cfl", "nzf", "jmp"],   # exp62/63: forced confidence dropped (unstable), + block-graph jump
    "spurious-checkshare-slope-jump": ["chk", "fas", "jmp"],      # exp63: the light version, no counterfactual passes
    "spurious-newmethods": ["pp_rhythm", "pp_anchor", "pp_minkpp", "pp_lid", "pp_li", "pp_lara", "pp_cie", "pp_recall"],   # user 2026-10-28: decisions = the 8 new methods
    "spurious-permodel-ensemble": "per-model",                    # exp67: each model its own feature set (PERMODEL), equal weights
    "spurious-permodel-cascade": "per-model",                     # exp68: jump-first cascade for Qwen / Olmo, as ensemble otherwise
    "spurious-permodel-light": "per-model",                       # exp67/68: no counterfactual / noise passes for any model
    "spurious-permodel-v4": "per-model",
    "spurious-permodel-lookback": "per-model",                    # exp72 idea 4c: lookback + prompt mass + jump + rewording cosine for Qwen / Olmo
    "spurious-permodel-lookback-prior": "per-model",              # exp72: the same with per-model thresholds at each model's robust share (idea 7)
    "spurious-permodel-max": "per-model",                         # exp73: per-model subsets scoring best on the leaderboard among those our data supports                          # exp69/70: all 8 except Qwen (jump, + T=1 sample rows); DeepSeek + TOHA
    "spurious-abs-forced-confidence": ["fa"],                     # fixed absolute cuts hand-set from main_v2 (exp41), no per-model split
    "spurious-abs-knn": ["ge"],
    "spurious-abs-tda": ["tda"],
    "spurious-abs-pearl": ["pearl"],
}
# per-model recipes (user, 2026-10-27: the CHOICE per model may use leaderboard diagnostics; references / thresholds only from our rows).
# Evidence (held-out AUC / acc on our unfinished rows, exp67 / exp68; leaderboard diagnostics in brackets):
#   Qwen     jmp 1.00/0.90, jmp+cfc 1.00/0.90, jmp-first cascade 1.00/0.90  [jmp 1.00/0.67, cfc 0.83/0.67]
#   Skywork  chk+fas+5WB+jmp 0.87/0.81 (equal weights beat every cascade)  [uninformative: both unfinished problems spurious]
#   Olmo     only 5 rows: chk+jmp 1.00/1.00, jmp-first cascade 1.00/1.00    [3 problems, almost every feature reversed]
#   DeepSeek chk+f1k+cfc+cfl 0.77/0.66, chk+fas 0.70/0.68, jmp 0.56          [chk f1k cfc cfl jmp 1.00; fa nzf 0.00]
_ALLWB = ["chk", "fas", "nrk", "f1k", "cfc", "cfl", "nzf", "jmp"]
PERMODEL = {
    "spurious-permodel-ensemble": {"Qwen/Qwen3.5-4B": dict(features=["jmp", "cfc"]),
                                   "Skywork/Skywork-OR1-Math-7B": dict(features=_ALLWB),
                                   "allenai/Olmo-3-7B-Think": dict(features=_ALLWB),
                                   "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": dict(features=["chk", "f1k", "cfc", "cfl"])},
    "spurious-permodel-cascade": {"Qwen/Qwen3.5-4B": dict(features=_ALLWB, cascade=dict(primary="jmp", secondary=[f for f in _ALLWB if f != "jmp"], tie_weight=0.25)),
                                  "Skywork/Skywork-OR1-Math-7B": dict(features=_ALLWB),
                                  "allenai/Olmo-3-7B-Think": dict(features=_ALLWB, cascade=dict(primary="jmp", secondary=[f for f in _ALLWB if f != "jmp"], tie_weight=0.25)),
                                  "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": dict(features=["chk", "f1k", "cfc", "cfl"])},
    "spurious-permodel-light": {"Qwen/Qwen3.5-4B": dict(features=["jmp"]),
                                "Skywork/Skywork-OR1-Math-7B": dict(features=["jmp", "chk"], cascade=dict(primary="jmp", secondary=["chk"], tie_weight=0.25)),
                                "allenai/Olmo-3-7B-Think": dict(features=["chk", "jmp"]),
                                "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": dict(features=["chk", "fas"])},
    "spurious-permodel-v4": {"Qwen/Qwen3.5-4B": dict(features=["jmp"]),
                             "Skywork/Skywork-OR1-Math-7B": dict(features=_ALLWB),
                             "allenai/Olmo-3-7B-Think": dict(features=_ALLWB),
                             "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": dict(features=["chk", "f1k", "cfc", "cfl", "tha"])},
    # exp72 idea 4c: the attention-to-the-problem features where our data (Qwen lble 0.83) or the leaderboard (Olmo) supports them
    "spurious-permodel-lookback": {"Qwen/Qwen3.5-4B": dict(features=["lble", "pmle", "jmp", "cfc"]),
                                   "Skywork/Skywork-OR1-Math-7B": dict(features=_ALLWB),
                                   "allenai/Olmo-3-7B-Think": dict(features=["lble", "pmle", "jmp", "cfc"]),
                                   "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": dict(features=["chk", "f1k", "cfc", "cfl"])}}
PERMODEL["spurious-permodel-lookback-prior"] = PERMODEL["spurious-permodel-lookback"]
# exp73 / exp73b: per model, the subset (<= 4 features) that does best on the leaderboard problems among those with our in-sample AUC >= 0.70,
# tie-broken by our held-out accuracy. Our held-out AUC / acc: Qwen 1.00/0.90, Skywork 0.82/0.81, Olmo 1.00/1.00 (5 rows), DeepSeek 0.79/0.70
# v5 (2026-10-29, exp84/85): Qwen -> lookback features, refs + median threshold with its 100 set100 traces (new labels: AUC 0.78 / acc 0.78
# vs max's jmp+cfc 0.75 / 0.63; the same leaderboard calls 10100); Skywork / Olmo / DeepSeek as permodel-max
PERMODEL["spurious-permodel-v5"] = {"Qwen/Qwen3.5-4B": dict(features=["lble", "pmle", "jmp", "cfc"]),
                                    "Skywork/Skywork-OR1-Math-7B": dict(features=["fa", "chk", "f1k"]),
                                    "allenai/Olmo-3-7B-Think": dict(features=["chk", "lble", "pmle"]),
                                    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": dict(features=["chk", "nrk", "cfl", "pmle"])}
# v6 (2026-10-29, exp87): v5 + Olmo -> f1k alone, sign FLIPPED to +1 for Olmo (its new labels, novel problems: AUC 0.84 forced / 0.89 pure;
# its shipped chk / lble / pmle are at or below chance there), refs + median threshold with Olmo's 50 set100 traces (label-free)
PERMODEL["spurious-permodel-v6"] = {**PERMODEL["spurious-permodel-v5"], "allenai/Olmo-3-7B-Think": dict(features=["f1k"], signs={"f1k": 1})}
# v7 (2026-10-30, exp90): v6 + an Olmo NOVELTY ROUTER. Min-K% (20%) of the problem text below the threshold -> novel branch = v6's f1k (+1);
# else familiar branch = v5's chk + lble + pmle. Router threshold fitted ONLY on date novelty of Olmo's 50 set100 problems (set24 = novel),
# accuracy-maximising (exp90: AUC 0.77; held-out Olmo acc 0.81 forced / 0.74 pure vs v5 0.71 / 0.61, v6 0.48 / 0.57). Branch thresholds =
# label-free medians over Olmo's calibration rows, as for every other model.
def _router_threshold():
    pp = [json.loads(_l) for _l in open(ROOT / "ec2_results/a10g/papers2/allenai__Olmo-3-7B-Think/papers2.jsonl")]
    nov = {_l.strip() for _l in open(ROOT / "vast_packages/screen/set24_olmo_novel.txt") if _l.strip()}
    r = np.array([-float(d["mink_20"]) for d in pp if d["dataset"] == "set100"]); yv = np.array([d["id"].split("_")[-1] in nov for d in pp if d["dataset"] == "set100"])
    u = np.unique(r); cand = np.r_[u[0] - 1, (u[1:] + u[:-1]) / 2, u[-1] + 1]
    return -float(cand[int(np.argmax([((r > t) == yv).mean() for t in cand]))])      # novel iff mink_20 < this
PERMODEL["spurious-permodel-v7"] = {**PERMODEL["spurious-permodel-v6"], "allenai/Olmo-3-7B-Think": dict(
    features=["chk", "f1k", "lble", "pmle"],
    router=dict(feature="mink_20", k=20, threshold=_router_threshold(), default="novel",
                novel=dict(features=["f1k"], signs={"f1k": 1}), familiar=dict(features=["chk", "lble", "pmle"], signs={"chk": 1, "lble": 1, "pmle": 1})))}
# v8 (2026-10-31, exp92 / exp94): v7 + Qwen as a JUDGE of Olmo's problems. Olmo's call = mean of two label-free percentiles (its router margin
# among its 50 set100 traces; Qwen's v7 margin on the same problem among Qwen's margins on those 50 problems) > 0.5. Held out on Olmo's new
# labels: 0.95 forced / 0.87 pure vs v7's router 0.81 / 0.74. The pools (r_ref, q_ref) are filled in below with the built recipe's own scorer.
PERMODEL["spurious-permodel-v8"] = {**PERMODEL["spurious-permodel-v7"], "allenai/Olmo-3-7B-Think": dict(PERMODEL["spurious-permodel-v7"]["allenai/Olmo-3-7B-Think"],
                                    judge=dict(model="Qwen/Qwen3.5-4B", cut=0.5, min_seconds_left=900))}
# v8-wblog (2026-10-31, exp97): v8's calls + the white-box battery LOGGED for every problem (leaderboard values for ~480 features)
PERMODEL["spurious-permodel-v8-wblog"] = PERMODEL["spurious-permodel-v8"]
# v8-wblog2 (2026-10-31, exp102-104 + exp93): v8's calls + battery + per-head arrays + wb2_log (anchoring, commitment / logit lens, PEARL,
# K-means graphs, directions, delta-hyperbolicity / compression / Poincare) + Olmo self-perturbation, all LOGGED ONLY
PERMODEL["spurious-permodel-v8-wblog2"] = PERMODEL["spurious-permodel-v8"]
# v8-ent4096 (2026-10-31; nomination S1, leaderboard 1.00): v8, but Qwen's UNFINISHED calls come from the forced-answer entropy at 4096 tokens
# (exp103 / exp107). Sign from our forced labels, threshold = median of Qwen's 100 set100 traces (label-free). Qwen's v8 margins are unchanged,
# so the Olmo judge is unchanged. The feature was chosen with leaderboard feedback, so this is not a held-out choice.
def _ent_override():
    import json as _j
    R = [_j.loads(_l) for _l in open(ROOT / "ec2_results/wb2/Qwen__Qwen3.5-4B.jsonl")]; v = {r["pid"]: r["force_ent__b4096"] for r in R}
    L = pd.read_csv(ROOT / "hard_pool/set100_labels_Qwen__Qwen3.5-4B.csv").set_index("pid")
    lab = {p_: int(l_ == "robust") for p_, l_ in L["label_forced"].items() if l_ in ("robust", "non_robust") and p_ in v}
    xs, ys = np.array([v[p_] for p_ in lab]), np.array(list(lab.values()))
    a = np.mean([(x1 > x0) + .5 * (x1 == x0) for x1 in xs[ys == 1] for x0 in xs[ys == 0]])
    return dict(feature="force_ent__b4096", sign=1 if a >= .5 else -1, threshold=float(np.median(list(v.values()))), our_auc=float(a), pool=len(v), labelled=len(lab))
def _seg25_override():
    """exp109 option (a): Olmo's unfinished calls = projection of the 25% segment mean (stored layer slot 2) on the shipped robust-minus-non-robust
    direction (fitted on our forced labels, analysis/build_wb2_ship.py). Sign from our labels; threshold = median of Olmo's 50 set100 traces."""
    import json as _j
    ship = dict(np.load(ROOT / "analysis/wb2_ship/wb2_ship_allenai__Olmo-3-7B-Think.npz")); dvec = ship["dir_seg25"][2].astype(np.float32)
    ids = [_j.loads(_l)["pid"] for _l in open(ROOT / "ec2_results/wb2/allenai__Olmo-3-7B-Think.jsonl")]
    v = {p_: float(np.load(ROOT / f"ec2_results/wb2/allenai__Olmo-3-7B-Think/{p_}.npz")["seg_means"][2, 0].astype(np.float32) @ dvec) for p_ in ids}
    L = pd.read_csv(ROOT / "hard_pool/set100_labels_allenai__Olmo-3-7B-Think.csv").set_index("pid")
    lab = {p_: int(l_ == "robust") for p_, l_ in L["label_forced"].items() if l_ in ("robust", "non_robust") and p_ in v}
    xs, ys = np.array([v[p_] for p_ in lab]), np.array(list(lab.values()))
    a = np.mean([(x1 > x0) + .5 * (x1 == x0) for x1 in xs[ys == 1] for x0 in xs[ys == 0]])
    return dict(feature="dirproj_seg25__s2", sign=1 if a >= .5 else -1, threshold=float(np.median(list(v.values()))), our_auc=float(a), pool=len(v), labelled=len(lab))
PERMODEL["spurious-permodel-v8-ent4096"] = {**PERMODEL["spurious-permodel-v8"], "Qwen/Qwen3.5-4B": dict(PERMODEL["spurious-permodel-v8"]["Qwen/Qwen3.5-4B"], override=_ent_override())}
PERMODEL["spurious-permodel-v8-ent4096-olmoseg25"] = {**PERMODEL["spurious-permodel-v8-ent4096"], "allenai/Olmo-3-7B-Think": dict(PERMODEL["spurious-permodel-v8"]["allenai/Olmo-3-7B-Think"], override=_seg25_override())}
PERMODEL["spurious-permodel-max"] = {"Qwen/Qwen3.5-4B": dict(features=["jmp", "cfc"]),
                                     "Skywork/Skywork-OR1-Math-7B": dict(features=["fa", "chk", "f1k"]),
                                     "allenai/Olmo-3-7B-Think": dict(features=["chk", "lble", "pmle"]),
                                     "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": dict(features=["chk", "nrk", "cfl", "pmle"])}


def SUB(base, mid):
    """The recipe one model actually uses: its own features (and cascade) when the recipe is per-model."""
    pm = base.get("per_model", {}).get(mid)
    return {**base, **pm} if pm else base


def SC(vals, base, mid):
    b = SUB(base, mid); return score_trace({f: vals[f] for f in b["features"]}, b, mid)


ABS = {"spurious-abs-forced-confidence": "fa", "spurious-abs-knn": "ge", "spurious-abs-tda": "tda", "spurious-abs-pearl": "pearl"}
EXTRA = {"spurious-confidence-cascade": dict(cascade=dict(primary="fa", secondary=["ge"], saturate_at=0.999, tie_weight=0.01), per_model_split=True),
         "spurious-shift-robust": dict(pooled_refs=True),
         "spurious-lapeigvals-probe": dict(lap_probe=True),
         "spurious-unfinished-calibrated": dict(finished_robust=True, per_model_split=False),
         "spurious-shift-robust-unfinished": dict(finished_robust=True, per_model_split=False),
         "spurious-whitebox": dict(finished_robust=True, per_model_split=False),
         "spurious-whitebox-jump": dict(finished_robust=True, per_model_split=False),
         "spurious-checkshare-slope-jump": dict(finished_robust=True, per_model_split=False),
         "spurious-confidence-cascade-tda": dict(cascade=dict(primary="fa", secondary=["ge", "tda"], saturate_at=0.999, tie_weight=0.01), per_model_split=True)}
# calibration sets beyond main_v2 (our data only): the friend's olympiad rows (DeepSeek, Skywork; same features at 6144, exp47) and/or only
# the traces still thinking at 6144 tokens (the state 13 of 14 leaderboard traces are in)
CALIB = {"spurious-slope-oly-refs": dict(olympiad=True, capped_only=False), "spurious-slope-capped-refs": dict(olympiad=True, capped_only=True),
         "spurious-unfinished-calibrated": dict(olympiad=True, capped_only=True, per_model_threshold=True),
         "spurious-shift-robust-unfinished": dict(olympiad=True, capped_only=True, per_model_threshold=True),
         "spurious-whitebox": dict(olympiad=True, capped_only=True, per_model_threshold=True),
         "spurious-whitebox-jump": dict(olympiad=True, capped_only=True, per_model_threshold=True),
         "spurious-checkshare-slope-jump": dict(olympiad=True, capped_only=True, per_model_threshold=True)}
for _k, _pm in PERMODEL.items():
    RECIPES[_k] = sorted({f for v in _pm.values() for f in v["features"]}, key=(_ALLWB + ["tha", "lble", "pmle", "fa"]).index)
    EXTRA[_k] = dict(finished_robust=True, per_model_split=False)
    CALIB[_k] = dict(olympiad=True, capped_only=True, per_model_threshold=True)
CALIB["spurious-permodel-v4"]["t1_models"] = ["Qwen/Qwen3.5-4B"]
CALIB["spurious-permodel-v5"]["extra_models"] = ["Qwen/Qwen3.5-4B"]   # exp85: Qwen's 100 set100 traces join its refs + threshold rows       # exp69 G2: Qwen's 10 rows + its unfinished T=1 samples
CALIB["spurious-permodel-v6"]["extra_models"] = ["Qwen/Qwen3.5-4B", "allenai/Olmo-3-7B-Think"]   # exp87: + Olmo's 50 set100 traces
CALIB["spurious-permodel-v7"]["extra_models"] = ["Qwen/Qwen3.5-4B", "allenai/Olmo-3-7B-Think"]   # as v6
CALIB["spurious-permodel-v8"]["extra_models"] = ["Qwen/Qwen3.5-4B", "allenai/Olmo-3-7B-Think"]   # as v6 / v7
CALIB["spurious-permodel-v8-wblog"]["extra_models"] = ["Qwen/Qwen3.5-4B", "allenai/Olmo-3-7B-Think"]
EXTRA["spurious-permodel-v8-wblog"]["log_battery"] = True
CALIB["spurious-permodel-v8-ent4096-olmoseg25"]["extra_models"] = ["Qwen/Qwen3.5-4B", "allenai/Olmo-3-7B-Think"]
EXTRA["spurious-permodel-v8-ent4096-olmoseg25"]["ship_wb2"] = True
CALIB["spurious-permodel-v8-ent4096"]["extra_models"] = ["Qwen/Qwen3.5-4B", "allenai/Olmo-3-7B-Think"]
CALIB["spurious-permodel-v8-wblog2"]["extra_models"] = ["Qwen/Qwen3.5-4B", "allenai/Olmo-3-7B-Think"]
EXTRA["spurious-permodel-v8-wblog2"].update(log_battery=True, log_wb2=True)
CALIB["spurious-permodel-lookback-prior"]["prior_q"] = True               # exp69 G4 / exp72 idea 7: q = each model's spurious share
EXTRA["spurious-newmethods"] = dict(finished_robust=True, per_model_split=False, log_extra=True, log_papers=True)
CALIB["spurious-newmethods"] = dict(olympiad=True, capped_only=True, per_model_threshold=True)
for _k in ("spurious-permodel-lookback", "spurious-permodel-lookback-prior", "spurious-permodel-max", "spurious-permodel-v5", "spurious-permodel-v6", "spurious-permodel-v7", "spurious-permodel-v8", "spurious-permodel-v8-wblog", "spurious-permodel-v8-wblog2", "spurious-permodel-v8-ent4096", "spurious-permodel-v8-ent4096-olmoseg25"):
    EXTRA[_k]["log_extra"] = True                                         # idea 8: log lookback, TOHA (all heads) and the jump for every model
F47 = pickle.load(open(A / "exp47_features.pkl", "rb"))
OLYKEY = {"fa": "4 output|forced p1 @6144", "chk": "6 text|check share", "fas": "4 output|confidence slope over budgets", "len": "0 shortcuts|-log trace length",
          "fsat": "4 output|share of saturated budgets", "pcr": "2 states|PC1 change 75%/25%"}
OLY_CAPPED = F47["F"]["0 shortcuts|finished by 6144"] == 0


def olyv(f):
    """Olympiad values of feature f, aligned with F47['ids']."""
    if f in OLYKEY: return F47["F"][OLYKEY[f]]
    if f in ("lble", "pmle"): return np.asarray(F51O["lookback late - early" if f == "lble" else "prompt mass late - early"], float)
    return np.array([WB61.get(rid, {}).get(f, np.nan) for rid in F47["ids"]])
CUT = sys.argv[1] if len(sys.argv) > 1 else "median"
Q = {"median": 0.5, "mv2rate": float((y[rows] == 0).mean())}[CUT]
RECIPES = {k + ("-" + CUT if CUT != "median" else ""): v for k, v in RECIPES.items()}
HFID = {m: HF[m] for m in MODELS}
sys.path.insert(0, str(CORE))
from spurious_features import decide_batch, ensemble_score, f_check_share, f_late_vs_early, f_seg25_pc1, f_tda_h0_spread, score_trace

_CACHE = {}


def lap_vector(path):
    """Training input of the LapEigvals probe from an extract_features.py attn file: top-8 forced-suffix-B eigenvalues at the last cutoff
    <= 6144 (= the submission's cut: min(end of thinking, 6144)), through the same transform the submission applies."""
    from spurious_features import lap_probe_features
    z = np.load(path); ce = list(z["cut_eff"]); j = ce.index(max(c for c in ce if c <= 6144))
    return lap_probe_features(z["topk_force"][j, list(z["variant_names"]).index("B")][..., :8])


def train_lap_probes(art):
    """Per model: standardise -> PCA -> logistic (C = 0.1); threshold = median training score (the test prior is 50/50). DeepSeek and
    Skywork train on all their olympiad rows (friend's labels), Qwen and Olmo on their main_v2 rows (official labels)."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    D47 = pickle.load(open(A / "exp47_features.pkl", "rb")); OLYA = ROOT / "extra_olympiad_problems_dataset" / "results_olympiad_attention" / "out_attn"
    info = {}
    for m in MODELS:
        hf = HFID[m]
        if hf in set(D47["model"]):
            k = [i for i in range(len(D47["ids"])) if D47["model"][i] == hf]
            X = np.stack([lap_vector(OLYA / m / "attn" / f"{D47['ids'][i]}.npz") for i in k]); yy = D47["y"][k].astype(int); src, nc = "olympiad", 5
        else:
            k = [i for i in rows if model[i] == m]
            X = np.stack([lap_vector(ROOT / "extraction_out" / m / "attn" / f"{allm.id.iloc[i]}.npz") for i in k]); yy = (y[k] == 1).astype(int); src, nc = "main_v2", 3
        mu_, sd = X.mean(0), X.std(0) + 1e-6; Z = (X - mu_) / sd; pca = PCA(nc, svd_solver="full").fit(Z); clf = LogisticRegression(C=0.1, max_iter=3000).fit(pca.transform(Z), yy)   # exact SVD: deterministic
        sc = clf.decision_function(pca.transform(Z)); thr = float(np.median(sc))
        np.savez(art / f"lap_probe_{hf.replace('/', '__')}.npz", mu=mu_.astype(np.float32), sd=sd.astype(np.float32), pca_mean=pca.mean_.astype(np.float32),
                 pca_components=pca.components_.astype(np.float32), coef=clf.coef_[0].astype(np.float32), intercept=np.float32(clf.intercept_[0]),
                 threshold=np.float32(thr), trained_on=f"{src} ({len(yy)} rows, {int(yy.sum())} robust)", shape=np.array(X.shape))
        info[hf] = dict(trained_on=src, rows=len(yy), robust=int(yy.sum()), features=int(X.shape[1]), pca=nc, threshold=thr,
                        train_auc_in_sample=float(roc_auc_score(yy, sc)), robust_calls_in_sample=int((sc > thr).sum()))
        print(f"   LapEigvals probe {hf.split('/')[1][:12]:12s} trained on {src} {len(yy)} rows ({int(yy.sum())} robust), {X.shape[1]} inputs -> PCA {nc} | "
              f"in-sample AUC {info[hf]['train_auc_in_sample']:.2f} | threshold {thr:+.3f} (median training score)")
    return info


_WIN, _TOK, _GEN = {}, {}, {}


def WIN(i):
    if i not in _WIN: _WIN[i] = np.load(P2 / model[i] / "tf" / f"{allm.id.iloc[i]}.npz")["win_states"]
    return _WIN[i]


def TOK(m):
    if m not in _TOK:
        from spurious_inference import load_tokenizer
        _TOK[m] = load_tokenizer(HF[m])
    return _TOK[m]


def GEN(i):
    if not _GEN:
        for f in (ROOT / "extraction_out").glob("*/generations.jsonl"):
            for l in open(f):
                d = json.loads(l); _GEN[d["id"]] = d["gen_ids"]
    return _GEN[allm.id.iloc[i]]


def features_at_cap(c):
    """{feature: array over allm rows} with every main_v2 trace truncated at c generated tokens."""
    if c in _CACHE:
        return _CACHE[c]
    V = {k: np.full(len(allm), np.nan) for k in NAMES}
    for i in rows:
        fj = force_json(model[i], allm.id.iloc[i]); te = fj.get("think_end"); n_gen = int(allm.n_gen.iloc[i])
        g = min(n_gen, c); L = te if (te is not None and te < c) else min(fj["L"], c)
        V["fa"][i], V["ab"][i] = F28[NAMES["fa"]][i], F28[NAMES["ab"]][i]
        V["pearl"][i], V["ge"][i] = F28[NAMES["pearl"]][i], F32["step64|k5|4k|global_efficiency"][i]
        V["len"][i] = -np.log(g)
        for k in list(NAMES_WB) + ["jmp", "tha"] + PP:             # white-box run (exp61 / exp63) + TOHA (exp70) + paper methods (exp81), capped at 6144
            V[k][i] = WB61.get(allm.id.iloc[i], {}).get(k, np.nan)
        j51 = int(np.where(rows == i)[0][0])
        V["lble"][i], V["pmle"][i] = F51["lookback late - early"][j51], F51["prompt mass late - early"][j51]
        for k in ("fas", "fsat", "pcr"):                           # exp44 reads exactly the first 6144 tokens (= CAP)
            V[k][i] = F44[NAMES[k][6:]][i]
        V["cr"][i] = F33["step64|K64|4k|cluster_cycle_rank_per_step"][i]   # reads only the first 4096 tokens
        if c >= 16384:                                             # every main_v2 trace fits in 16k: the saved values
            V["tda"][i] = F38["C_layer|H0_betti_spread|L3"][i]
            V["chk"][i] = F29["text:full:check_share"][i]
        else:                                                      # greedy => the capped trace is the first c tokens: recompute
            V["tda"][i] = f_tda_h0_spread(WIN(i)[: -(-g // 16), 3], L)
            V["chk"][i] = f_check_share(TOK(model[i]).decode(GEN(i)[:g], skip_special_tokens=True))
        W = WIN(i)[: -(-g // 16)]
        V["seg"][i], V["traj"][i] = f_seg25_pc1(W, L), f_late_vs_early(W)
    _CACHE[c] = V
    return V


MV2_CAPPED = {}
for i in rows:
    _te = force_json(model[i], allm.id.iloc[i]).get("think_end"); MV2_CAPPED[i] = _te is None or _te >= 6144
# per-model mean of the layer-3 window states over main_v2 traces (exp32 centring); the kNN feature only reads the first 4096 tokens
MU = {}
for m in MODELS:
    W = [np.load(P2 / m / "tf" / f"{allm.id.iloc[i]}.npz")["win_states"][:, 3].astype(np.float32) for i in rows if model[i] == m]
    MU[m] = np.concatenate(W).mean(0)

import os
for name, feats in RECIPES.items():
    if os.environ.get("ONLY") and name != os.environ["ONLY"]: continue      # rebuild/score one recipe
    cap = CAP                                                     # one cap for every submission (user, 2026-10-24)
    V = features_at_cap(cap)
    d = SOL / name; d.mkdir(parents=True, exist_ok=True)
    cal = CALIB.get(name.replace("-" + CUT, ""), {})
    crow = rows[[MV2_CAPPED[i] for i in rows]] if cal.get("capped_only") else rows       # main_v2 rows used for calibration
    ok = OLY_CAPPED if cal.get("capped_only") else np.ones(len(F47["model"]), bool)
    oly = {f: {m: [float(v) for v in olyv(f)[ok & (F47["model"] == HFID[m])] if np.isfinite(v)] for m in MODELS} for f in feats} if cal.get("olympiad") else {}
    ref = {f: {HFID[m]: sorted([float(v) for v in V[f][crow][model[crow] == m] if np.isfinite(v)] + (oly[f][m] if oly else [])) for m in MODELS} for f in feats}
    t1m = [m for m in cal.get("t1_models", []) if set(PERMODEL.get(name.replace("-" + CUT, ""), {}).get(m, {}).get("features", feats)) <= {"jmp"}]
    for m in t1m:                                                  # T=1 sample rows join this model's references (exp69 G2)
        for f in ("jmp",):
            ref[f][m] = sorted(ref[f][m] + [r[f] for r in T1ROWS.get(m, [])])
    for m in cal.get("extra_models", []):                          # set100 traces join this model's references (label-free)
        for f in feats:
            ref[f][m] = sorted(ref[f][m] + [float(r[f]) for r in EXTRA_ROWS.get(m, []) if f in r and np.isfinite(r[f])])
    for f in feats:                                                # all models pooled: used by recipes with pooled_refs
        ref[f]["*"] = sorted(v for m in MODELS for v in ref[f][HFID[m]])
    # every recipe decides by ranking within each model's batch (top half robust; the threshold only settles the middle problem of
    # an odd batch): immune to shifts between our training traces and the worker's, never all-True / all-False for a model
    base = dict(features=feats, signs={f: SIGN[f] for f in feats}, ref=ref, **{"per_model_split": True, **EXTRA.get(name.replace("-" + CUT, ""), {})})
    if name.replace("-" + CUT, "") in PERMODEL:
        base["per_model"] = PERMODEL[name.replace("-" + CUT, "")]
    if "tha" in feats:
        base["toha_heads"] = {_DSID: TOHA_HEADS}
    S = np.array([SC({f: V[f][i] for f in feats}, base, HFID[model[i]]) for i in rows]); thr = float(np.quantile(S, Q))
    if cal:                                                        # threshold = median score over the calibration set (main_v2 subset + olympiad)
        Sc = [SC({f: V[f][i] for f in feats}, base, HFID[model[i]]) for i in crow]; Mc = [HFID[model[i]] for i in crow]; Yc = [int(y[i] == 1) for i in crow]
        if cal.get("olympiad"):
            Sc += [SC({f: olyv(f)[k] for f in feats}, base, F47["model"][k]) for k in np.where(ok)[0]]
            Mc += [F47["model"][k] for k in np.where(ok)[0]]; Yc += [int(F47["y"][k] == 1) for k in np.where(ok)[0]]
        for m in t1m:                                              # and its calibration (threshold) rows
            Sc += [SC({f: r.get(f, np.nan) for f in feats}, base, m) for r in T1ROWS.get(m, [])]; Mc += [m] * len(T1ROWS.get(m, [])); Yc += [r["robust"] for r in T1ROWS.get(m, [])]
        for m in cal.get("extra_models", []):                      # ... and its threshold rows (median: label-free)
            ex = EXTRA_ROWS.get(m, []); Sc += [SC({f: r.get(f, np.nan) for f in feats}, base, m) for r in ex]; Mc += [m] * len(ex); Yc += [0] * len(ex)
        thr = float(np.quantile(Sc, Q)); base["calibration"] = {**cal, "n_rows": len(Sc)}
        if cal.get("per_model_threshold"):                         # exp57: each model's own median over its calibration rows
            qm = {m: (1 - float(np.mean([yy for yy, mm in zip(Yc, Mc) if mm == m])) if cal.get("prior_q") else Q) for m in set(Mc)}
            base["threshold_by_model"] = {m: float(np.quantile([s for s, mm in zip(Sc, Mc) if mm == m], qm[m])) for m in sorted(set(Mc))}
            if cal.get("prior_q"): base["calibration"]["q_by_model"] = {m: round(qm[m], 3) for m in sorted(qm)}
            base["calibration"]["rows_by_model"] = {m: int(sum(mm == m for mm in Mc)) for m in sorted(set(Mc))}
            for m, pm in base.get("per_model", {}).items():         # exp90 router: each branch's own median over this model's calibration rows
                if "router" in pm:
                    pm = base["per_model"][m] = copy.deepcopy(pm)
                    rows_m = [{f: V[f][i] for f in feats} for i in crow if HFID[model[i]] == m] + [{f: r.get(f, np.nan) for f in feats} for r in EXTRA_ROWS.get(m, [])]
                    for bn in ("novel", "familiar"):
                        br = pm["router"][bn]; bb = {**base, **pm, **br}
                        br["threshold"] = float(np.quantile([score_trace({f: v[f] for f in br["features"]}, bb, m) for v in rows_m], Q))
                    base["calibration"].setdefault("router_rows", {})[m] = len(rows_m)
            for m, pm in base.get("per_model", {}).items():         # v8 judge: label-free percentile pools on this model's set100 problems
                if "judge" in pm:
                    pm = base["per_model"][m] = copy.deepcopy(pm); J = pm["judge"]; jm = J["model"]
                    _mk = {d["id"].split("_")[-1]: d["mink_20"] for d in map(json.loads, open(ROOT / f"ec2_results/a10g/papers2/{m.replace('/', '__')}/papers2.jsonl"))}
                    own = {r["pid"]: r for r in EXTRA_ROWS.get(m, []) if r.get("still_thinking", True)}
                    jud = {r["pid"]: r for r in EXTRA_ROWS.get(jm, [])}; pids = sorted(p_ for p_ in own if p_ in jud)
                    rm = []
                    for p_ in pids:
                        v = {f: own[p_].get(f, np.nan) for f in feats}; R_ = pm["router"]
                        br = R_["novel"] if (_mk.get(p_, np.nan) < R_["threshold"]) else R_["familiar"]
                        rm.append(score_trace({f: v[f] for f in br["features"]}, {**base, **pm, **br}, m) - br["threshold"])
                    qb = {**base, **base["per_model"][jm]}; qt = base["threshold_by_model"][jm]
                    qm = [score_trace({f: jud[p_].get(f, np.nan) for f in qb["features"]}, qb, jm) - qt for p_ in pids]
                    J.update(r_ref=sorted(float(x) for x in rm), q_ref=sorted(float(x) for x in qm), n_ref=len(pids))
    key = name.replace("-" + CUT, "")
    if key in ABS:                                                 # one raw cut for all models, maximising main_v2 balanced accuracy (exp41)
        f = ABS[key]; x = V[f][rows].astype(float); yy_ = y[rows] == 1
        v = np.unique(x[np.isfinite(x)]); cand = (v[1:] + v[:-1]) / 2
        def _bal(c):
            pr = np.where(np.isfinite(x), x > c, True)
            return 0.5 * (pr[yy_].mean() + (~pr[~yy_]).mean())
        base.update(abs_feature=f, abs_cut=float(cand[int(np.argmax([_bal(c) for c in cand]))]), per_model_split=False)
        S = x.copy(); thr = base["abs_cut"]
    if base.get("abs_feature"):
        pred_tr = np.where(np.isfinite(S), S > thr, True)
    elif base.get("per_model_split"):                              # each model's main_v2 rows as one batch, as at test time
        pred_tr = np.zeros(len(rows), bool)
        for m in MODELS:
            idx = np.where(model[rows] == m)[0]; pred_tr[idx] = decide_batch(S[idx], thr, True)
    else:
        pred_tr = S > thr
    acc = float((pred_tr == (y[rows] == 1)).mean())
    recipe = dict(name=name, **base, threshold=thr, q_nonrobust=Q, cap=cap, expected_problems=14, time_budget_s=3500,
                  train_rows=int(len(rows)), train_accuracy_in_sample=acc, source_features={f: NAMES[f] for f in feats},
                  approximations=(["ab: references at 16k (its fraction-of-thinking cutoffs move under a cap)"] if "ab" in feats and cap < 16384 else []))
    if name.replace("-" + CUT, "") == "spurious-confidence-cascade":
        recipe["note"] = "kept as the 0.71 build"
    for stale in ("robust_core.py", "recipe.json"):            # earlier layout
        (d / stale).unlink(missing_ok=True)
    shutil.rmtree(d / "artifacts", ignore_errors=True)
    art = d / "spurious_artifacts"; art.mkdir(exist_ok=True)
    if "ge" in feats or "cr" in feats:
        for m in MODELS: np.save(art / f"mu_{HFID[m].replace('/', '__')}.npy", MU[m].astype(np.float32))
    if base.get("log_papers") or any(f in PP for f in feats):              # per-model LID reference set + recall directions (exp81)
        for m in MODELS: shutil.copy(A / "exp81_ship" / f"papers_{HFID[m].replace('/', '__')}.npz", art)
    if base.get("lap_probe"):                                       # exp53 probes: olympiad rows for DeepSeek / Skywork, main_v2 for Qwen / Olmo
        recipe["lap_probe_info"] = train_lap_probes(art)
    if "cr" in feats:                                                # exp33 clusters, refitted identically (verified equal to the saved feature)
        from sklearn.cluster import KMeans
        from spurious_features import kmeans_nodes
        for m in MODELS:
            ii = [i for i in rows if model[i] == m]
            nd = [kmeans_nodes(np.load(P2 / m / "tf" / f"{allm.id.iloc[i]}.npz")["win_states"][:, 3], MU[m], force_json(m, allm.id.iloc[i])["L"]) for i in ii]
            km = KMeans(n_clusters=64, n_init=4, random_state=0).fit(np.concatenate([v for v in nd if v is not None]))
            np.save(art / f"km64_{HFID[m].replace('/', '__')}.npy", km.cluster_centers_.astype(np.float32))
    if base.get("log_wb2") or base.get("ship_wb2"):                  # exp102-104 logging / wb2-feature overrides: label-free PCA / K-means + our-label directions
        for _f in sorted((A / "wb2_ship").glob("wb2_ship_*.npz")): shutil.copy(_f, art)
    json.dump(recipe, open(art / "recipe.json", "w"), indent=1)
    for f in CODE: shutil.copy(CORE / f, d / f)
    # identity of this build, logged first thing by solution.py (downloads from Codabench are renamed)
    import hashlib, datetime
    h = hashlib.sha256()
    for f in sorted(p for p in d.rglob("*") if p.is_file() and p.name != "SUBMISSION_INFO.json" and "__pycache__" not in p.parts):
        h.update(f.relative_to(d).as_posix().encode()); h.update(f.read_bytes())
    info = dict(zip_name=f"{name.removeprefix('spurious-')}-small.zip", solution=name, recipe_features=feats, cut=CUT, q_nonrobust=round(Q, 3),
                threshold=round(thr, 4), cap=cap, built=datetime.datetime.now().isoformat(timespec="seconds"), build_id=h.hexdigest()[:12])
    json.dump(info, open(d / "SUBMISSION_INFO.json", "w"), indent=1)
    pred = pred_tr
    yy = y[rows] == 1; bal = 0.5 * (pred[yy].mean() + (~pred[~yy]).mean())
    per = {HFID[m].split("/")[1][:8]: f"{pred[model[rows] == m].sum()}/{(model[rows] == m).sum()}" for m in MODELS}
    print(f"{name:36s} cap {cap:5d} | features {feats} | threshold {thr:+.3f} (q={Q:.2f}) | main-v2 robust {pred.sum()}/43 {per} | balanced acc {bal:.2f} (in-sample)")
    if not base.get("abs_feature"):                                # one problem per call: the per-model split is inactive, the threshold decides
        p1 = S > thr; b1 = 0.5 * (p1[yy].mean() + (~p1[~yy]).mean())
        per1 = {HFID[m].split("/")[1][:8]: f"{p1[model[rows] == m].sum()}/{(model[rows] == m).sum()}" for m in MODELS}
        print(f"{'':36s} single-problem calls (threshold only): robust {p1.sum()}/43 {per1} | balanced acc {b1:.2f} | "
              f"{int(np.isclose(S, thr).sum())} scores within 1e-8 of the threshold")
        unf = np.array([MV2_CAPPED[i] for i in rows], bool)                    # still thinking at the cap
        acc = lambda k: f"{(p1[k] == yy[k]).mean():.2f} ({(p1[k] == yy[k]).sum()}/{k.sum()})"
        print(f"{'':36s} accuracy {acc(np.ones(len(rows), bool))} | unfinished {acc(unf)} | finished {acc(~unf)} | "
              + ", ".join(f"{HFID[m].split('/')[1][:8]} {acc(model[rows] == m)}" for m in MODELS))
