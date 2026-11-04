"""exp109 (user 2026-11-01): score every candidate on train-main-v2 (official labels; difficulty matched to the leaderboard), from the final queue's
outputs on the submission's own greedy traces (cap 6144): v8's submission features, battery (+ per-head arrays), wb2 (+ wb2_log geometry with
the shipped npz), entropy trajectory (exp107 features). Exactly as on the leaderboard: sign from OUR forced labels, threshold = median of our
label-free pool, finished within 6144 -> robust. Candidates: v8's per-model recipe (Olmo: both router branches, Min-K% not computed on
main-v2), ent4096 for Qwen, the exp105 pre-registered leads, and every logged scalar ('does our-data AUC predict main-v2 AUC').
Also merges each lead's leaderboard result (exp105_lb.csv). Writes exp109_mainv2.csv, exp109_corr.csv."""
import json, sys, types, warnings, zipfile, importlib.util, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

src = open("/home/sabyasachi19/PycharmProjects/AimoAGAIN/analysis/exp99_cv_candidates.py").read()
exec(src[:src.index("out, nested = [], []")])
spec = importlib.util.spec_from_file_location("wb2_log", A + "submission_core/wb2_log.py"); W2 = importlib.util.module_from_spec(spec)
_t = sys.modules.get("torch"); sys.modules["torch"] = types.SimpleNamespace(no_grad=lambda: (lambda f: f)); spec.loader.exec_module(W2)
if _t is not None: sys.modules["torch"] = _t
else: del sys.modules["torch"]
e107 = open(A + "analysis/exp107_entropy_traj.py").read(); g107 = {"np": np, "BUD": list(range(256, 6145, 256))}
exec(e107[e107.index("def feats_one"):e107.index("rows_s, rows_f = [], []")], g107); feats_one = g107["feats_one"]
sys.path.insert(0, A + "submission_core")
from spurious_features import score_trace
REC = json.load(zipfile.ZipFile(A + "submitted_solutions/permodel-v8-ent4096-small.zip").open("spurious_artifacts/recipe.json"))
MV2 = {}
for l in open(A + "vast_packages/mv2/mv2_pool.jsonl"):
    d = json.loads(l); MV2[(d["model"], d["pid"])] = d
LB105 = pd.read_csv(A + "analysis/exp105_lb.csv")
PRE = {"Qwen": ["force_ent__b4096", "lookback_le_head#75", "pearl_flip__r0.25", "force_kl_mean", "force_p1__b6144", "anchor_pm_slope__l22", "dirproj_prompt_last__s5", "lens_emerge__b3072", "tok_ent_mean", "fc_p1__b5120"],
       "Skywork": ["kg_distinct__k32_d1", "dirproj_trace_mean__s5", "pearl_flip__r0.1", "norm_mean__d2", "lap_top1_head#292", "fc_p1_lasthalf", "pmass_le__all"],
       "allenai": ["lap_top1_head#577", "dirproj_seg25__s2", "kg_cycle_rank__k16_d1", "anchor_pl_mean__l31", "norm_mean__d0", "pearl_kl__r0.25", "speed_cv__all", "tok_ent_mean", "drift_code_typos__l4"],
       "deepseek": ["delta_cos__d1", "anchor_pl_slope__l16", "anchor_pl_lastq__l8", "pearl_flip__r0.25", "norm_last_over_prev", "tok_ent_gt2"]}


def table(tag, wb, wb2, ent, ship):
    """pid -> {feature: value} from battery jsonl + npz heads, wb2 jsonl + geometry, entropy npz features"""
    V = {}
    for l in open(A + f"ec2_results/{wb}/{tag}.jsonl"): d = json.loads(l); V.setdefault(d["pid"], {}).update(d)
    for l in open(A + f"ec2_results/{wb2}/{tag}.jsonl"): d = json.loads(l); V.setdefault(d["pid"], {}).update(d)
    for p in list(V):
        try:
            z = np.load(A + f"ec2_results/{wb}/{tag}/{p}.npz")
            for k in ("lookback_le_head", "pmass_le_head", "toha2048_head"):
                if k in z.files:
                    for j, x in enumerate(z[k].reshape(-1)): V[p][f"{k}#{j}"] = float(x)
            if "lap_topk" in z.files:
                for j, x in enumerate(z["lap_topk"][..., 0].reshape(-1)): V[p][f"lap_top1_head#{j}"] = float(x)
        except FileNotFoundError: pass
        try: V[p].update(W2.geometry(dict(np.load(A + f"ec2_results/{wb2}/{tag}/{p}.npz")), ship))
        except (FileNotFoundError, KeyError): pass
        try: V[p].update(feats_one(np.load(A + f"ec2_results/{ent}/{tag}/{p}.npz")))
        except FileNotFoundError: pass
    return V


rows, corr = [], []
for m, tag in TAGS.items():
    ship = dict(np.load(A + f"analysis/wb2_ship/wb2_ship_{tag}.npz"))
    P = table(tag, "wb", "wb2", "entropy", ship); M = table(tag, "wb_mv2", "wb2_mv2", "entropy_mv2", ship)
    try:                                                              # CKA drift (exp106) exists only for our pool -> skipped on main-v2
        pass
    except Exception: pass
    F = {json.loads(l)["pid"]: json.loads(l) for l in open(A + f"ec2_results/feats_mv2/{tag}.jsonl")}
    lab = pd.Series(labels(m)["forced"]); lab = lab[lab.index.isin(P)]; y = lab.values
    mids = [p for p in M if (HF[m], p) in MV2 and MV2[(HF[m], p)]["is_robust"] is not None and p in F]
    ym = np.array([int(MV2[(HF[m], p)]["is_robust"]) for p in mids]); fin = np.array([not F[p]["still_thinking"] for p in mids])
    print(f"\n=== {m}: main-v2 labelled {len(mids)} ({ym.sum()} R, {fin.sum()} finished within 6144) | our pool {len(P)} traces, {len(y)} labelled", flush=True)
    def evaluate(name, calls, kind):
        c = np.where(fin, 1, calls).astype(int)
        r = dict(model=m, candidate=name, kind=kind, n=len(mids), mv2_acc=float((c == ym).mean()), mv2_acc_unfinished=float((c[~fin] == ym[~fin]).mean()) if (~fin).any() else np.nan,
                 mv2_calls="".join(map(str, c)), mv2_truth="".join(map(str, ym)))
        lb = LB105[(LB105.model == m) & (LB105.feature == name)]
        if len(lb): r["lb_B"] = int(lb.lb_B.iloc[0]); r["lb_n"] = int(lb.n_lb.iloc[0])
        rows.append(r); return r
    # v8's own recipe (and branches for Olmo)
    pm = REC["per_model"][HF[m]]; rec = {**REC, **pm}; thr = REC["threshold_by_model"][HF[m]]
    if "router" in pm:
        for br in ("novel", "familiar"):
            b = pm["router"][br]; s = [score_trace({f: F[p].get(f, np.nan) for f in b["features"]}, {**rec, **b}, HF[m]) for p in mids]
            evaluate(f"v8 Olmo {br} branch", np.array(s) > b["threshold"], "v8")
    else:
        s = [score_trace({f: F[p].get(f, np.nan) for f in pm["features"]}, rec, HF[m]) for p in mids]
        evaluate("v8 recipe", np.array(s) > thr, "v8")
    if "override" in pm:                                              # ent4096 (Qwen): v8 call replaced by the override on unfinished problems
        O = pm["override"]; v = np.array([M[p].get(O["feature"], np.nan) for p in mids], float)
        evaluate("ent4096 (v8 + Qwen override)", np.where(np.isfinite(v), O["sign"] * (v - O["threshold"]) > 0, np.array(s) > thr), "v8")
    evaluate("all robust", np.ones(len(mids), int), "baseline")
    feats = sorted({k for p in mids for k, x in M[p].items() if isinstance(x, (int, float)) and not isinstance(x, bool)} & {k for p in P for k in P[p]})
    for f in feats:
        x = np.array([P[p].get(f, np.nan) if isinstance(P[p].get(f), (int, float)) else np.nan for p in P], float)
        xl = np.array([P[p].get(f, np.nan) if p in P and isinstance(P[p].get(f), (int, float)) else np.nan for p in lab.index], float)
        if np.isfinite(xl).sum() < 8 or np.nanstd(x) == 0: continue
        a_our = auc(xl[np.isfinite(xl)], y[np.isfinite(xl)]); sg = 1 if a_our >= .5 else -1; med = float(np.nanmedian(x))
        v = np.array([M[p].get(f, np.nan) for p in mids], float)
        if not np.isfinite(v).all(): continue
        if f in PRE[m]: evaluate(f, sg * (v - med) > 0, "pre-registered")
        u = ~fin
        if u.sum() >= 4 and 0 < ym[u].sum() < u.sum(): corr.append(dict(model=m, feature=f, auc_our=a_our, auc_mv2=auc(v[u], ym[u])))
T = pd.DataFrame(rows); T.to_csv(A + "analysis/exp109_mainv2.csv", index=False); C = pd.DataFrame(corr); C.to_csv(A + "analysis/exp109_corr.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
for m in TAGS:
    t = T[T.model == m]
    if len(t): print(f"\n=== {m} (main-v2 truth {t.mv2_truth.iloc[0]})"); print(t.drop(columns=["model", "mv2_truth"]).round(2).to_string(index=False))
print("\n=== does our-data AUC predict main-v2 AUC (unfinished main-v2 rows, every shared scalar)?")
C["strength"] = (C.auc_our - .5).abs(); C["agree"] = np.sign(C.auc_our - .5) == np.sign(C.auc_mv2 - .5)
for m, g in C.groupby("model"):
    print(f"  {m:8s} n {len(g)} | spearman(our, main-v2) {g.auc_our.corr(g.auc_mv2, method='spearman'):+.2f} | sign agreement |our-.5|>=.2: {g[g.strength >= .2].agree.mean():.2f} ({(g.strength >= .2).sum()}) | >=.3: {g[g.strength >= .3].agree.mean():.2f} ({(g.strength >= .3).sum()})")
