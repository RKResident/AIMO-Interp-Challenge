"""Reproduce the report's train-main-v2 numbers on CPU in a few seconds, from the shipped recipes and our published feature values.

Inputs
  reproduce/features.csv                   per-trace feature values of our 6144-token greedy traces (columns: README, "Reproducing the report")
  submitted_solutions/*.zip                the two nominated submissions; their spurious_artifacts/recipe.json holds every reference value,
                                           sign, threshold, router and override the submissions use
  submission_core/spurious_features.py     the scoring function the submissions run (score_trace)
  train-main-v2 parquet (organisers')      official labels for the 43 Main rows; joined by pid = 'mv2_' + sha256(problem.strip())[:8]

What it prints (Main columns only; leaderboard columns are quoted in the report as received)
  Table 1  baselines and the two submissions                 Table 2  components (v8 base, router, H4096, Olmo direction)
  Table 3  white-box ablation, per model                     Appendix B  drop-one-feature
  Table 4  H_b by truncation / Olmo direction by position     Appendix C  per-feature AUC
Each value is printed next to the number in the report. Olmo's Qwen judge is not replayed (it needs a Qwen trace of each Olmo problem); as in
the report, Olmo's own router branch decides on Main.

  python reproduce/replay.py --main /path/to/train-main-v2.parquet"""
import argparse, copy, hashlib, json, os, sys, zipfile
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "submission_core"))
from spurious_features import score_trace  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--main", required=True, help="the organisers' train-main-v2 parquet (columns model_id, problem, is_robust)")
args = ap.parse_args()

QWEN, SKY, OLMO, DS = "Qwen/Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B", "allenai/Olmo-3-7B-Think", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
MODELS = [QWEN, SKY, OLMO, DS]
SHORT = {QWEN: "Qwen", SKY: "Skywork", OLMO: "Olmo", DS: "DeepSeek"}

# ---------------------------------------------------------------- data
F = pd.read_csv(os.path.join(ROOT, "reproduce/features.csv"))
off = pd.read_parquet(args.main)
off["pid"] = "mv2_" + off.problem.map(lambda s: hashlib.sha256(s.strip().encode()).hexdigest()[:8])
lab = {(r.model_id, r.pid): r.is_robust for r in off.itertuples() if pd.notna(r.is_robust)}
MAIN = F[F.set == "main"].copy()
MAIN["y"] = [lab.get((m, p), np.nan) for m, p in zip(MAIN.model, MAIN.pid)]
if MAIN.y.isna().any():
    sys.exit(f"{int(MAIN.y.isna().sum())} Main rows have no official label in {args.main}; use the current train-main-v2 release "
             "(the 2026-11-02 version on the organisers' Hugging Face page has all 43).")
MAIN["y"] = MAIN.y.astype(int)
print(f"Main: {len(MAIN)} rows, {int(MAIN.y.sum())} robust, {int((~MAIN.finished).sum())} unfinished")


def recipe(z):
    return json.load(zipfile.ZipFile(os.path.join(ROOT, "submitted_solutions", z)).open("spurious_artifacts/recipe.json"))


S1 = recipe("permodel-v8-ent4096-small.zip")
S2 = recipe("permodel-v8-ent4096-olmoseg25-small.zip")
V8 = copy.deepcopy(S1); V8["per_model"][QWEN].pop("override")            # v8 = S1 without Qwen's H4096 readout (the recipes differ only there)
OQ, OO = S1["per_model"][QWEN]["override"], S2["per_model"][OLMO]["override"]


# ---------------------------------------------------------------- the submission's decision, per trace
def call(R, row):
    """1 = robust. Mirrors spurious_inference: finished -> robust; per-model rank ensemble (Olmo: Min-K% router branch); override on
    unfinished traces when its feature is available."""
    if R is None: return 1
    if R == "finished_only": return int(row.finished)
    if row.finished: return 1
    m = row.model; pm = R["per_model"][m]; rec = {**R, **pm}
    feats, thr, r_ = pm["features"], R["threshold_by_model"][m], rec
    if "router" in pm:
        rt = pm["router"]; x = row.mink_20
        br = rt[("novel" if x < rt["threshold"] else "familiar") if np.isfinite(x) else rt.get("default", "novel")]
        feats, r_, thr = br["features"], {**rec, **br}, br["threshold"]
    c = int(score_trace({f: getattr(row, f) for f in feats}, r_, m) > thr)
    O = pm.get("override")
    if O and np.isfinite(getattr(row, O["feature"], np.nan)): c = int(O["sign"] * (getattr(row, O["feature"]) - O["threshold"]) > 0)
    return c


def correct(R, model=None):
    rows = MAIN if model is None else MAIN[MAIN.model == model]
    return int(sum(call(R, r) == r.y for r in rows.itertuples()))


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# ---------------------------------------------------------------- ablation variants (as in the report)
INTERNAL, OUTPUT = {"cfc", "lble", "pmle", "jmp"}, {"fa", "nrk", "f1k", "cfl"}


def with_overrides(qwen=False, olmo=False, router=True):
    R = copy.deepcopy(V8)
    if qwen: R["per_model"][QWEN]["override"] = OQ
    if olmo: R["per_model"][OLMO]["override"] = OO
    if not router:                                    # Olmo: one feature list, with the routed f1k sign (+1)
        pm = R["per_model"][OLMO]; pm.pop("router"); pm.pop("judge", None); pm["signs"] = {**R["signs"], "f1k": 1}
    return R


def zero_thresholds(R):
    R["threshold_by_model"] = {k: 0.0 for k in R["threshold_by_model"]}
    return R


def text_only(R):
    R = copy.deepcopy(R)
    for m in R["per_model"]: R["per_model"][m] = {"features": ["chk"]}
    return zero_thresholds(R)


def strip_internal(R):
    """drop hidden-state / attention features, the judge and the hidden-state direction; a model left empty falls back to chk"""
    R = copy.deepcopy(R)
    for pm in R["per_model"].values():
        pm["features"] = [x for x in pm["features"] if x not in INTERNAL] or ["chk"]
        if "router" in pm:
            for b in ("novel", "familiar"):
                br = pm["router"][b]; br["features"] = [x for x in br["features"] if x not in INTERNAL] or ["chk"]
                br["signs"] = {x: br["signs"].get(x, R["signs"][x]) for x in br["features"]}; br["threshold"] = 0.0
        pm.pop("judge", None)
        if pm.get("override", {}).get("feature", "").startswith("dirproj"): pm.pop("override")
    return zero_thresholds(R)


def strip_output(R):
    """drop output-distribution readouts (fa, nrk, f1k, cfl, the Min-K% router, H4096); internals, the direction stay"""
    R = copy.deepcopy(R)
    for pm in R["per_model"].values():
        pm["features"] = [x for x in pm["features"] if x not in OUTPUT] or ["chk"]
        pm.pop("router", None)
        if pm.get("override", {}).get("feature", "").startswith("force_"): pm.pop("override")
    return zero_thresholds(R)


def show(title, rows, per_model=False):
    print(f"\n=== {title}")
    for name, R, expect in rows:
        k = correct(R); a, b = wilson(k, len(MAIN))
        pm = "  " + " ".join(f"{SHORT[m]} {correct(R, m)}" for m in MODELS) if per_model else ""
        flag = "" if expect is None or k == expect else f"   <-- report says {expect}"
        print(f"  {name:48s} {k:2d}/43  [{a:.2f}, {b:.2f}]{pm}   (report: {expect}){flag}")


show("Table 1 (Main column)", [("All robust", None, 31), ("Finished only", "finished_only", 23), ("S1", S1, 33), ("S2", S2, 36),
                                ("Text only (finished rule + chk)", text_only(V8), 34)])
show("Table 2 (Main column)", [("v8", V8, 32), ("v8 without router / judge", with_overrides(router=False), 34), ("S1 = v8 + H4096", S1, 33),
                                ("v8 + Olmo direction", with_overrides(olmo=True), 35), ("S2", S2, 36)])
show("Table 3 (Main, per model: Qwen / Skywork / Olmo / DeepSeek)",
     [("Text only (10/9/8/7)", text_only(V8), 34), ("Text + output, no internals (9/10/6/8)", strip_internal(S1), 33),
      ("S1 text + internal (8/9/6/7)", strip_output(S1), 30), ("S2 text + internal (8/9/8/7)", strip_output(S2), 32)], per_model=True)

drop = []
for m, rep in ((SKY, {"fa": 32, "chk": 33, "f1k": 33}), (DS, {"chk": 32, "nrk": 31, "cfl": 31, "pmle": 31})):
    for f in S1["per_model"][m]["features"]:
        R = copy.deepcopy(S1); R["per_model"][m]["features"] = [x for x in R["per_model"][m]["features"] if x != f]
        drop.append((f"{SHORT[m]}: drop {f}", R, rep[f]))
for f in S1["per_model"][OLMO]["router"]["familiar"]["features"]:
    R = copy.deepcopy(S1); br = R["per_model"][OLMO]["router"]["familiar"]; br["features"] = [x for x in br["features"] if x != f]
    br["signs"] = {x: br["signs"][x] for x in br["features"]}; drop.append((f"Olmo familiar branch: drop {f}", R, 34))
R = copy.deepcopy(S1); br = R["per_model"][OLMO]["router"]["novel"]; br["features"], br["signs"] = ["chk"], {"chk": 1}
drop.append(("Olmo novel branch: f1k -> chk", R, 35))
R = copy.deepcopy(S1); R["per_model"][QWEN].pop("override"); drop.append(("Qwen: drop H4096", R, 32))
show("Appendix B, drop one feature from S1 (Main column)", drop)


# ---------------------------------------------------------------- AUC tables (unfinished traces, oriented by the shipped sign)
def auc(df, f, sign):
    d = df[~df.finished & df[f].notna()]
    y = (d.label == "robust").astype(int).values if "y" not in d else d.y.values
    x = sign * d[f].values; a, b = x[y == 1], x[y == 0]
    if not len(a) or not len(b): return np.nan
    return float(((a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()) / (len(a) * len(b)))


def sub(s, m): return MAIN[MAIN.model == m] if s == "main" else F[(F.set == f"ours_{s}") & (F.model == m)]


print("\n=== Table 4: Qwen H_b (sign -), ours (first sets) / Main")
for b, rep in ((512, "0.44 / 0.75"), (1024, "0.49 / 0.64"), (2048, "0.61 / 0.71"), (3072, "0.69 / 0.75"), (4096, "0.77 / 0.71"), (6144, "0.79 / --")):
    print(f"  b={b:5d}  {auc(sub('first', QWEN), f'force_ent__b{b}', -1):.2f} / {auc(sub('main', QWEN), f'force_ent__b{b}', -1):.2f}   (report: {rep})")
print("=== Table 4: Olmo direction (sign +), ours (first sets; in-sample) / Main")
for s, rep in ((25, "0.99 / 1.00"), (50, "0.97 / 1.00"), (75, "0.93 / 1.00")):
    print(f"  {s}% of trace  {auc(sub('first', OLMO), f'dirproj_seg{s}__s2', 1):.2f} / {auc(sub('main', OLMO), f'dirproj_seg{s}__s2', 1):.2f}   (report: {rep})")

print("\n=== Appendix C: per-feature AUC, ours first / ours matched / Main")
SHIP = {QWEN: [("force_ent__b4096", -1)] + [(f, S1["signs"][f]) for f in ("lble", "pmle", "jmp", "cfc")],
        SKY: [(f, S1["signs"][f]) for f in ("fa", "chk", "f1k")],
        OLMO: [("f1k", 1), ("chk", 1), ("lble", 1), ("pmle", 1), ("dirproj_seg25__s2", 1)],
        DS: [(f, S1["signs"][f]) for f in ("chk", "nrk", "cfl", "pmle")]}
for m, fs in SHIP.items():
    for f, s in fs:
        print(f"  {SHORT[m]:9s} {f:18s} {'+' if s > 0 else '-'}  " + " / ".join(f"{auc(sub(k, m), f, s):.2f}" for k in ("first", "matched", "main")))
