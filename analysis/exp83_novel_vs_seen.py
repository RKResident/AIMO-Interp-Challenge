"""exp83 (user, 2026-10-29: the leaderboard is >= 60% completely novel problems): every feature scored separately on problems each model
has probably NOT seen vs probably seen, and a check of Min-K%++ itself as a novelty detector (the user does not trust it yet).
Novelty per (model, problem) from the problem's date vs the model's release (exp75 web check):
  Olmo-3 (data cutoff Dec 2024): 2025+ novel | Skywork-OR1 (Apr 2025): 2026 novel, 2025 uncertain | DeepSeek-R1-0528 (May 2025): 2026 novel,
  2025 uncertain | Qwen3.5 (Mar 2026): 2026 uncertain, <= 2025 seen. Every problem dated <= 2024 is seen for all four.
Rows: unfinished labelled rows - main_v2 + olympiad (all features) and the set100 Qwen labels (forced class; paper features only).
Dates: olympiad pids from olympiad/selected_problems.csv; main_v2 problems matched to olympiad/candidates.csv by text (unmatched = unknown);
set100 from the pool source."""
import json, re, sys
import numpy as np, pandas as pd
A = "/home/sabyasachi19/PycharmProjects/AimoAGAIN/"; rng = np.random.default_rng(0)
src = open(A + "analysis/exp73_shift_and_permodel_search.py").read(); exec(src[:src.index('print("(1a) UNFINISHED AT 6144")')])
A = "/home/sabyasachi19/PycharmProjects/AimoAGAIN/"                                   # the exec above redefines A
norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())[:160]
C = pd.read_csv(A + "olympiad/candidates.csv"); YEAR = {norm(p): s for p, s in zip(C.problem, C.source)}
SEL = pd.read_csv(A + "olympiad/selected_problems.csv").set_index("pid")["source"].to_dict()
year_of = lambda s: int(re.search(r"(20\d\d)", s).group(1)) if isinstance(s, str) and re.search(r"(20\d\d)", s) else None


def novelty(model, yr):
    if yr is None: return "unknown"
    if yr <= 2024: return "seen"
    m = model.lower()
    if "olmo" in m or "allenai" in m: return "novel"
    if "skywork" in m or "deepseek" in m: return "novel" if yr >= 2026 else "uncertain"
    if "qwen" in m: return "uncertain" if yr >= 2026 else "seen"
    return "unknown"


O = U.copy()
O["source"] = [SEL.get(i.split("_")[-1]) if d == "olympiad" else YEAR.get(norm(p)) for i, d, p in zip(O.id, O.dataset, O.problem)]
O["year"] = [year_of(s) for s in O.source]; O["novelty"] = [novelty(m, y) for m, y in zip(O.model, O.year)]
P2 = pd.concat([pd.DataFrame([json.loads(l) for l in open(f)]).assign(model=f.split("/")[-2])
                for f in __import__("glob").glob(A + "ec2_results/a10g/papers2/*/papers2.jsonl")], ignore_index=True)
L2 = pd.read_csv(A + "analysis/exp81_pp_rows.csv")[["id", "pp_lid"]]
P2 = P2.merge(L2, on="id", how="left").rename(columns={"pp_lid": "lid_ref"})
PFE = ["li_sum", "li1_sum", "lid_ref", "minkpp_20", "minkpp_10", "rhythm_cooc_excess", "sup_logkl_top_minus_rnd", "sup_ans_top", "lara_dc_mid",
       "cie_gw", "rec_comp__ans_mean", "rec_comp__last_mean", "rec_formula__last_lastq"]
O = O.merge(P2[["id"] + PFE], on="id", how="left")
# set100 Qwen labelled rows (forced class)
Lq = pd.read_csv(A + "hard_pool/set100_labels_Qwen__Qwen3.5-4B.csv"); Lq = Lq[Lq.label_forced.isin(["robust", "non_robust"])]
Pool = pd.read_csv(A + "hard_pool/pool_screen.csv").set_index("pid")
S = P2[(P2.dataset == "set100") & (P2.model == "Qwen__Qwen3.5-4B")].copy(); S["pid"] = S.id.str.split("_").str[-1]
S = S.merge(Lq[["pid", "label_forced"]], on="pid"); S["robust"] = (S.label_forced == "robust").astype(int); S["model"] = "Qwen"
S["source"] = [Pool.loc[p, "source"] for p in S.pid]; S["year"] = [year_of(s) or (2025 if any(k in s for k in ("BeyondAIME", "final_answer", "smt_2025")) else 2024) for s in S.source]
S["novelty"] = [novelty("qwen", y) for y in S.year]; S["dataset"] = "set100"
ROWS = pd.concat([O.assign(src="orig"), S[["id", "model", "robust", "source", "year", "novelty", "dataset"] + PFE].assign(src="set100")], ignore_index=True)
print("labelled unfinished rows by novelty:")
print(ROWS.groupby(["model", "novelty"]).size().unstack(fill_value=0).to_string())
print("\nsources of the novel rows:", ROWS[ROWS.novelty == "novel"].source.value_counts().to_dict())


def auc(s, y):
    ok = np.isfinite(s); s, y = s[ok], y[ok]; p, n = s[y == 1], s[y == 0]
    return float(((p[:, None] > n[None]) + .5 * (p[:, None] == n[None])).mean()) if len(p) and len(n) else np.nan


def within(s, y, g):
    num = den = 0
    for m in np.unique(g):
        k = (g == m) & np.isfinite(s); p, n = s[k & (y == 1)], s[k & (y == 0)]
        num += ((p[:, None] > n[None]) + .5 * (p[:, None] == n[None])).sum(); den += len(p) * len(n)
    return num / den if den else np.nan


def perm(s, y, g, n=1000):
    a = abs(within(s, y, g) - .5); null = []
    for _ in range(n):
        yy = y.copy()
        for m in np.unique(g):
            k = g == m; yy[k] = rng.permutation(yy[k])
        null.append(abs(within(s, yy, g) - .5))
    return float(np.mean(np.array(null) >= a - 1e-12))


FE = [f for f in FEATS if f in ROWS] + PFE
print(f"\nAUC (raw, higher value -> robust), pooled within model | novel vs seen | n (robust) per class")
print(f"   {'feature':26s}  novel (p)        seen (p)        uncertain | same direction?")
for c in FE:
    out = []
    for cl in ("novel", "seen", "uncertain"):
        k = (ROWS.novelty == cl).values & ROWS[c].notna().values; s = ROWS[c].astype(float).values[k]; y = ROWS.robust.astype(int).values[k]; g = ROWS.model.values[k]
        out.append((within(s, y, g), perm(s, y, g) if k.sum() >= 6 else np.nan, int(k.sum()), int(y.sum())))
    (an, pn, nn, rn), (as_, ps, ns, rs), (au, _, nu, _) = out
    same = "" if not (np.isfinite(an) and np.isfinite(as_)) else ("same" if (an - .5) * (as_ - .5) > 0 else "OPPOSITE")
    print(f"   {c[:26]:26s}  {an:.2f} ({pn:.2f}) n{nn:<3d}  {as_:.2f} ({ps:.2f}) n{ns:<3d}  {au:.2f} n{nu:<3d} | {same}")
# Min-K%++ as a novelty detector: does it separate our seen vs novel problems (within model)?
print("\nMIN-K%++ AS A NOVELTY DETECTOR (is it lower on problems the model cannot have seen?) - AUC(novel vs seen), within model:")
P2["pid"] = P2.id.str.split("_").str[-1]
meta = O[["id", "novelty", "year"]].drop_duplicates("id")
Q = P2.merge(meta, on="id", how="left")
qs = Q.dataset == "set100"
Q.loc[qs, "year"] = [year_of(Pool.loc[p, "source"]) or (2025 if any(k in Pool.loc[p, "source"] for k in ("BeyondAIME", "final_answer", "smt_2025")) else 2024) for p in Q.loc[qs, "pid"]]
Q.loc[qs, "novelty"] = [novelty(m, y) for m, y in zip(Q.loc[qs, "model"], Q.loc[qs, "year"])]
for c in ("minkpp_20", "minkpp_10", "mink_20"):
    k = Q.novelty.isin(["novel", "seen"]).values
    yy = (Q.novelty[k] == "seen").astype(int).values; s = Q[c].astype(float).values[k]; g = Q.model.values[k]
    print(f"   {c:10s}: AUC(higher score -> seen) {within(s, yy, g):.2f} (p {perm(s, yy, g):.3f}) | rows " +
          ", ".join(f"{m[:8]} {int(((g == m) & (yy == 1)).sum())} seen / {int(((g == m) & (yy == 0)).sum())} novel" for m in np.unique(g)))
print("   (a working memorization detector gives AUC well above 0.5 here; leaderboard Qwen / Skywork values sit below every reference, exp82)")
ROWS.to_csv(A + "analysis/exp83_rows.csv", index=False)
