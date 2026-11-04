"""Per-feature evidence for the shipped recipes (2026-11-05, report appendix). For each model and each feature the submissions use, the
robust-vs-non-robust AUC on UNFINISHED traces, oriented by the shipped sign (> 0.5 = the feature points the way the recipe uses it), on:
  first    our first labelled sets (set100 / olympiad, forced labels at 6144) - the signs were chosen here, so this is in-sample
  matched  the confidence-matched set (forced labels at 6144) - not used for any choice
  main     train-main-v2 unfinished rows (official labels)
Olmo's f1k is reported with both signs (novel branch +1, familiar / global -1). No leaderboard numbers.
  python analysis/report_featauc.py      -> analysis/report_featauc.csv"""
src = open("/home/sabyasachi19/PycharmProjects/AimoAGAIN/analysis/exp121_selection.py").read()
src = src[:src.index("# ---------------------------------------------------------------- leaderboard rows")]
src = src.replace('analysis/exp121.log", "w"', 'analysis/report_featauc.log", "w"')
exec(src)
from scipy.stats import mannwhitneyu

SHIP = {"Qwen": [("force_ent__b4096", -1, "S1/S2 readout")] + [(f, FIN["signs"][f], "fallback ensemble") for f in ("lble", "pmle", "jmp", "cfc")],
        "Skywork": [(f, FIN["signs"][f], "ensemble") for f in ("fa", "chk", "f1k")],
        "allenai": [("f1k", 1, "novel branch"), ("chk", 1, "familiar branch"), ("lble", 1, "familiar branch"), ("pmle", 1, "familiar branch"),
                    ("dirproj_seg25__s2", 1, "S2 direction")],
        "deepseek": [(f, FIN["signs"][f], "ensemble") for f in ("chk", "nrk", "cfl", "pmle")]}


def auc1(rows, f, sign):
    v = np.array([r.get(f, np.nan) if isnum(r.get(f, np.nan)) else np.nan for r in rows if not r["_fin"]], float)
    y = np.array([r["_y"] for r in rows if not r["_fin"]]); ok = np.isfinite(v); v, y = sign * v[ok], y[ok]
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if not n1 or not n0: return np.nan, n1, n0
    return float(mannwhitneyu(v[y == 1], v[y == 0]).statistic / (n1 * n0)), n1, n0


out = []
for m, fs in SHIP.items():
    for f, s, role in fs:
        r = dict(model=m, feature=f, sign=s, role=role)
        for k in ("first", "matched", "main"):
            a, n1, n0 = auc1(DATA2[m][k], f, s); r[f"auc_{k}"], r[f"n_{k}"] = a, f"{n1}R/{n0}NR"
        out.append(r); say(f"{m:9s} {f:20s} {s:+d} {role:18s} first {r['auc_first']:.2f} ({r['n_first']})  matched {r['auc_matched']:.2f} ({r['n_matched']})  main {r['auc_main']:.2f} ({r['n_main']})")
pd.DataFrame(out).to_csv(A + "analysis/report_featauc.csv", index=False)
