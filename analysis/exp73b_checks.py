"""exp73b: (i) is the NEGATIVE correlation between our feature-AUC profile and the leaderboard's (exp73 1c: -0.72 / -0.62) beyond chance?
Null: leaderboard labels shuffled within model (13 unfinished problems, robust counts kept). (ii) held-out (leave-one-problem-out) accuracy
on our rows of the per-model subsets the exp73 search favours, vs the current permodel-lookback choices."""
src = open("/home/sabyasachi19/PycharmProjects/AimoAGAIN/analysis/exp73_shift_and_permodel_search.py").read()
exec(src[:src.index('print("(1a) UNFINISHED AT 6144")')])
rng = np.random.default_rng(0)
SG = {"fa": 1, "chk": 1, "fas": 1, "nrk": -1, "f1k": -1, "cfc": -1, "cfl": -1, "nzf": -1, "jmp": 1, "lble": 1, "pmle": 1}
okU = np.arange(14) != 7
ours = np.array([within(SG[f] * U[f].values.astype(float), U.robust.values.astype(int), U.model.values) for f in FEATS])
def prof_lb(lab):
    return np.array([within(SG[f] * X[f][okU], lab, lm[okU]) for f in FEATS])
obs = {k: np.corrcoef(ours, prof_lb(LAB[k][okU]))[0, 1] for k in LAB}
null = []
for _ in range(3000):
    lab = LAB["A"][okU].copy()
    for m in np.unique(lm[okU]):
        ii = lm[okU] == m; lab[ii] = rng.permutation(lab[ii])
    p = prof_lb(lab)
    if np.isfinite(p).all(): null.append(np.corrcoef(ours, p)[0, 1])
null = np.array(null)
print("(i) correlation of our feature-AUC profile with the leaderboard's: observed A %.2f, B %.2f | null (shuffled labels) mean %.2f, "
      "5th pct %.2f | P(null <= observed A) = %.3f, B %.3f" % (obs["A"], obs["B"], null.mean(), np.percentile(null, 5), (null <= obs["A"]).mean(), (null <= obs["B"]).mean()))
print("\n(ii) held-out accuracy on OUR rows (leave-one-problem-out within the model) | AUC")
CAND = {"Qwen": [["jmp"], ["jmp", "cfc"], ["lble", "pmle", "jmp", "cfc"], ["cfc", "lble"]],
        "Skywork": [["chk", "fas", "nrk", "f1k", "cfc", "cfl", "nzf", "jmp"], ["chk", "f1k"], ["chk", "nrk", "f1k"], ["fa", "chk", "f1k"]],
        "allenai": [["lble", "pmle", "jmp", "cfc"], ["pmle"], ["chk", "pmle"], ["chk", "lble", "pmle"]],
        "deepseek": [["chk", "f1k", "cfc", "cfl"], ["chk", "nrk", "cfl"], ["chk", "nrk", "f1k", "cfc"], ["chk", "nrk", "cfl", "pmle"]]}
for m, cands in CAND.items():
    for fs in cands:
        s, c = heldout({mm: fs for mm in MODELS})
        k = mdl == m
        print(f"   {m:9s} {'+'.join(fs):34s} held-out AUC {auc(s[k], y[k]):.2f} acc {(c[k] == (y[k] == 1)).mean():.2f} (n={k.sum()})")
