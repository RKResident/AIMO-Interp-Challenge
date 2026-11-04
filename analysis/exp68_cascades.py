"""exp68: cascade combiners vs the equal-weight ensemble, same protocol as exp67 (88 unfinished rows, leave-one-PROBLEM-out, references and
per-model median thresholds refitted in every fold). Cascades (signed within-model percentiles p_f - 0.5):
  lex w     shipped form: score = p_primary + w * mean(p_secondary)  (w = 0.01 lexicographic tie-break; 0.25 / 0.5 softer)
  gate b    primary decides unless it is in the middle band |p_primary| < b; there the secondaries' mean decides
Primary fixed in advance (chk or jmp) or chosen inside the fold (best training AUC). Reported overall and per model (AUC / accuracy)."""
import numpy as np
exec(open("/home/sabyasachi19/PycharmProjects/AimoAGAIN/analysis/exp67_ensembles_permodel.py").read().split("SETS = {")[0])


def combos(feats):
    """held-out scores for eqw and the cascade variants; returns {name: (score, calls)}."""
    names = ["eqw"] + [f"lex{w} P={p}" for p in ("first", "fold") for w in (0.01, 0.25, 0.5)] + [f"gate{b} P={p}" for p in ("first", "fold") for b in (0.1, 0.2)]
    S = {n: np.full(len(U), np.nan) for n in names}; C = {n: np.zeros(len(U), bool) for n in names}
    for g in np.unique(grp):
        te = grp == g; tr = ~te; itr, ite = np.where(tr)[0], np.where(te)[0]
        Ptr, Pte = signed_pct(tr, itr, feats), signed_pct(tr, ite, feats)
        best = int(np.argmax([auc(Ptr[:, k], y[itr]) for k in range(len(feats))]))
        def put(n, s_tr, s_te):
            th = {m: np.median(s_tr[mdl[itr] == m]) for m in MODELS if (mdl[itr] == m).any()}
            S[n][ite] = s_te; C[n][ite] = [s > th.get(mdl[i], 0) for s, i in zip(s_te, ite)]
        put("eqw", Ptr.mean(1), Pte.mean(1))
        for pn, k in (("first", 0), ("fold", best)):
            rest = [j for j in range(len(feats)) if j != k]
            sec_tr, sec_te = Ptr[:, rest].mean(1), Pte[:, rest].mean(1)
            for w in (0.01, 0.25, 0.5):
                put(f"lex{w} P={pn}", Ptr[:, k] + w * sec_tr, Pte[:, k] + w * sec_te)
            for b in (0.1, 0.2):
                put(f"gate{b} P={pn}", np.where(np.abs(Ptr[:, k]) >= b, Ptr[:, k], sec_tr), np.where(np.abs(Pte[:, k]) >= b, Pte[:, k], sec_te))
    return S, C


SETS = {"chk+jmp": ["chk", "jmp"], "jmp+chk": ["jmp", "chk"], "chk+fas+jmp": ["chk", "fas", "jmp"], "jmp+chk+fas": ["jmp", "chk", "fas"],
        "chk+fas+5WB+jmp": ["chk", "fas", "nrk", "f1k", "cfc", "cfl", "nzf", "jmp"], "jmp+chk+fas+5WB": ["jmp", "chk", "fas", "nrk", "f1k", "cfc", "cfl", "nzf"],
        "fa+ge (old 0.71 cascade family: fa first)": ["fa", "chk", "fas"]}
print("held-out AUC / accuracy. 'P=first' = the first listed feature is the primary; 'P=fold' = chosen by training AUC inside each fold")
for nm, fs in SETS.items():
    S, C = combos(fs)
    print(f"\n{nm}")
    for n in S:
        per = "  ".join(f"{m[:4]} {auc(S[n][(U.model == m).values], y[(U.model == m).values]):.2f}/{(C[n][(U.model == m).values] == (y[(U.model == m).values] == 1)).mean():.2f}" for m in MODELS)
        print(f"  {n:16s} all {auc(S[n], y):.2f}/{(C[n] == (y == 1)).mean():.2f} | {per}")
