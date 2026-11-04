"""exp103 (user 2026-10-31): the OLD white-box approaches the retries skipped, rerun on the new labelled traces (run_wb2.py outputs):
  anchoring cosine (window vs prompt mean / last prompt token; mean, last quarter, slope, drop; every layer), answer-commitment curve (forced
  entropy / p1 / same-as-final per budget 512..6144, commit budget, answer changes, KL between budgets), logit lens at the forced position
  (emergence depth, late-layer entropy, mid-layer prob of the answer), PEARL prose noise (KL, flips, NLL change, drift at 2 rates),
  K-means state graphs (64-token windows at 25/50/75% depth, K = 16 / 32, label-free per model: cycle rank, entropy rate, self-loops,
  distinct states, returns, spectral gap), and difference-of-means DIRECTIONS (segment 25/50/75%, trace mean, last prompt token; every 2nd layer).
Evaluation as exp99/exp102: stratified 5-fold x 20 repeats. Per family, NESTED choice of the member (layer / budget / rate / K) by training
accuracy (sign by training AUC, threshold = label-free pool median), plus every single scalar with its sign fitted in the training folds.
Directions are fitted in the training folds (mean robust - mean non-robust), projections thresholded at the pool median.
Splits novel / seen (Qwen, Olmo), finished / unfinished; forced and pure. No leaderboard values exist for these features (not logged).
Writes analysis/exp103_singles.csv, exp103_families.csv."""
import json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

src = open("/home/sabyasachi19/PycharmProjects/AimoAGAIN/analysis/exp99_cv_candidates.py").read()
exec(src[:src.index("out, nested = [], []")])
RNG = np.random.default_rng(3); REP = 20


def kmeans(X, k, it=30, seed=0):
    r = np.random.default_rng(seed); C = X[r.choice(len(X), k, replace=False)].copy()
    for _ in range(it):
        lab = np.argmax(X @ C.T, 1)
        for j in range(k):
            w = lab == j
            if w.any(): c = X[w].mean(0); C[j] = c / (np.linalg.norm(c) + 1e-8)
    return C


def walk_feats(seq, k):
    n = len(seq); T = np.zeros((k, k))
    for a, b in zip(seq, seq[1:]): T[a, b] += 1
    E = (T + T.T > 0) & ~np.eye(k, dtype=bool); V = np.unique(seq).size; ne = E.sum() / 2
    P = T / np.maximum(T.sum(1, keepdims=True), 1); pi = np.bincount(seq, minlength=k) / n
    with np.errstate(all="ignore"): er = float(-np.nansum(pi[:, None] * P * np.log(np.where(P > 0, P, 1))))
    D = np.diag((E).sum(1)); Lp = D - E; ev = np.sort(np.linalg.eigvalsh(Lp[np.ix_(np.unique(seq), np.unique(seq))].astype(float)))
    seen, ret = set(), 0
    for i, s in enumerate(seq):
        if s in seen and seq[i - 1] != s: ret += 1
        seen.add(s)
    return dict(cycle_rank=float(ne - V + 1), entropy_rate=er, self_loops=float(np.trace(T) / max(1, n - 1)), distinct=float(V / n),
                returns=float(ret / n), spectral_gap=float(ev[1]) if len(ev) > 1 else 0.0)


def unitn(X): return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)


singles, fams_out = [], []
for m, tag in TAGS.items():
    try: B = pd.DataFrame([json.loads(l) for l in open(A + f"ec2_results/wb2/{tag}.jsonl")]).drop_duplicates("pid").set_index("pid")
    except FileNotFoundError: print(f"{m}: no wb2 output yet"); continue
    Z = {p: np.load(A + f"ec2_results/wb2/{tag}/{p}.npz") for p in B.index}
    Z = {p: {k: z[k] for k in z.files} for p, z in Z.items()}
    # aggregates over layers for the anchoring family
    L = max(int(c.split("__l")[-1]) for c in B.columns if c.startswith("anchor_pm_mean__l")) + 1
    for base in ("anchor_pm_mean", "anchor_pm_lastq", "anchor_pm_slope", "anchor_pm_drop", "anchor_pl_mean", "anchor_pl_lastq", "anchor_pl_slope", "anchor_pl_drop"):
        cols = [f"{base}__l{l}" for l in range(L)]; B[f"{base}__all"] = B[cols].mean(1); B[f"{base}__ge40"] = B[cols[int(.4 * L):]].mean(1)
    # K-means state graphs (label-free, per model, all windows of all traces)
    ok = [p for p in B.index if "w64_3layers" in Z[p]]
    for li in range(3):
        allw = np.concatenate([unitn(Z[p]["w64_3layers"][li].astype(np.float32)) for p in ok])
        for k in (16, 32):
            C = kmeans(allw, k, seed=li)
            for p in ok:
                seq = np.argmax(unitn(Z[p]["w64_3layers"][li].astype(np.float32)) @ C.T, 1)
                for nm, v in walk_feats(seq, k).items(): B.loc[p, f"kg_{nm}__k{k}_d{li}"] = v
    fams = {}
    for c in B.columns:
        if c in ("model", "n_gen", "still_thinking") or not pd.api.types.is_numeric_dtype(B[c]): continue
        if c.startswith("anchor") and "__l" in c: fams.setdefault(c.split("__")[0] + " (layer)", []).append(c)
        elif c.startswith(("force_", "lens_")) and "__b" in c: fams.setdefault(c.split("__")[0] + " (budget)", []).append(c)
        elif c.startswith("pearl"): fams.setdefault(c.split("__")[0] + " (rate)", []).append(c)
        elif c.startswith("kg_"): fams.setdefault(c.split("__")[0] + " (K, depth)", []).append(c)
    fams["anchor aggregates"] = [c for c in B.columns if c.startswith("anchor") and c.endswith(("__all", "__ge40"))]
    fams["commitment"] = ["commit_budget", "commit_frac", "n_answer_changes", "force_kl_mean", "force_kl_last", "force_ent_slope"]
    feats = sorted({c for v in fams.values() for c in v}); FI = {f: j for j, f in enumerate(feats)}
    POOL = B[feats].astype(float).values; MED = np.nanmedian(POOL, 0)
    # direction arrays: [pid] -> {name: [layers, d]}
    DIRS = {}
    for p in ok:
        z = Z[p]; DIRS.setdefault("dir_prompt_last", {})[p] = z["prompt_last"].astype(np.float32); DIRS.setdefault("dir_trace_mean", {})[p] = z["trace_mean"].astype(np.float32)
        for si, f in enumerate((25, 50, 75)): DIRS.setdefault(f"dir_seg{f}", {})[p] = z["seg_means"][:, si].astype(np.float32)
    LAB = labels(m)
    for rule in ("forced", "pure"):
        lab = pd.Series(LAB[rule]); lab = lab[lab.index.isin(B.index)]; y = lab.values; n = len(y)
        if n < 10 or y.min() == y.max(): continue
        X = B.loc[lab.index, feats].astype(float).values
        nv = np.array([(p in NOV_O) if m == "allenai" else (SRC.get(p, "").split("/")[-1] in NOVSRC) if m == "Qwen" else True for p in lab.index])
        fn = (B.loc[lab.index, "still_thinking"] == False).values  # noqa: E712
        OS = np.full((REP, n, len(feats)), np.nan); OF = {k: np.full((REP, n), np.nan) for k in list(fams) + list(DIRS)}; picks = {k: {} for k in OF}
        for r in range(REP):
            folds = np.zeros(n, int)
            for cls in (0, 1):
                w = np.where(y == cls)[0]; RNG.shuffle(w); folds[w] = np.arange(len(w)) % 5
            for k_ in range(5):
                te, tr = folds == k_, folds != k_
                a = np.array([auc(X[tr, j], y[tr]) if np.isfinite(X[tr, j]).sum() > 3 else .5 for j in range(len(feats))]); sg = np.where(a >= .5, 1, -1)
                C = sg * (X - MED) > 0; OS[r, te] = C[te]
                acc_tr = (C[tr] == y[tr, None]).mean(0)
                for fam, cols in fams.items():
                    js = [FI[c] for c in cols]; j = js[int(np.nanargmax(acc_tr[js]))]; OF[fam][r, te] = C[te, j]; picks[fam][feats[j]] = picks[fam].get(feats[j], 0) + 1
                for dn, d in DIRS.items():                                  # direction per layer fitted on the training rows; nested layer choice
                    M = np.stack([d[p] for p in lab.index]); Pl = np.stack(list(d.values()))     # [n, layers, dim], pool
                    best = (-1, None, None)
                    for l in range(M.shape[1]):
                        v = M[tr & (y == 1), l].mean(0) - M[tr & (y == 0), l].mean(0); v /= np.linalg.norm(v) + 1e-8
                        pr, pp = M[:, l] @ v, Pl[:, l] @ v; t = np.median(pp); at = ((pr[tr] > t) == y[tr]).mean()
                        if at > best[0]: best = (at, l, pr[te] > t)
                    OF[dn][r, te] = best[2]; picks[dn][best[1]] = picks[dn].get(best[1], 0) + 1

        def summ(Cm, mk): return float(np.mean([(Cm[r, mk] == y[mk]).mean() for r in range(REP)])) if mk.sum() >= 4 else np.nan
        SPL = (("all", np.ones(n, bool)), ("novel", nv), ("seen", ~nv), ("finished", fn), ("unfinished", ~fn))
        for j, f in enumerate(feats):
            row = dict(model=m, feature=f, rule=rule, n=n, auc_all=auc(X[:, j], y))
            for sp, mk in SPL:
                if sp == "all" or not mk.all(): row[f"acc_{sp}"] = summ(OS[:, :, j], mk)
            singles.append(row)
        for fam, Cm in OF.items():
            row = dict(model=m, family=fam, rule=rule, n=n, n_robust=int(y.sum()), n_members=len(fams.get(fam, [])) or "layers",
                       top_pick=max(picks[fam], key=picks[fam].get), top_pick_share=max(picks[fam].values()) / (REP * 5))
            for sp, mk in SPL:
                if sp == "all" or not mk.all(): row[f"acc_{sp}"] = summ(Cm, mk)
            row["sd_all"] = float(np.std([(Cm[r] == y).mean() for r in range(REP)])); fams_out.append(row)
        print(f"{m} {rule}: n={n} done", flush=True)
S = pd.DataFrame(singles); F = pd.DataFrame(fams_out)
S.to_csv(A + "analysis/exp103_singles.csv", index=False); F.to_csv(A + "analysis/exp103_families.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.max_colwidth", 34)
for m in F.model.unique():
    print(f"\n=== {m}: NESTED family accuracy (member chosen in each training fold)")
    X = F[F.model == m].pivot_table(index="family", columns="rule", values=["acc_all", "acc_novel", "acc_unfinished"], aggfunc="first").round(2)
    X.columns = [f"{a}_{b}" for a, b in X.columns]
    tp = F[(F.model == m) & (F.rule == "forced")].set_index("family")[["top_pick", "top_pick_share"]]
    print(X.join(tp).sort_values("acc_all_forced", ascending=False).to_string())
    print(f"--- {m}: best single scalars (sign fitted in-fold), forced")
    print(S[(S.model == m) & (S.rule == "forced")].sort_values("acc_all", ascending=False).head(12).round(2).to_string(index=False))
