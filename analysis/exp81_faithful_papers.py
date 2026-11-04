"""exp81: the FAITHFUL paper features (submission_core/spurious_papers.py via run_papers2.py on the A10G) - scoring, LID, signs, shipping.
1. LID (arXiv 2402.18048): per model, the reference set R = every trace's final forced-answer-position state (our labelled traces + the
   set100 screening traces of that model); layer = argmax_l sum_i LID_l(x_i) + 1 (the paper's rule, leave-self-out, T = min(50, n // 2));
   each row's LID against R without itself.
2. Every feature scored on the 88 unfinished original rows (pooled within-model AUC, per-model AUC, permutation p, overlap with fa / chk /
   fas, residual AUC over that base) and on the set100 Qwen labels; the 8 primary scores (spurious_papers.PP_MAP) are flagged.
3. Signs of the primaries = direction of their pooled within-model AUC on the 88 unfinished rows (our data only).
4. Writes exp81_pp_rows.csv (primaries by row id, for the builder), exp81_pp_signs.json, exp81_ship/papers_<model>.npz (LID layer +
   R in fp16 + recall directions at their recall-chosen layers) and exp81.log."""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
A = Path("/home/sabyasachi19/PycharmProjects/AimoAGAIN/"); R = A / "ec2_results/a10g/papers2"; DIRS = A / "ec2_results/a10g/recall_dirs"
sys.path.insert(0, str(A / "submission_core"))
from spurious_papers import PP_MAP, lid_mle                                          # noqa: E402
rng = np.random.default_rng(0); torch.set_num_threads(4)


def auc(s, y):
    ok = np.isfinite(s); s, y = s[ok], y[ok]; p, n = s[y == 1], s[y == 0]
    return float(((p[:, None] > n[None]) + .5 * (p[:, None] == n[None])).mean()) if len(p) and len(n) else np.nan


def within(s, y, g):
    num = den = 0
    for m in np.unique(g):
        k = (g == m) & np.isfinite(s); p, n = s[k & (y == 1)], s[k & (y == 0)]
        num += ((p[:, None] > n[None]) + .5 * (p[:, None] == n[None])).sum(); den += len(p) * len(n)
    return num / den if den else np.nan


def perm(s, y, g, n=2000):
    a = abs(within(s, y, g) - .5); null = []
    for _ in range(n):
        yy = y.copy()
        for m in np.unique(g):
            k = g == m; yy[k] = rng.permutation(yy[k])
        null.append(abs(within(s, yy, g) - .5))
    return float(np.mean(np.array(null) >= a - 1e-12))


rows, ship = [], {}
(A / "analysis/exp81_ship").mkdir(exist_ok=True)
for d in sorted(glob.glob(str(R / "*/"))):
    m = d.rstrip("/").split("/")[-1]; recs = [json.loads(l) for l in open(d + "/papers2.jsonl")]
    St = np.stack([np.load(f"{d}/states/{r['id']}.npy").astype(np.float32) for r in recs])          # [n, L, d]
    n, L, _ = St.shape; T = max(3, min(50, n // 2)); X = torch.tensor(St)
    tot = []
    for l in range(L):
        Dl = torch.cdist(X[:, l], X[:, l]); Dl.fill_diagonal_(float("inf")); Q = Dl.topk(T, largest=False).values.clamp(min=1e-6)
        tot.append(float((1.0 / torch.log(Q[:, -1:] / Q[:, :-1]).mean(-1).clamp(min=1e-6)).sum()))
    lay = int(min(L - 1, int(np.argmax(tot)) + 1))                                     # the paper's rule: argmax + 1
    Rl = X[:, lay]
    for k, r in enumerate(recs):
        keep = torch.ones(n, dtype=torch.bool); keep[k] = False; r["lid_ref"] = lid_mle(Rl[k], Rl[keep], T); r["model"] = m; rows.append(r)
    z = np.load(DIRS / m / "directions.npz"); info = json.load(open(DIRS / m / "directions.json")); names, lays, U = [], [], []
    for nm in sorted(z.files):
        fam, t = nm.split("__"); b = info[fam]["best_layer"][t]; names.append(nm); lays.append(b - 1); U.append(z[nm][b])
    np.savez_compressed(A / "analysis/exp81_ship" / f"papers_{m}.npz", lid_layer=lay, lid_R=Rl.numpy().astype(np.float16), lid_T=T,
                        rec_names=np.array(names), rec_layers=np.array(lays), rec_U=np.stack(U).astype(np.float16))
    print(f"{m}: {n} reference traces, LID layer {lay} of {L} (summed-LID argmax + 1), T = {T}")
D = pd.DataFrame(rows)
for f, k in PP_MAP.items(): D[f] = D[k]
FE = [c for c in D.columns if c not in ("id", "dataset", "robust", "cut", "think_end", "model", "n_sent", "n_answer_tokens", "minkpp_span_found",
                                         "lara_error", "lara_n_neighbours", "lid_T") and not c.startswith("pp_") and pd.api.types.is_numeric_dtype(D[c])]
B = pd.read_csv(A / "analysis/exp56_rows.csv")[["id", "fa", "chk", "fas"]]
O = D[D.robust.notna() & D.think_end.isna()].merge(B, on="id", how="left").reset_index(drop=True)
y = O.robust.astype(int).values; g = O.model.values
for e in ("fa", "chk", "fas"): O[e + "_r"] = O.groupby("model")[e].rank(pct=True)
O["base"] = O[["fa_r", "chk_r", "fas_r"]].mean(axis=1).rank(pct=True)
PRIM = set(PP_MAP.values())
print(f"\nORIGINAL LABELS, unfinished rows: {len(O)} ({y.sum()} robust) | base fa+chk+fas pooled AUC {within(O.base.values, y, g):.2f}")
print(f"   {'feature':26s} pooled  perm p | " + " ".join(f"{m[:8]:>8s}" for m in np.unique(g)) + " | rho fa   chk   fas | resid AUC   (raw: higher value -> robust; * = primary)")
res = {}
for c in FE:
    s = O[c].astype(float).values; a = within(s, y, g); p = perm(s, y, g)
    x = O.groupby("model")[c].rank(pct=True).values; b = O.base.values; ok = np.isfinite(x)
    resid = np.full(len(x), np.nan)
    if ok.sum() > 5: resid[ok] = x[ok] - np.polyval(np.polyfit(b[ok], x[ok], 1), b[ok])
    rho = [pd.Series(x).corr(O[e + "_r"], method="spearman") for e in ("fa", "chk", "fas")]; res[c] = (a, p)
    print(f"   {c[:26]:26s} {a:6.2f}  {p:6.3f} | " + " ".join(f"{auc(s[g == m], y[g == m]):8.2f}" for m in np.unique(g)) +
          f" | {rho[0]:+.2f} {rho[1]:+.2f} {rho[2]:+.2f} | {within(resid, y, g):.2f} {'*' if c in PRIM else ''}")
ps = np.array([v[1] for v in res.values()]); print(f"   {len(res)} features | share p<0.05 {np.mean(ps < .05):.2f} (null 0.05) | smallest p {ps.min():.3f}")
L1 = A / "hard_pool/set100_labels_Qwen__Qwen3.5-4B.csv"
if L1.exists():
    Lq = pd.read_csv(L1); S = D[(D.dataset == "set100") & (D.model == "Qwen__Qwen3.5-4B")].copy(); S["pid"] = S.id.str.split("_").str[-1]
    S = S.merge(Lq[["pid", "label_forced"]], on="pid"); S = S[S.label_forced.isin(["robust", "non_robust"])]; ys = (S.label_forced == "robust").astype(int).values
    print(f"\nSET100 QWEN (forced class): {len(S)} labelled ({ys.sum()} robust) | raw AUC, and whether it agrees in direction with the original rows")
    for c in sorted(PRIM) + ["li_mid", "minkpp_10", "lid_ref"]:
        if c in S:
            a = auc(S[c].astype(float).values, ys); print(f"   {c:26s} {a:.2f}  (original {res[c][0]:.2f}) {'same direction' if (a - .5) * (res[c][0] - .5) > 0 else 'OPPOSITE'}")
signs = {f: (1 if res[k][0] >= .5 else -1) for f, k in PP_MAP.items()}
print("\nensemble signs (from the 88 unfinished original rows):", signs)
json.dump(signs, open(A / "analysis/exp81_pp_signs.json", "w"), indent=1)
D[["id", "model", "dataset", "robust"] + list(PP_MAP)].to_csv(A / "analysis/exp81_pp_rows.csv", index=False)
print(f"wrote exp81_pp_rows.csv ({len(D)} rows), exp81_pp_signs.json, exp81_ship/ ({len(list((A / 'analysis/exp81_ship').glob('*.npz')))} models)")
