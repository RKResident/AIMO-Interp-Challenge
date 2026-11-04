"""MemLens on DeepSeek (agg + its main_v2 rows), from the saved extras (res_last = residual stream at the forced-answer position
'...\\boxed{' at every layer, 5 cutoffs x 2 suffix variants). Logit lens restricted to the 10 digit tokens: per layer the digit
distribution, its entropy and max confidence, and their layer-to-layer differences (24 channels x 36 layers).
CPU use is capped (few threads) - run under taskset."""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "3"
from common import *
from collections import Counter
from sklearn.decomposition import PCA
import torch
torch.set_num_threads(2)
EX = ROOT / "extra_deepsek" / "extra"
H = np.load(A / "memlens" / "digit_head_deepseek-ai__DeepSeek-R1-0528-Qwen3-8B.npz")
Wd, wn, eps, did = H["digit_rows"], H["norm_weight"], float(H["rms_norm_eps"]), list(H["digit_ids"])
allm, _ = load_all(); y = allm.y.values
rows_ds = np.where(allm.model == DS)[0]
agg = datasets(allm)["agg"]; dsm = np.array([i for i in datasets(allm)["main_v2"] if allm.model.values[i] == DS])
Lthink = np.array([force_json(r.model, r.id)["L"] if r.model == DS else 0 for r in allm.itertuples()])
L3 = np.log(allm[["n_prompt", "n_gen", "L_think"]].astype(float).values + 1)


def lens(res):  # res [..., 36, d] -> digit probs [..., 36, 10]
    x = res.astype(np.float32)
    x = x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps) * wn
    lg = x @ Wd.T
    lg -= lg.max(-1, keepdims=True); p = np.exp(lg); return p / p.sum(-1, keepdims=True)


TR, SUM, CUTS, CHECK = {}, {}, {}, []
for i in np.concatenate([agg, dsm]):
    pid = allm.id.values[i]; e = np.load(EX / f"{pid}.npz"); ce = list(e["cut_eff"]); CUTS[i] = ce
    P = lens(e["res_last"])                                          # [cut, var, 36, 10]
    ent = -(P * np.log(P + 1e-12)).sum(-1); mx = P.max(-1)            # [cut, var, 36]
    traj = np.concatenate([P, ent[..., None], mx[..., None]], -1)     # [cut, var, 36, 12]
    traj = np.concatenate([traj, np.diff(traj, axis=2, prepend=traj[:, :, :1])], -1)   # + layer-to-layer differences -> 24 channels
    TR[i] = traj.mean(1)                                              # average the two suffix variants -> [cut, 36, 24]
    fin = P.argmax(-1)[..., -1:]                                     # final-layer top digit
    stable = (P.argmax(-1) == fin)
    emerg = 36 - np.flip(stable, -1).cumprod(-1).sum(-1)              # first layer from which the top digit equals the final one
    SUM[i] = np.stack([emerg / 36, ent.mean(-1), ent[..., 24:].mean(-1), mx[..., -1], mx[..., 18:27].mean(-1), mx[..., -1] - mx[..., 18:27].mean(-1)], -1).mean(1)  # [cut, 6]
    fj = force_json(DS, pid)                                          # validation: final-layer lens vs the model's real first answer token
    for k, c in enumerate(ce):
        r = fj["results"].get(f"{c}|B")
        if r and r["top"] and r["top"][0]["ids"][0] in did:
            CHECK.append(did[int(P[k, list(e["variant_names"]).index("B"), -1].argmax())] == r["top"][0]["ids"][0])
print(f"logit-lens check: final-layer digit argmax == model's actual first answer digit in {np.mean(CHECK):.3f} of {len(CHECK)} cases")
SUM_NAMES = ["emergence layer", "entropy mean", "entropy late", "final max conf", "mid max conf", "final-minus-mid conf"]

# ---------------- groups (organizer permutations): robust / genuine fragility / information loss
a = pd.read_csv(ROOT / "augmented-sample-math-agg.csv"); nums = lambda s: set(re.findall(r"\d+(?:\.\d+)?", str(s)))
grp, keep = {}, {}
for r in a.itertuples():
    try: perms = json.loads(r.permutations_causing_decay)
    except Exception: perms = []
    pid = str(r.problem_id)
    if bool(r.model_is_robust) and not perms: grp.setdefault(pid, "robust"); continue
    keep[pid] = keep.get(pid, False) or any(nums(r.original_problem) <= nums(p.get("question", "") if isinstance(p, dict) else p) for p in perms)
for pid, k in keep.items(): grp[pid] = "genuine" if k else "info_loss"
g = np.array([grp.get(str(p)) if i in set(agg) else None for i, p in enumerate(allm.parent_id.astype(str).values)])
digit_ans = np.array([bool(re.match(r"^\d", nrm(force_json(DS, allm.id.values[i]).get("final_boxed")) or "")) if i in TR else False for i in range(len(allm))])


def at(i, which):  # which: "first" (<=1024) or "last" (largest budget reached)
    return 0 if which == "first" else len(CUTS[i]) - 1


def mats(which):
    T = np.full((len(allm), 36 * 24), np.nan, np.float32); Sm = np.full((len(allm), 6), np.nan, np.float32)
    for i in TR:
        T[i] = TR[i][at(i, which)].reshape(-1); Sm[i] = SUM[i][at(i, which)]
    return T, Sm


def cv(X, rows, yy, kind, reps=8, seed=0):
    s, c = np.zeros(len(rows)), np.zeros(len(rows))
    for r in range(reps):
        for tr, te in folds(rows, yy, np.arange(len(allm)), seed + r):
            if len(np.unique(yy[rows[tr]])) < 2: continue
            m = (make_pipeline(StandardScaler(), PCA(8, random_state=0), StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)) if kind == "high"
                 else make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)))
            s[te] += m.fit(X[rows[tr]], yy[rows[tr]]).decision_function(X[rows[te]]); c[te] += 1
    return s / np.maximum(c, 1)


class CNN(torch.nn.Module):  # MemLens discriminator, scaled down for n ~ 100 (hidden 32 instead of 128)
    def __init__(s, ch=24, h=32):
        super().__init__()
        s.net = torch.nn.Sequential(torch.nn.Conv1d(ch, h, 3, padding=1), torch.nn.ReLU(), torch.nn.Dropout(0.3),
                                    torch.nn.Conv1d(h, h, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv1d(h, h, 3, padding=1), torch.nn.ReLU())
        s.out = torch.nn.Linear(h, 1)

    def forward(s, x): return s.out(s.net(x).mean(-1)).squeeze(-1)


def cv_cnn(T3, rows, yy, reps=3):
    s, c = np.zeros(len(rows)), np.zeros(len(rows))
    for r in range(reps):
        for tr, te in folds(rows, yy, np.arange(len(allm)), r):
            Xtr, Xte = T3[rows[tr]], T3[rows[te]]
            mu, sd = Xtr.mean((0, 1)), Xtr.std((0, 1)) + 1e-6
            xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32).transpose(1, 2); xe = torch.tensor((Xte - mu) / sd, dtype=torch.float32).transpose(1, 2)
            yt = torch.tensor(yy[rows[tr]], dtype=torch.float32)
            torch.manual_seed(r); m = CNN(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
            for _ in range(150):
                opt.zero_grad(); loss = torch.nn.functional.binary_cross_entropy_with_logits(m(xt), yt); loss.backward(); opt.step()
            m.eval()
            with torch.no_grad(): s[te] += m(xe).numpy()
            c[te] += 1
    return s / np.maximum(c, 1)


def bins2d(rows):
    a_ = np.log(allm.L_think.astype(float).values[rows] + 1); b_ = np.log(allm.n_prompt.astype(float).values[rows] + 1)
    return np.digitize(a_, np.quantile(a_, [1 / 3, 2 / 3])) * 3 + np.digitize(b_, np.quantile(b_, [1 / 3, 2 / 3]))


def report(name, scorer, X, rows, yy):
    s = scorer(X, rows, yy); a1 = auc_ci(yy[rows], s)
    rng = np.random.default_rng(0); nl = []
    for k in range(4):
        y2 = yy.copy(); y2[rows] = rng.permutation(yy[rows]); nl.append(roc_auc_score(y2[rows], scorer(X, rows, y2)))
    Xf = np.nan_to_num(X.astype(np.float64)).reshape(len(X), -1).copy()
    Xf[rows] -= LinearRegression().fit(L3[rows], Xf[rows]).predict(L3[rows])
    ar = roc_auc_score(yy[rows], scorer(Xf.reshape(X.shape).astype(np.float32), rows, yy))
    b2 = bins2d(rows); w = within_bin_auc(yy[rows], s, b2)
    print(f"    {name:40s} n={len(rows):3d} AUC {a1[0]:.2f} [{a1[1]:.2f},{a1[2]:.2f}] null {np.mean(nl):.2f} | lengths removed {ar:.2f} | 2-D within-bin {w:.2f}", flush=True)


for which in ("first", "last"):
    T, Sm = mats(which)
    T3 = np.nan_to_num(T).reshape(len(allm), 36, 24)
    print(f"\n=== cutoff: {which} ({'<= 1024 tokens' if which == 'first' else 'largest budget reached, up to 12288'}) ===")
    for cname, keepset in (("robust vs all non-robust", ("robust", "genuine", "info_loss")), ("robust vs genuine fragility", ("robust", "genuine")),
                           ("robust vs information loss", ("robust", "info_loss"))):
        rows = np.array([i for i in agg if g[i] in keepset]); yy = np.zeros(len(allm)); yy[rows] = (g[rows] == "robust")
        print(f"  [{cname}]")
        report("summaries (6: emergence, entropy, conf)", lambda X, r, t: cv(X, r, t, "low"), Sm, rows, yy)
        report("full trajectory 24x36 (PCA-8 + logistic)", lambda X, r, t: cv(X, r, t, "high"), T, rows, yy)
        report("1-D CNN on the trajectory", lambda X, r, t: cv_cnn(X, r, t), T3, rows, yy)
        if cname.startswith("robust vs all"):
            sub = rows[digit_ans[rows]]
            report("summaries, integer-answer subset", lambda X, r, t: cv(X, r, t, "low"), Sm, sub, yy)
    # single summaries (sign-free view)
    rows = np.array([i for i in agg]); yy = np.zeros(len(allm)); yy[rows] = (g[rows] == "robust")
    print("  single summaries, robust vs all (AUC; >0.5 = higher in robust): " + "; ".join(f"{n} {roc_auc_score(yy[rows], Sm[rows, j]):.2f}" for j, n in enumerate(SUM_NAMES)))
    print(f"  DeepSeek main_v2 (n={len(dsm)}, robust {int(y[dsm].sum())}): single summaries " + "; ".join(f"{n} {roc_auc_score(y[dsm], Sm[dsm, j]):.2f}" for j, n in enumerate(SUM_NAMES)))
