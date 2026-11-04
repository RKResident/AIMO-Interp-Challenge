"""Problem-grouped CV pooled over the 4 models with model-aligned feature families. No gpt_proxy rows."""
src_code = open(__file__.replace("run_pooled_families.py", "run_experiments.py")).read().split("# ----------------------------------------------------------------------------------------- 2/3")[0]
exec(src_code)
SL = ["Qwen__Qwen3.5-4B", "Skywork__Skywork-OR1-Math-7B", "allenai__Olmo-3-7B-Think"]
D = {DS: (meta, F)}
for s in SL: D[s] = load(s)
NB = 6


def binned(a, axis_L, fn):
    """a [..., L, H, ...] -> list of per-depth-bin summaries along the layer axis"""
    parts = np.array_split(np.arange(a.shape[axis_L]), NB)
    return [fn(np.take(a, p, axis=axis_L)) for p in parts]


def aligned(F, n):
    out = {}
    cl, pm = F["cut_last"], F["prompt_mean"]                 # [n,4,6,d], [n,6,d]
    for ci, tag in enumerate(("b1024", "b4096", "f50", "f100")):
        c = cl[:, ci]
        nrm = np.log(np.linalg.norm(c, axis=-1) + 1e-6)
        cos = (c * pm).sum(-1) / (np.linalg.norm(c, axis=-1) * np.linalg.norm(pm, axis=-1) + 1e-9)
        out[f"hid:norm+cos@{tag}"] = np.hstack([nrm, cos])
        ln = F["lap_no"][:, ci]; lf = F["lap_force"][:, ci].mean(1)                   # [n,L,H,8]
        lmn = F["lap_mean_no"][:, ci]; lmf = F["lap_mean_force"][:, ci].mean(1)          # [n,L,H]
        for nm, T in (("noforce", ln), ("force", lf)):
            cols_ = []
            for b in binned(T, 1, lambda x: np.stack([np.nanmean(x[..., 0], (1, 2)), np.nanmean(x, (1, 2, 3)), np.nanstd(x[..., 0], (1, 2))], 1)):
                cols_.append(b)
            out[f"lap:topk_{nm}@{tag}"] = np.hstack(cols_)
        for nm, T in (("mean_noforce", lmn), ("mean_force", lmf)):
            out[f"lap:{nm}@{tag}"] = np.hstack(binned(T, 1, lambda x: np.stack([np.nanmean(x, (1, 2)), np.nanstd(x, (1, 2))], 1)))
    return out


rows_meta, feats = [], {}
for mname, (m, F_) in D.items():
    a = aligned(F_, len(m))
    m = m.assign(model=mname)
    rows_meta.append(m)
    for k, v in a.items():
        feats.setdefault(k, []).append(v)
allm = pd.concat(rows_meta, ignore_index=True)
fam = {k: np.vstack(v).astype(np.float32) for k, v in feats.items()}
keep = allm.source.isin(["main_v2", "agg"]).values & allm.is_robust.notna().values            # no gpt_proxy, no unlabeled
common = [c for c in tcols + fc + sc if all(c in mm.columns and mm[c].notna().mean() > .8 for mm, _ in D.values())]
fam["scalars"] = allm[common].astype(float).values
fam["length (control)"] = np.log(allm[["n_gen", "n_prompt"]].astype(float).values + 1)
fam["hid:all_cuts"] = np.hstack([fam[f"hid:norm+cos@{t}"] for t in ("b1024", "b4096", "f50", "f100")])
fam["lap:force_all_cuts"] = np.hstack([fam[f"lap:topk_force@{t}"] for t in ("b1024", "b4096", "f50", "f100")])
fam["lap:mean_force_all_cuts"] = np.hstack([fam[f"lap:mean_force@{t}"] for t in ("b1024", "b4096", "f50", "f100")])
fam["scalars+hid+lap"] = np.hstack([fam["scalars"], fam["hid:all_cuts"], fam["lap:force_all_cuts"]])
grp = (allm.model + allm.source).values
allm["y"] = allm.is_robust.map({True: 1, False: 0, "True": 1, "False": 0}).astype(float)
mv = np.where((allm.source == "main_v2").values & keep)[0]
ag = np.where((allm.source == "agg").values & keep)[0]
yv = allm.y.values
prob = allm.base_id.values
print(f"main_v2 rows {len(mv)} ({int(yv[mv].sum())} robust), problems {len(set(prob[mv]))}; agg rows (DeepSeek) {len(ag)}")
Xz = {k: src_standardize(v, grp) for k, v in fam.items()}


def cv(X, add_agg, C, y_use, reps=10):
    up = np.unique(prob[mv]); s = np.zeros(len(mv)); c = np.zeros(len(mv))
    for r in range(reps):
        order = np.random.default_rng(r).permutation(up); fold = {p: i % 5 for i, p in enumerate(order)}
        f = np.array([fold[p] for p in prob[mv]])
        for k in range(5):
            te, tr = mv[f == k], mv[f != k]
            if add_agg: tr = np.concatenate([tr, ag])
            yy = y_use[tr]
            if len(np.unique(yy)) < 2: continue
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(X[tr], yy)
            s[f == k] += m.decision_function(X[te]); c[f == k] += 1
    return s / np.maximum(c, 1)


# problem-level label shuffles (the labels of a problem agree across models, so shuffle problems, not rows)
def shuffled(seed):
    up = np.unique(prob[mv]); lab = {p: yv[mv][prob[mv] == p][0] for p in up}
    perm_ = dict(zip(up, np.random.default_rng(seed).permutation(up)))
    y2 = yv.copy(); y2[mv] = [lab[perm_[p]] for p in prob[mv]]; return y2


res = []
for nm, X in Xz.items():
    for add_agg in (False, True):
        C = 0.05
        s = cv(X, add_agg, C, yv); a, lo, hi = auc_ci(yv[mv], s)
        nulls = []
        for sd in range(4):
            y2 = shuffled(sd); nulls.append(roc_auc_score(y2[mv], cv(X, add_agg, C, y2, reps=4)))
        per = {k.split('__')[-1][:6]: round(roc_auc_score(yv[mv][allm.model.values[mv] == k], s[allm.model.values[mv] == k]), 2)
               for k in D if len(np.unique(yv[mv][allm.model.values[mv] == k])) == 2}
        res.append((nm, add_agg, X.shape[1], a, lo, hi, np.mean(nulls), per))
        print(f"{nm:26s} d={X.shape[1]:3d} {'+agg' if add_agg else '    '}  AUC {a:.2f} [{lo:.2f},{hi:.2f}]  shuffled-label null {np.mean(nulls):.2f}  {per}", flush=True)
pd.DataFrame(res, columns=["family", "add_agg", "dim", "auc", "lo", "hi", "null", "per_model"]).to_csv(ROOT / "results_pooled_grouped_families.csv", index=False)
