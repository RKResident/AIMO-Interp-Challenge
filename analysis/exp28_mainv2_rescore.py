"""main_v2 re-score (43 rows, 22 problems, 4 models). Single features WITHOUT fitting (avoids small-sample CV bias), each against the
'finished within the cap' baseline: pooled AUC, AUC within finished / within unfinished traces, problem level raw and with
(finished fraction, mean log length) regressed out, with permutation p and BH-FDR over all features.
Also per-model probes on averaged trajectories / prompt state (leave-one-problem-out within model, OOF scores pooled). CPU capped."""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "3"
from common import *
from sklearn.decomposition import PCA
P2 = ROOT / "extraction_out_generations_p2" / "extraction_out_generations"
R3 = ROOT / "extraction_out_generations_pearl_more" / "extraction_out_generations"
allm, feats = load_all(); y = allm.y.values
main = datasets(allm)["main_v2"]; rows = main
model = allm.model.values; grp = allm.grp.values
fin = (allm.stop.values == "eos").astype(float); lgn = np.log(allm.n_gen.astype(float).values)
Lthink = np.array([force_json(r.model, r.id)["L"] for r in allm.itertuples()])
F = {}                                    # name -> full-length array (NaN where missing)
import faulthandler, time as _time; faulthandler.dump_traceback_later(300, repeat=True)   # stack dump every 5 min (shows where it is if it hangs)
_T0 = _time.time()
def stage(msg): print(f"[{_time.time() - _T0:7.0f}s] {msg}", flush=True)
stage(f"meta loaded: {len(allm)} rows, main_v2 rows {len(rows)}")


def put(name, idx, val):
    if name not in F: F[name] = np.full(len(allm), np.nan)
    F[name][idx] = val


CACHE = A / "exp28_features_cache.pkl"
if CACHE.exists():
    F = pickle.load(open(CACHE, "rb")); stage(f"features loaded from cache {CACHE.name}")
else:
    # ---- scalar columns already in the meta table (trace stats, forced answer, consistency)
    for c in [c for c in allm.columns if c.startswith(("lp_", "ent_", "top1_", "fa_", "sc_", "agree_", "prob_")) and allm[c].dtype != object]:
        F[f"meta:{c}"] = allm[c].astype(float).values
    for nm, df, pre in (("text", pd.read_pickle(A / "exp3_features.pkl"), ("t_", "e_")), ("dyn", pd.read_pickle(A / "exp4_features.pkl"), ("h", "c", "l_")),
                        ("ab", pd.read_pickle(A / "exp8_features.pkl"), None)):
        for c in df.columns:
            if pre is None and not c.endswith(("_mean", "_max", "_bmean")): continue
            if pre is not None and not c.startswith(pre): continue
            F[f"{nm}:{c}"] = df[c].astype(float).values
    D4 = pd.read_pickle(A / "exp4_features.pkl"); F["anchor:cos_prob_mean_all"] = D4[[f"h{j}_cos_prob_mean" for j in range(6)]].mean(1).values
    stage("window-state geometry (exp18)")
    # ---- window-state geometry (exp18), averaged over the 6 layers
    R18 = pd.read_pickle(A / "exp18_features.pkl")
    for key in sorted({c.rsplit("|", 2)[0] + "|" + c.rsplit("|", 1)[1] for c in R18.columns}):
        setting, fname = key.split("|")[0] + "|" + key.split("|")[1], key.rsplit("|", 1)[1]
        cols = [c for c in R18.columns if c.startswith(setting + "|") and c.endswith("|" + fname)]
        if cols: F[f"traj:{setting}:{fname}"] = R18[cols].mean(1).values
    feat19, probe19 = pickle.load(open(A / "exp19_cache.pkl", "rb"))
    for nm in ("bag", "segment", "seg25", "seg50", "seg75"):
        X = np.array([x if x is not None else np.full(6 * (12 if nm != "bag" and nm != "sorted" else 7), np.nan) for x in feat19[nm]], dtype=object)
        arr = np.full((len(allm), max(len(x) for x in feat19[nm] if x is not None)), np.nan)
        for i, x in enumerate(feat19[nm]):
            if x is not None: arr[i] = x
        per = arr.shape[1] // 6
        for j in range(per): F[f"subsample:{nm}:f{j}"] = arr[:, j::per].mean(1)
    stage("TOHA")
    # ---- TOHA (all models): mean forest length over heads, per depth third, at 2048 (win) / 1024 (ans)
    for i in rows:
        p = P2 / model[i] / "toha" / f"{allm.id.values[i]}.npz"
        if not p.exists(): continue
        z = np.load(p); ce = list(z["cut_eff"])
        for b, kind in ((1024, "win"), (2048, "win"), (1024, "ans"), (2048, "ans")):
            if b in ce and Lthink[i] > b:
                t = z["win"][ce.index(b)] if kind == "win" else z["ans"][ce.index(b), 0]          # [L,H,5]
                for si, sn in enumerate(("total", "mean_edge", "max_edge", "direct", "prompt_dist")):
                    put(f"toha:{kind}@{b}:{sn}", i, t[..., si].mean())
                    for th, part in enumerate(np.array_split(np.arange(t.shape[0]), 3)): put(f"toha:{kind}@{b}:{sn}:third{th}", i, t[part][..., si].mean())
    stage("Lookback Lens")
    # ---- Lookback Lens (round 3)
    for i in rows:
        p = R3 / model[i] / "lookback" / f"{allm.id.values[i]}.npz"
        if not p.exists(): continue
        z = np.load(p); lr = z["lookback"].astype(np.float32); sp = int(z["span"]); te = int(z["think_end"]); nthink = lr.shape[0] if te < 0 else max(1, -(-te // sp))
        for b in (1024, 2048, None):
            if b is not None and (te if te >= 0 else int(z["n_gen"])) <= b: continue
            x = lr[: b // sp] if b else lr[:nthink]; tag = f"@{b}" if b else "@think"
            put(f"lookback:mean{tag}", i, x.mean())
            for th, part in enumerate(np.array_split(np.arange(x.shape[1]), 3)): put(f"lookback:mean{tag}:third{th}", i, x[:, part].mean())
    stage("PEARL")
    # ---- PEARL (round 3)
    RATES = [0.02, 0.05, 0.1, 0.2, 0.3]
    cosd = lambda A_, B_: 1 - (A_ * B_).sum(-1) / (np.linalg.norm(A_, axis=-1) * np.linalg.norm(B_, axis=-1) + 1e-9)
    for i in rows:
        p = R3 / model[i] / "pearl" / f"{allm.id.values[i]}.npz"
        if not p.exists(): continue
        stage(f"  pearl row {i} {model[i].split('/')[-1]}")
        z = np.load(p, allow_pickle=True); kind, rate = z["kind"], z["rate"]; o = np.where(kind == "original")[0][0]
        curves = {"prefix_kl": [z["prefix_kl"][(kind == "noise") & np.isclose(rate, r)].mean() for r in RATES],
                  "prefix_flip": [1 - z["prefix_top1_same"][(kind == "noise") & np.isclose(rate, r)].mean() for r in RATES],
                  "prefix_dnll": [z["prefix_nll"][(kind == "noise") & np.isclose(rate, r)].mean() - z["prefix_nll"][o] for r in RATES]}
        for st in ("last_state", "mean_state", "prefix_state"):
            S = z[st].astype(np.float32); curves[st] = [cosd(S[(kind == "noise") & np.isclose(rate, r)], S[o][None]).mean() for r in RATES]
        for nm, c in curves.items():
            c = np.array(c); put(f"pearl:{nm}:falloff", i, np.diff(np.concatenate([[0], c])).max()); put(f"pearl:{nm}:area", i, np.trapz(c, RATES)); put(f"pearl:{nm}:at10", i, c[2])
    stage("MemLens + extras")
    # ---- MemLens (all models) + round-3 extras (aligned summaries), first and last extras cutoff
    HEAD = {m: np.load(A / "memlens" / f"digit_head_{m}.npz") for m in MODELS}
    EXD = {DS: ROOT / "extra_deepsek" / "extra", **{m: R3 / m / "extra" for m in MODELS[1:]}}
    for i in rows:
        p = EXD[model[i]] / f"{allm.id.values[i]}.npz"
        if not p.exists(): continue
        stage(f"  extras row {i} {model[i].split('/')[-1]} {p.name}")
        e = np.load(p); H = HEAD[model[i]]
        for k, tag in ((0, "first"), (len(e["cut_eff"]) - 1, "last")):
            res = e["res_last"][k].astype(np.float32)                                            # [var, L, d]
            x = res / np.sqrt((res ** 2).mean(-1, keepdims=True) + float(H["rms_norm_eps"])) * (float(H["norm_offset"]) + H["norm_weight"])
            lg = x @ H["digit_rows"].T; lg -= lg.max(-1, keepdims=True); P_ = np.exp(lg); P_ /= P_.sum(-1, keepdims=True)
            nl = P_.shape[1]; ent = -(P_ * np.log(P_ + 1e-12)).sum(-1); mx = P_.max(-1)
            stable = P_.argmax(-1) == P_.argmax(-1)[:, -1:]; emerg = (nl - np.flip(stable, -1).cumprod(-1).sum(-1)) / nl
            put(f"memlens:{tag}:emergence", i, emerg.mean()); put(f"memlens:{tag}:entropy", i, ent.mean()); put(f"memlens:{tag}:final_conf", i, mx[:, -1].mean())
            rn = np.linalg.norm(res, axis=-1).mean(0)
            for th, part in enumerate(np.array_split(np.arange(nl), 3)): put(f"extra:{tag}:resid_norm_third{th}", i, np.log(rn[part]).mean())
            put(f"extra:{tag}:suffix_lookback", i, e["lookback"][k].mean()); put(f"extra:{tag}:prompt_mass", i, e["prompt_mass"][k].mean())
            put(f"extra:{tag}:kl_adj", i, np.nanmean(e["kl_adj"][k])); put(f"extra:{tag}:kl_headmean", i, np.nanmean(e["kl_headmean"][k]))
            ff = np.nanmean(e["fft"][k], (0, 2))                                                  # [L, nb] (mean over variants and heads)
            cen = (ff * np.arange(ff.shape[-1])).sum(-1) / (ff.sum(-1) + 1e-12)                   # spectral centroid per layer
            put(f"extra:{tag}:fft_centroid", i, np.nanmean(cen)); put(f"extra:{tag}:fft_lowband", i, np.nanmean(ff[:, :8]) / (np.nanmean(ff) + 1e-12))
    pickle.dump(F, open(CACHE, "wb")); stage(f"features cached -> {CACHE.name}")
print(f"features: {len(F)}", flush=True)

# ---------------------------------------------------------------- scoring
prob = allm.base_id.values; up = np.unique(prob[rows]); yp = np.array([y[rows][prob[rows] == p][0] for p in up])
fp = np.array([fin[rows][prob[rows] == p].mean() for p in up]); lp_ = np.array([lgn[rows][prob[rows] == p].mean() for p in up])
Cp = np.c_[fp, lp_]
rng = np.random.default_rng(0); PERMS = np.array([rng.permutation(len(up)) for _ in range(5000)])


from scipy.stats import rankdata
YPERM = yp[PERMS].astype(float)                                         # [5000, n_problems] permuted labels


def perm_p(x):                                                          # rank-based AUC for all permutations at once (= roc_auc_score)
    rk = rankdata(x); n1 = yp.sum(); n0 = len(yp) - n1; c = n1 * (n1 + 1) / 2
    a = (rk[yp == 1].sum() - c) / (n1 * n0); nl = (YPERM @ rk - c) / (n1 * n0)
    return a, float(np.mean(np.abs(nl - .5) >= abs(a - .5)))


out = []
F["baseline:finished"] = fin; F["baseline:-log_length"] = -lgn
for k_, (nm, v) in enumerate(F.items()):
    if k_ % 50 == 0: stage(f"scoring feature {k_}/{len(F)}")
    v = v.astype(float); ok = np.isfinite(v[rows])
    if ok.sum() < 30 or len(np.unique(v[rows][ok])) < 3 and not nm.startswith("baseline"): continue
    r = rows[ok]
    vz = zgroup(np.where(np.isin(np.arange(len(allm)), r), v, np.nan)[:, None], grp)[:, 0]
    if len(np.unique(y[r])) < 2: continue
    a_pool = roc_auc_score(y[r], vz[r]); a_fin = within_bin_auc(y[r], vz[r], fin[r].astype(int))
    xp = np.array([np.nanmean(np.where(prob[r] == p, vz[r], np.nan)) if (prob[r] == p).any() else np.nan for p in up])
    okp = np.isfinite(xp)
    if okp.sum() < 15 or len(np.unique(yp[okp])) < 2: continue
    if okp.all():
        ap, pp_ = perm_p(xp); xr = xp - LinearRegression().fit(Cp, xp).predict(Cp); apr, ppr = perm_p(xr)
    else:
        ap, pp_, apr, ppr = roc_auc_score(yp[okp], xp[okp]), np.nan, np.nan, np.nan
    out.append(dict(feature=nm, n=int(ok.sum()), pooled=a_pool, within_finished=a_fin, problem=ap, p_problem=pp_, problem_controlled=apr, p_controlled=ppr))
T = pd.DataFrame(out); stage(f"scored {len(T)} features")
q = T.p_controlled.values.copy(); okq = np.isfinite(q); o = np.argsort(q[okq]); adj = q[okq][o] * okq.sum() / np.arange(1, okq.sum() + 1)
qq = np.full(len(q), np.nan); qq[np.where(okq)[0][o]] = np.minimum(1, np.minimum.accumulate(adj[::-1])[::-1]); T["q_controlled"] = qq
T.to_csv(A / "exp28_mainv2_single_features.csv", index=False)
print("\n=== baselines ===")
print(T[T.feature.str.startswith("baseline")].round(3).to_string(index=False))
print(f"\n=== {len(T)} features; strongest AFTER removing finished-fraction + length at the problem level (sign-free) ===")
T["dist"] = (T.problem_controlled - .5).abs()
print(T.sort_values("dist", ascending=False).head(20)[["feature", "n", "pooled", "within_finished", "problem", "problem_controlled", "p_controlled", "q_controlled"]].round(3).to_string(index=False))
print(f"features with controlled p < 0.05: {int((T.p_controlled < .05).sum())} (expected by chance ~{0.05 * okq.sum():.0f}); with q < 0.1: {int((T.q_controlled < .1).sum())}")
print("PEARL drift hint:"); print(T[T.feature.str.startswith("pearl:mean_state")].round(3)[["feature", "pooled", "within_finished", "problem", "problem_controlled", "p_controlled"]].to_string(index=False))

# ---------------------------------------------------------------- 4. per-model probes, pooled OOF scores
print("\n=== per-model probes (leave-one-problem-out within each model, PCA-3 + logistic, OOF scores z-scored within model and pooled) ===")
def probe_vectors(kind):
    V = {}
    for i in rows:
        m = model[i]
        if kind == "prompt_state":
            j = np.where(np.where(model == m)[0] == i)[0][0]; pl = feats[m]["prompt_last"]; V[i] = pl[j, int(pl.shape[1] * 2 / 3)]
        else:
            p = P2 / m / "tf" / f"{allm.id.values[i]}.npz"
            if not p.exists(): continue
            ws = np.load(p)["win_states"].astype(np.float32)[:, 3]                 # middle sampled layer, 16-token means
            nw = max(1, min(len(ws), Lthink[i] // 16))
            V[i] = ws[:min(nw, 128)].mean(0) if kind == "traj_first2048" else ws[:nw].mean(0)
    return V
for kind in ("prompt_state", "traj_first2048", "traj_whole_thinking"):
    V = probe_vectors(kind); s = np.full(len(allm), np.nan)
    for m in MODELS:
        rm = np.array([i for i in rows if model[i] == m and i in V])
        if len(rm) < 6 or len(np.unique(y[rm])) < 2: continue
        X = np.stack([V[i] for i in rm])
        for p in np.unique(prob[rm]):
            te = prob[rm] == p; tr = ~te
            if len(np.unique(y[rm][tr])) < 2: continue
            mdl = make_pipeline(StandardScaler(), PCA(min(3, tr.sum() - 1), random_state=0), LogisticRegression(C=0.3, max_iter=2000)).fit(X[tr], y[rm][tr])
            s[rm[te]] = mdl.decision_function(X[te])
    r = rows[np.isfinite(s[rows])]; sz = zgroup(np.where(np.isin(np.arange(len(allm)), r), s, np.nan)[:, None], grp)[:, 0]
    print(f"  {kind:22s} n={len(r)} pooled OOF AUC {roc_auc_score(y[r], sz[r]):.2f} | within finished {within_bin_auc(y[r], sz[r], fin[r].astype(int)):.2f}")
