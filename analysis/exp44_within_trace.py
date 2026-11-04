"""exp44: WITHIN-TRACE (self-referenced) versions of our signals, on the first 6144 generated tokens (the submission cap), main_v2 only.
Motivation: with one problem per are_robust call, decisions rest on thresholds calibrated on main_v2, and the leaderboard runs show
test traces sitting at systematically different absolute levels (harder problems, different GPU / stack). A feature that compares
parts of the SAME trace cancels such shifts, so a fixed threshold can transfer.
For each candidate: within-model pooled AUC, controlled problem-level AUC + permutation p, eta^2 = share of its variance explained by
the model (low -> comparable across models -> one cut can serve all), and the AUC of the RAW pooled values (what one global cut sees).
Candidates (all within one trace):
  forced answer : slope of top-1 prob over budgets <= 6144; p1(6144) - p1(2048); log entropy ratio 6144 / 2048; share of saturated budgets
  A/B           : |H_A - H_B| / (H_A + H_B) at 6144 and max over budgets <= 6144
  token entropy : mean generated-token entropy, last third / first third of the first 6144 tokens
  kNN graph     : global efficiency of the 5-NN graph (self-centred 64-token states), tokens 3-6k minus 0-3k
  TDA (H0)      : H0 Betti spread (MST), tokens 3-6k minus 0-3k; whole-6k value self-centred
  trajectory    : late-vs-early cosine; first-PC share at 75% / 25% of the 6k window
  dispersion    : mean distance of states from the trace's own centroid, late / early
  PEARL shape   : drift at the highest typo rate / drift at the lowest (prompt only)"""
import os, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "2"
from common import *
from scipy.stats import rankdata
sys.path.insert(0, str(ROOT / "submission_core"))
from spurious_features import mst_edge_lengths, f_late_vs_early

P2 = ROOT / "extraction_out_generations_p2" / "extraction_out_generations"
R3 = ROOT / "extraction_out_generations_pearl_more" / "extraction_out_generations"
allm, _ = load_all(); y = allm.y.values; rows = datasets(allm)["main_v2"]; model = allm.model.values; prob = allm.base_id.values
fin = (allm.stop.values == "eos").astype(float); lgn = np.log(allm.n_gen.astype(float).values)
CAP = 6144
F = {}


def put(k, i, v): F.setdefault(k, np.full(len(allm), np.nan))[i] = v


def nodes(W):                                                      # 64-token mean states, self-centred, unit norm
    S = W[: len(W) // 4 * 4].reshape(-1, 4, W.shape[1]).mean(1); S = S - S.mean(0)
    return S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-6)


def global_eff(S, k=5):
    n = len(S)
    if n < 3 * k: return np.nan
    C = S @ S.T; np.fill_diagonal(C, -np.inf); nb = np.argsort(-C, axis=1)[:, :k]
    A = np.zeros((n, n), bool); A[np.repeat(np.arange(n), k), nb.ravel()] = True; A |= A.T; tot = 0.0
    for s in range(n):
        dist = np.full(n, -1); dist[s] = 0; fr = [s]; d = 0
        while fr:
            d += 1; nx = np.unique(np.nonzero(A[fr].any(0) & (dist < 0))[0]); dist[nx] = d; fr = list(nx)
        tot += (1.0 / dist[dist > 0]).sum()
    return tot / (n * (n - 1))


def h0_spread(S):
    if len(S) < 10: return np.nan
    D = np.clip(1 - S @ S.T, 0, None); np.fill_diagonal(D, 0); d = mst_edge_lengths(D); eps = d.max()
    x = np.linspace(0, 1, 200); beta = (d[None, :] > (x * eps)[:, None]).sum(1).astype(float); c = (x * beta).sum() / beta.sum()
    return float(np.sqrt(((x - c) ** 2 * beta).sum() / beta.sum()))


def pc1_share(H):
    Hc = H - H.mean(0); ev = np.clip(np.linalg.eigvalsh(Hc @ Hc.T), 0, None); return ev[-1] / (ev.sum() + 1e-12)


BUD = [512, 1024, 2048, 3072, 4096, 6144]
for i in rows:
    fj = force_json(model[i], allm.id.iloc[i]); te = fj.get("think_end"); n_gen = int(allm.n_gen.iloc[i])
    L = te if (te is not None and te < CAP) else min(fj["L"], CAP); g = min(n_gen, CAP)
    R = fj["results"]; lab = fj["labels"]
    p1 = [R[f"{lab[f'b{b}']}|B"]["top"][0]["probs"][0] for b in BUD]; ent = [R[f"{lab[f'b{b}']}|B"]["first_entropy"] for b in BUD]
    eA = [R[f"{lab[f'b{b}']}|A"]["first_entropy"] for b in BUD]
    put("forced|p1 slope over budgets", i, np.polyfit(np.log(BUD), p1, 1)[0])
    put("forced|p1(6144) - p1(2048)", i, p1[5] - p1[2])
    put("forced|log entropy ratio 6144/2048", i, np.log((ent[5] + 1e-9) / (ent[2] + 1e-9)))
    put("forced|share of budgets saturated", i, np.mean(np.array(p1) >= .99))
    rel = [abs(a - b) / (a + b + 1e-9) for a, b in zip(eA, ent)]
    put("A/B|relative shift at 6144", i, rel[5]); put("A/B|max relative shift", i, max(rel))
    z = np.load(OUT / model[i] / "tf" / f"{allm.id.iloc[i]}.npz"); ge_ = z["gen_entropy"].astype(float)[:g]
    t3 = max(1, len(ge_) // 3); put("tokens|entropy last/first third", i, np.log((ge_[-t3:].mean() + 1e-9) / (ge_[:t3].mean() + 1e-9)))
    ws = np.load(P2 / model[i] / "tf" / f"{allm.id.iloc[i]}.npz")["win_states"][: -(-g // 16)].astype(np.float32)
    W3 = ws[:, 3]; half = len(W3) // 2
    if len(W3) >= 2 * 4 * 15:
        put("kNN|global eff late - early", i, global_eff(nodes(W3[half:])) - global_eff(nodes(W3[:half])))
        put("TDA|H0 spread late - early", i, h0_spread(nodes(W3[half:])) - h0_spread(nodes(W3[:half])))
    put("TDA|H0 spread, self-centred, 6k", i, h0_spread(nodes(W3)))
    put("trajectory|late-vs-early cosine (6k)", i, f_late_vs_early(ws))
    nw = len(W3)
    if nw >= 128:
        a, b = int(.25 * nw - 32), int(.75 * nw - 32)
        put("trajectory|PC1 share 75% / 25%", i, np.log(np.mean([pc1_share(ws[b:b + 64, l]) for l in range(6)]) / np.mean([pc1_share(ws[max(0, a):max(0, a) + 64, l]) for l in range(6)])))
    Wc = W3 - W3.mean(0); dist = np.linalg.norm(Wc, axis=1); q = max(1, len(dist) // 4)
    put("dispersion|late / early", i, np.log(dist[-q:].mean() / (dist[:q].mean() + 1e-9)))
    p = R3 / model[i] / "pearl" / f"{allm.id.iloc[i]}.npz"
    if p.exists():
        zp = np.load(p, allow_pickle=True); kind, rate = zp["kind"], zp["rate"]; S = zp["mean_state"].astype(np.float32); o = S[kind == "original"][0]
        cd = lambda X: (1 - (X * o).sum(-1) / (np.linalg.norm(X, axis=-1) * np.linalg.norm(o, axis=-1) + 1e-9)).mean()   # noqa: E731
        put("PEARL|drift at 0.3 / drift at 0.02", i, np.log(cd(S[(kind == "noise") & np.isclose(rate, .3)]) / (cd(S[(kind == "noise") & np.isclose(rate, .02)]) + 1e-12)))
# references: the absolute-level versions we ship (for comparison)
F28 = pickle.load(open(A / "exp28_features_cache.pkl", "rb")); F32 = pickle.load(open(A / "exp32_features.pkl", "rb"))
F["REF absolute|forced conf 6144"] = F28["meta:fa_b6144_B_p1"]; F["REF absolute|kNN global eff (model-mean centred)"] = F32["step64|k5|4k|global_efficiency"]
F["REF absolute|PEARL drift area"] = F28["pearl:mean_state:area"]

up = np.unique(prob[rows]); yp = np.array([y[rows][prob[rows] == p][0] for p in up]).astype(int)
fp = np.array([fin[rows][prob[rows] == p].mean() for p in up]); lp = np.array([lgn[rows][prob[rows] == p].mean() for p in up])
rng = np.random.default_rng(0)


def auc(x, yy):
    rk = rankdata(x); n1 = yy.sum(); n0 = len(yy) - n1
    return (rk[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


print(f"{'feature':48s} {'rows':>4s} {'within-model':>12s} {'controlled (p)':>16s} {'eta2 model':>10s} {'raw global':>10s}")
for nm, v in F.items():
    x = v[rows]; ok = np.isfinite(x); r = rows[ok]
    if ok.sum() < 20 or len(np.unique(x[ok])) < 3: continue
    z = np.full(len(allm), np.nan)
    for m in np.unique(model[r]):
        rr = r[model[r] == m]; z[rr] = (v[rr] - v[rr].mean()) / (v[rr].std() + 1e-12)
    xp = np.array([np.nanmean(z[rows][prob[rows] == p]) if np.isfinite(z[rows][prob[rows] == p]).any() else np.nan for p in up]); okp = np.isfinite(xp)
    C = np.c_[np.ones(okp.sum()), fp[okp], lp[okp]]; xr = xp[okp] - C @ np.linalg.lstsq(C, xp[okp], rcond=None)[0]
    a = auc(xr, yp[okp]); rk = rankdata(xr); y_ = yp[okp]; n1 = y_.sum(); n0 = len(y_) - n1
    YP = rng.permuted(np.tile(y_, (5000, 1)), axis=1).astype(float); nl = (YP @ rk - n1 * (n1 + 1) / 2) / (n1 * n0)
    grand = v[r].mean(); ss_t = ((v[r] - grand) ** 2).sum(); ss_b = sum(((model[r] == m).sum() * (v[r][model[r] == m].mean() - grand) ** 2) for m in np.unique(model[r]))
    print(f"{nm:48s} {len(r):4d} {roc_auc_score(y[r], z[r]):12.2f} {a:9.2f} ({np.mean(np.abs(nl - .5) >= abs(a - .5)):.3f}) {ss_b / (ss_t + 1e-12):10.2f} {roc_auc_score(y[r], v[r]):10.2f}")
pickle.dump(F, open(A / "exp44_features.pkl", "wb"))
