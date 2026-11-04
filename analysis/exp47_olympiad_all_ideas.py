"""exp47: every idea family we have tried (IDEAS_TRIED.md sections 0-7), re-scored on the olympiad set: 75 labelled (model, problem) rows,
DeepSeek 43 + Skywork 32, labels conditioned on a solved original (test-like). Reported on all rows and on the 49 rows whose problem is NOT
also in main_v2 (verbatim or number-swapped), which is the independent check.

Sources per row:
  ours (Vast olympiad_features.zip, greedy trace capped at 6144): 16-token window states (6 traj layers), per-token entropy / log-prob /
      top-1, forced answers (p1, entropy, top-5, 12-token answer string; suffixes A and B) at budgets 512..6144 and thinking fractions,
      states at cutoffs, last-prompt state at every layer, problem-token log-prob, hidden states at the forced suffix, text;
  attention (Vast, friend's extract_features.py on the same traces): LapEigvals per layer / head / cutoff; Lookback-Lens per 16 tokens;
  friend's run (his 10240-token traces): PEARL prose noise + 5 semantic variants, TOHA, extras (attention FFT / KL / lookback at the
      forced suffix, residual-stream activations).
Score per feature: within-model AUC (higher value -> robust; < 0.5 = the feature points to spurious), label permutation inside each model.
Probes (prompt state, reasoning states, forced-suffix states, extras activations) are leave-one-problem-out logistic regressions on
per-model standardised PCA features, so their AUC is out-of-sample. MemLens (needs a logit lens at the boxed position) and the decay
direction (needs agg permutation pairs) cannot be computed from this data and are left out."""
import csv, glob, json, os, re, sys, difflib
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "2"
import numpy as np
from pathlib import Path
from scipy.stats import rankdata
sys.path.insert(0, "/home/sabyasachi19/PycharmProjects/AimoAGAIN/submission_core")
from spurious_features import f_check_share, f_cluster_cycle_rank, f_pearl_area, rips_persistence, mst_edge_lengths

ROOT = Path("/home/sabyasachi19/PycharmProjects/AimoAGAIN"); OLY = ROOT / "extra_olympiad_problems_dataset"
OURS = OLY / "results_olympiad_features" / "out_t1"; ATT = OLY / "results_olympiad_attention" / "out_attn"; FR = OLY / "olympiad-features"
ART = Path("/home/sabyasachi19/PycharmProjects/aimo/getting-started/solutions/spurious-cluster-graph/spurious_artifacts")
rows = [json.loads(l) for f in sorted(OURS.glob("*/rows_shard*.jsonl")) for l in open(f)]
inp = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "t1_run" / "inputs.jsonl")}
N = len(rows); y = np.array([r["robust"] for r in rows]).astype(int); model = np.array([r["model_id"] for r in rows])
slug = lambda m: m.replace("/", "__")                                                     # noqa: E731
norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()                                    # noqa: E731
mvp = [norm(r["problem"])[:400] for r in inp.values() if r["dataset"] == "main_v2"]
indep = np.array([max(difflib.SequenceMatcher(None, norm(inp[r["id"]]["problem"])[:400], q).ratio() for q in mvp) <= .8 for r in rows])
problem = np.array([norm(inp[r["id"]]["problem"]) for r in rows])
F, FAM = {}, {}


def put(fam, name, i, v):
    k = f"{fam}|{name}"; FAM[k] = fam
    F.setdefault(k, np.full(N, np.nan))[i] = v


def cos(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def first_number(s):
    m = re.search(r"-?\d+(?:\.\d+)?(?:/\d+)?", s.replace(",", "")); return m.group(0) if m else s.strip()[:6]


MARK = {"hesitation": r"\b(wait|hmm+|hold on|actually)\b", "verification": r"\b(check|verify|double-check|confirm|let me make sure)\b",
        "restart": r"\b(alternatively|another approach|let me (?:re)?start|let's try|start over|scratch that)\b",
        "answer mentions": r"(\\boxed|final answer|the answer is)"}
KM = {m: np.load(ART / f"km64_{slug(m)}.npy") for m in set(model)}; MU = {m: np.load(ART / f"mu_{slug(m)}.npy") for m in set(model)}
PROBE = {k: [] for k in ("prompt state (last token, 4 layers)", "reasoning mean state (6 layers)", "forced-suffix state B @6144",
                         "state at last cutoff (mean of last 64)", "extras residual @ forced suffix (mid layer)")}

for i, r in enumerate(rows):
    g = r["greedy"]; m = r["model_id"]; z = np.load(OURS / slug(m) / f"{r['id']}.npz")
    ws = z["t0_ws16"].astype(np.float32); ent = z["t0_tok_entropy"].astype(np.float32); lp = z["t0_tok_logprob"].astype(np.float32)
    top1 = z["t0_tok_top1"].astype(np.float32); L = g["L"]; n = g["n_gen"]; text = g["text"]
    # 0. shortcuts
    put("0 shortcuts", "prompt length (-)", i, -len(inp[r["id"]]["prompt_ids"])); put("0 shortcuts", "-log trace length", i, g["len"])
    put("0 shortcuts", "finished by 6144", i, float(g["think_end"] is not None))
    # 1. the prompt
    plp = z["t0_problem_logprob"].astype(np.float32) if "t0_problem_logprob" in z.files else np.array([np.nan])
    put("1 prompt", "problem log-prob (mean)", i, plp.mean()); put("1 prompt", "Min-K% (lowest 20% log-probs)", i, np.sort(plp)[: max(1, len(plp) // 5)].mean())
    pl = z["t0_prompt_last_all_layers"].astype(np.float32); nl = len(pl)
    PROBE["prompt state (last token, 4 layers)"].append(np.concatenate([pl[int(f * (nl - 1))] for f in (.25, .5, .75, 1.0)]))
    # 2. hidden-state geometry
    cl, cm, pm = z["t0_cut_last"].astype(np.float32), z["t0_cut_mean64"].astype(np.float32), z["t0_problem_mean"].astype(np.float32)
    d = [np.linalg.norm(cl[k + 1] - cl[k]) / (np.linalg.norm(cl[k]) + 1e-9) for k in range(len(cl) - 1)]
    put("2 states", "drift between cutoffs (mean)", i, np.mean(d) if d else np.nan)
    put("2 states", "anchoring cosine (last cutoff vs problem)", i, np.mean([cos(cm[-1, l], pm[l]) for l in range(pm.shape[0])]))
    W3 = ws[:, 3]; dW = np.diff(W3, axis=0); sp = np.linalg.norm(dW, axis=1)
    put("2 states", "speed (mean step)", i, sp.mean()); put("2 states", "speed variation (CV)", i, sp.std() / (sp.mean() + 1e-9))
    put("2 states", "tortuosity (path / displacement, -)", i, -sp.sum() / (np.linalg.norm(W3[-1] - W3[0]) + 1e-9))
    put("2 states", "late-vs-early cosine", i, g["traj"]); put("2 states", "segment PC1 share @25%", i, g["seg"])
    put("2 states", "PC1 change 75%/25%", i, g["pcr"])
    S64 = W3[: len(W3) // 4 * 4].reshape(-1, 4, W3.shape[1]).mean(1)
    put("2 states", "step similarity (consecutive 64-token, -)", i, -np.mean([cos(S64[k], S64[k + 1]) for k in range(min(len(S64), 64) - 1)]))
    dp = ws[:, 1:] - ws[:, :-1]
    put("2 states", "depth-path straightness", i, np.mean(np.linalg.norm(ws[:, -1] - ws[:, 0], axis=1) / (np.linalg.norm(dp, axis=2).sum(1) + 1e-9)))
    PROBE["reasoning mean state (6 layers)"].append(ws.mean(0).ravel()); PROBE["state at last cutoff (mean of last 64)"].append(cm[-1].ravel())
    # graphs / topology on the averaged trajectory
    put("6 graphs+TDA", "kNN global efficiency", i, g["ge"]); put("6 graphs+TDA", "TDA H0 Betti spread", i, g["tda"])
    put("6 graphs+TDA", "cluster cycle rank (K=64)", i, f_cluster_cycle_rank(W3, MU[m], KM[m], L))
    nodes = W3[: max(4, min(len(W3), L // 16)) // 4 * 4].reshape(-1, 4, W3.shape[1]).mean(1)
    nodes = nodes / (np.linalg.norm(nodes, axis=1, keepdims=True) + 1e-9)
    if len(nodes) >= 10:
        D = np.clip(1 - nodes @ nodes.T, 0, None); np.fill_diagonal(D, 0); h0, h1 = rips_persistence(D)
        put("6 graphs+TDA", "H1 loop count (-)", i, -len(h1)); put("6 graphs+TDA", "H1 total persistence (-)", i, -float((h1[:, 1] - h1[:, 0]).sum()) if len(h1) else 0.0)
    # 3. attention
    a = ATT / slug(m)
    if (a / "attn" / f"{r['id']}.npz").exists():
        za = np.load(a / "attn" / f"{r['id']}.npz"); j = int(np.argmax(za["cut_eff"])); vB = list(za["variant_names"]).index("B")
        put("3 attention", "LapEigvals mean, no forcing", i, np.nanmean(za["mean_noforce"][j].astype(np.float32)))
        put("3 attention", "LapEigvals mean, forced B", i, np.nanmean(za["mean_force"][j, vB].astype(np.float32)))
        put("3 attention", "LapEigvals top-1, no forcing", i, np.nanmean(za["topk_noforce"][j, :, :, 0].astype(np.float32)))
    if (a / "lookback" / f"{r['id']}.npz").exists():
        zl = np.load(a / "lookback" / f"{r['id']}.npz"); lb = zl["lookback"].astype(np.float32); q = max(1, len(lb) // 4)
        put("3 attention", "lookback ratio (mean)", i, lb.mean()); put("3 attention", "lookback ratio, last quarter", i, lb[-q:].mean())
        put("3 attention", "lookback late - early", i, lb[-q:].mean() - lb[:q].mean()); put("3 attention", "prompt attention mass (mean)", i, zl["prompt_mass"].astype(np.float32).mean())
    fr = FR / slug(m)
    if (fr / "toha" / f"{r['id']}.npz").exists():
        zt = np.load(fr / "toha" / f"{r['id']}.npz", allow_pickle=True); sn = list(zt["stat_names"])
        put("3 attention", "TOHA window total (first cut, -)", i, -np.nanmean(zt["win"][0, :, :, sn.index("total")]))
        put("3 attention", "TOHA answer total (first cut, -)", i, -np.nanmean(zt["ans"][0, 0, :, :, sn.index("total")]))
        put("3 attention", "TOHA answer: share direct to prompt", i, np.nanmean(zt["ans"][0, 0, :, :, sn.index("frac_direct_to_prompt")]))
    if (fr / "extra" / f"{r['id']}.npz").exists():
        ze = np.load(fr / "extra" / f"{r['id']}.npz", allow_pickle=True); vB = list(ze["variant_names"]).index("B"); j = len(ze["cut_eff"]) - 1
        fft = ze["fft"][j, vB].astype(np.float32); put("3 attention", "attention FFT low-frequency share", i, np.nanmean(fft[..., :8].sum(-1) / (fft.sum(-1) + 1e-9)))
        put("3 attention", "KL between adjacent layers", i, np.nanmean(ze["kl_adj"][j, vB])); put("3 attention", "KL head vs layer mean", i, np.nanmean(ze["kl_headmean"][j, vB]))
        put("3 attention", "lookback at forced suffix", i, np.nanmean(ze["lookback"][j, vB]))
        rl = ze["res_last"][j, vB].astype(np.float32); PROBE["extras residual @ forced suffix (mid layer)"].append(rl[len(rl) // 2])
    else:
        PROBE["extras residual @ forced suffix (mid layer)"].append(None)
    # 4. output distribution
    fo = g["forced"]
    for b in (512, 1024, 2048, 4096, 6144):
        put("4 output", f"forced p1 @{b}", i, fo[f"b{b}"]["B"]["p1"])
    put("4 output", "forced entropy @6144 (-)", i, -fo["b6144"]["B"]["entropy"])
    put("4 output", "forced p1 at end of thinking (f100)", i, fo["f100"]["B"]["p1"])
    put("4 output", "confidence slope over budgets", i, g["fas"]); put("4 output", "share of saturated budgets", i, g["fsat"])
    put("4 output", "A/B entropy shift (max)", i, g["ab"])
    put("4 output", "A and B agree on top-1 @6144", i, float(fo["b6144"]["A"]["top5_ids"][0] == fo["b6144"]["B"]["top5_ids"][0]))
    ans = [first_number(fo[f"b{b}"]["B"].get("continuation", "")) for b in (512, 1024, 2048, 3072, 4096, 6144)]
    put("4 output", "answer @512 = answer @6144 (guessable)", i, float(ans[0] == ans[-1]))
    stab = next((k for k in range(len(ans)) if all(x == ans[-1] for x in ans[k:])), len(ans))
    put("4 output", "answer stabilises early (-budget index)", i, -stab)
    put("4 output", "token entropy mean (-)", i, -ent.mean()); put("4 output", "token log-prob mean", i, lp.mean())
    put("4 output", "token top-1 mean", i, top1.mean()); put("4 output", "entropy bursts: share of tokens > 2 nats (-)", i, -np.mean(ent > 2))
    put("4 output", "entropy last third / first third (-)", i, -np.log((ent[-len(ent) // 3:].mean() + 1e-6) / (ent[: len(ent) // 3].mean() + 1e-6)))
    fh = z["t0_force_hidden"].astype(np.float32); ce = list(z["t0_cut_eff"]); jB = list(z["t0_force_suffixes"]).index("B")
    PROBE["forced-suffix state B @6144"].append(fh[ce.index(g["cuts"]["b6144"]), jB].ravel())
    # 5. input perturbation (friend's PEARL files: prompt + 128 reasoning tokens)
    if (fr / "pearl" / f"{r['id']}.npz").exists():
        zp = np.load(fr / "pearl" / f"{r['id']}.npz", allow_pickle=True); kind = zp["kind"]; ms = zp["mean_state"].astype(np.float32)
        noise = np.where(kind == "noise")[0]; o = np.where(kind == "original")[0][0]
        put("5 perturbation", "PEARL drift area (-)", i, -f_pearl_area(np.concatenate([ms[[o]], ms[noise]])))
        put("5 perturbation", "PEARL KL area (-)", i, -float(np.mean(zp["kl_vs_original"][noise])))
        put("5 perturbation", "PEARL prefix NLL change (-)", i, -float(np.mean(zp["prefix_nll"][noise] - zp["prefix_nll"][o])))
        put("5 perturbation", "PEARL top-1 kept under noise", i, float(np.mean(zp["top1_same"][noise])))
        sem = [k for k in range(len(kind)) if kind[k] not in ("noise", "original")]
        if sem:
            put("5 perturbation", "semantic variants: state shift (-)", i, -np.mean([1 - cos(ms[k].ravel(), ms[o].ravel()) for k in sem]))
            put("5 perturbation", "semantic variants: KL (-)", i, -float(np.mean(zp["kl_vs_original"][sem])))
            put("5 perturbation", "semantic variants: top-1 kept", i, float(np.mean(zp["top1_same"][sem])))
    # 6. text
    words = max(1, len(text.split()))
    for k, pat in MARK.items():
        put("6 text", f"{k} per 1k words", i, 1000 * len(re.findall(pat, text, flags=re.I)) / words)
    put("6 text", "check share", i, g["chk"])
    sents = [s.strip().lower() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) > 20]
    put("6 text", "repeated sentences (share, -)", i, -(1 - len(set(sents)) / max(1, len(sents))))

# probes: per model (hidden sizes differ), leave-one-problem-out logistic regression on standardised PCA(5) inputs fitted on the training rows
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
for nm, X in PROBE.items():
    out = np.full(N, np.nan)
    for m in set(model):
        idx = np.array([k for k in range(N) if model[k] == m and X[k] is not None])
        if len(idx) < 15: continue
        Xm = np.stack([X[k] for k in idx]).astype(np.float64)
        for p in np.unique(problem[idx]):
            te = problem[idx] == p; tr = ~te
            mu_, sd = Xm[tr].mean(0), Xm[tr].std(0) + 1e-6
            pca = PCA(5).fit((Xm[tr] - mu_) / sd)
            clf = LogisticRegression(C=0.1, max_iter=2000).fit(pca.transform((Xm[tr] - mu_) / sd), y[idx][tr])
            out[idx[te]] = clf.decision_function(pca.transform((Xm[te] - mu_) / sd))
    k = f"probe|{nm}"; F[k] = out; FAM[k] = {"prompt": "1 prompt", "forced-suffix": "4 output", "extras": "3 attention"}.get(nm.split()[0], "2 states")

rng = np.random.default_rng(0)


def within(x, yy, mm):
    num = den = 0
    for m in np.unique(mm):
        k = (mm == m) & np.isfinite(x); p, n = x[k & (yy == 1)], x[k & (yy == 0)]
        num += (p[:, None] > n[None]).sum() + .5 * (p[:, None] == n[None]).sum(); den += len(p) * len(n)
    return num / den if den else np.nan


def perm(x, yy, mm, n=2000):
    w = within(x, yy, mm); nl = np.empty(n)
    for t in range(n):
        yp = yy.copy()
        for m in np.unique(mm): k = mm == m; yp[k] = rng.permutation(yy[k])
        nl[t] = within(x, yp, mm)
    return w, float(np.mean(np.abs(nl - .5) >= abs(w - .5)))


res = []
for k, v in F.items():
    if np.isfinite(v).sum() < 30 or len(np.unique(v[np.isfinite(v)])) < 2: continue
    wa, pa = perm(v, y, model); wi, pi = perm(v[indep], y[indep], model[indep])
    res.append((FAM[k], k.split("|", 1)[1], int(np.isfinite(v).sum()), wa, pa, wi, pi))
print(f"{N} rows (DeepSeek {np.sum(model == 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B')}, Skywork {np.sum(model == 'Skywork/Skywork-OR1-Math-7B')}); "
      f"independent of main_v2: {indep.sum()} ({y[indep].sum()} robust)")
print(f"{'family':16s} {'feature':46s} {'n':>3s} {'all 75 (p)':>14s} {'independent 49 (p)':>20s}")
for fam, nm, n, wa, pa, wi, pi in sorted(res):
    flag = " *" if pi < .05 else ""
    print(f"{fam:16s} {nm:46s} {n:3d} {wa:7.2f} ({pa:.3f}) {wi:12.2f} ({pi:.3f}){flag}")
nsig = sum(pi < .05 for *_, pi in res); print(f"\n{nsig} of {len(res)} features at p < 0.05 on the independent rows (chance: {0.05 * len(res):.1f})")
import pickle; pickle.dump(dict(F=F, FAM=FAM, y=y, model=model, indep=indep, problem=problem, ids=[r["id"] for r in rows]),
                           open(ROOT / "analysis" / "exp47_features.pkl", "wb"))
