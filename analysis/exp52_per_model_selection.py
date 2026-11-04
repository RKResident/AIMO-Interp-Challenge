"""exp52: per-model feature choice for DeepSeek and Skywork vs the shared recipe, with the choice made INSIDE the cross-validation.
Data per model: its main_v2 rows (official labels) + its olympiad rows (friend's greedy labels), every feature at the 6144 cap with the
same definitions (exp47 olympiad / exp47b + exp51 main_v2). Folds: leave one PROBLEM out (a problem that appears in both sets is one group).
In each training fold, per model: every candidate's within-training AUC and sign; the top-k features by |AUC - 0.5| (k = 1, 2, 3, 5),
with their training signs; a held-out row's score = mean of its signed percentiles among the training values (as the submission scores).
Shared recipe (fa + chk + fas, signs +1) is scored under the same scheme. Out-of-fold AUC per model and pooled within model, with a
label-permutation p for the per-model vs shared difference. Also: what each model picks on ALL its rows (for a submission)."""
import json, os, pickle, re, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "2"
import numpy as np
from collections import Counter
A = "/home/sabyasachi19/PycharmProjects/AimoAGAIN/analysis/"
sys.path.insert(0, A)
from common import *                                                                    # noqa: E402,F401

D47 = pickle.load(open(A / "exp47_features.pkl", "rb")); MB = pickle.load(open(A / "exp47b_mainv2_values.pkl", "rb")); MA = pickle.load(open(A / "exp51_attention.pkl", "rb"))
CAND = {"fa": ("4 output|forced p1 @6144", "4 output|forced p1 @6144", None), "chk": ("6 text|check share", "6 text|check share", None),
        "fas": ("4 output|confidence slope over budgets", "4 output|confidence slope over budgets", None),
        "fsat": ("4 output|share of saturated budgets", "4 output|share of saturated budgets", None),
        "len": ("0 shortcuts|-log trace length", "0 shortcuts|-log trace length", None),
        "verification": ("6 text|verification per 1k words", "6 text|verification per 1k words", None),
        "hesitation": ("6 text|hesitation per 1k words", "6 text|hesitation per 1k words", None),
        "restart": ("6 text|restart per 1k words", "6 text|restart per 1k words", None),
        "entropy bursts": ("4 output|entropy bursts: share of tokens > 2 nats (-)", "4 output|entropy bursts: share of tokens > 2 nats (-)", None),
        "token entropy": ("4 output|token entropy mean (-)", "4 output|token entropy mean (-)", None),
        "speed variation": ("2 states|speed variation (CV)", "2 states|speed variation (CV)", None),
        "late-vs-early cos": ("2 states|late-vs-early cosine", "2 states|late-vs-early cosine", None),
        "kNN": ("6 graphs+TDA|kNN global efficiency", "6 graphs+TDA|kNN global efficiency", None),
        "A/B shift": ("4 output|A/B entropy shift (max)", "4 output|A/B entropy shift (max)", None),
        "lookback late-early": (None, None, "lookback late - early"), "prompt mass late-early": (None, None, "prompt mass late - early"),
        "lookback last quarter": (None, None, "lookback ratio, last quarter"), "prompt mass": (None, None, "prompt attention mass (mean)")}
allm, _ = load_all(); rows = datasets(allm)["main_v2"]
norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()[:300]                            # noqa: E731
inp = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "t1_run" / "inputs.jsonl")}
mv_model = np.array([HF[m] for m in allm.model.values[rows]])
T = {}                                                                                  # model -> dict(X, y, group, src)
for m in ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "Skywork/Skywork-OR1-Math-7B"):
    X, Y, G, S = [], [], [], []
    for j, i in enumerate(rows):
        if mv_model[j] != m: continue
        X.append([MB["M"][o][j] if o in MB["M"] else (MA["M"][a][j] if a else np.nan) for _, (k47, o, a) in CAND.items()])
        Y.append(int(allm.y.values[i] == 1)); G.append(norm(inp[allm.id.iloc[i]]["problem"])); S.append("main_v2")
    for k in range(len(D47["ids"])):
        if D47["model"][k] != m: continue
        X.append([D47["F"][k47][k] if k47 else MA["O"][a][k] for _, (k47, o, a) in CAND.items()])
        Y.append(int(D47["y"][k])); G.append(norm(inp[D47["ids"][k]]["problem"])); S.append("olympiad")
    T[m] = dict(X=np.array(X, float), y=np.array(Y), g=np.array(G), src=np.array(S))
NAMES = list(CAND); SHARED = [NAMES.index(f) for f in ("fa", "chk", "fas")]
missing = {m: [NAMES[c] for c in range(len(NAMES)) if np.isnan(T[m]["X"][:, c]).mean() > .3] for m in T}
print("rows:", {m.split("/")[0][:8]: f"{len(T[m]['y'])} ({T[m]['y'].sum()} robust; main_v2 {np.sum(T[m]['src'] == 'main_v2')}, olympiad {np.sum(T[m]['src'] == 'olympiad')}; "
               f"{len(set(T[m]['g']))} problems)" for m in T}, "| candidates mostly missing:", missing)


def auc(x, y):
    ok = np.isfinite(x); x, y = x[ok], y[ok]; p, n = x[y == 1], x[y == 0]
    return ((p[:, None] > n[None]).sum() + .5 * (p[:, None] == n[None]).sum()) / (len(p) * len(n)) if len(p) and len(n) else np.nan


def pct(x, ref):
    ref = ref[np.isfinite(ref)]
    return ((ref < x).sum() + .5 * (ref == x).sum()) / len(ref) if np.isfinite(x) and len(ref) else np.nan


def oof(m, choose, y=None):
    X, g = T[m]["X"], T[m]["g"]; y = T[m]["y"] if y is None else y; out = np.full(len(y), np.nan); picks = Counter()
    for p in np.unique(g):
        te = g == p; tr = ~te
        feats = choose(X[tr], y[tr]); picks.update(NAMES[c] for c, _ in feats)
        for i in np.where(te)[0]:
            u = [(pct(X[i, c], X[tr, c]) if s > 0 else 1 - pct(X[i, c], X[tr, c])) for c, s in feats]
            u = [v for v in u if np.isfinite(v)]; out[i] = np.mean(u) if u else .5
    return out, picks


def top_k(k):
    def ch(Xtr, ytr):
        a = np.array([auc(Xtr[:, c], ytr) for c in range(Xtr.shape[1])]); a = np.where(np.isnan(a), .5, a)
        order = np.argsort(-np.abs(a - .5))[:k]; return [(int(c), 1 if a[c] >= .5 else -1) for c in order]
    return ch


shared = lambda Xtr, ytr: [(c, 1) for c in SHARED]                                      # noqa: E731
rng = np.random.default_rng(0)
print(f"\n{'scheme':38s} {'DeepSeek':>9s} {'Skywork':>9s} {'pooled within':>14s}")
R = {}
for nm, ch in [("shared: fa + chk + fas (signs +1)", shared)] + [(f"per-model top-{k} (chosen in each fold)", top_k(k)) for k in (1, 2, 3, 5)]:
    res = {m: oof(m, ch) for m in T}; R[nm] = res
    per = {m: auc(res[m][0], T[m]["y"]) for m in T}
    num = sum(auc(res[m][0], T[m]["y"]) * T[m]["y"].sum() * (len(T[m]["y"]) - T[m]["y"].sum()) for m in T)
    den = sum(T[m]["y"].sum() * (len(T[m]["y"]) - T[m]["y"].sum()) for m in T)
    print(f"{nm:38s} {per['deepseek-ai/DeepSeek-R1-0528-Qwen3-8B']:9.2f} {per['Skywork/Skywork-OR1-Math-7B']:9.2f} {num / den:14.2f}"
          + ("" if nm.startswith("shared") else "   picks: " + "; ".join(f"{m.split('/')[0][:4]}: " + ", ".join(f"{f} {c}" for f, c in res[m][1].most_common(4)) for m in T)))
# what each model picks on all of its rows, top 5 with signs
print("\nper-model ranking on ALL rows (AUC, sign):")
for m in T:
    a = [(NAMES[c], auc(T[m]["X"][:, c], T[m]["y"])) for c in range(len(NAMES))]
    a = sorted([x for x in a if np.isfinite(x[1])], key=lambda t: -abs(t[1] - .5))
    print(f"  {m.split('/')[0][:8]}: " + ", ".join(f"{f} {v:.2f}" for f, v in a[:8]))
pickle.dump(dict(T=T, NAMES=NAMES), open(A / "exp52_tables.pkl", "wb"))
