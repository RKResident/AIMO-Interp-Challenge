"""exp84: the submission's own features (run_feats.py on the A10G) on Qwen's set100 screening traces vs Qwen's NEW labels (forced and pure
classes), split into likely-novel (2025+ sources) and likely-seen problems. AUC (raw: higher value -> robust) with a permutation p, next to
the direction the shipped recipes use (SIGN) and the old-rows AUC (exp83, 88 unfinished rows).
  python exp84_qwen_new_labels.py [model]   (default Qwen/Qwen3.5-4B; Olmo: novel = its set24_olmo_novel.txt problems, after its release)"""
import json, sys, numpy as np, pandas as pd
A = "/home/sabyasachi19/PycharmProjects/AimoAGAIN/"; rng = np.random.default_rng(0)
MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-4B"; TAG = MODEL.replace("/", "__")
F = pd.DataFrame([json.loads(l) for l in open(A + f"ec2_results/a10g/feats/{TAG}.jsonl")])
L = pd.read_csv(A + f"hard_pool/set100_labels_{TAG}.csv"); P = pd.read_csv(A + "hard_pool/pool_screen.csv").set_index("pid")
NOV = {"hmmt_nov_2025", "smt_2025", "final_answer_comps", "BeyondAIME"}
D = F.merge(L, on="pid")
if "Olmo" in MODEL: NOVP = {l.strip() for l in open(A + "vast_packages/screen/set24_olmo_novel.txt") if l.strip()}; D["novel"] = D.pid.isin(NOVP)
else: D["novel"] = [P.loc[p, "source"].split("/")[-1] in NOV for p in D.pid]
SIGN = {"fa": 1, "chk": 1, "fas": 1, "nrk": -1, "f1k": -1, "cfc": -1, "cfl": -1, "nzf": -1, "jmp": 1, "lble": 1, "pmle": 1}


def auc(s, y):
    ok = np.isfinite(s); s, y = s[ok], y[ok]; p, n = s[y == 1], s[y == 0]
    return float(((p[:, None] > n[None]) + .5 * (p[:, None] == n[None])).mean()) if len(p) and len(n) else np.nan


def pv(s, y, n=2000): a = abs(auc(s, y) - .5); return float(np.mean([abs(auc(s, rng.permutation(y)) - .5) >= a - 1e-12 for _ in range(n)]))


for rule in ("forced", "pure"):
    X = D[D[f"label_{rule}"].isin(["robust", "non_robust"])].copy(); X["y"] = (X[f"label_{rule}"] == "robust").astype(int)
    print(f"\n{rule} class: {len(X)} labelled {MODEL} problems ({X.y.sum()} robust) | novel {int(X.novel.sum())} ({X[X.novel].y.sum()} robust), "
          f"seen {int((~X.novel).sum())} ({X[~X.novel].y.sum()} robust)")
    print(f"   {'feature':6s} sign | all (p)       novel (p)     seen (p)")
    for f in SIGN:
        out = []
        for k in (np.ones(len(X), bool), X.novel.values, ~X.novel.values):
            s, y = X[f].astype(float).values[k], X.y.values[k]
            out.append((auc(s, y), pv(s, y) if len(y) >= 6 and 0 < y.sum() < len(y) else np.nan))
        agree = "" if not np.isfinite(out[1][0]) else ("matches sign" if (out[1][0] - .5) * SIGN[f] > 0 else "AGAINST sign")
        print(f"   {f:6s} {SIGN[f]:+d}   | " + "  ".join(f"{a:.2f} ({p:.2f})" for a, p in out) + f"   novel: {agree}")
D.to_csv(A + ("analysis/exp84_rows.csv" if "Qwen" in MODEL else f"analysis/exp84_rows_{TAG}.csv"), index=False)
