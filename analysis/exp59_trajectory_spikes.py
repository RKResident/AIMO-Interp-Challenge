"""exp59: trajectory spikes (user's observation: DeepSeek's hidden-state paths jump far away and come back, the other models move smoothly).
For every labelled greedy trace (main_v2 + olympiad, capped at 6144) and, on main_v2, the two T=1 samples:
  states   16-token window means at the 6 saved layers (the submission's window states), each layer standardised per trace
  spike    distance of a window from the median of its +-8 neighbouring windows (itself excluded), in units of the trace's median such
           distance; a window is a spike when that ratio exceeds 3 (also 5 for 'large')
Part 1: spike rate per model (and per layer), and the decoded text at the largest spikes - what is the model writing when it jumps?"""
import json, os, re, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "2"
import numpy as np
import pandas as pd
sys.path.insert(0, "/home/sabyasachi19/PycharmProjects/AimoAGAIN/analysis"); sys.path.insert(0, "/home/sabyasachi19/PycharmProjects/AimoAGAIN/submission_core")
from common import *                                                                    # noqa: E402,F401

D = pd.read_csv(A / "exp56_rows.csv"); slug = lambda m: m.replace("/", "__")             # noqa: E731
OLR = ROOT / "extra_olympiad_problems_dataset" / "results_olympiad_features" / "out_t1"; MV1 = ROOT / "results_mainv2_t1" / "out_t1"
inp = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "t1_run" / "inputs.jsonl")}


def load_states(row, which="t0"):
    z = np.load((MV1 if row.dataset == "main_v2" else OLR) / slug(row.model_id) / f"{row.id}.npz")
    if f"{which}_ws16" not in z.files: return None, None
    ids = z[f"{which}_gen_ids"] if f"{which}_gen_ids" in z.files else np.array(inp[row.id]["greedy_gen_ids"][:6144])
    return z[f"{which}_ws16"].astype(np.float32), ids


def spike_ratio(ws, half=8):
    """[n_win, n_layer] distance of each window from its neighbourhood median, divided by the trace's median distance (per layer)."""
    n, L, d = ws.shape
    X = (ws - ws.mean(0)) / (ws.std(0) + 1e-6)
    out = np.zeros((n, L), np.float32)
    for t in range(n):
        nb = [k for k in range(max(0, t - half), min(n, t + half + 1)) if k != t]
        if not nb: continue
        ref = np.median(X[nb], axis=0); out[t] = np.linalg.norm(X[t] - ref, axis=-1)
    return out / (np.median(out, axis=0) + 1e-9)


if __name__ == "__main__":
    from spurious_inference import load_tokenizer
    stats, examples = [], []
    for _, r in D.iterrows():
        ws, ids = load_states(r)
        if ws is None or len(ws) < 32: continue
        s = spike_ratio(ws); sp3 = s > 3; sp5 = s > 5
        stats.append(dict(model=r.model, dataset=r.dataset, robust=r.robust, finished=r.finished,
                          rate3=sp3.any(1).mean(), rate5=sp5.any(1).mean(), per_layer=sp3.mean(0), max_ratio=float(s.max())))
        if r.model == "deepseek" and len(examples) < 400:
            top = np.argsort(-s.max(1))[:3]
            for t in top: examples.append((r.id, int(t), float(s[t].max()), ids))
    S = pd.DataFrame(stats)
    print("share of 16-token windows that are spikes (any layer), per model:")
    print(S.groupby("model")[["rate3", "rate5", "max_ratio"]].median().rename(columns={"rate3": "ratio > 3", "rate5": "ratio > 5", "max_ratio": "largest ratio"}).round(3).to_string())
    print("\nspike share (ratio > 3) by layer (6 saved layers, shallow -> deep), median over traces:")
    for m, g in S.groupby("model"):
        print(f"  {m:9s} " + " ".join(f"{v:.3f}" for v in np.median(np.stack(g.per_layer.values), 0)))
    tok = load_tokenizer("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    examples.sort(key=lambda e: -e[2])
    print("\nDeepSeek: text in the 16-token windows with the largest spikes (with the window before):")
    for rid, t, ratio, ids in examples[:14]:
        before = tok.decode([int(x) for x in ids[max(0, 16 * t - 16):16 * t]], skip_special_tokens=True)
        win = tok.decode([int(x) for x in ids[16 * t:16 * t + 16]], skip_special_tokens=True)
        print(f"  ratio {ratio:5.1f} | ...{re.sub(chr(10), ' / ', before)[-60:]!r} >>> {re.sub(chr(10), ' / ', win)!r}")


# Part 2: the same measurement in the space the explorer draws: RAW layer-3 states (no per-dimension standardisation), 64-token means,
# full dimension and the per-model 2D PCA; plus which hidden dimensions carry DeepSeek's excursions (massive activations?)
def raw_spikes(ws3, half=4):
    n = len(ws3) // 4 * 4; X = ws3[:n].reshape(-1, 4, ws3.shape[-1]).mean(1); m = len(X); d = np.zeros(m)
    for t in range(m):
        nb = [k for k in range(max(0, t - half), min(m, t + half + 1)) if k != t]
        d[t] = np.linalg.norm(X[t] - np.median(X[nb], axis=0))
    return X, d / (np.median(d) + 1e-9)


def part2():
    rows = []
    for _, r in D.iterrows():
        ws, _ = load_states(r)
        if ws is None or len(ws) < 64: continue
        X, s = raw_spikes(ws[:, 3]); top = np.argsort(-np.abs(X - np.median(X, 0)).max(0))[:3]
        rows.append(dict(model=r.model, rate3=(s > 3).mean(), rate5=(s > 5).mean(), maxr=s.max(), norm_cv=np.linalg.norm(X, axis=1).std() / np.linalg.norm(X, axis=1).mean(),
                         maxdim_share=float((np.abs(X - np.median(X, 0)).max(0)[top[0]] ** 2) / ((X - np.median(X, 0)) ** 2).sum(1).max())))
    S = pd.DataFrame(rows)
    print("\nRAW layer-3 states, 64-token steps: spike share (step > 3x / 5x the trace's median jump), largest jump, variation of the state norm,")
    print("and how much of the largest excursion sits in ONE hidden dimension (massive-activation check)")
    print(S.groupby("model")[["rate3", "rate5", "maxr", "norm_cv", "maxdim_share"]].median().round(3).to_string())


if __name__ == "__main__":
    part2()
