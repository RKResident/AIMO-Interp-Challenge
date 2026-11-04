"""White-box feature battery (user 2026-10-31: retry trajectories / state graphs / TDA / depth paths / PC1 / LapEigvals / TOHA / MLP write on
the new data, at ALL layers). One function, used by the GPU extraction (vast_packages/screen/run_wb.py) and by a logging-only submission, so
leaderboard values come from identical code. Everything is computed on the first 6144 generated tokens (the submission's trace).
Per trace:
  hidden states, every decoder layer l (one teacher-forced pass, output_hidden_states): 16- and 64-token window means of the reasoning
    traj   late_vs_early (unit-mean cosine first vs last quarter, 16-token windows), tortuosity, speed mean / CV, straightness, slowness (64-token)
    pc1    first-PC variance share of the 64-window (1024-token) segment at 25 / 50 / 75%, and pcr = log(75% / 25%)
    graph  cosine kNN graph (k 5, 10) of the trace-mean-centred 64-token states: global efficiency, clustering, normalised-Laplacian Fiedler
           value; epsilon-recurrence at distance quantiles 10% / 20%: determinism and laminarity (RQA)
    tda    Vietoris-Rips persistence of the same distances: H0 total / max persistence, H0 Betti spread, H1 count / total persistence
  depth paths: per 64-token window, the path across layers (unit states): straightness and length, mean over windows and late-minus-early;
    the same for the last prompt token; for all layers and for layers >= 40% depth
  attention (softmax layers): lookback ratio and prompt mass per layer (late-minus-early, last quarter; per-head arrays), 64-token-block graph
    per layer (jump, far share, setup share), TOHA mean MST edge per head (window 128 before 2048 and before the cut), LapEigvals top-8 per
    layer / head at the forced suffix
  MLP write size per layer at the forced answer position: ||mlp_out|| / ||residual in||
Returns (scalars: dict name -> float, arrays: dict name -> np.ndarray)."""
from __future__ import annotations

import math
import numpy as np
import torch

import spurious_inference as SI
from spurious_features import f_block_jump, rips_persistence


def _hops(A: np.ndarray) -> np.ndarray:
    """unweighted all-pairs shortest path lengths (BFS by boolean matrix products; inf = unreachable)"""
    n = len(A); D = np.full((n, n), np.inf); np.fill_diagonal(D, 0); R = np.eye(n, dtype=bool); F = R.copy(); A_ = A.astype(np.float32)
    for d in range(1, n):
        F = ((F.astype(np.float32) @ A_) > 0) & ~R
        if not F.any(): break
        D[F] = d; R |= F
    return D


def _wmeans(h: torch.Tensor, w: int) -> np.ndarray:
    n = h.shape[0] // w
    return h[: n * w].reshape(n, w, -1).mean(1).float().cpu().numpy() if n else np.zeros((0, h.shape[1]), np.float32)


def _unit(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-9)


def _pc1(H):
    Hc = H - H.mean(0); ev = np.clip(np.linalg.eigvalsh(Hc @ Hc.T), 0, None); return float(ev[-1] / (ev.sum() + 1e-12))


def traj_feats(W16, W64) -> dict:
    f = {}
    if len(W16) >= 8:
        U = _unit(W16); q = len(U) // 4; f["late_vs_early"] = float(U[:q].mean(0) @ U[-q:].mean(0))
    if len(W64) >= 4:
        d = np.diff(W64, axis=0); seg = np.linalg.norm(d, axis=1); disp = np.linalg.norm(W64[-1] - W64[0])
        f["tortuosity"] = float(seg.sum() / (disp + 1e-9))
        du = np.diff(_unit(W64), axis=0); sp = np.linalg.norm(du, axis=1); f["speed"] = float(sp.mean()); f["speed_cv"] = float(sp.std() / (sp.mean() + 1e-12))
        dn = _unit(d); f["straightness"] = float((dn[1:] * dn[:-1]).sum(1).mean()) if len(dn) > 1 else np.nan
        U64 = _unit(W64); f["slowness"] = float((U64[1:] * U64[:-1]).sum(1).mean())
    nw = len(W16)
    if nw >= 64:
        for frac in (0.25, 0.5, 0.75):
            st = int(np.clip(frac * nw - 32, 0, nw - 64)); f[f"pc1_{int(frac * 100)}"] = _pc1(W16[st:st + 64].astype(np.float64))
        if nw >= 128 and f.get("pc1_25", 0) > 0: f["pcr"] = float(np.log(f["pc1_75"] / f["pc1_25"]))
    return f


def graph_feats(W64) -> dict:
    f = {}
    n = len(W64)
    if n < 12:
        return f
    U = _unit(W64 - W64.mean(0)); D = np.clip(1 - U @ U.T, 0, 2); np.fill_diagonal(D, 0)
    for k in (5, 10):
        A = np.zeros((n, n), bool); nn = np.argsort(D, axis=1)[:, 1:k + 1]
        A[np.repeat(np.arange(n), k), nn.ravel()] = True; A = A | A.T
        sp = _hops(A)
        inv = np.where(np.isfinite(sp) & (sp > 0), 1 / np.where(sp > 0, sp, 1), 0); f[f"knn{k}_geff"] = float(inv.sum() / (n * (n - 1)))
        deg = A.sum(1); tri = np.diag((A.astype(float) @ A @ A)) / 2
        cl = np.where(deg > 1, 2 * tri / np.maximum(deg * (deg - 1), 1), 0); f[f"knn{k}_clust"] = float(cl.mean())
        Dg = np.where(deg > 0, 1 / np.sqrt(np.maximum(deg, 1)), 0); Ln = np.eye(n) - Dg[:, None] * A * Dg[None]
        ev = np.sort(np.linalg.eigvalsh(Ln)); f[f"knn{k}_fiedler"] = float(ev[1])
    off = D[~np.eye(n, dtype=bool)]
    for q in (0.1, 0.2):
        R = D <= np.quantile(off, q); np.fill_diagonal(R, False); tot = R.sum()
        if tot == 0: continue
        diag_pts = 0
        for o in range(1, n):
            dline = np.diagonal(R, o).astype(int)
            runs = np.diff(np.r_[0, dline, 0]); s, e = np.where(runs == 1)[0], np.where(runs == -1)[0]
            diag_pts += sum(int(b - a) for a, b in zip(s, e) if b - a >= 2)
        f[f"rqa{int(q * 100)}_det"] = float(2 * diag_pts / tot)
        lam = 0
        for c in range(n):
            col = R[:, c].astype(int); runs = np.diff(np.r_[0, col, 0]); s, e = np.where(runs == 1)[0], np.where(runs == -1)[0]
            lam += sum(int(b - a) for a, b in zip(s, e) if b - a >= 2)
        f[f"rqa{int(q * 100)}_lam"] = float(lam / tot)
    h0, h1 = rips_persistence(D)
    h0 = h0[np.isfinite(h0[:, 1])]; h1 = h1[h1[:, 1] > h1[:, 0]]
    if len(h0):
        p0 = h0[:, 1] - h0[:, 0]; f["h0_total"] = float(p0.sum()); f["h0_max"] = float(p0.max())
        eps = h0[:, 1].max()
        if eps > 0:
            grid = np.linspace(0, eps, 200); x = grid / eps; beta = ((h0[:, :1] <= grid[None]) & (grid[None] < h0[:, 1:2])).sum(0).astype(float)
            c = (x * beta).sum() / beta.sum(); f["h0_spread"] = float(np.sqrt(((x - c) ** 2 * beta).sum() / beta.sum()))
    f["h1_count"] = float(len(h1)); f["h1_total"] = float((h1[:, 1] - h1[:, 0]).sum()) if len(h1) else 0.0
    return f


def depth_feats(Ws: list, prompt_last: np.ndarray, lo: int) -> dict:
    """Ws: per-layer [n_windows, d] arrays (same n); paths across layers lo..L-1 of unit states."""
    f = {}
    n = min(len(w) for w in Ws[lo:]) if Ws[lo:] else 0
    if n >= 4 and len(Ws) - lo >= 3:
        S = np.stack([_unit(w[:n]) for w in Ws[lo:]], 1)                       # [n, layers, d]
        d = np.diff(S, axis=1); ln = np.linalg.norm(d, axis=2); dn = d / (ln[..., None] + 1e-9)
        st = (dn[:, 1:] * dn[:, :-1]).sum(2).mean(1); L_ = ln.sum(1); q = max(1, n // 4)
        f.update(depth_straight=float(st.mean()), depth_len=float(L_.mean()), depth_straight_le=float(st[-q:].mean() - st[:q].mean()), depth_len_le=float(L_[-q:].mean() - L_[:q].mean()))
    P = _unit(prompt_last[lo:])
    if len(P) >= 3:
        d = np.diff(P, axis=0); ln = np.linalg.norm(d, axis=1); dn = d / (ln[:, None] + 1e-9)
        f.update(prompt_depth_straight=float((dn[1:] * dn[:-1]).sum(1).mean()), prompt_depth_len=float(ln.sum()))
    return f


@torch.no_grad()
def hidden_pass(model, prompt_ids, gen):
    ids = torch.tensor([prompt_ids + gen], device=model.device); n_p = len(prompt_ids)
    out = model(input_ids=ids, output_hidden_states=True, use_cache=False, logits_to_keep=1)
    hs = out.hidden_states[1:]                                                      # decoder layer outputs
    W16, W64, PL = [], [], []
    for h in hs:
        g = h[0, n_p:n_p + len(gen)]
        W16.append(_wmeans(g, 16)); W64.append(_wmeans(g, 64)); PL.append(h[0, n_p - 1].float().cpu().numpy())
    del out, hs
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return W16, W64, np.stack(PL)


@torch.no_grad()
def block_graphs(model, layers, prompt_ids, trace):
    """64-token-block graph per softmax-attention layer (all of them, not only the middle third)."""
    SI._install_attention_hook()
    att = [i for i, layer in enumerate(layers) if getattr(layer, "self_attn", None) is not None]
    SI._BG.update(on=True, layers=frozenset(att), res={}, n_p=len(prompt_ids), nq=-(-len(trace) // 64))
    try:
        SI.forward_last_logits(model, input_ids=torch.tensor([prompt_ids + trace], device=model.device), use_cache=False)
    finally:
        SI._BG["on"] = False
    res = {l: SI._BG["res"][l].detach().double().numpy() for l in sorted(SI._BG["res"])}; SI._BG["res"] = {}
    return res


@torch.no_grad()
def mlp_write(model, layers, ids):
    """||mlp_out|| / ||residual input|| at the last position, per layer."""
    rec, hooks = {}, []
    for i, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None: continue
        hooks.append(layer.register_forward_pre_hook(lambda m, a, i=i: rec.__setitem__(("in", i), a[0][0, -1].float().norm().item())))
        hooks.append(mlp.register_forward_hook(lambda m, a, o, i=i: rec.__setitem__(("mlp", i), (o[0] if isinstance(o, tuple) else o)[0, -1].float().norm().item())))
    try:
        SI.forward_last_logits(model, input_ids=torch.tensor([ids], device=model.device), use_cache=False)
    finally:
        for h in hooks: h.remove()
    return np.array([rec.get(("mlp", i), np.nan) / max(rec.get(("in", i), np.nan), 1e-9) for i in range(len(layers))])


@torch.no_grad()
def battery(model, tokenizer, layers, prompt_ids: list, gen: list, suffix_ids: list, log=print) -> tuple[dict, dict]:
    import time
    gen = gen[:6144]; S, A = {}, {}; t0 = time.time()
    L = len(layers); lo40 = int(math.floor(0.4 * L))
    W16, W64, PL = hidden_pass(model, prompt_ids, gen); t1 = time.time()
    per = {}
    for l in range(L):
        fl = {**traj_feats(W16[l], W64[l]), **graph_feats(W64[l])}
        for k, v in fl.items(): per.setdefault(k, np.full(L, np.nan))[l] = v
    for k, arr in per.items():
        A[f"layer_{k}"] = arr
        S[f"{k}__all"] = float(np.nanmean(arr)); S[f"{k}__ge40"] = float(np.nanmean(arr[lo40:]))
        for frac in (0.25, 0.5, 0.75):
            S[f"{k}__L{int(frac * 100)}"] = float(arr[min(L - 1, int(frac * L))])
    for tag, lo in (("all", 0), ("ge40", lo40)):
        for k, v in depth_feats(W64, PL, lo).items(): S[f"{k}__{tag}"] = v
    A["W64"] = np.stack([w[: min(len(x) for x in W64)] for w in W64]).astype(np.float16)
    t2 = time.time()
    lr, pm = SI.lookback_spans(model, layers, prompt_ids, gen)
    if lr is not None:
        q = max(1, lr.shape[0] // 4)
        A["lookback_le_head"] = (lr[-q:].mean(0) - lr[:q].mean(0)); A["pmass_le_head"] = (pm[-q:].mean(0) - pm[:q].mean(0))
        A["lookback_lastq_head"] = lr[-q:].mean(0); A["pmass_lastq_head"] = pm[-q:].mean(0)
        for nm in ("lookback_le", "pmass_le", "lookback_lastq", "pmass_lastq"):
            v = A[f"{nm}_head"].mean(1); S[f"{nm}__all"] = float(v.mean()); S[f"{nm}__ge40"] = float(v[int(0.4 * len(v)):].mean())
            for j, x in enumerate(v): S[f"{nm}__att{j}"] = float(x)
    G = block_graphs(model, layers, prompt_ids, gen)
    jfs = np.array([f_block_jump(g) for g in G.values()]) if G else np.zeros((0, 3))
    if len(jfs):
        A["block_jump_far_setup_layer"] = jfs
        for c, nm in enumerate(("jmp", "far", "setup")):
            S[f"{nm}__allatt"] = float(np.nanmean(jfs[:, c])); S[f"{nm}__ge40att"] = float(np.nanmean(jfs[int(0.4 * len(jfs)):, c]))
            for j, x in enumerate(jfs[:, c]): S[f"{nm}__att{j}"] = float(x)
    t3 = time.time()
    for cut, nm in ((min(2048, len(gen)), "toha2048"), (len(gen), "tohacut")):
        th = SI.toha_heads(model, layers, prompt_ids, gen, cut)
        if th is not None: A[f"{nm}_head"] = th; S[f"{nm}__mean"] = float(np.nanmean(th))
    lap = SI.lap_forced_topk(model, layers, prompt_ids, gen, suffix_ids, k=8)
    if lap is not None:
        A["lap_topk"] = lap
        S["lap_top1__all"] = float(lap[..., 0].mean()); S["lap_sum8__all"] = float(lap.sum(-1).mean())
        for j in range(lap.shape[0]): S[f"lap_top1__att{j}"] = float(lap[j, :, 0].mean()); S[f"lap_sum8__att{j}"] = float(lap[j].sum(-1).mean())
    mw = mlp_write(model, layers, prompt_ids + gen + suffix_ids); A["mlp_write_layer"] = mw
    S["mlp_write__all"] = float(np.nanmean(mw)); S["mlp_write__ge40"] = float(np.nanmean(mw[lo40:])); S["mlp_write__last25"] = float(np.nanmean(mw[int(0.75 * L):]))
    for l in range(L): S[f"mlp_write__l{l}"] = float(mw[l])
    t4 = time.time()
    log(f"  battery: hidden {t1 - t0:.1f}s, per-layer feats {t2 - t1:.1f}s, attention {t3 - t2:.1f}s, toha/lap/mlp {t4 - t3:.1f}s | {len(S)} scalars")
    return S, A
