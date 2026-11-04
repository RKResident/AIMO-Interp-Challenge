"""Features from the papers in papers_to_see/ (user, 2026-10-28), implemented as faithfully as a single 6144-token teacher-forced trace allows
(faithfulness reviewed against each paper's text; deviations are marked ADAPTATION). Shared by the submission and the A10G extraction.

PASS 1: prompt + trace + forced-answer suffix + forced answer tokens, with hooks:
  - rhythm (arXiv 2510.13554, Sec 4.1/5.1): 5 evenly spaced layers of the middle third of the softmax-attention layers; head span
    d = mean_t sum_s A[t,s](t-s) over response queries (Eq 7); local / global heads = bottom / top 30% of d pooled over those layers (Eq 8);
    WAAD_t = sum_s A_loc[t,s] min(t-s, W), W = 10 (Eq 9; W = 64 kept as a variant); FAI_s = mean_{t in [s+10, s+100]} A_glob[t,s] (Eq 10);
    anchors = top 40% FAI (Eq 16); preplan tokens = top 40% of |WAAD_t - WAAD_{t+1}|. Statistic: share of anchors with a preplan token at the
    same position or up to 3 tokens before (FAI follows or coincides with WAAD), minus its chance rate.
  - receiver heads (Thought Anchors arXiv 2506.19143, Sec 4/4.1): sentence x sentence mean attention for EVERY head of EVERY softmax layer;
    vertical score of sentence j = mean over query sentences >= j + 4 (sentences with < 2 such queries excluded); receiver heads = the 32
    heads with the most kurtotic vertical scores (ADAPTATION: kurtosis from this single trace, the paper averages over many traces).
  - hidden states: every layer at the forced-answer positions (logit lens, LID); trace states at the CIE layer and the recall-direction
    layers; the final-norm input at the prompt and trace positions (Min-K%++, Thought-Anchors KL).
PASS 2: empty-problem prompt + suffix + the same answer tokens (layer-wise usable information, arXiv 2412.10246 Alg 1:
  I_l = H_l(Q | empty) - H_l(Q | C), H = mean over the answer tokens of -log2 p_l, logit lens; LI = sum_l I_l).
PASS 3/4: attention suppression (Thought Anchors Sec 5.1): all heads, every query after the sentence is blocked from its tokens; effect =
  mean over later sentences of the mean log KL(clean || suppressed) of their tokens' next-token distributions. For the top receiver sentence
  of the first half and a length-matched random control (ADAPTATION: the paper's all-pairs matrix and its normalisation need one pass per
  sentence; the control difference stands in for the normalisation). The forced-answer log-prob change is kept as a labelled variant.
PROMPT-ONLY PASSES: LaRA (arXiv 2605.29888, Sec 3.2): mean-pooled problem-token states at every layer for the problem q0, its blanked
  version (every number -> [BLANK]) and the same for K = 3 automatic rewordings q1..q3 (ADAPTATION: the paper's neighbours are other,
  semantically similar questions; a single-problem call only has rewordings). Delta_i = h(q_i) - h(q_i^-); DC_l = cos(Delta_0, mean_i Delta_i);
  RSM_l = (|Delta_0| - mu) / sigma over the K neighbours.
Also: Min-K%++ (arXiv 2404.02936, Eq 3-4) on the problem tokens, k in {10, 20, 30, 50, 100}; LID (arXiv 2402.18048, Eq 3) of the
forced-answer-position state at a per-model layer against a SHIPPED reference set of training traces (T = min(50, n // 2) neighbours);
CIE-style (arXiv 2605.25603, Eq 15-16) entropic Gromov-Wasserstein between the external graph (next-sentence edge + 0.5 max(0, cos) of
sentence-mean states) and the internal graph (sentence attention), both forward-edged and row-normalised over outgoing edges
(ADAPTATION: the paper's internal graph comes from transcoder circuits + a trained GNN, unavailable; its feature term is dropped, alpha = 1);
recall-direction projections (exp78) of the trace states."""
from __future__ import annotations

import math
import re

import numpy as np
import torch

_P = {"on": False, "sup": None}
# the ensemble's one primary score per method (user, 2026-10-28: decisions from the new methods); every other value is logged only
PP_MAP = {"pp_rhythm": "rhythm_cooc_excess", "pp_anchor": "sup_logkl_top_minus_rnd", "pp_minkpp": "minkpp_20", "pp_lid": "lid_ref",
          "pp_li": "li_sum", "pp_lara": "lara_dc_mid", "pp_cie": "cie_gw", "pp_recall": "rec_comp__last_mean"}


def _prob_chunk(qg, k, scaling, window, qpos, kpos):
    sc = torch.einsum("grqd,gkd->grqk", qg, k).float() * scaling
    mask = kpos[None, :] > qpos[:, None]
    if window:
        mask = mask | (kpos[None, :] <= qpos[:, None] - window)
    sc.masked_fill_(mask, float("-inf"))
    sc.sub_(torch.logsumexp(sc, dim=-1, keepdim=True)).exp_()
    return sc


def sup_mask(module, query, key, attention_mask):
    """additive mask for the suppression passes (None when inactive): causal (+ sliding window) and later queries blocked from one span"""
    if _P["sup"] is None:
        return None
    q, T = query.shape[2], key.shape[2]; dev = query.device
    kpos = torch.arange(T, device=dev); qpos = T - q + torch.arange(q, device=dev)
    allowed = kpos[None, :] <= qpos[:, None]; win = getattr(module, "sliding_window", None)
    if win:
        allowed = allowed & (kpos[None, :] > qpos[:, None] - win)
    b0, b1 = _P["sup"]; allowed[(qpos >= b1)[:, None] & ((kpos >= b0) & (kpos < b1))[None, :]] = False
    m = torch.zeros(q, T, dtype=query.dtype, device=dev).masked_fill(~allowed, float("-inf"))[None, None]
    if attention_mask is not None and attention_mask.dtype != torch.bool:
        m = m + attention_mask[..., -T:].to(m.dtype)
    return m


def attn_hook(li, query, key, sc, win):
    """called from the submission's SDPA wrapper after every softmax-attention layer"""
    if not _P["on"] or li not in _P["layers"]:
        return
    q, kk = query[0], key[0]; H, T, D = q.shape; Hkv = kk.shape[0]; n_p, c, nS = _P["n_p"], _P["c"], _P["nS"]
    qg = q.reshape(Hkv, H // Hkv, T, D); dev = q.device; kpos = torch.arange(T, device=dev); ksid = _P["ksid"][:T]; qsid = _P["qsid"]
    rh = li in _P["rhythm"]
    S = torch.zeros(H, nS + 1, nS + 1, device=dev)
    if rh:
        dist = torch.zeros(H, device=dev); w10 = torch.zeros(H, c, device=dev); w64 = torch.zeros(H, c, device=dev)
        fnum = torch.zeros(H, c, device=dev); fcnt = torch.zeros(c, device=dev)
    for s0 in range(n_p, n_p + c, 256):
        e0 = min(n_p + c, s0 + 256); qpos = torch.arange(s0, e0, device=dev)
        p = _prob_chunk(qg[:, :, s0:e0], kk, sc, win, qpos, kpos).reshape(H, e0 - s0, T)
        S.index_add_(1, qsid[s0 - n_p:e0 - n_p], torch.zeros(H, e0 - s0, nS + 1, device=dev).index_add_(2, ksid, p))
        if rh:
            d = (qpos[:, None] - kpos[None, :]).clamp(min=0).float()
            dist += (p * d).sum((1, 2)); w10[:, s0 - n_p:e0 - n_p] = (p * d.clamp(max=10)).sum(-1); w64[:, s0 - n_p:e0 - n_p] = (p * d.clamp(max=64)).sum(-1)
            band = ((d >= 10) & (d <= 100)).float()[:, n_p:n_p + c]
            fnum += (p[:, :, n_p:n_p + c] * band).sum(1); fcnt += band.sum(0); del d, band
        del p
    r = dict(S=S.cpu())
    if rh:
        r.update(dist=(dist / c).cpu(), w10=w10.cpu(), w64=w64.cpu(), fai=(fnum / fcnt.clamp(min=1)).cpu(), fcnt=fcnt.cpu())
    _P["res"][li] = r


def sentences(tok, ids, min_len=4, max_len=120):
    """sentence id (1..nS) per trace token: a boundary after a token containing a newline or ending in . ? ! (ADAPTATION: the paper's
    exact chunking rule is not specified; merged up to min_len tokens, split at max_len)"""
    sid, cur, n = [], 1, 0
    for t in ids:
        s = tok.decode([t]); sid.append(cur); n += 1
        if (("\n" in s) or re.search(r"[.?!]\s*$", s)) and n >= min_len or n >= max_len:
            cur += 1; n = 0
    return np.array(sid, np.int64), cur - (1 if n == 0 else 0)


def _lens(head, norm_m, H):
    return torch.log_softmax(head(norm_m(H.to(head.weight.dtype))).float(), -1)


def _kurt(x):
    x = x - x.mean(-1, keepdim=True); return (x ** 4).mean(-1) / (x ** 2).mean(-1).clamp(min=1e-12) ** 2


def lid_mle(x, R, T):
    """Levina-Bickel MLE (LID paper Eq 3) of x against the reference rows R (Euclidean), T neighbours"""
    d = torch.cdist(x[None].float(), R.float())[0]; Q = d.topk(T, largest=False).values.clamp(min=1e-6)
    return float(1.0 / torch.log(Q[-1] / Q[:-1]).mean().clamp(min=1e-6))


def entropic_gw(C1, C2, eps_rel=0.05, outer=20, inner=60):
    """square-loss entropic Gromov-Wasserstein between two (possibly asymmetric) graphs with uniform node measures; returns (GW value at
    the optimised coupling, GW value at the identity coupling) - the nodes are the same sentences, so the gap measures how far the best
    matching moves away from sentence-to-itself"""
    n = C1.shape[0]; p = torch.full((n,), 1.0 / n, dtype=torch.float64); C1 = C1.double(); C2 = C2.double()
    const = (C1 ** 2 @ p)[:, None] + (C2 ** 2 @ p)[None, :]
    loss = lambda T: float(((const - 2 * C1 @ T @ C2.T) * T).sum())
    T = p[:, None] * p[None, :]
    for _ in range(outer):
        L = const - 2 * C1 @ T @ C2.T; eps = max(1e-12, eps_rel * float(L.abs().mean()))
        K = torch.exp(-(L - L.min()) / eps); u = torch.ones(n, dtype=torch.float64)
        for _ in range(inner):
            v = p / (K.T @ u).clamp(min=1e-300); u = p / (K @ v).clamp(min=1e-300)
        T = u[:, None] * K * v[None, :]
    return loss(T), loss(torch.eye(n, dtype=torch.float64) / n)


def blank_numbers(text):
    return re.sub(r"\d+(?:\.\d+)?", "[BLANK]", text)


def _problem_states(model, layers, build_prompt, tok, text):
    """mean-pooled hidden states over the problem tokens, every layer [L, d] (LaRA's h_l)"""
    ids, _, span = build_prompt(tok, text)
    a, b = span if span else (0, len(ids)); out = {}
    hk = [l.register_forward_hook(lambda m, i_, o, i=i: out.__setitem__(i, (o[0] if isinstance(o, tuple) else o)[0, a:b].float().mean(0)))
          for i, l in enumerate(layers)]
    try:
        with torch.no_grad():
            model(input_ids=torch.tensor([ids], device=model.device), use_cache=False, logits_to_keep=1)
    finally:
        for h in hk: h.remove()
    return torch.stack([out[i] for i in range(len(layers))])


@torch.no_grad()
def paper_features(model, tok, layers, norm_m, build_prompt, auto_variants, prompt_ids, gen, cut, problem, sfx, lid_ref=None, recall=None,
                   want_states=False):
    """all paper features for one trace (cut = end of thinking or 6144). lid_ref = dict(layer, R [n, d] tensor, exclude [n] bool or None);
    recall = dict(layers [n_dirs], U [n_dirs, d], names). Returns (features dict, final-position states [L, d] if want_states)."""
    dev = model.device; head = model.get_output_embeddings(); L = len(layers); n_p = len(prompt_ids); trace = list(gen[:cut]); c = len(trace)
    base = prompt_ids + trace + sfx
    # forced answer tokens: greedy continuation of the suffix (until '}' or 12 tokens)
    g = model.generate(input_ids=torch.tensor([base], device=dev), max_new_tokens=12, do_sample=False, temperature=None, top_p=None, top_k=None)
    ans = g[0, len(base):].tolist(); txt = ""
    for k_, t in enumerate(ans):
        txt += tok.decode([t])
        if "}" in txt: ans = ans[:k_ + 1]; break
    ans = ans or g[0, len(base):len(base) + 1].tolist()
    ids = base + ans[:-1]; A_ = len(ans)                                             # positions len(base)-1 .. end predict the answer tokens
    att = [i for i, l in enumerate(layers) if getattr(l, "self_attn", None) is not None]
    mid = att[len(att) // 3:max(len(att) // 3 + 1, 2 * len(att) // 3)]
    rhythm = sorted({mid[int(round(x))] for x in np.linspace(0, len(mid) - 1, min(5, len(mid)))})
    sid, nS = sentences(tok, trace)
    cie_l = L // 2; rec_layers = set(recall["layers"]) if recall else set()
    _P.update(on=True, res={}, n_p=n_p, c=c, nS=nS, layers=frozenset(att), rhythm=frozenset(rhythm), qsid=torch.tensor(sid, device=dev),
              ksid=torch.tensor(np.r_[np.zeros(n_p, np.int64), sid, np.zeros(len(ids) - n_p - c, np.int64)], device=dev))
    ANS, TR, FIN, hk = {}, {}, {}, []
    for i, layer in enumerate(layers):
        def fn(m, a, o, i=i):
            h = (o[0] if isinstance(o, tuple) else o)[0]
            ANS[i] = h[len(base) - 1:].detach()
            if i == cie_l or i in rec_layers: TR[i] = h[n_p:n_p + c].detach()
        hk.append(layer.register_forward_hook(fn))
    hk.append(norm_m.register_forward_pre_hook(lambda m, a: FIN.__setitem__("h", a[0][0, :n_p + c].detach())))
    try:
        model(input_ids=torch.tensor([ids], device=dev), use_cache=False, logits_to_keep=1)
    finally:
        _P["on"] = False
        for h in hk: h.remove()
    f = dict(n_sent=int(nS), n_answer_tokens=A_)
    q = torch.tensor(ans, device=dev)
    # ---- layer-wise usable information (Alg 1): with context vs empty problem, mean over the answer tokens, every layer
    LC = torch.stack([_lens(head, norm_m, ANS[i])[torch.arange(A_), q] for i in range(L)])          # [L, A] log p_l(q_t | C)
    e_ids = build_prompt(tok, "")[0] + sfx + ans[:-1]; E = {}; hk = []
    for i, layer in enumerate(layers):
        hk.append(layer.register_forward_hook(lambda m, a, o, i=i: E.__setitem__(i, (o[0] if isinstance(o, tuple) else o)[0, -A_:].detach())))
    try:
        model(input_ids=torch.tensor([e_ids], device=dev), use_cache=False, logits_to_keep=1)
    finally:
        for h in hk: h.remove()
    L0 = torch.stack([_lens(head, norm_m, E[i])[torch.arange(A_), q] for i in range(L)])
    I = ((LC - L0).mean(1) / math.log(2)).cpu().numpy(); I1 = ((LC[:, 0] - L0[:, 0]) / math.log(2)).cpu().numpy()
    f.update(li_sum=float(I.sum()), li_mid=float(I[L // 3:2 * L // 3].mean()), li_last=float(I[-1]), li1_sum=float(I1.sum()),
             li_drop_ext=float(np.maximum(0, I[:-1] - I[1:]).sum()))                  # li_drop_ext: our extension, not a paper quantity
    lp_ans = torch.log_softmax(head(norm_m(ANS[L - 1][0:1].to(head.weight.dtype))).float(), -1)[0]
    # ---- Min-K%++ on the problem tokens (Eq 3-4)
    span = build_prompt(tok, problem)[2]
    lpP = _lens(head, norm_m, FIN["h"][:n_p]); x = torch.tensor(prompt_ids, device=dev)
    mu = (lpP.exp() * lpP).sum(-1); sd = ((lpP.exp() * (lpP - mu[:, None]) ** 2).sum(-1)).clamp(min=1e-12).sqrt()
    tl = lpP[:-1].gather(1, x[1:, None])[:, 0]; z = ((tl - mu[:-1]) / sd[:-1]).cpu().numpy(); raw = tl.cpu().numpy()
    a0, a1 = (span[0] - 1, span[1] - 1) if span else (0, n_p - 1); zz, rr = np.sort(z[a0:a1]), np.sort(raw[a0:a1])
    for k in (10, 20, 30, 50, 100):
        m_ = max(1, int(np.ceil(len(zz) * k / 100))); f[f"minkpp_{k}"] = float(zz[:m_].mean()); f[f"mink_{k}"] = float(rr[:m_].mean())
    f["minkpp_span_found"] = bool(span)
    # ---- rhythm (Eq 7-10, 16)
    R = _P["res"]; RL = [l for l in rhythm if l in R and "dist" in R[l]]
    if RL:
        dist = torch.cat([R[l]["dist"] for l in RL]); lo, hi = dist.quantile(.3), dist.quantile(.7)
        ok = R[RL[0]]["fcnt"].numpy() > 0
        for W in ("w10", "w64"):
            w_all = torch.cat([R[l][W] for l in RL]); fai_all = torch.cat([R[l]["fai"] for l in RL])
            waad = w_all[dist <= lo].mean(0).numpy(); fai = fai_all[dist >= hi].mean(0).numpy()
            if W == "w10":
                anchors = fai >= np.quantile(fai[ok], .6); dlt = np.abs(np.diff(waad, append=waad[-1])); pre = dlt >= np.quantile(dlt, .6)
                near = np.zeros_like(pre)
                for g_ in range(0, 4): near[g_:] |= pre[:len(pre) - g_]                  # a preplan token at t, t-1, t-2 or t-3
                rate = float(near[anchors & ok].mean()) if (anchors & ok).any() else np.nan; chance = float(near[ok].mean())
                f.update(rhythm_cooc=rate, rhythm_cooc_excess=rate - chance, anchor_fai_top40=float(fai[anchors & ok].mean()), fai_mean=float(fai[ok].mean()),
                         waad10_mean=float(waad.mean()), preplan_delta_mean=float(dlt.mean()))
                cc = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if len(a) > 10 and a.std() > 0 and b.std() > 0 else np.nan
                f["rhythm_corr_lag_max"] = float(np.nanmax([cc(waad[ok][:len(waad[ok]) - g_], fai[ok][g_:]) for g_ in range(0, 9)]))   # FAI after WAAD
            else:
                f["waad64_mean"] = float(waad.mean())
    # ---- receiver heads (all softmax layers) and suppression
    S = torch.cat([R[l]["S"] for l in sorted(R)])                                   # [heads, nS+1 (query sentence), nS+1 (key sentence)]
    size = torch.tensor(np.bincount(sid, minlength=nS + 1), dtype=torch.float).clamp(min=1)
    Ad = S / size[None, :, None] / size[None, None, :]
    j = torch.arange(nS + 1); later = (j[:, None] >= j[None, :] + 4).float(); later[:, 0] = 0; later[0, :] = 0
    nq = later.sum(0); valid = (nq >= 2); valid[0] = False
    inc = (Ad * later).sum(1) / nq.clamp(min=1)
    vs = torch.where(valid)[0]
    if len(vs) >= 4:
        rec = _kurt(inc[:, vs]).topk(min(32, inc.shape[0])).indices; score = inc[rec].mean(0)
        sv = score[vs].numpy(); srt = np.sort(sv)[::-1]
        f.update(anchor_conc3=float(srt[:3].sum() / max(1e-12, srt.sum())), anchor_kurt=float(_kurt(torch.tensor(sv)[None])[0]))
        half = [int(s) for s in vs.tolist() if s <= nS // 2 and (sid == s).sum() >= 4] or [int(s) for s in vs.tolist()]
        s_top = max(half, key=lambda s: float(score[s])); ln = (sid == s_top).sum()
        cand = [s for s in half if s != s_top and abs((sid == s).sum() - ln) <= max(3, ln // 2)] or [s for s in half if s != s_top] or [s_top]
        s_rnd = int(np.random.default_rng(int(np.sum(trace[:50])) % 2**31).choice(cand))
        fin_clean = FIN["h"][n_p:]
        for nm, s in (("top", s_top), ("rnd", s_rnd)):
            pos = np.where(sid == s)[0]; b0, b1 = n_p + int(pos[0]), n_p + int(pos[-1]) + 1; FS = {}
            hh = norm_m.register_forward_pre_hook(lambda m, a: FS.__setitem__("h", a[0][0, n_p:n_p + c].detach())); _P["sup"] = (b0, b1)
            try:
                o = model(input_ids=torch.tensor([ids], device=dev), use_cache=False, logits_to_keep=A_)
            finally:
                _P["sup"] = None; hh.remove()
            kls = []
            for s0 in range(b1 - n_p, c, 512):                                          # positions after the sentence, chunked over the vocab
                lc = _lens(head, norm_m, fin_clean[s0:s0 + 512]); ls_ = _lens(head, norm_m, FS["h"][s0:s0 + 512])
                kls.append((lc.exp() * (lc - ls_)).sum(-1).clamp(min=1e-10).log().cpu()); del lc, ls_
            kl = torch.cat(kls).numpy() if kls else np.array([np.nan]); sj = sid[b1 - n_p:]
            per_sent = [kl[sj == t].mean() for t in np.unique(sj)] if len(sj) else [np.nan]
            f[f"sup_logkl_{nm}"] = float(np.nanmean(per_sent))
            f[f"sup_ans_{nm}"] = float(lp_ans[q[0]] - torch.log_softmax(o.logits[0, 0].float(), -1)[q[0]])   # forced-answer variant
        f["sup_logkl_top_minus_rnd"] = f["sup_logkl_top"] - f["sup_logkl_rnd"]
        # ---- CIE-style entropic GW: external (next-sentence + 0.5 max(0, cos)) vs internal (sentence attention), forward edges, outgoing norm
        X = torch.zeros(nS + 1, TR[cie_l].shape[1], device=dev).index_add_(0, torch.tensor(sid, device=dev), TR[cie_l].float())[1:]
        X = X / size[1:, None].to(dev); cs = torch.nn.functional.normalize(X, dim=-1); C = (cs @ cs.T).clamp(min=0).cpu()
        n = nS; i_ = torch.arange(n); up = (i_[None, :] > i_[:, None]).float()      # edge i -> j for j > i
        Aext = ((i_[None, :] == i_[:, None] + 1).float() + 0.5 * C) * up
        Aint = Ad.mean(0)[1:, 1:].T * up                                              # attention that later sentence j pays to i
        Aext = Aext / Aext.sum(1, keepdim=True).clamp(min=1e-12); Aint = Aint / Aint.sum(1, keepdim=True).clamp(min=1e-12)
        if n >= 6:
            m_ = min(n, 300); gw, gw_id = entropic_gw(Aext[:m_, :m_], Aint[:m_, :m_])
            f.update(cie_gw=gw, cie_gw_identity=gw_id, cie_gw_gap=gw_id - gw)
    # ---- LID against the shipped reference set (final forced-answer position)
    if lid_ref is not None:
        Rr = lid_ref["R"].to(dev); ex = lid_ref.get("exclude")
        if ex is not None: Rr = Rr[~torch.as_tensor(ex, device=dev)]
        T_ = max(3, min(50, len(Rr) // 2)); f["lid_ref"] = lid_mle(ANS[lid_ref["layer"]][0], Rr, T_); f["lid_T"] = T_
    # ---- LaRA: DC / RSM from blanking shifts of the problem and its rewordings
    try:
        vars_ = [v for v in auto_variants(problem)][:3]
        hs = [_problem_states(model, layers, build_prompt, tok, t) for t in [problem] + vars_]
        hb = [_problem_states(model, layers, build_prompt, tok, blank_numbers(t)) for t in [problem] + vars_]
        D = [a - b for a, b in zip(hs, hb)]                                            # [L, d] each
        if len(D) >= 2:
            sbar = torch.stack(D[1:]).mean(0); dc = torch.nn.functional.cosine_similarity(D[0], sbar, dim=-1).cpu().numpy()
            nrm = torch.stack([d.norm(dim=-1) for d in D]).cpu().numpy(); rsm = (nrm[0] - nrm[1:].mean(0)) / (nrm[1:].std(0) + 1e-6)
            f.update(lara_dc_mid=float(dc[L // 3:2 * L // 3].mean()), lara_dc_all=float(dc.mean()), lara_rsm_mid=float(rsm[L // 3:2 * L // 3].mean()),
                     lara_n_neighbours=len(D) - 1)
    except Exception as e:                                                             # noqa: BLE001 - logged, other features still count
        f["lara_error"] = repr(e)[:80]
    # ---- recall-direction projections (exp78)
    if recall:
        U = recall["U"].to(dev)
        for k, (nm, l) in enumerate(zip(recall["names"], recall["layers"])):
            pr = (TR[l].float() @ U[k].float()).cpu().numpy(); qn = max(1, c // 4)
            f[f"rec_{nm}_mean"] = float(pr.mean()); f[f"rec_{nm}_lastq"] = float(pr[-qn:].mean())
    states = torch.stack([ANS[i][0] for i in range(L)]).half().cpu() if want_states else None
    _P["res"] = {}
    return f, states
