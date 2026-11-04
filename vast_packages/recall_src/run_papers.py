"""Features from the papers in papers_to_see/ (user, 2026-10-28), on our labelled traces (graph_inputs.jsonl: 118 greedy traces) and the
set100 screening traces (traces_set100.jsonl). One model per process (run_papers.sh). Per trace (cut at end of thinking / 6144 tokens):
PASS 1 - one teacher-forced pass over prompt + trace + forced-answer suffix, with hooks:
  attention (middle third of the softmax-attention layers, recomputed in 256-query chunks, trace queries only):
    * WAAD / FAI rhythm (arXiv 2510.13554): heads split by mean attention distance into local (bottom third) and global (top third);
      WAAD_t = sum_s A_loc[t,s] min(t-s, 64); FAI_s = mean over t in [s+16, s+256] of A_glob[t,s]. Features: corr(WAAD, FAI) at lag 0 and the
      max over lags -8..8, mean FAI of the top-5% anchor tokens, mean WAAD, WAAD spike rate.
    * sentence-to-sentence attention (Thought Anchors, arXiv 2506.19143): receiver heads = the 16 heads with the most kurtotic incoming
      sentence attention; incoming score per sentence; concentration of the top 3.
  hidden states: logit lens at the forced-answer position, every layer (information deficiency, arXiv 2412.10246); trace states at layers
    L/3, L/2, 0.7L, 2L/3 (LID arXiv 2402.18048, directional collapse LaRA arXiv 2605.29888, sentence features for CIE arXiv 2605.25603);
    final-norm input at the prompt positions (Min-K%++ arXiv 2404.02936).
PASS 2 - empty-context pass (template with an empty problem + suffix): per-layer lens log p of the clean answer token -> usable information
  I_l = log2 p_l(q | C) - log2 p_l(q | empty); LI = sum I_l, Def = sum max(0, I_{l-1} - I_l); entropy deficiency sum max(0, H_l - H_{l-1}).
PASS 3/4 - attention suppression (Thought Anchors): every query after the top receiver sentence (first half of the trace) is blocked from its
  tokens; delta = log p_clean(answer) - log p_suppressed(answer). Control: the same for a random first-half sentence of similar length.
CIE-style discrepancy: external graph = next-sentence edge + max(0, cos) of sentence mean states (L/2); internal graph = mean sentence
  attention; both row-normalised over earlier sentences; the nodes are the SAME sentences, so the graphs are compared under the identity
  coupling (mean total-variation distance per row, and the Spearman correlation of the edges) - an approximation of the paper's FGW.
Output <out>/<model>/papers.jsonl (resumable by id)."""
from __future__ import annotations

import argparse, json, re, sys, time, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "submission_core"))
import numpy as np
import torch

import spurious_inference as SI
from spurious_features import SUFFIXES_NOTHINK, SUFFIXES_THINK, build_prompt, find_think_end, think_end_token_id
from run_t1 import gpu, log, final_norm

_A = {"on": False, "sup": None, "orig": None}


def _attn_layer(li, query, key, sc, win):
    q, kk = query[0], key[0]; H, T, D = q.shape; Hkv = kk.shape[0]; n_p, c, nS = _A["n_p"], _A["c"], _A["nS"]
    qg = q.reshape(Hkv, H // Hkv, T, D); kpos = torch.arange(T, device=q.device); ksid = _A["ksid"][:T]; qsid = _A["qsid"]
    dist = torch.zeros(H, device=q.device); waad = torch.zeros(H, c, device=q.device); fnum = torch.zeros(H, c, device=q.device)
    fcnt = torch.zeros(c, device=q.device); S = torch.zeros(H, nS + 1, nS + 1, device=q.device)
    for s0 in range(n_p, n_p + c, 256):
        e0 = min(n_p + c, s0 + 256); qpos = torch.arange(s0, e0, device=q.device)
        p, _ = SI._lb_prob_chunk(qg[:, :, s0:e0], kk, sc, win, qpos, kpos); p = p.reshape(H, e0 - s0, T)
        d = (qpos[:, None] - kpos[None, :]).clamp(min=0).float()
        dist += (p * d).sum((1, 2)); waad[:, s0 - n_p:e0 - n_p] = (p * d.clamp(max=64)).sum(-1)
        band = ((d >= 16) & (d <= 256)).float()[:, n_p:n_p + c]
        fnum += (p[:, :, n_p:n_p + c] * band).sum(1); fcnt += band.sum(0)
        ps = torch.zeros(H, e0 - s0, nS + 1, device=q.device).index_add_(2, ksid, p)
        S.index_add_(1, qsid[s0 - n_p:e0 - n_p], ps); del p, ps, d, band
    _A["res"][li] = dict(dist=(dist / c).cpu(), waad=waad.half().cpu(), fai=(fnum / fcnt.clamp(min=1)).half().cpu(), fcnt=fcnt.cpu(), S=S.cpu())


def _sdpa(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kw):
    li = getattr(module, "layer_idx", None)
    if _A["sup"] is not None:                                                  # suppression pass: block later queries from one sentence
        q, T = query.shape[2], key.shape[2]; kpos = torch.arange(T, device=query.device); qpos = T - q + torch.arange(q, device=query.device)
        allowed = kpos[None, :] <= qpos[:, None]; win = getattr(module, "sliding_window", None)
        if win: allowed = allowed & (kpos[None, :] > qpos[:, None] - win)
        b0, b1 = _A["sup"]; allowed[(qpos >= b1)[:, None] & ((kpos >= b0) & (kpos < b1))[None, :]] = False
        m = torch.zeros(q, T, dtype=query.dtype, device=query.device).masked_fill(~allowed, float("-inf"))[None, None]
        return _A["orig"](module, query, key, value, m, dropout=dropout, scaling=scaling, **{k: v for k, v in kw.items() if k != "is_causal"})
    out = _A["orig"](module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kw)
    if _A["on"] and li in _A["layers"]:
        _attn_layer(li, query, key, scaling if scaling is not None else query.shape[-1] ** -0.5, getattr(module, "sliding_window", None))
    return out


def install():
    if _A["orig"] is None:
        from transformers import AttentionInterface
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        _A["orig"] = sdpa_attention_forward; AttentionInterface.register("sdpa", _sdpa)


def sentences(tok, ids, min_len=4, max_len=120):
    """sentence id per trace token: a boundary after a token containing a newline or ending in . ? ! (merged up to min_len, split at max_len)"""
    sid, cur, n = [], 1, 0
    for t in ids:
        s = tok.decode([t]); sid.append(cur); n += 1
        if (("\n" in s) or re.search(r"[.?!]\s*$", s)) and n >= min_len or n >= max_len: cur += 1; n = 0
    return np.array(sid), cur - (1 if n == 0 else 0)


def lens(model, ctx, H):
    head = model.get_output_embeddings(); h = ctx["norm"](H.to(head.weight.dtype)); return torch.log_softmax(head(h).float(), -1)


def kurt(x):
    x = x - x.mean(-1, keepdim=True); return (x ** 4).mean(-1) / (x ** 2).mean(-1).clamp(min=1e-12) ** 2


def lid(X, k=20, gap=0):
    X = X.float(); D = torch.cdist(X, X)
    if gap:
        i = torch.arange(len(X), device=X.device); D[(i[:, None] - i[None, :]).abs() < gap] = float("inf")
    else:
        D.fill_diagonal_(float("inf"))
    Q = D.topk(k, largest=False).values.clamp(min=1e-6)                       # ascending distances to the k nearest
    l = 1.0 / (torch.log(Q[:, -1:] / Q[:, :-1]).mean(-1).clamp(min=1e-6))
    return float(l.mean()), float(l.quantile(.9))


def dc(X):
    X = X.float(); X = X - X.mean(0, keepdim=True); s = torch.linalg.svdvals(X); s = s / s.sum().clamp(min=1e-12)
    return float(1 - (-(s * torch.log(s.clamp(min=1e-12))).sum()) / np.log(min(X.shape)))


@torch.no_grad()
def one(model, tok, ctx, r):
    dev = model.device; sfx = ctx["sfx"]; prompt = r["prompt_ids"]; gen = r["greedy_gen_ids"][:6144]
    te = find_think_end(tok, gen, ctx["tid"]); c = min(max(1, te if te is not None else len(gen)), 6144); trace = gen[:c]; n_p = len(prompt)
    ids = prompt + trace + sfx; L = len(ctx["layers"])
    sid, nS = sentences(tok, trace)
    _A.update(on=True, res={}, n_p=n_p, c=c, nS=nS, qsid=torch.tensor(sid, device=dev),
              ksid=torch.tensor(np.r_[np.zeros(n_p, np.int64), sid, np.zeros(len(sfx), np.int64)], device=dev), layers=ctx["attn_mid"])
    CAP, LAST, FIN, hk = {}, {}, {}, []
    for i, layer in enumerate(ctx["layers"]):
        def fn(m, a, o, i=i):
            h = o[0] if isinstance(o, tuple) else o
            LAST[i] = h[0, -1].detach()
            if i in ctx["cap"]: CAP[i] = h[0, n_p:n_p + c].detach().half()
        hk.append(layer.register_forward_hook(fn))
    hk.append(ctx["norm"].register_forward_pre_hook(lambda m, a: FIN.__setitem__("p", a[0][0, :n_p].detach())))
    try:
        with torch.no_grad(): lg = SI.forward_last_logits(model, input_ids=torch.tensor([ids], device=dev), use_cache=False).logits[0, -1]
    finally:
        _A["on"] = False
        for h in hk: h.remove()
    lp = torch.log_softmax(lg.float().reshape(-1), -1); q = int(lp.argmax()); f = dict(p1=float(lp[q].exp()), cut=c, think_end=te, n_sent=int(nS))
    # ---- information deficiency (logit lens at the answer position, with vs without context)
    LC = lens(model, ctx, torch.stack([LAST[i] for i in range(L)]))
    e_ids = build_prompt(tok, "")[0] + sfx; LAST0 = {}; hk = []
    for i, layer in enumerate(ctx["layers"]):
        hk.append(layer.register_forward_hook(lambda m, a, o, i=i: LAST0.__setitem__(i, (o[0] if isinstance(o, tuple) else o)[0, -1].detach())))
    try:
        with torch.no_grad(): model(input_ids=torch.tensor([e_ids], device=dev), use_cache=False, logits_to_keep=1)
    finally:
        for h in hk: h.remove()
    L0 = lens(model, ctx, torch.stack([LAST0[i] for i in range(L)]))
    I = ((LC[:, q] - L0[:, q]) / np.log(2)).cpu().numpy(); Hl = (-(LC.exp() * LC).sum(-1)).cpu().numpy()
    f.update(li_sum=float(I.sum()), li_def=float(np.maximum(0, I[:-1] - I[1:]).sum()), li_last=float(I[-1]), li_mid=float(I[L // 3:2 * L // 3].mean()),
             ent_def=float(np.maximum(0, Hl[1:] - Hl[:-1]).sum()), ent_mid=float(Hl[L // 3:2 * L // 3].mean()))
    # ---- Min-K%++ on the problem tokens (vocabulary-normalised token scores, bottom k%)
    span = build_prompt(tok, r.get("problem") or "")[2] if r.get("problem") else None
    lpP = lens(model, ctx, FIN["p"]); x = torch.tensor(prompt, device=dev)
    mu = (lpP.exp() * lpP).sum(-1); sd = ((lpP.exp() * (lpP - mu[:, None]) ** 2).sum(-1)).clamp(min=1e-12).sqrt()
    tl = lpP[:-1].gather(1, x[1:, None])[:, 0]; z = ((tl - mu[:-1]) / sd[:-1]).cpu().numpy(); raw = tl.cpu().numpy()
    a0, a1 = (span[0] - 1, span[1] - 1) if span else (max(0, n_p // 4), n_p - 6)             # positions predicting problem tokens
    zz, rr = np.sort(z[a0:a1]), np.sort(raw[a0:a1])
    for k in (10, 20):
        m_ = max(1, int(np.ceil(len(zz) * k / 100))); f[f"minkpp_{k}"] = float(zz[:m_].mean()); f[f"mink_{k}"] = float(rr[:m_].mean())
    # ---- WAAD / FAI rhythm and sentence attention
    R = _A["res"]; ls = sorted(R)
    dist = torch.cat([R[l]["dist"] for l in ls]); lo, hi = dist.quantile(1 / 3), dist.quantile(2 / 3)
    WA = torch.cat([R[l]["waad"].float() for l in ls]); FA = torch.cat([R[l]["fai"].float() for l in ls])
    waad = WA[dist <= lo].mean(0).numpy(); fai = FA[dist >= hi].mean(0).numpy(); ok = R[ls[0]]["fcnt"].numpy() > 0
    w_, f_ = waad[ok], fai[ok]
    cc = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if len(a) > 10 and a.std() > 0 and b.std() > 0 else np.nan
    lagc = [cc(w_[:len(w_) - g], f_[g:]) if g >= 0 else cc(w_[-g:], f_[:len(f_) + g]) for g in range(-8, 9)]   # FAI g tokens after WAAD
    top = np.sort(f_)[-max(1, len(f_) // 20):]
    f.update(rhythm_corr0=cc(w_, f_), rhythm_corrmax=float(np.nanmax(np.abs(lagc))), anchor_fai=float(top.mean()), fai_mean=float(f_.mean()),
             waad_mean=float(w_.mean()), waad_spike=float((w_ > w_.mean() + 2 * w_.std()).mean()))
    S = torch.cat([R[l]["S"] for l in ls])                                           # [heads, nS+1 (query sentence), nS+1 (key sentence)]
    size = torch.tensor(np.bincount(sid, minlength=nS + 1), dtype=torch.float).clamp(min=1)
    A = S / size[None, :, None]                                                     # mean attention mass from a query token of sentence i
    Ad = A / size[None, None, :]                                                    # per key token (paper's A_sent)
    j = torch.arange(nS + 1); later = (j[:, None] >= j[None, :] + 4).float(); later[:, 0] = 0
    inc = (Ad * later).sum(1) / later.sum(0).clamp(min=1)                            # [heads, sentences]: incoming from >= 4 sentences later
    rec = kurt(inc[:, 1:]).topk(min(16, inc.shape[0])).indices; sc_ = inc[rec].mean(0)[1:].numpy()     # sentences 1..nS
    srt = np.sort(sc_)[::-1]; f.update(anchor_conc3=float(srt[:3].sum() / max(1e-12, srt.sum())), anchor_kurt=float(kurt(torch.tensor(sc_)[None])[0]))
    # ---- suppression of the top first-half receiver sentence (and a random control)
    half = [s for s in range(1, nS // 2 + 1) if (sid == s).sum() >= 4] or list(range(1, max(2, nS // 2 + 1)))
    s_top = max(half, key=lambda s: sc_[s - 1]); rng = np.random.default_rng(int(np.sum(trace[:50])) % 2**31)
    ln = (sid == s_top).sum(); cand = [s for s in half if s != s_top and abs((sid == s).sum() - ln) <= max(3, ln // 2)] or [s for s in half if s != s_top] or [s_top]
    s_rnd = int(rng.choice(cand))
    for nm, s in (("sup_top", s_top), ("sup_rnd", s_rnd)):
        pos = np.where(sid == s)[0]; _A["sup"] = (n_p + int(pos[0]), n_p + int(pos[-1]) + 1)
        try:
            with torch.no_grad(): lg2 = SI.forward_last_logits(model, input_ids=torch.tensor([ids], device=dev), use_cache=False).logits[0, -1]
        finally:
            _A["sup"] = None
        f[nm] = float(lp[q] - torch.log_softmax(lg2.float().reshape(-1), -1)[q])
    f["sup_top_minus_rnd"] = f["sup_top"] - f["sup_rnd"]
    # ---- LID, directional collapse, CIE-style graph discrepancy
    l3, l2, l7, l23 = ctx["cap_named"]
    X7 = CAP[l7][::2]; f["lid_mean"], f["lid_p90"] = lid(X7); f["lid_nt_mean"], f["lid_nt_p90"] = lid(X7, gap=16)
    for nm, l in (("dc_l3", l3), ("dc_l2", l2), ("dc_l23", l23)): f[nm] = dc(CAP[l][::2])
    Xs = torch.zeros(nS + 1, CAP[l2].shape[1], device=dev).index_add_(0, torch.tensor(sid, device=dev), CAP[l2].float())
    Xs = Xs[1:] / torch.tensor(np.bincount(sid, minlength=nS + 1)[1:], device=dev).clamp(min=1)[:, None].float()
    cs = torch.nn.functional.normalize(Xs, dim=-1); C = (cs @ cs.T).clamp(min=0).cpu()
    n = nS; i_ = torch.arange(n); low = (i_[:, None] > i_[None, :]).float()       # row i: edges to earlier sentences j < i
    Aext = (C + (i_[:, None] == i_[None, :] + 1).float()) * low; Aint = A.mean(0)[1:, 1:] * low
    Aext = Aext / Aext.sum(1, keepdim=True).clamp(min=1e-12); Aint = Aint / Aint.sum(1, keepdim=True).clamp(min=1e-12)
    rows = low.sum(1) > 0
    f["cie_tv"] = float(0.5 * (Aext - Aint).abs().sum(1)[rows].mean()) if rows.any() else np.nan
    m_ = low.bool()
    rk = lambda v: torch.argsort(torch.argsort(v)).float()
    f["cie_rho"] = float(torch.corrcoef(torch.stack([rk(Aext[m_]), rk(Aint[m_])]))[0, 1]) if m_.sum() > 10 else np.nan
    _A["res"] = {}; del CAP, LAST, FIN
    return f


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", required=True); ap.add_argument("--out", default=str(HERE / "out_papers"))
    ap.add_argument("--traces", default="graph_inputs.jsonl,traces_set100.jsonl"); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    rows = [json.loads(l) for f in a.traces.split(",") for l in open(HERE / f)]
    import transformers
    for m in a.models.split(","):
        R = [r for r in rows if r["model_id"] == m]
        if a.limit: R = R[:a.limit]
        odir = Path(a.out) / m.replace("/", "__"); odir.mkdir(parents=True, exist_ok=True); out = odir / "papers.jsonl"
        done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set(); todo = [r for r in R if r["id"] not in done]
        log(f"{m}: {len(R)} traces, {len(todo)} to do | torch {torch.__version__} transformers {transformers.__version__} | {gpu()}")
        if not todo: continue
        model, tok = SI.load_model_and_tokenizer(m, local_files_only=True); layers, nL = SI.find_decoder_layers(model); tid = think_end_token_id(tok)
        has_think = tid is not None or "<think>" in build_prompt(tok, "x")[1]
        att = [i for i, l in enumerate(layers) if getattr(l, "self_attn", None) is not None]
        mid = frozenset(att[len(att) // 3: max(len(att) // 3 + 1, 2 * len(att) // 3)])
        named = (nL // 3, nL // 2, int(0.7 * nL), 2 * nL // 3)
        ctx = dict(layers=layers, tid=tid, norm=final_norm(model), attn_mid=mid, cap=set(named), cap_named=named,
                   sfx=tok((SUFFIXES_THINK if has_think else SUFFIXES_NOTHINK)["B"], add_special_tokens=False)["input_ids"])
        install(); log(f"=== {m}: {nL} layers, attention layers {len(att)}, hooked {sorted(mid)} | {gpu()}")
        for k, r in enumerate(todo):
            t0 = time.time()
            try:
                f = one(model, tok, ctx, r)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); log(f"    OOM on {r['id']} - skipped | {gpu()}"); continue
            except Exception:                                                         # noqa: BLE001
                log(f"    FAILED {r['id']}:\n{traceback.format_exc()}"); continue
            rec = dict(id=r["id"], dataset=r["dataset"], robust=r["robust"], **f)
            with open(out, "a") as fo: fo.write(json.dumps(rec) + "\n")
            log(f"    [{k + 1}/{len(todo)}] {r['id']}: {f['cut']} tok, {f['n_sent']} sent | rhythm {f['rhythm_corr0']:+.2f} anchorFAI {f['anchor_fai']:.4f} | "
                f"minK++20 {f['minkpp_20']:+.2f} | LID {f['lid_mean']:.1f} | DC {f['dc_l2']:.3f} | LI {f['li_sum']:+.1f} def {f['li_def']:.1f} | "
                f"sup {f['sup_top']:+.3f}/{f['sup_rnd']:+.3f} | CIE {f['cie_tv']:.3f} | {time.time() - t0:.1f}s | {gpu()}")
            torch.cuda.empty_cache()
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
