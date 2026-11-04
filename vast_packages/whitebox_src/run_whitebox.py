"""White-box / counterfactual run (Vast): for every labelled row of whitebox_inputs.jsonl (main_v2 43 + olympiad 75) the GREEDY trace we
already have (capped at 6144, never regenerated) is teacher forced under the original prompt and under perturbed prompts, and the model's
internals are measured. Cut c = end of thinking or 6144 tokens, whichever comes first (the forced-answer budget the submission uses).

  1 counterfactual teacher forcing  for each variant prompt (the friend's rephrase / rename / distract / domain / typos where available,
      plus automatic typos / distractor / variable renaming): the same trace after the variant prompt. Per-token log-prob of the trace,
      16-token window states (6 layers), and the forced answer at c (suffix B): full first-token distribution -> probability of the
      ORIGINAL answer token, top-1, KL to the original, 12-token answer string.
  2 internal-noise sensitivity      at the forced-answer suffix (original prompt, cut c): Gaussian noise added to the residual stream
      output of a layer at 25 / 50 / 75% depth, scale alpha x the per-position RMS (alpha 0.05, 0.1, 0.2, 0.4), 3 seeds -> probability of
      the clean answer token and whether the top-1 flips.
  3 attribution                     gradient x input of the clean answer-token logit w.r.t. the input embeddings (prompt + trace up to c +
      suffix), with gradient checkpointing -> |attribution| per prompt token, per 16-token trace window.
  4 attention coverage + reasoning graph   Lookback Lens as the submission computes it, plus per prompt token the attention mass it receives
      from the generated queries (mean over heads): per layer [L, n_prompt] and per 16-token span [n_span, n_prompt] (mean over layers).
      Reasoning graphs: nodes = the prompt's number tokens, its prose tokens, and reasoning segments; edge (q, k) = mean attention mass
      a query segment's tokens put on key node k (exact softmax). Segments: sentences (graph_sentence_layer [L, n, 2 + n], mean over
      heads), steps = paragraphs (graph_step_layer and per head graph_step_head [L, H, n, 2 + n]), and fixed 64-token blocks
      (graph_block64_layer, a length-neutral baseline); the segment id of every reasoning token is saved (seg_sentence, seg_step).
  5 reasoning ablation              the forced answer at the end after removing reasoning: middle half deleted, only the last 1024 tokens,
      only the first 1024 tokens, no reasoning at all.
  6 logit lens                      at the forced-answer position (clean): final norm + LM head applied to every layer's hidden state ->
      probability and rank of the clean answer token per layer.
  7 MLP layers                      (a) direct logit attribution of the clean answer token to every layer's attention and MLP output
      (dla_attn / dla_mlp [L]); (b) along the reasoning, per layer and 16-token span: ||MLP output|| / ||residual||, the share of MLP
      activation mass in the top 1% of neurons, the normalised entropy of the MLP activations; (c) the top-64 MLP neurons at the answer
      position, clean and under every counterfactual variant (Jaccard overlap per layer); (d) the forced answer with the MLPs of the
      early / middle / late third of the layers zeroed or halved.
Everything heavy goes to <out>/<model>/<id>.npz; a summary line per row to <out>/<model>/rows_shard<i>.jsonl (resumable).
Usage (run.sh): CUDA_VISIBLE_DEVICES=0 python -u run_whitebox.py --models <model> --shard 0 --nshard 1"""
from __future__ import annotations

import argparse, copy, json, sys, time, traceback, zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "submission_core"))
import numpy as np
import torch

import spurious_inference as SI
from spurious_features import (SEL_LAYER_FRACS, SUFFIXES_NOTHINK, SUFFIXES_THINK, TRAJ_LAYERS, build_prompt, find_think_end, think_end_token_id)
from run_t1 import gpu, log, prefetch, final_norm, tf_pass_safe

ALPHAS = (0.05, 0.1, 0.2, 0.4); NOISE_DEPTHS = (0.25, 0.5, 0.75); NOISE_SEEDS = 3


# ----------------------------------------------------------------------------------------------------------------- MLP helpers
def _mixer(layer):
    for nme in ("self_attn", "linear_attn", "attn", "mixer", "token_mixer"):
        if getattr(layer, nme, None) is not None:
            return getattr(layer, nme)
    return None


def _ffn(layer):
    for nme in ("mlp", "feed_forward", "ffn"):
        if getattr(layer, nme, None) is not None:
            return getattr(layer, nme)
    return None


def _down(ffn):
    for nme in ("down_proj", "fc2", "w2", "c_proj"):
        if ffn is not None and getattr(ffn, nme, None) is not None:
            return getattr(ffn, nme)
    return None


def _out(o):
    return o[0] if isinstance(o, tuple) else o


class Components:
    """Hooks for one suffix pass: per layer, the token mixer (attention) output and the MLP output at the last position, the MLP hidden
    activations (input of down_proj) at the last position, and the residual entering the final norm."""
    def __init__(self, layers, norm):
        self.layers, self.norm, self.res = layers, norm, {}

    def __call__(self):
        hs = []
        for i, layer in enumerate(self.layers):
            mx, ff = _mixer(layer), _ffn(layer); dn = _down(ff)
            if mx is not None:
                hs.append(mx.register_forward_hook(lambda m, a, o, i=i: self.res.__setitem__(("attn", i), _out(o)[0, -1].float())))
            if ff is not None:
                hs.append(ff.register_forward_hook(lambda m, a, o, i=i: self.res.__setitem__(("mlp", i), _out(o)[0, -1].float())))
            if dn is not None:
                hs.append(dn.register_forward_pre_hook(lambda m, a, i=i: self.res.__setitem__(("act", i), a[0][0, -1].float())))
        if self.norm is not None:
            hs.append(self.norm.register_forward_pre_hook(lambda m, a: self.res.__setitem__(("final", 0), a[0][0, -1].float())))
        return hs

    def dla(self, model, target):
        """Direct logit attribution of the target token: each component's output through the (linearised) final norm and the LM head."""
        W = model.get_output_embeddings().weight[target].float(); g = getattr(self.norm, "weight", None)
        g = g.float() if g is not None else torch.ones_like(W); rms = self.res[("final", 0)].pow(2).mean().sqrt() + 1e-6
        f = lambda v: float((v * g * W).sum() / rms)                                 # noqa: E731
        L = len(self.layers)
        return (np.array([f(self.res[("attn", i)]) if ("attn", i) in self.res else np.nan for i in range(L)], np.float32),
                np.array([f(self.res[("mlp", i)]) if ("mlp", i) in self.res else np.nan for i in range(L)], np.float32))

    def topk_neurons(self, k=64):
        L = len(self.layers)
        return np.stack([self.res[("act", i)].abs().topk(k).indices.cpu().numpy().astype(np.int32) if ("act", i) in self.res else np.full(k, -1, np.int32)
                         for i in range(L)])


def mlp_scale_hook(layers, band, scale):
    """Scale the MLP output of the layers in `band` (zero-ablation at scale 0) during a suffix pass."""
    def make():
        return [_ffn(layers[i]).register_forward_hook(lambda m, a, o: (_out(o) * scale,) + tuple(o[1:]) if isinstance(o, tuple) else o * scale)
                for i in band if _ffn(layers[i]) is not None]
    return make


# ----------------------------------------------------------------------------------------------------------------- forced answers
def prefix_cache(model, ids):
    """KV cache of `ids`, prefilled in 1024-token chunks (256 on CUDA OOM)."""
    for chunk in (1024, 256):
        try:
            cache = None
            with torch.no_grad():
                for s in range(0, len(ids), chunk):
                    o = SI.forward_last_logits(model, input_ids=torch.tensor([ids[s:s + chunk]], device=model.device), past_key_values=cache, use_cache=True)
                    cache = o.past_key_values; del o
            return cache
        except torch.cuda.OutOfMemoryError:
            if chunk == 256:
                raise
        torch.cuda.empty_cache(); log(f"    OOM prefilling {len(ids)} tokens in 1024 chunks -> 256 | {gpu()}")


def suffix_pass(model, cache, prefix_ids, sids, hidden=False, hook=None):
    """Forward the suffix on a COPY of the prefix cache (recompute if the cache cannot be copied). -> (log-probs [V], hidden states, kv)."""
    h = hook() if hook else None
    h = h if isinstance(h, list) else ([h] if h is not None else [])
    try:
        with torch.no_grad():
            try:
                branch = copy.deepcopy(cache)
                o = model(input_ids=torch.tensor([sids], device=model.device), past_key_values=branch, use_cache=True, output_hidden_states=hidden, logits_to_keep=1)
            except (RuntimeError, TypeError, AttributeError) as e:
                if isinstance(e, torch.cuda.OutOfMemoryError): raise
                o = model(input_ids=torch.tensor([prefix_ids + sids], device=model.device), use_cache=True, output_hidden_states=hidden, logits_to_keep=1)
    finally:
        for x in h: x.remove()
    lp = torch.log_softmax(o.logits[0, -1].float(), -1)
    hs = [x[0, -1].float() for x in o.hidden_states] if hidden else None
    return lp, hs, o.past_key_values


def continuation(model, tok, kv, first, n=12):
    toks, nxt = [], int(first)
    with torch.no_grad():
        for _ in range(n):
            toks.append(nxt)
            o = SI.forward_last_logits(model, input_ids=torch.tensor([[nxt]], device=model.device), past_key_values=kv, use_cache=True)
            kv, nxt = o.past_key_values, int(o.logits[0, -1].argmax())
    return tok.decode(toks, skip_special_tokens=True)


def answer_record(model, tok, prompt, trace, sfx, clean_lp=None, clean_top=None, hidden=False, comp=None):
    """Forced answer after prompt + trace (+ suffix B): its own top-1 and answer string, and versus the clean run: probability of the clean
    answer token, flip, KL(clean || this)."""
    ids = prompt + trace; cache = prefix_cache(model, ids)
    lp, hs, kv = suffix_pass(model, cache, ids, sfx, hidden=hidden, hook=comp); p = lp.exp(); top = int(lp.argmax())
    rec = dict(p1=float(p[top]), top1=top, entropy=float(-(p * lp).sum()), answer=continuation(model, tok, kv, top))
    if clean_lp is not None:
        cp = clean_lp.exp()
        rec.update(p_clean_answer=float(p[clean_top]), flip=bool(top != clean_top), kl_from_clean=float((cp * (clean_lp - lp)).sum()))
    del kv
    return rec, lp, hs, cache


# ----------------------------------------------------------------------------------------------------------------- attention coverage
_COV = {"on": False}


def _cov_layer(li, query, key, sc, win):
    q, kk = query[0], key[0]; H, T, D = q.shape; Hkv = kk.shape[0]; n_p, Wn = _COV["n_p"], 16; nwin = -(-(T - n_p) // Wn)
    qg = q.reshape(Hkv, H // Hkv, T, D); kpos = torch.arange(T, device=q.device)
    lr_sum = torch.zeros(H, nwin, device=q.device); m_sum = torch.zeros(H, nwin, device=q.device); cnt = torch.zeros(nwin, device=q.device)
    cov_span = torch.zeros(nwin, n_p, device=q.device)
    G = {nm: (torch.zeros(H, g["nq"], g["nk"], device=q.device), torch.zeros(g["nq"], device=q.device)) for nm, g in _COV["graphs"].items()}
    for s0 in range(n_p, T, 256):
        e0 = min(T, s0 + 256); qpos = torch.arange(s0, e0, device=q.device)
        p, mask = SI._lb_prob_chunk(qg[:, :, s0:e0], kk, sc, win, qpos, kpos); p = p.reshape(H, e0 - s0, T)
        mctx = p[..., :n_p].sum(-1); vis = ~mask
        cctx = vis[:, :n_p].sum(-1).float(); cnew = vis[:, n_p:].sum(-1).float()
        actx = mctx / cctx.clamp(min=1); anew = (1 - mctx).clamp(min=0) / cnew.clamp(min=1)
        lr = torch.where(cctx > 0, actx / (actx + anew).clamp(min=1e-12), torch.zeros_like(actx))
        widx = torch.div(qpos - n_p, Wn, rounding_mode="floor")
        lr_sum.index_add_(1, widx, lr); m_sum.index_add_(1, widx, mctx); cnt.index_add_(0, widx, torch.ones_like(widx, dtype=torch.float))
        cov_span.index_add_(0, widx, p[..., :n_p].mean(0))                  # attention on each prompt token, mean over heads
        # reasoning graphs: attention mass of each query on each key node (prompt numbers, prompt prose, then reasoning segments:
        # sentences, paragraphs / steps, fixed 64-token blocks), summed per query segment
        for nm, g in _COV["graphs"].items():
            pk = torch.zeros(H, e0 - s0, g["nk"], device=q.device).index_add_(2, g["kseg"][:T], p)
            qb = g["qseg"][s0 - n_p:e0 - n_p]
            G[nm][0].index_add_(1, qb, pk); G[nm][1].index_add_(0, qb, torch.ones_like(qb, dtype=torch.float))
        del p
    c = cnt.clamp(min=1)
    _COV["res"][li] = (lr_sum / c, m_sum / c, cov_span / c[:, None],                 # graphs leave the GPU layer by layer
                       {nm: (a / n.clamp(min=1)[None, :, None]).half().cpu() for nm, (a, n) in G.items()})


def _cov_sdpa(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kw):
    out = _COV["orig"](module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kw)
    if _COV["on"]:
        li = getattr(module, "layer_idx", None)
        if li in _COV["layers"]:
            _cov_layer(li, query, key, scaling if scaling is not None else query.shape[-1] ** -0.5, getattr(module, "sliding_window", None))
    return out


def attention_coverage(model, layers, prompt, trace, prompt_is_number, segmentations):
    if "orig" not in _COV:
        from transformers import AttentionInterface
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        _COV["orig"] = sdpa_attention_forward; AttentionInterface.register("sdpa", _cov_sdpa)
    n_p = len(prompt); pk = np.where(np.asarray(prompt_is_number), 0, 1)       # prompt nodes: 0 = number tokens, 1 = prose
    graphs = {}
    for nm, seg in segmentations.items():                                       # seg: segment id of every reasoning token
        seg = np.asarray(seg, np.int64); n = int(seg.max()) + 1
        graphs[nm] = dict(qseg=torch.tensor(seg, device=model.device), kseg=torch.tensor(np.r_[pk, 2 + seg], device=model.device), nq=n, nk=n + 2)
    _COV.update(on=True, layers=frozenset(i for i, l in enumerate(layers) if getattr(l, "self_attn", None) is not None), res={}, n_p=n_p, graphs=graphs)
    # MLP 2 + 5 in the same pass: per generated token and layer, ||MLP output|| / ||residual after the layer||, and the sparsity of the MLP
    # hidden activations (share of |activation| mass in the top 1% of neurons; normalised entropy of |activation|); means per 16-token span
    T = n_p + len(trace); nsp = -(-len(trace) // 16); sp = torch.tensor(np.arange(len(trace)) // 16, device=model.device)
    spc = torch.bincount(sp, minlength=nsp).clamp(min=1).float(); MS = {}

    def span_mean(v):                                                           # v [T] -> [nsp] over the generated positions
        return torch.zeros(nsp, device=v.device).index_add_(0, sp, v[n_p:].float()) / spc

    def mk(i):
        def mlp_out(m, a_, o):
            MS[("mlp", i)] = _out(o)[0].float().norm(dim=-1)
        def layer_out(m, a_, o):
            MS[("res", i)] = _out(o)[0].float().norm(dim=-1)
        def act_in(m, a_):
            x = a_[0][0].float().abs(); k = max(1, x.shape[-1] // 100)
            MS[("top1", i)] = span_mean(x.topk(k, dim=-1).values.sum(-1) / (x.sum(-1) + 1e-9))
            pr = x / (x.sum(-1, keepdim=True) + 1e-9); MS[("ent", i)] = span_mean(-(pr * (pr + 1e-12).log()).sum(-1) / np.log(x.shape[-1]))
        return mlp_out, layer_out, act_in
    hk = []
    for i, layer in enumerate(layers):
        ff = _ffn(layer); dn = _down(ff); f1, f2, f3 = mk(i)
        if ff is not None: hk.append(ff.register_forward_hook(f1))
        hk.append(layer.register_forward_hook(f2))
        if dn is not None: hk.append(dn.register_forward_pre_hook(f3))
    try:
        with torch.no_grad():
            SI.forward_last_logits(model, input_ids=torch.tensor([prompt + trace], device=model.device), use_cache=False)
    finally:
        _COV["on"] = False
        for h in hk: h.remove()
    L_ = len(layers)
    mlp_spans = dict(
        mlp_write_ratio=np.stack([span_mean(MS[("mlp", i)] / (MS[("res", i)] + 1e-6)).cpu().numpy() if ("mlp", i) in MS else np.full(nsp, np.nan) for i in range(L_)]).astype(np.float16),
        mlp_top1pct_mass=np.stack([MS[("top1", i)].cpu().numpy() if ("top1", i) in MS else np.full(nsp, np.nan) for i in range(L_)]).astype(np.float16),
        mlp_act_entropy=np.stack([MS[("ent", i)].cpu().numpy() if ("ent", i) in MS else np.full(nsp, np.nan) for i in range(L_)]).astype(np.float16))
    MS.clear()
    if not _COV["res"]:
        return mlp_spans
    ls = sorted(_COV["res"]); R = _COV["res"]; _COV["res"] = {}
    lb = torch.stack([R[l][0] for l in ls]).permute(2, 0, 1); pm = torch.stack([R[l][1] for l in ls]).permute(2, 0, 1)
    cs = torch.stack([R[l][2] for l in ls])                                # [L, n_span, n_prompt]
    out = {}
    for nm in segmentations:                                                    # [L, H, nq, 2 + nq]; per head only for the step graph
        B = torch.stack([R[l][3][nm] for l in ls])
        out[f"graph_{nm}_layer"] = B.float().mean(1).half().numpy()
        if nm == "step":
            out["graph_step_head"] = B.numpy()
    return dict(**out, **mlp_spans,
                lookback=lb.half().cpu().numpy(), prompt_mass=pm.half().cpu().numpy(), cov_layer=cs.mean(1).half().cpu().numpy(),
                cov_span=cs.mean(0).half().cpu().numpy(), attn_layers=np.array(ls, np.int16))


def segment_trace(tok, ids, min_len=4, max_len=256, max_steps=64):
    """Segment ids per reasoning token at two granularities, from the model's own tokens:
      sentence  ends after a token containing a newline, or after a token ending in . ? ! when the next token starts with whitespace
                (so 3.5 or x.y never split); fragments shorter than min_len tokens join the next sentence; runs over max_len are cut
      step      ends at a blank line (paragraph); falls back to single newlines when a trace has fewer than 8 paragraphs; when there are
                more than max_steps steps, neighbouring steps are merged into max_steps groups of about equal length (the per-head graph
                is saved at this level, so its size must be bounded)"""
    pieces = [tok.decode([t]) for t in ids]; n = len(ids)

    def assign(ends):
        seg, cur, start = np.zeros(n, np.int64), 0, 0
        for i in range(n):
            seg[i] = cur
            if (i in ends and i - start + 1 >= min_len) or i - start + 1 >= max_len:
                cur += 1; start = i + 1
        return seg
    sent_end = {i for i, p in enumerate(pieces) if "\n" in p or (p.rstrip()[-1:] in ".?!" and p.rstrip() and i + 1 < n and pieces[i + 1][:1].isspace())}
    para = {i for i, p in enumerate(pieces) if "\n\n" in p or (p.endswith("\n") and i + 1 < n and pieces[i + 1].startswith("\n"))}
    if len(para) < 8:
        para = {i for i, p in enumerate(pieces) if "\n" in p}
    step = assign(para)
    if step.max() + 1 > max_steps:                                              # merge neighbouring steps into <= max_steps groups of ~equal
        sizes = np.bincount(step); target = len(ids) / max_steps; g, acc, mp = 0, 0, {}  # length (group borders stay paragraph borders)
        for k, n_k in enumerate(sizes):
            mp[k] = g; acc += n_k
            if acc >= target and g < max_steps - 1:
                g += 1; acc = 0
        step = np.array([mp[k] for k in step])
    return assign(sent_end), step


# ----------------------------------------------------------------------------------------------------------------- attribution
def attribution(model, prompt, trace, sfx, target):
    """|grad x input| of the clean answer-token logit w.r.t. the input embeddings, gradient checkpointing, parameters frozen."""
    for p_ in model.parameters():
        p_.requires_grad_(False)
    ids = torch.tensor([prompt + trace + sfx], device=model.device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.train()
    try:
        with torch.enable_grad():
            emb = model.get_input_embeddings()(ids).detach().requires_grad_(True)
            try:
                o = model(inputs_embeds=emb, use_cache=False, logits_to_keep=1)
            except TypeError:
                o = model(inputs_embeds=emb, use_cache=False)
            o.logits[0, -1, target].float().backward()
            a = (emb.grad.float() * emb.float()).sum(-1).abs()[0].detach().cpu().numpy()
    finally:
        model.eval(); model.gradient_checkpointing_disable()
    return a


# ----------------------------------------------------------------------------------------------------------------- noise hook
def noise_hook(layer, alpha, seed):
    def make():
        g = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
        def fn(_m, _i, o):
            h = o[0] if isinstance(o, tuple) else o
            rms = h.float().pow(2).mean(-1, keepdim=True).sqrt()
            h2 = (h.float() + alpha * rms * torch.randn(h.shape, generator=g, device=h.device)).to(h.dtype)
            return (h2,) + tuple(o[1:]) if isinstance(o, tuple) else h2
        return layer.register_forward_hook(fn)
    return make


# ----------------------------------------------------------------------------------------------------------------- one row
def run_row(model, tok, ctx, r):
    prompt = r["prompt_ids"]; gen = r["greedy_gen_ids"][:6144]; te = find_think_end(tok, gen, ctx["tid"]); L = max(1, te if te is not None else len(gen))
    c = min(L, 6144); trace = gen[:c]; sfx = ctx["suffixes"]["B"]; span = build_prompt(tok, r["problem"])[2]
    out, arr, T0 = dict(id=r["id"], dataset=r["dataset"], model_id=r["model_id"], robust=r["robust"], n_gen=len(gen), think_end=te, cut=c), {}, time.time()
    # clean run: trace log-probs + window states, forced answer with all hidden states (logit lens), its cache for the noise runs
    (ws0, ex0), mode = tf_pass_safe(model, ctx, prompt, gen, span, [c]); out["tf_mode"] = mode
    arr.update(ws16=ws0.astype(np.float16), tok_logprob=ex0["gen_logprob"].astype(np.float16))
    comp = Components(ctx["layers"], ctx["norm"])
    clean, clp, hs, cache = answer_record(model, tok, prompt, trace, sfx, hidden=True, comp=comp); top = clean["top1"]; out["clean"] = clean
    try:                                                                    # MLP 1: direct logit attribution of the clean answer token
        arr["dla_attn"], arr["dla_mlp"] = comp.dla(model, top); arr["mlp_topk_clean"] = comp.topk_neurons()
    except Exception as e:                                                  # noqa: BLE001
        out["dla_error"] = repr(e)[:300]; log(f"    DLA failed: {e!r}")
    head, norm = model.get_output_embeddings(), ctx["norm"]
    with torch.no_grad():
        lens = [torch.log_softmax(head(norm(h[None].to(head.weight.dtype))).float()[0], -1) for h in hs]
    arr["lens_p"] = np.array([float(x[top].exp()) for x in lens], np.float32); arr["lens_rank"] = np.array([int((x > x[top]).sum()) for x in lens], np.int32)
    out["t_clean"] = round(time.time() - T0, 1); t = time.time()
    # 2 internal noise at the forced-answer suffix
    nz = []
    for d in NOISE_DEPTHS:
        li = min(ctx["n_layers"] - 1, int(round(d * ctx["n_layers"])))
        for a in ALPHAS:
            for s in range(NOISE_SEEDS):
                lp, _, kv = suffix_pass(model, cache, prompt + trace, sfx, hook=noise_hook(ctx["layers"][li], a, 1000 * s + int(a * 100) + li)); del kv
                nz.append((d, a, s, float(lp[top].exp()), bool(int(lp.argmax()) != top)))
    arr["noise"] = np.array([x[:4] + (float(x[4]),) for x in nz], np.float32)        # depth, alpha, seed, p(clean answer), flipped
    # MLP 4: ablate the MLPs of one layer band (early / middle / late third) at the forced-answer suffix, zero and half scale
    nL = ctx["n_layers"]; mab = []
    for bname, band in (("early", range(0, nL // 3)), ("middle", range(nL // 3, 2 * nL // 3)), ("late", range(2 * nL // 3, nL))):
        for sc in (0.0, 0.5):
            lp, _, kv = suffix_pass(model, cache, prompt + trace, sfx, hook=mlp_scale_hook(ctx["layers"], band, sc)); del kv
            mab.append(dict(band=bname, scale=sc, p_clean_answer=float(lp[top].exp()), flip=bool(int(lp.argmax()) != top)))
    out["mlp_ablation"] = mab
    del cache; torch.cuda.empty_cache(); out["t_noise"] = round(time.time() - t, 1); t = time.time()
    # 1 counterfactual teacher forcing
    cf = []
    for k, v in enumerate(r["variants"]):
        pv = build_prompt(tok, v["text"])[0]
        (wsv, exv), _ = tf_pass_safe(model, ctx, pv, gen, None, [c])
        cvc = Components(ctx["layers"], ctx["norm"])
        rec, _, _, cv = answer_record(model, tok, pv, trace, sfx, clp, top, comp=cvc); del cv
        try:                                                                # MLP 3: neuron overlap with the clean answer position
            arr[f"cf{k}_mlp_topk"] = cvc.topk_neurons()
            if "mlp_topk_clean" in arr:
                rec["mlp_neuron_jaccard"] = [float(len(set(a_) & set(b_)) / len(set(a_) | set(b_))) for a_, b_ in zip(arr["mlp_topk_clean"], arr[f"cf{k}_mlp_topk"])]
        except Exception as e:                                              # noqa: BLE001
            rec["mlp_topk_error"] = repr(e)[:200]
        dl = exv["gen_logprob"][:c].astype(np.float32) - ex0["gen_logprob"][:c].astype(np.float32)
        m = min(len(wsv), len(ws0)); cosw = (wsv[:m] * ws0[:m]).sum(-1) / (np.linalg.norm(wsv[:m], axis=-1) * np.linalg.norm(ws0[:m], axis=-1) + 1e-9)
        q = max(1, len(dl) // 4)
        rec.update(kind=v["kind"], n_prompt=len(pv), dlogprob_mean=float(dl.mean()), dlogprob_first_quarter=float(dl[:q].mean()), dlogprob_last_quarter=float(dl[-q:].mean()),
                   state_cos_mean=[float(x) for x in cosw.mean(0)], state_cos_last_quarter=[float(x) for x in cosw[-max(1, m // 4):].mean(0)])
        cf.append(rec); arr[f"cf{k}_dlogprob"] = dl.astype(np.float16); arr[f"cf{k}_state_cos"] = cosw.astype(np.float16)
        torch.cuda.empty_cache()
    out["counterfactual"] = cf; out["t_counterfactual"] = round(time.time() - t, 1); t = time.time()
    # 5 reasoning ablation
    ab = {}
    for nm, tr in (("drop middle half", trace[: c // 4] + trace[3 * c // 4:]), ("last 1024 only", trace[-1024:]), ("first 1024 only", trace[:1024]), ("no reasoning", [])):
        rec, _, _, cv = answer_record(model, tok, prompt, tr, sfx, clp, top); del cv; ab[nm] = rec
    out["ablation"] = ab; out["t_ablation"] = round(time.time() - t, 1); t = time.time()
    # 4 attention coverage (+ lookback as shipped)
    try:
        isnum = [any(ch.isdigit() for ch in tok.decode([x])) for x in prompt]
        sent, step = segment_trace(tok, trace)
        arr.update(seg_sentence=sent.astype(np.int16), seg_step=step.astype(np.int16))
        cov = attention_coverage(model, ctx["layers"], prompt, trace, isnum, {"sentence": sent, "step": step, "block64": np.arange(len(trace)) // 64})
        if cov: arr.update({f"att_{k}": v for k, v in cov.items()})
        out["coverage"] = "ok" if cov else "no attention captured"
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); out["coverage"] = "OOM"
    except Exception as e:                                              # noqa: BLE001 - keep the row, record why
        out["coverage"] = f"error: {e!r}"[:300]; log(f"    coverage failed: {e!r}")
    out["t_coverage"] = round(time.time() - t, 1); t = time.time()
    # 3 attribution (falls back to the first 2048 reasoning tokens on OOM)
    for cc in (c, min(c, 2048)):
        try:
            arr["attribution"] = attribution(model, prompt, trace[:cc], sfx, top).astype(np.float32); out["attribution_cut"] = cc; break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); log(f"    OOM in attribution at {cc} tokens | {gpu()}"); out["attribution_cut"] = None
        except Exception as e:                                          # noqa: BLE001 - keep the row, record why
            out["attribution_cut"] = None; out["attribution_error"] = repr(e)[:300]; log(f"    attribution failed: {e!r}"); break
    out["t_attribution"] = round(time.time() - t, 1)
    pt = [tok.decode([x]) for x in prompt]
    arr["prompt_is_number"] = np.array([any(ch.isdigit() for ch in s) for s in pt]); arr["prompt_ids"] = np.array(prompt, np.int32)
    arr["problem_span"] = np.array(span if span else (-1, -1), np.int32)
    return out, arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=str(HERE / "whitebox_inputs.jsonl")); ap.add_argument("--out", default=str(HERE / "out_whitebox"))
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1); ap.add_argument("--models", default="")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.inputs)]; rows = [r for i, r in enumerate(rows) if i % a.nshard == a.shard]
    order = ["Qwen/Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B", "allenai/Olmo-3-7B-Think", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"]
    allm = [m for m in order if any(r["model_id"] == m for r in rows)]; models = [m for m in allm if not a.models or m in a.models.split(",")]
    import transformers
    log(f"shard {a.shard}/{a.nshard}: {len(rows)} rows, models {models} | torch {torch.__version__} transformers {transformers.__version__} | {gpu()}")
    if a.shard == 0 and models:
        prefetch(allm[allm.index(models[0]):])
    for m in models:
        mrows = [r for r in rows if r["model_id"] == m]; odir = Path(a.out) / m.replace("/", "__"); odir.mkdir(parents=True, exist_ok=True)
        summ = odir / f"rows_shard{a.shard}.jsonl"; done = {json.loads(l)["id"] for l in open(summ)} if summ.exists() else set()
        todo = [r for r in mrows if r["id"] not in done]
        log(f"=== {m}: {len(mrows)} rows, {len(done)} done, {len(todo)} to do")
        if not todo: continue
        from huggingface_hub import snapshot_download
        snapshot_download(m)
        model, tok = SI.load_model_and_tokenizer(m, local_files_only=True); layers, n_layers = SI.find_decoder_layers(model)
        sel = sorted({min(n_layers - 1, max(0, int(round(f * n_layers)) - 1)) for f in SEL_LAYER_FRACS}); tid = think_end_token_id(tok)
        has_think = tid is not None or "<think>" in build_prompt(tok, "x")[1]
        sfx = {k: tok(v, add_special_tokens=False)["input_ids"] for k, v in (SUFFIXES_THINK if has_think else SUFFIXES_NOTHINK).items()}
        ctx = dict(layers=layers, n_layers=n_layers, sel_layers=sel, traj_layers=[l for l in TRAJ_LAYERS.get(m, []) if l < n_layers] or sel,
                   tid=tid, suffixes=sfx, norm=final_norm(model))
        log(f"{m}: {n_layers} layers | final norm {'found' if ctx['norm'] is not None else 'MISSING'} | {gpu()}")
        for j, r in enumerate(todo):
            t0 = time.time()
            if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
            try:
                out, arr = run_row(model, tok, ctx, r)
                np.savez_compressed(odir / f"{r['id']}.npz", **arr)
                with open(summ, "a") as f:
                    f.write(json.dumps(out) + "\n")
                cf = out["counterfactual"]
                log(f"  [{j + 1}/{len(todo)}] {r['id']} ({r['dataset']}, robust={r['robust']}) {time.time() - t0:.0f}s "
                    f"(clean {out['t_clean']} noise {out['t_noise']} cf {out['t_counterfactual']} ablation {out['t_ablation']} coverage {out['t_coverage']} "
                    f"attribution {out['t_attribution']}) | clean p1 {out['clean']['p1']:.3f} {out['clean']['answer']!r} | variants "
                    + ", ".join(f"{x['kind']}: p {x['p_clean_answer']:.2f}{' FLIP' if x['flip'] else ''}" for x in cf)
                    + f" | coverage {out['coverage']} | attribution cut {out['attribution_cut']} | {gpu()}")
            except Exception:
                log(f"  [{j + 1}/{len(todo)}] {r['id']} FAILED:\n{traceback.format_exc()}")
                with open(odir / f"failures_shard{a.shard}.jsonl", "a") as f:
                    f.write(json.dumps(dict(id=r["id"], error=traceback.format_exc())) + "\n")
                torch.cuda.empty_cache()
        del model, ctx, layers
        import gc; gc.collect(); torch.cuda.empty_cache(); log(f"{m} released | {gpu()}")
    log("shard finished")


if __name__ == "__main__":
    main()
