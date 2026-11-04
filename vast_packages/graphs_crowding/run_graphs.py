"""Reasoning-graph checks + embedding-space crowding (Vast, one process per model; run.sh). No generation.
(The recall-vs-thinking direction stages are held back for now: vast_packages/recall_graphs_src/run_recall_graphs_with_recall.py.)

For every labelled row (the white-box run's 118 greedy traces, cut at end of thinking / 6144) and the main_v2 T=1 samples (2 per row,
re-tokenised from text): one teacher-forced pass with
  - reasoning graphs at EVERY attention layer: 64-token blocks, keys = prompt (node 0) + reasoning blocks; head-mean graph per layer;
    the same graph with the layer's attention sinks removed (reasoning tokens receiving > 10x the median incoming attention, top 4);
    per head and layer: mean backward jump, share > 20 blocks back, share on the first 4 blocks (when > 20 back);
  - embedding-space crowding (arXiv 2601.22536): per position, top-100 next-token distribution, step crowding
    sum_i p_i sum_{j!=i} p_j |cos(e_i, e_j)| with input embeddings and with output (unembedding) rows; plus entropy for comparison.
Outputs <out>/<model>/: graphs/<id>__<trace>.npz, graph_rows.jsonl (resumable)."""
from __future__ import annotations

import argparse, json, sys, time, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "submission_core"))
import numpy as np
import torch

import spurious_inference as SI
from spurious_features import SUFFIXES_NOTHINK, SUFFIXES_THINK, build_prompt, find_think_end, think_end_token_id
from run_t1 import gpu, log, prefetch, final_norm

ORDER = ["Qwen/Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B", "allenai/Olmo-3-7B-Think", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"]


# ----------------------------------------------------------------------------------------------------------------- G: graphs + projections + crowding
_G = {"on": False}


def _g_layer(li, query, key, sc, win):
    q, kk = query[0], key[0]; H, T, D = q.shape; Hkv = kk.shape[0]; n_p, nq = _G["n_p"], _G["nq"]
    qg = q.reshape(Hkv, H // Hkv, T, D); kpos = torch.arange(T, device=q.device); kseg = _G["kseg"][:T]; qblk = _G["qblk"]
    tokm = torch.zeros(nq, T, device=q.device); per = torch.zeros(H, nq, nq + 1, device=q.device); cnt = torch.zeros(nq, device=q.device)
    for s0 in range(n_p, T, 256):
        e0 = min(T, s0 + 256); qpos = torch.arange(s0, e0, device=q.device)
        p, _ = SI._lb_prob_chunk(qg[:, :, s0:e0], kk, sc, win, qpos, kpos); p = p.reshape(H, e0 - s0, T); qb = qblk[s0 - n_p:e0 - n_p]
        tokm.index_add_(0, qb, p.mean(0)); cnt.index_add_(0, qb, torch.ones_like(qb, dtype=torch.float))
        per.index_add_(1, qb, torch.zeros(H, e0 - s0, nq + 1, device=q.device).index_add_(2, kseg, p)); del p
    c = cnt.clamp(min=1); tokm /= c[:, None]; per /= c[None, :, None]
    gm = torch.zeros(nq, nq + 1, device=q.device).index_add_(1, kseg, tokm)
    inc = tokm[:, n_p:].sum(0); k = min(4, inc.numel()); tv, ti = inc.topk(k); sink = ti[tv > 10 * inc.median()]
    tm2 = tokm.clone(); tm2[:, n_p + sink] = 0; gns = torch.zeros(nq, nq + 1, device=q.device).index_add_(1, kseg, tm2)
    R = per[:, :, 1:]; qi = torch.arange(nq, device=q.device); d = (qi[:, None] - qi[None, :]).float(); low = d > 0
    W = R * low; s = W.sum(-1); ok = (s > 0) & (qi[None, :] >= 1)
    jump = (W * d).sum(-1) / s.clamp(min=1e-12); far = (W * (d > 20)).sum(-1) / s.clamp(min=1e-12)
    setup = (W * ((d > 20) & (qi[None, :] < 4))).sum(-1) / s.clamp(min=1e-12)
    summ = torch.stack([(x * ok).sum(-1) / ok.sum(-1).clamp(min=1) for x in (jump, far, setup)], -1)
    _G["res"][li] = (gm.half().cpu(), gns.half().cpu(), (sink - 0).cpu().numpy(), summ.cpu())


def _g_sdpa(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kw):
    out = _G["orig"](module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kw)
    if _G["on"] and getattr(module, "layer_idx", None) in _G["layers"]:
        _g_layer(module.layer_idx, query, key, scaling if scaling is not None else query.shape[-1] ** -0.5, getattr(module, "sliding_window", None))
    return out


def graph_pass(model, ctx, prompt, trace):
    if "orig" not in _G:
        from transformers import AttentionInterface
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        _G["orig"] = sdpa_attention_forward; AttentionInterface.register("sdpa", _g_sdpa)
    n_p = len(prompt); blk = np.arange(len(trace)) // 64; nq = int(blk.max()) + 1; dev = model.device
    _G.update(on=True, res={}, n_p=n_p, nq=nq, qblk=torch.tensor(blk, device=dev), kseg=torch.tensor(np.r_[np.zeros(n_p, np.int64), 1 + blk], device=dev),
              layers=frozenset(i for i, l in enumerate(ctx["layers"]) if getattr(l, "self_attn", None) is not None))
    FIN, hk = {}, []
    hk.append(ctx["norm"].register_forward_pre_hook(lambda m, a: FIN.__setitem__("h", a[0][0, n_p - 1:-1].detach())))
    try:
        with torch.no_grad():
            SI.forward_last_logits(model, input_ids=torch.tensor([prompt + trace], device=dev), use_cache=False)
    finally:
        _G["on"] = False
        for h in hk: h.remove()
    ls = sorted(_G["res"]); R = _G["res"]; _G["res"] = {}
    out = dict(attn_layers=np.array(ls, np.int16), graph=torch.stack([R[l][0] for l in ls]).numpy(), graph_nosink=torch.stack([R[l][1] for l in ls]).numpy(),
               head_summary=torch.stack([R[l][3] for l in ls]).numpy().astype(np.float32),
               sinks=np.array([np.pad(R[l][2][:4], (0, 4 - len(R[l][2][:4])), constant_values=-1) for l in ls], np.int32))
    out.update(crowding(model, ctx, FIN["h"], trace))
    return out


def crowding(model, ctx, h, trace, K=100, chunk=64):
    """Embedding-space crowding (arXiv 2601.22536, Def. 3.1-3.2) of the next-token distribution at every reasoning position (the state that
    predicts trace token t), top-K tokens, input and output embeddings; and the entropy of the full distribution."""
    head, norm = model.get_output_embeddings(), ctx["norm"]; Ein = model.get_input_embeddings().weight; Eout = head.weight
    cin, cout, ent = [], [], []
    with torch.no_grad():
        for s in range(0, h.shape[0], chunk):
            lg = head(norm(h[s:s + chunk].to(head.weight.dtype))).float(); lp = torch.log_softmax(lg, -1); p = lp.exp()
            ent.append(-(p * lp).sum(-1)); pk, ik = p.topk(K, -1)
            for E, acc in ((Ein, cin), (Eout, cout)):
                e = torch.nn.functional.normalize(E[ik].float(), dim=-1); C = (e @ e.transpose(1, 2)).abs()
                C = C - torch.diag_embed(torch.diagonal(C, dim1=1, dim2=2))
                acc.append((pk * (C @ pk[..., None])[..., 0]).sum(-1))
            del lg, lp, p
    return dict(crowd_in=torch.cat(cin).cpu().numpy().astype(np.float16), crowd_out=torch.cat(cout).cpu().numpy().astype(np.float16),
                entropy=torch.cat(ent).cpu().numpy().astype(np.float16))


def graph_stage(model, tok, ctx, rows, odir):
    gdir = odir / "graphs"; gdir.mkdir(exist_ok=True); summ = odir / "graph_rows.jsonl"
    done = {json.loads(l)["key"] for l in open(summ)} if summ.exists() else set()
    jobs = []
    for r in rows:
        jobs.append((r, "greedy", r["greedy_gen_ids"][:6144]))
        for k, txt in enumerate(r.get("sample_texts", [])):
            jobs.append((r, f"t1s{k}", tok(txt, add_special_tokens=False)["input_ids"][:6144]))
    todo = [j for j in jobs if f"{j[0]['id']}__{j[1]}" not in done]; log(f"  graphs: {len(jobs)} traces, {len(todo)} to do")
    for n, (r, nm, gen) in enumerate(todo):
        key = f"{r['id']}__{nm}"; t = time.time(); te = find_think_end(tok, gen, ctx["tid"]); c = min(max(1, te if te is not None else len(gen)), 6144)
        try:
            out = graph_pass(model, ctx, r["prompt_ids"], gen[:c])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); log(f"    OOM on {key} ({c} tokens) - retrying on the first 4096 | {gpu()}"); c = min(c, 4096)
            out = graph_pass(model, ctx, r["prompt_ids"], gen[:c])
        np.savez_compressed(gdir / f"{key}.npz", **out)
        hs = out["head_summary"]; mid = hs[len(hs) // 3:2 * len(hs) // 3].mean((0, 1))
        rec = dict(key=key, id=r["id"], trace=nm, dataset=r["dataset"], robust=r["robust"], cut=c, think_end=te, mid_jump=float(mid[0]), mid_far=float(mid[1]),
                   mid_setup=float(mid[2]), crowd_in=float(out["crowd_in"].astype(np.float32).mean()), crowd_out=float(out["crowd_out"].astype(np.float32).mean()),
                   entropy=float(out["entropy"].astype(np.float32).mean()), seconds=round(time.time() - t, 1))
        with open(summ, "a") as f: f.write(json.dumps(rec) + "\n")
        log(f"    [{n + 1}/{len(todo)}] {key}: {c} tok | mid jump {rec['mid_jump']:.2f} far {rec['mid_far']:.3f} setup {rec['mid_setup']:.3f} | "
            f"crowding in {rec['crowd_in']:.3f} out {rec['crowd_out']:.3f} | {rec['seconds']}s | {gpu()}")
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=""); ap.add_argument("--out", default=str(HERE / "out_graphs"))
    ap.add_argument("--graph_inputs", default=str(HERE / "graph_inputs.jsonl")); a = ap.parse_args()
    grows = [json.loads(l) for l in open(a.graph_inputs)]
    models = [m for m in ORDER if not a.models or m in a.models.split(",")]
    import transformers
    log(f"models {models} | {len(grows)} graph rows | torch {torch.__version__} transformers {transformers.__version__} | {gpu()}")
    prefetch(ORDER[ORDER.index(models[-1]) + 1:ORDER.index(models[-1]) + 2] if models else [])   # the next model, while this one runs
    for m in models:
        odir = Path(a.out) / m.replace("/", "__"); odir.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import snapshot_download
        snapshot_download(m)
        model, tok = SI.load_model_and_tokenizer(m, local_files_only=True); layers, n_layers = SI.find_decoder_layers(model); tid = think_end_token_id(tok)
        has_think = tid is not None or "<think>" in build_prompt(tok, "x")[1]
        sfx = {k: tok(v, add_special_tokens=False)["input_ids"] for k, v in (SUFFIXES_THINK if has_think else SUFFIXES_NOTHINK).items()}
        ctx = dict(layers=layers, n_layers=n_layers, tid=tid, suffixes=sfx, norm=final_norm(model))
        log(f"=== {m}: {n_layers} layers | final norm {'found' if ctx['norm'] is not None else 'MISSING'} | {gpu()}")
        T0 = time.time()
        try:
            graph_stage(model, tok, ctx, [r for r in grows if r["model_id"] == m], odir)
        except Exception:                                                       # noqa: BLE001 - log, keep going with the next model
            log(f"FAILED {m}:\n{traceback.format_exc()}")
        log(f"=== {m} done in {(time.time() - T0) / 60:.1f} min")
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
