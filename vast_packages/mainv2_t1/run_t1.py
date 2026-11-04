"""T=0 vs T=1 run (Vast): for every row of inputs.jsonl (main_v2 + labelled olympiad), the greedy trace (given, teacher forced) and
K sampled traces at the label-generation settings (temperature 1.0, top_k 40, top_p 0.95), all capped at 6144 generated tokens.

Per trace it saves
  - gen_ids, stop reason, think end, decoded text;
  - the submission's features, computed by the submission's own code (spurious_features / spurious_inference): forced-answer top-1
    probability and entropy for suffixes A and B at every cutoff of cutoff_labels(L), A/B shift, kNN global efficiency, TDA H0 spread,
    segment PC1 share, late-vs-early cosine, PC1 change 75%/25%, check share, length;
  - for suffix B at every cutoff: the top-5 answer tokens and a 12-token greedy continuation (the forced answer string, for agreement
    between greedy and samples);
  - per generated token: entropy of the model's (untempered) next-token distribution and the log-prob of the token actually produced;
  - the 16-token window states (float16, all 6 layers), so any lower cap can be recomputed exactly offline;
  - main_v2's 'tf' / 'force' extras with the same definitions as extract_features.py: per-token top-1 prob, problem-token log-prob
    and entropy, last prompt state at every layer, mean problem state, states at each cutoff (last token, mean of last 64), and the
    hidden states at the last forced-suffix token per cutoff and suffix;
  - versus the greedy trace: the first token where the sample departs, and the per-window cosine of layer-3 states.

Usage (one process per GPU; run.sh does this):
  CUDA_VISIBLE_DEVICES=0 python run_t1.py --inputs inputs.jsonl --out out_t1 --shard 0 --nshard 4
Resumable: rows already in <out>/<model>/rows_shard<i>.jsonl are skipped."""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import threading
import time
import traceback
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "submission_core"))

import numpy as np
import torch
from transformers import GenerationConfig, StoppingCriteriaList

import spurious_inference as SI
from spurious_features import (BUDGETS, SEL_LAYER_FRACS, SUFFIXES_NOTHINK, SUFFIXES_THINK, TRAJ_LAYERS, build_prompt, cutoff_labels,
                               f_check_share, f_forced_saturated_share, f_forced_slope, f_global_eff, f_late_vs_early, f_pc1_change,
                               f_seg25_pc1, f_tda_h0_spread, find_think_end, think_end_token_id)

LoopStoppingCriteria = SI.LoopStoppingCriteria
SHIFT_BUDGETS = (512, 1024, 2048, 3072, 4096, 6144)
T0 = time.time()


def log(msg: str) -> None:
    print(f"[t1 {time.time() - T0:8.1f}s] {msg}", flush=True)


def gpu() -> str:
    if not torch.cuda.is_available():
        return "no cuda"
    f, t = torch.cuda.mem_get_info()
    return f"{torch.cuda.get_device_name(0)} {(t - f) / 2**30:.1f}/{t / 2**30:.1f} GiB used, peak {torch.cuda.max_memory_allocated() / 2**30:.1f}"


def prefetch(model_ids):
    """Download the weights of the models still to come in the background, one after another."""
    def run():
        from huggingface_hub import snapshot_download
        for m in model_ids:
            try:
                t = time.time(); snapshot_download(m); log(f"prefetch {m} ready ({time.time() - t:.0f}s)")
            except Exception as e:                                  # noqa: BLE001 - the main loop downloads it again if needed
                log(f"prefetch {m} failed: {e!r}")
    th = threading.Thread(target=run, daemon=True); th.start(); return th


def eos_ids(model, tok):
    g = model.generation_config.eos_token_id
    e = set([g] if isinstance(g, int) else (g or [])) | {tok.eos_token_id}
    e.discard(None)
    return sorted(e)


def sample_traces(model, tok, prompt, k, cap, seed):
    """K sampled traces of one prompt in one generate call (identical prompts -> no padding); per-row stop reasons as in the
    submission's generate_batch. A fresh GenerationConfig so the checkpoints' own recommended sampling defaults do not leak in."""
    eos = eos_ids(model, tok)
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos[0]
    cfg = GenerationConfig(do_sample=True, temperature=1.0, top_k=40, top_p=0.95, max_new_tokens=cap, eos_token_id=eos, pad_token_id=pad)
    res, todo, chunk = [], k, k
    while todo:
        n = min(chunk, todo)
        try:
            ids = torch.tensor([prompt] * n, device=model.device)
            stopper = LoopStoppingCriteria(len(prompt), n)
            torch.manual_seed(seed + len(res))
            with torch.no_grad():
                out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=cfg,
                                     stopping_criteria=StoppingCriteriaList([stopper]))
        except torch.cuda.OutOfMemoryError:
            if chunk == 1:
                raise
            oom = True
        else:
            oom = False
        if oom:
            torch.cuda.empty_cache(); chunk = max(1, chunk // 2); log(f"    OOM sampling {n} rows -> retry in chunks of {chunk} | {gpu()}"); continue
        for r in range(n):
            g = out[r, len(prompt):].tolist(); lim = stopper.stop_len[r]
            e = next((j for j, t in enumerate(g[:lim]) if t in eos), None)
            if e is not None:
                res.append((g[:e + 1], "eos"))
            elif lim is not None:
                res.append((g[:lim], "loop"))
            else:
                res.append((g[:cap], "cap"))
        todo -= n
        del out; torch.cuda.empty_cache()
    return res


def final_norm(model):
    """The norm applied right before the LM head (the last '*norm' module outside the decoder layers)."""
    cand = [(n, m) for n, m in model.named_modules() if n.split(".")[-1] in ("norm", "final_layernorm", "ln_f") and "layers" not in n]
    return cand[-1][1] if cand else None


def tf_pass(model, ctx, prompt, gen, span, cuts, chunk=None, logit_chunk=512):
    """One teacher-forced pass over prompt + gen with forward hooks; returns (ws [n_win, 6, d] exactly as the submission's
    window_states, dict of main_v2 'tf' extras with the same names and definitions as extract_features.py:
      gen_logprob / gen_entropy / gen_top1   per generated token (untempered next-token distribution)
      problem_logprob / problem_entropy      per problem token
      prompt_last_all_layers [n_layers, d]   last prompt token at every layer
      cut_last / cut_mean64 [n_cut, 6, d]    sel layers at each cutoff: last token, mean of the last 64 generated tokens
      problem_mean [6, d]                    sel layers, mean over the problem tokens).
    chunk=None runs the whole sequence in one forward (what the submission does); on CUDA OOM it is rerun in 2048-token chunks over a
    KV cache, so no T x T attention buffer is ever needed."""
    n_p, n_g = len(prompt), len(gen); full = prompt + gen; T = n_p + n_g
    keep = sorted(set(ctx["traj_layers"]) | set(ctx["sel_layers"])); cap = {"seq": {l: [] for l in keep}, "last": {}, "norm": []}
    state = {"start": 0}

    def hook(l):
        def fn(_m, _i, o):
            h = (o[0] if isinstance(o, tuple) else o)[0]; s0 = state["start"]
            if s0 <= n_p - 1 < s0 + h.shape[0]:
                cap["last"][l] = h[n_p - 1 - s0].detach().clone()
            if l in cap["seq"]:
                cap["seq"][l].append(h.detach().clone())      # native dtype (bf16 on the GPU)
        return fn

    def norm_hook(_m, _i, o):
        cap["norm"].append((o[0] if isinstance(o, tuple) else o)[0].detach().clone())

    hs = [ctx["layers"][l].register_forward_hook(hook(l)) for l in range(ctx["n_layers"])]
    if ctx["norm"] is not None:
        hs.append(ctx["norm"].register_forward_hook(norm_hook))
    try:
        with torch.no_grad():
            if chunk is None:
                SI.forward_last_logits(model, input_ids=torch.tensor([full], device=model.device), use_cache=False)
            else:
                cache = None
                for s0 in range(0, T, chunk):
                    state["start"] = s0
                    o = SI.forward_last_logits(model, input_ids=torch.tensor([full[s0:s0 + chunk]], device=model.device),
                                               past_key_values=cache, use_cache=True)
                    cache = o.past_key_values; del o
                del cache
    finally:
        for h in hs:
            h.remove()
    seq = {l: torch.cat(v, 0) for l, v in cap["seq"].items()}        # [T, d] per kept layer
    ws = []
    for l in ctx["traj_layers"]:                                     # identical to spurious_inference.window_states
        g = seq[l][n_p:n_p + n_g].float(); k = n_g // 16
        parts = [g[:k * 16].view(k, 16, -1).mean(1)] if k else []
        if n_g % 16:
            parts.append(g[k * 16:].mean(0, keepdim=True))
        ws.append(torch.cat(parts, 0).cpu())
    ws = torch.stack(ws, 1).numpy()
    h16 = lambda t: t.to(torch.float16).cpu().numpy()                 # noqa: E731
    sel = ctx["sel_layers"]; cpos = [n_p + c - 1 for c in cuts]
    ex = dict(cut_eff=np.array(cuts, np.int32), sel_layers=np.array(sel, np.int16),
              prompt_last_all_layers=h16(torch.stack([cap["last"][l] for l in range(ctx["n_layers"])])),
              cut_last=h16(torch.stack([torch.stack([seq[l][p].float() for l in sel]) for p in cpos])),
              cut_mean64=h16(torch.stack([torch.stack([seq[l][max(n_p, p - 63):p + 1].float().mean(0) for l in sel]) for p in cpos])),
              problem_mean=h16(torch.stack([seq[l][span[0]:span[1]].float().mean(0) for l in sel])) if span else None)
    del seq
    if cap["norm"]:
        H = torch.cat(cap["norm"], 0); head = model.get_output_embeddings()

        def stats(start, targets):
            lps, ents, tops = [], [], []
            tg = torch.tensor(targets, device=H.device)
            with torch.no_grad():
                for s0 in range(0, len(targets), logit_chunk):
                    lp = torch.log_softmax(head(H[start + s0:start + s0 + logit_chunk][: len(targets) - s0]).float(), -1); p_ = lp.exp()
                    lps.append(lp.gather(1, tg[s0:s0 + logit_chunk, None])[:, 0]); ents.append(-(p_ * lp).sum(-1)); tops.append(p_.max(-1).values)
                    del lp, p_
            return [torch.cat(x).cpu().numpy() for x in (lps, ents, tops)]
        ex["gen_logprob"], ex["gen_entropy"], ex["gen_top1"] = stats(n_p - 1, gen)
        if span and span[1] - span[0] > 1:
            ex["problem_logprob"], ex["problem_entropy"], _ = stats(span[0] - 1, prompt[span[0]:span[1]])
        del H
    cap.clear(); torch.cuda.empty_cache()
    return ws, {k: v for k, v in ex.items() if v is not None}


def tf_pass_safe(model, ctx, prompt, gen, span, cuts):
    """Full pass (as the submission); on CUDA OOM retried over a KV cache in ever smaller chunks (models whose attention falls back to
    a kernel that materialises query x key scores, e.g. Olmo-3's sliding-window layers, need this on 24 GB cards)."""
    for chunk in (None, 1024, 256):
        try:
            return tf_pass(model, ctx, prompt, gen, span, cuts, chunk=chunk), ("full" if chunk is None else f"chunked {chunk}")
        except torch.cuda.OutOfMemoryError:
            if chunk == 256:
                raise
        torch.cuda.empty_cache()                                   # outside the except: the failed pass's tensors are released by now
        log(f"    OOM in the teacher-forced pass ({'full' if chunk is None else chunk}) -> smaller chunks | {gpu()}")


def forced_safe(model, tok, prompt, gen, cuts, suffixes, sel_layers):
    """forced() with its prefix prefill in 1024-token chunks, retried at 256 on CUDA OOM."""
    for chunk in (1024, 256):
        try:
            return forced(model, tok, prompt, gen, cuts, suffixes, sel_layers, prefill=chunk)
        except torch.cuda.OutOfMemoryError:
            if chunk == 256:
                raise
        torch.cuda.empty_cache()
        log(f"    OOM in the forced answers (prefill {chunk}) -> 256 | {gpu()}")


def forced(model, tok, prompt, gen, cuts, suffixes, sel_layers, cont=12, prefill=1024):
    """{cut: {suffix: {p1, entropy, top5_ids, top5_p, [continuation for B]}}}: the prefix cache is extended cut by cut and copied per
    suffix (as the submission's forced_answers); for suffix B, `cont` greedy tokens after the suffix give the forced answer string.
    Also returns {cut: {suffix: [6, d]}}: sel-layer hidden states at the last suffix token (main_v2's force/<id>.npz)."""
    full = prompt + gen; n_p = len(prompt); cache, pos, out_, hid = None, 0, {}, {}
    with torch.no_grad():
        for c in sorted(set(cuts)):
            target = n_p + c
            for s in range(pos, target, prefill):
                o = SI.forward_last_logits(model, input_ids=torch.tensor([full[s:min(target, s + prefill)]], device=model.device),
                                           past_key_values=cache, use_cache=True)
                cache = o.past_key_values
            pos = target; out_[c] = {}; hid[c] = {}
            for name, sids in suffixes.items():
                try:
                    branch = copy.deepcopy(cache)
                    o = SI.forward_last_logits(model, input_ids=torch.tensor([sids], device=model.device), past_key_values=branch,
                                               use_cache=True, output_hidden_states=True)
                except (RuntimeError, TypeError, AttributeError):
                    o = SI.forward_last_logits(model, input_ids=torch.tensor([full[:target] + sids], device=model.device), use_cache=True,
                                               output_hidden_states=True)
                hid[c][name] = torch.stack([o.hidden_states[l + 1][0, -1] for l in sel_layers]).to(torch.float16).cpu().numpy()
                lp = torch.log_softmax(o.logits[0, -1].float(), -1); p = lp.exp(); top = torch.topk(p, 5)
                rec = dict(p1=float(top.values[0]), entropy=float(-(p * lp).sum()), top5_ids=top.indices.tolist(),
                           top5_p=[float(v) for v in top.values])
                if name == "B" and cont:
                    toks, kv, nxt = [], o.past_key_values, int(top.indices[0])
                    for _ in range(cont):
                        toks.append(nxt)
                        o2 = SI.forward_last_logits(model, input_ids=torch.tensor([[nxt]], device=model.device), past_key_values=kv, use_cache=True)
                        kv, nxt = o2.past_key_values, int(o2.logits[0, -1].argmax())
                    rec["continuation_ids"] = toks; rec["continuation"] = tok.decode(toks, skip_special_tokens=True)
                    del kv
                out_[c][name] = rec
                del o
    del cache; torch.cuda.empty_cache()
    return out_, hid


def trace_record(model, tok, ctx, prompt, gen, stop, span=None, prompt_extras=True):
    """Everything for one trace (see the module docstring)."""
    t = time.time(); te = find_think_end(tok, gen, ctx["tid"]); L = max(1, te if te is not None else len(gen))
    text = tok.decode(gen, skip_special_tokens=True)
    rec = dict(n_gen=len(gen), stop=stop, think_end=te, L=L, text=text, gen_ids=gen,
               chk=f_check_share(text), len=-float(np.log(max(1, len(gen)))))
    lab = cutoff_labels(L); lab = {k: v for k, v in lab.items() if not (k.startswith("b") and int(k[1:]) > 6144)}
    cuts = sorted(set(lab.values()))
    (ws, ex), rec["tf_mode"] = tf_pass_safe(model, ctx, prompt, gen, span, cuts)
    rec.update(seg=f_seg25_pc1(ws, L), traj=f_late_vs_early(ws), ge=f_global_eff(ws[:, 3], ctx["mu"], L) if ctx["mu"] is not None else None,
               tda=f_tda_h0_spread(ws[:, 3], L), pcr=f_pc1_change(ws))
    fa, fh = forced_safe(model, tok, prompt, gen, cuts, ctx["suffixes"], ctx["sel_layers"])
    rec["cuts"] = lab; rec["forced"] = {k: fa[c] for k, c in lab.items()}
    p1s = [fa[lab[f"b{b}"]]["B"]["p1"] for b in SHIFT_BUDGETS]
    rec.update(fa=fa[lab["b6144"]]["B"]["p1"], fas=f_forced_slope(p1s), fsat=f_forced_saturated_share(p1s),
               ab=max(abs(fa[c]["A"]["entropy"] - fa[c]["B"]["entropy"]) for c in lab.values()))
    ent, lp = ex.get("gen_entropy"), ex.get("gen_logprob")
    # full-resolution 16-token window states, all 6 layers: any lower cap c is the first ceil(c / 16) windows, so every feature can be
    # recomputed exactly offline at a smaller budget
    arrays = dict(ws16=ws.astype(np.float16),
                  tok_entropy=(ent if ent is not None else np.zeros(0)).astype(np.float16),
                  tok_logprob=(lp if lp is not None else np.zeros(0)).astype(np.float16), gen_ids=np.array(gen, np.int32),
                  tok_top1=ex.get("gen_top1", np.zeros(0)).astype(np.float16), cut_eff=ex["cut_eff"], cut_last=ex["cut_last"],
                  cut_mean64=ex["cut_mean64"], force_suffixes=np.array(sorted(ctx["suffixes"])),
                  force_hidden=np.stack([np.stack([fh[c][v] for v in sorted(ctx["suffixes"])]) for c in cuts]))   # [n_cut, 2, 6, d]
    if prompt_extras:                                                 # prompt-only: identical for every trace of a row, saved once
        arrays.update({k: ex[k] for k in ("prompt_last_all_layers", "problem_mean", "problem_logprob", "problem_entropy") if k in ex})
    rec["seconds"] = round(time.time() - t, 1)
    if ent is not None:
        rec.update(mean_tok_entropy=float(ent.mean()), mean_tok_logprob=float(lp.mean()))
    return rec, arrays


def versus_greedy(g_gen, g_ws3, s_gen, s_ws3):
    """First token where the sample departs from the greedy trace, and the per-16-token-window cosine of their layer-3 states."""
    n = min(len(g_gen), len(s_gen)); d = next((i for i in range(n) if g_gen[i] != s_gen[i]), n)
    m = min(len(g_ws3), len(s_ws3)); a, b = g_ws3[:m].astype(np.float32), s_ws3[:m].astype(np.float32)
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
    return d, cos.astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=str(HERE / "inputs.jsonl")); ap.add_argument("--out", default=str(HERE / "out_t1"))
    ap.add_argument("--k", type=int, default=2, help="sampled traces per row (0: greedy features only)"); ap.add_argument("--cap", type=int, default=6144); ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--models", default="", help="comma-separated subset of model ids (default: all in the inputs)")
    ap.add_argument("--datasets", default="main_v2,olympiad")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.inputs)]
    rows = [r for i, r in enumerate(rows) if i % a.nshard == a.shard and r["dataset"] in a.datasets.split(",")]
    order = ["Qwen/Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B", "allenai/Olmo-3-7B-Think", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"]
    models = [m for m in order if any(r["model_id"] == m for r in rows) and (not a.models or m in a.models.split(","))]
    import transformers
    log(f"shard {a.shard}/{a.nshard}: {len(rows)} rows, models {models} | K={a.k} cap={a.cap} | torch {torch.__version__} "
        f"transformers {transformers.__version__} | {gpu()}")
    if a.shard == 0 and models:                                   # one downloader for every shard (the others wait on its files);
        allm = [m for m in order if any(r["model_id"] == m for r in rows)]   # with --models (one process per model, run.sh) it also
        prefetch(allm[allm.index(models[0]):])                    # fetches the later models; an interrupted download resumes
    for m in models:
        mrows = [r for r in rows if r["model_id"] == m]
        odir = Path(a.out) / m.replace("/", "__"); odir.mkdir(parents=True, exist_ok=True)
        summ = odir / f"rows_shard{a.shard}.jsonl"
        done = {json.loads(l)["id"] for l in open(summ)} if summ.exists() else set()
        todo = [r for r in mrows if r["id"] not in done]
        log(f"=== {m}: {len(mrows)} rows in shard, {len(done)} already done, {len(todo)} to do")
        if not todo:
            continue
        from huggingface_hub import snapshot_download
        snapshot_download(m)
        model, tok = SI.load_model_and_tokenizer(m, local_files_only=True)
        layers, n_layers = SI.find_decoder_layers(model)
        sel = sorted({min(n_layers - 1, max(0, int(round(f * n_layers)) - 1)) for f in SEL_LAYER_FRACS})
        tid = think_end_token_id(tok)
        has_think = tid is not None or "<think>" in build_prompt(tok, "x")[1]
        sfx = {k: tok(v, add_special_tokens=False)["input_ids"] for k, v in (SUFFIXES_THINK if has_think else SUFFIXES_NOTHINK).items()}
        mu_p = HERE / "mu" / f"mu_{m.replace('/', '__')}.npy"
        ctx = dict(layers=layers, n_layers=n_layers, sel_layers=sel, traj_layers=[l for l in TRAJ_LAYERS.get(m, []) if l < n_layers] or sel, tid=tid, suffixes=sfx,
                   mu=np.load(mu_p) if mu_p.exists() else None, norm=final_norm(model))
        log(f"{m}: {n_layers} layers, traj layers {ctx['traj_layers']}, final norm {'found' if ctx['norm'] is not None else 'MISSING'}, "
            f"mu {'loaded' if ctx['mu'] is not None else 'MISSING'}, model generation_config {model.generation_config.to_diff_dict()} | {gpu()}")
        for j, r in enumerate(todo):
            t_row = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()                 # the row log's 'peak' is this row's
            try:
                p_build = build_prompt(tok, r["problem"])[0]
                same = p_build == r["prompt_ids"]
                greedy = r["greedy_gen_ids"][: a.cap]
                seed = a.seed + zlib.crc32(r["id"].encode()) % 100000
                t = time.time(); samples = sample_traces(model, tok, p_build, a.k, a.cap, seed); t_gen = time.time() - t
                span = build_prompt(tok, r["problem"])[2] if same else None    # problem-token span (needs the rebuilt ids)
                g_rec, g_arr = trace_record(model, tok, ctx, r["prompt_ids"], greedy, "given", span)
                recs, arrs = [g_rec], {f"t0_{k}": v for k, v in g_arr.items()}
                for s, (gen, stop) in enumerate(samples):
                    rec, arr = trace_record(model, tok, ctx, p_build, gen, stop, span, prompt_extras=False)
                    d, cos = versus_greedy(greedy, g_arr["ws16"][:, 3], gen, arr["ws16"][:, 3])
                    rec.update(depart_token=d, mean_cos_vs_greedy=float(np.mean(cos.astype(np.float32))) if len(cos) else None)
                    recs.append(rec); arrs.update({f"t1s{s}_{k}": v for k, v in arr.items()}); arrs[f"t1s{s}_cos_vs_greedy"] = cos
                for x in recs:                                    # the heavy fields live in the npz
                    x.pop("gen_ids", None)
                np.savez_compressed(odir / f"{r['id']}.npz", **arrs)
                line = dict(id=r["id"], dataset=r["dataset"], model_id=m, robust=r["robust"], prompt_rebuild_matches=same, seed=seed,
                            sample_seconds=round(t_gen, 1), seconds=round(time.time() - t_row, 1), greedy=recs[0], samples=recs[1:])
                with open(summ, "a") as f:
                    f.write(json.dumps(line) + "\n")
                log(f"  [{j + 1}/{len(todo)}] {r['id']} ({r['dataset']}, robust={r['robust']}) prompt match {same} | sampling {t_gen:.0f}s, "
                    f"row {time.time() - t_row:.0f}s | stops {[s for _, s in samples]} lens {[len(g) for g, _ in samples]} | "
                    f"fa greedy {g_rec['fa']:.3f} samples {[round(x['fa'], 3) for x in recs[1:]]} | "
                    f"answer@6144 greedy {g_rec['forced']['b6144']['B'].get('continuation', '')!r} samples "
                    f"{[x['forced']['b6144']['B'].get('continuation', '') for x in recs[1:]]} | {gpu()}")
            except Exception:                                      # noqa: BLE001 - log and move on; the row is retried on resume
                log(f"  [{j + 1}/{len(todo)}] {r['id']} FAILED:\n{traceback.format_exc()}")
                with open(odir / f"failures_shard{a.shard}.jsonl", "a") as f:
                    f.write(json.dumps(dict(id=r["id"], error=traceback.format_exc())) + "\n")
                torch.cuda.empty_cache()
        # free EVERY reference to this model before the next loads (ctx holds its layers and norm: a 7B + an 8B do not fit in 24 GB)
        del model, ctx, layers
        import gc
        gc.collect(); torch.cuda.empty_cache(); log(f"{m} released | {gpu()}")
    log("shard finished")


if __name__ == "__main__":
    main()
