"""Recall direction + reasoning-graph checks + embedding-space crowding (Vast, one process per model; run.sh). No long generation.

R  recall vs thinking (GSM-Symbolic, recall_pairs.jsonl: 100 GSM8K originals + 3 templated variants each). Per prompt:
     - the answer forced with NO reasoning (prompt + "</think> The final answer is \\boxed{"), greedy, parsed; correct vs the item's gold;
     - hidden states at every layer: last prompt token (H_last), forced-answer position (H_ans), mean over the first 64 greedily generated
       reasoning tokens (H_gen); logit lens at the answer position (rank / prob of the gold answer's first token per layer).
R2 direction, on the box: recall pairs = (original, variant) where the original is answered right with no reasoning and the variant wrong.
     Per layer and position type: difference of means (unit vector); 5-fold grouped CV AUC; best layer per type (layer 0 excluded).
     Fallback if < 8 pairs: all originals vs all variants (logged). Causal checks at the best H_ans layer: steering the variants along the
     direction (alpha x the mean projection gap, all positions): share answering the ORIGINAL's answer; projecting it out of the originals:
     no-reasoning accuracy.
G  for every labelled row (the white-box run's 118 greedy traces, cut at end of thinking / 6144) and the main_v2 T=1 samples (2 per row,
     re-tokenised from text): one teacher-forced pass with
     - reasoning graphs at EVERY attention layer: 64-token blocks, keys = prompt (node 0) + reasoning blocks; head-mean graph per layer;
       the same graph with the layer's attention sinks removed (reasoning tokens receiving > 10x the median incoming attention, top 4);
       per head and layer: mean backward jump, share > 20 blocks back, share on the first 4 blocks (when > 20 back);
     - projection of every reasoning token's residual (all layers) on the R2 directions (H_gen and H_last types);
     - embedding-space crowding (arXiv 2601.22536): per position, top-100 next-token distribution, step crowding
       sum_i p_i sum_{j!=i} p_j |cos(e_i, e_j)| with input embeddings and with output (unembedding) rows; plus entropy for comparison.
Outputs <out>/<model>/: recall.jsonl + recall.npz, direction.npz + direction.json, graphs/<id>__<trace>.npz, graph_rows.jsonl (resumable)."""
from __future__ import annotations

import argparse, json, re, sys, time, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "submission_core"))
import numpy as np
import torch

import spurious_inference as SI
from spurious_features import SUFFIXES_NOTHINK, SUFFIXES_THINK, build_prompt, find_think_end, think_end_token_id
from run_t1 import gpu, log, prefetch, final_norm
from run_whitebox import continuation

ORDER = ["Qwen/Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B", "allenai/Olmo-3-7B-Think", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"]
POS_TYPES = ("last", "ans", "gen")


# ----------------------------------------------------------------------------------------------------------------- R: recall prompts
def parse_answer(s):
    s = s.split("}")[0].replace(",", "").replace("$", "").replace("\\%", "").replace("%", "").strip()
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return m.group(0) if m else s


def same(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return a.strip() == b.strip()


def recall_stage(model, tok, ctx, pairs, odir):
    head, norm = model.get_output_embeddings(), ctx["norm"]; sfx = ctx["suffixes"]["B"]; recs, HL, HA, HG, LR, LP = [], [], [], [], [], []
    for j, p in enumerate(pairs):
        t = time.time(); pids = build_prompt(tok, p["question"])[0]; ids = pids + sfx
        with torch.no_grad():
            o = model(input_ids=torch.tensor([ids], device=model.device), output_hidden_states=True, use_cache=True, logits_to_keep=1)
            hs = o.hidden_states; top = int(o.logits[0, -1].argmax())
            HL.append(torch.stack([h[0, len(pids) - 1] for h in hs]).float().cpu().half()); ha = torch.stack([h[0, -1] for h in hs])
            HA.append(ha.float().cpu().half())
            gid = tok(p["gold"], add_special_tokens=False)["input_ids"][0]
            lg = torch.log_softmax(head(norm(ha.to(head.weight.dtype))).float(), -1)
            LR.append((lg > lg[:, gid:gid + 1]).sum(-1).cpu().numpy().astype(np.int32)); LP.append(lg[:, gid].exp().cpu().numpy().astype(np.float32))
            ans = continuation(model, tok, o.past_key_values, top, n=10); del o, hs
            g = model.generate(input_ids=torch.tensor([pids], device=model.device), max_new_tokens=64, do_sample=False, temperature=None, top_p=None, top_k=None)
            o2 = model(input_ids=g, output_hidden_states=True, use_cache=False, logits_to_keep=1)
            HG.append(torch.stack([h[0, len(pids):].float().mean(0) for h in o2.hidden_states]).cpu().half()); del o2
        a = parse_answer(ans); rec = dict(key=p["key"], pid=p["pid"], kind=p["kind"], gold=p["gold"], answer_raw=ans[:40], answer=a, correct=same(a, p["gold"]))
        if p["kind"] == "variant":
            rec["says_original_answer"] = same(a, p["original_gold"])
        recs.append(rec)
        if j % 50 == 0 or j == len(pairs) - 1:
            log(f"  recall {j + 1}/{len(pairs)} {p['key']}: '{ans[:20]!r}' -> {a} (gold {p['gold']}) {'ok' if rec['correct'] else 'x'} | {time.time() - t:.1f}s | {gpu()}")
    np.savez_compressed(odir / "recall.npz", H_last=torch.stack(HL).numpy(), H_ans=torch.stack(HA).numpy(), H_gen=torch.stack(HG).numpy(),
                        lens_rank=np.stack(LR), lens_p=np.stack(LP), keys=np.array([r["key"] for r in recs]))
    with open(odir / "recall.jsonl", "w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
    o_ok = np.mean([r["correct"] for r in recs if r["kind"] == "original"]); v_ok = np.mean([r["correct"] for r in recs if r["kind"] == "variant"])
    v_orig = np.mean([r.get("says_original_answer", False) for r in recs if r["kind"] == "variant"])
    log(f"  recall stage: no-reasoning accuracy originals {o_ok:.2f}, variants {v_ok:.2f}; variants answering the ORIGINAL's answer {v_orig:.2f}")


# ----------------------------------------------------------------------------------------------------------------- R2: direction
def auc(s, y):
    p, n = s[y == 1], s[y == 0]
    return float(((p[:, None] > n[None]) + .5 * (p[:, None] == n[None])).mean()) if len(p) and len(n) else float("nan")


def direction_stage(model, tok, ctx, pairs, odir):
    Z = np.load(odir / "recall.npz"); recs = [json.loads(l) for l in open(odir / "recall.jsonl")]; idx = {r["key"]: i for i, r in enumerate(recs)}
    orig = {r["pid"]: r for r in recs if r["kind"] == "original"}
    pos, neg = [], []
    for r in recs:
        if r["kind"] == "variant" and orig[r["pid"]]["correct"] and not r["correct"]:
            neg.append(idx[r["key"]]); pos.append(idx[orig[r["pid"]]["key"]])
    pos = sorted(set(pos)); mode = "filtered"
    if len(pos) < 8:
        pos = [i for i, r in enumerate(recs) if r["kind"] == "original"]; neg = [i for i, r in enumerate(recs) if r["kind"] == "variant"]; mode = "fallback all"
    sel = np.array(pos + neg); y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]; grp = np.array([recs[i]["pid"] for i in sel])
    ug = np.unique(grp); rng = np.random.default_rng(0); fold = {g: k % 5 for k, g in enumerate(rng.permutation(ug))}; fo = np.array([fold[g] for g in grp])
    D, info = {}, dict(mode=mode, n_pos=len(pos), n_neg=len(neg), cv_auc={}, best_layer={}, gap={})
    for t in POS_TYPES:
        X = Z[f"H_{t}"][sel].astype(np.float32); Lp = X.shape[1]; dirs = np.zeros((Lp, X.shape[2]), np.float32); cv = []
        for l in range(Lp):
            Xl = X[:, l]; s = np.zeros(len(sel))
            for k in range(5):
                tr, te = fo != k, fo == k
                if not te.any() or y[tr].min() == y[tr].max(): continue
                d = Xl[tr][y[tr] == 1].mean(0) - Xl[tr][y[tr] == 0].mean(0); s[te] = Xl[te] @ (d / (np.linalg.norm(d) + 1e-9))
            cv.append(auc(s, y)); d = Xl[y == 1].mean(0) - Xl[y == 0].mean(0); dirs[l] = d / (np.linalg.norm(d) + 1e-9)
        b = int(np.nanargmax(np.r_[[-1.0], cv[1:]])); D[t] = dirs
        info["cv_auc"][t] = [round(float(v), 3) for v in cv]; info["best_layer"][t] = b
        pr = X[:, b] @ dirs[b]; info["gap"][t] = float(pr[y == 1].mean() - pr[y == 0].mean())
        log(f"  direction {t}: best layer {b} CV AUC {cv[b]:.3f} (mode {mode}, {len(pos)} pos / {len(neg)} neg)")
    # causal checks at the best H_ans layer (hidden_states index b = output of decoder layer b - 1)
    b = info["best_layer"]["ans"]; u = torch.tensor(D["ans"][b], device=model.device); gap = info["gap"]["ans"]; layer = ctx["layers"][b - 1]
    sfx = ctx["suffixes"]["B"]; byk = {p["key"]: p for p in pairs}

    def forced(q, fn):
        h = layer.register_forward_hook(fn) if fn else None
        try:
            with torch.no_grad():
                ids = build_prompt(tok, q)[0] + sfx; o = model(input_ids=torch.tensor([ids], device=model.device), use_cache=True, logits_to_keep=1)
                return parse_answer(continuation(model, tok, o.past_key_values, int(o.logits[0, -1].argmax()), n=10))
        finally:
            if h: h.remove()

    def add(alpha):
        def fn(m, a, o):
            h = o[0] if isinstance(o, tuple) else o; h2 = h + (alpha * gap * u).to(h.dtype)
            return (h2,) + tuple(o[1:]) if isinstance(o, tuple) else h2
        return fn

    def ablate(m, a, o):
        h = o[0] if isinstance(o, tuple) else o; uu = u.to(h.dtype); h2 = h - (h @ uu)[..., None] * uu
        return (h2,) + tuple(o[1:]) if isinstance(o, tuple) else h2
    vars_ = [recs[i] for i in neg][:40] if mode == "filtered" else [r for r in recs if r["kind"] == "variant"][:40]
    origs = [recs[i] for i in pos][:40]; steer = {}
    for alpha in (0.0, 1.0, 2.0):
        outs = [forced(byk[r["key"]]["question"], add(alpha) if alpha else None) for r in vars_]
        steer[alpha] = dict(says_original=float(np.mean([same(a, byk[r["key"]]["original_gold"]) for a, r in zip(outs, vars_)])),
                            correct=float(np.mean([same(a, r["gold"]) for a, r in zip(outs, vars_)])))
        log(f"  steer variants alpha {alpha}: answer = original's {steer[alpha]['says_original']:.2f}, correct {steer[alpha]['correct']:.2f}")
    acc0 = float(np.mean([same(forced(byk[r["key"]]["question"], None), r["gold"]) for r in origs]))
    acc1 = float(np.mean([same(forced(byk[r["key"]]["question"], ablate), r["gold"]) for r in origs]))
    log(f"  ablate direction on originals: no-reasoning accuracy {acc0:.2f} -> {acc1:.2f}")
    info.update(steer=steer, ablate_acc=[acc0, acc1])
    np.savez_compressed(odir / "direction.npz", **{t: D[t] for t in POS_TYPES}); json.dump(info, open(odir / "direction.json", "w"))


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


def graph_pass(model, ctx, prompt, trace, dirs):
    if "orig" not in _G:
        from transformers import AttentionInterface
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        _G["orig"] = sdpa_attention_forward; AttentionInterface.register("sdpa", _g_sdpa)
    n_p = len(prompt); blk = np.arange(len(trace)) // 64; nq = int(blk.max()) + 1; dev = model.device
    _G.update(on=True, res={}, n_p=n_p, nq=nq, qblk=torch.tensor(blk, device=dev), kseg=torch.tensor(np.r_[np.zeros(n_p, np.int64), 1 + blk], device=dev),
              layers=frozenset(i for i, l in enumerate(ctx["layers"]) if getattr(l, "self_attn", None) is not None))
    U = {t: torch.tensor(dirs[t], device=dev) for t in ("gen", "last")} if dirs is not None else None
    PJ, FIN, hk = {}, {}, []
    if U is not None:
        for i, layer in enumerate(ctx["layers"]):
            def fn(m, a, o, i=i):
                h = (o[0] if isinstance(o, tuple) else o)[0, n_p:].float()
                PJ[i] = torch.stack([h @ U[t][i + 1] for t in ("gen", "last")]).half().cpu()
            hk.append(layer.register_forward_hook(fn))
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
    if PJ: out["proj"] = torch.stack([PJ[i] for i in range(len(ctx["layers"]))], 1).numpy()       # [2 (gen, last), L, n_trace]
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


def graph_stage(model, tok, ctx, rows, odir, dirs):
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
            out = graph_pass(model, ctx, r["prompt_ids"], gen[:c], dirs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); log(f"    OOM on {key} ({c} tokens) - retrying on the first 4096 | {gpu()}"); c = min(c, 4096)
            out = graph_pass(model, ctx, r["prompt_ids"], gen[:c], dirs)
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
    ap.add_argument("--models", default=""); ap.add_argument("--out", default=str(HERE / "out_recall_graphs"))
    ap.add_argument("--stages", default="R,R2,G"); ap.add_argument("--pairs", default=str(HERE / "recall_pairs.jsonl"))
    ap.add_argument("--graph_inputs", default=str(HERE / "graph_inputs.jsonl")); a = ap.parse_args()
    pairs = [json.loads(l) for l in open(a.pairs)]; grows = [json.loads(l) for l in open(a.graph_inputs)]
    models = [m for m in ORDER if not a.models or m in a.models.split(",")]; stages = set(a.stages.split(","))
    import transformers
    log(f"models {models} | stages {sorted(stages)} | {len(pairs)} recall prompts, {len(grows)} graph rows | torch {torch.__version__} transformers {transformers.__version__} | {gpu()}")
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
            if "R" in stages and not (odir / "recall.npz").exists():
                recall_stage(model, tok, ctx, pairs, odir)
            if "R2" in stages and (odir / "recall.npz").exists() and not (odir / "direction.npz").exists():
                direction_stage(model, tok, ctx, pairs, odir)
            if "G" in stages:
                dz = np.load(odir / "direction.npz") if (odir / "direction.npz").exists() else None
                graph_stage(model, tok, ctx, [r for r in grows if r["model_id"] == m], odir, {t: dz[t] for t in ("gen", "last")} if dz is not None else None)
        except Exception:                                                       # noqa: BLE001 - log, keep going with the next model
            log(f"FAILED {m}:\n{traceback.format_exc()}")
        log(f"=== {m} done in {(time.time() - T0) / 60:.1f} min")
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
