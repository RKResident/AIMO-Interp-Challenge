"""Recall-vs-computation directions (A10G; one model at a time; run_recall.sh). No long generation.
R  every prompt of recall_inputs.jsonl (families gsm / comp / arith / formula, see build_recall_inputs.py): the answer forced with NO
   reasoning (prompt + "</think> The final answer is \\boxed{", greedy, parsed, correct vs gold); hidden states at every layer at
   the last prompt token (H_last), the forced-answer position (H_ans) and the mean over the first 64 greedily generated tokens (H_gen).
   Saved in shards of 250 prompts (resumable).
D  per family: recall rows vs computation rows.
     gsm / comp: (original right with no reasoning AND its variant wrong) -> the original is "recall", that variant "compute";
                 if fewer than 8 such pairs: all originals vs all variants (logged).
     arith / formula: kind recall vs kind compute (all rows).
   Per position type and layer: difference of means (unit vector), 5-fold CV grouped by pair/problem -> AUC; cosines between the families'
   directions per layer. -> directions.npz, directions.json
P  projection of our labelled traces (graph_inputs.jsonl: the 118 greedy traces, cut at end of thinking / 6144) on every family x type
   direction at every layer, via forward hooks (nothing large is stored): mean over the trace, over its first and last quarter.
   -> projections.jsonl
Outputs <out>/<model>/."""
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

ORDER = ["Qwen/Qwen3.5-4B", "allenai/Olmo-3-7B-Think", "Skywork/Skywork-OR1-Math-7B", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"]
TYPES = ("last", "ans", "gen"); SHARD = 250


def parse_answer(s):
    s = s.split("}")[0].replace(",", "").replace("$", "").replace("\\%", "").replace("%", "").strip()
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return m.group(0) if m else s


def same(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return False


# ----------------------------------------------------------------------------------------------------------------- R
def recall_stage(model, tok, ctx, rows, odir):
    """pass A: every prompt except the comp variants; pass B: the comp variants whose original was answered right with no reasoning
    (only those can form a recall pair)"""
    A = [r for r in rows if not (r["family"] == "comp" and r["kind"] == "variant")]
    _recall_pass(model, tok, ctx, A, odir, "A")
    right = {json.loads(l)["group"] for f in sorted((odir / "shards").glob("shardA*.jsonl")) for l in open(f)
             if json.loads(l)["family"] == "comp" and json.loads(l)["kind"] == "original" and json.loads(l)["correct"]}
    B = [r for r in rows if r["family"] == "comp" and r["kind"] == "variant" and r["group"] in right]
    log(f"  R pass B: {len(right)} competition originals answered right with no reasoning -> {len(B)} variants")
    _recall_pass(model, tok, ctx, B, odir, "B")


def _recall_pass(model, tok, ctx, rows, odir, tag):
    sdir = odir / "shards"; sdir.mkdir(exist_ok=True); sfx = ctx["suffixes"]["B"]
    for s0 in range(0, len(rows), SHARD):
        f = sdir / f"shard{tag}{s0 // SHARD:03d}.npz"
        if f.exists(): continue
        t0 = time.time(); HL, HA, HG, recs = [], [], [], []
        for p in rows[s0:s0 + SHARD]:
            pids = build_prompt(tok, p["question"])[0]
            rids = tok(p["reasoning"], add_special_tokens=False)["input_ids"] if p.get("reasoning") else []    # forced family: teacher-forced solution
            with torch.no_grad():
                o = model(input_ids=torch.tensor([pids + rids + sfx], device=model.device), output_hidden_states=True, use_cache=True, logits_to_keep=1)
                hs = o.hidden_states; top = int(o.logits[0, -1].argmax())
                HL.append(torch.stack([h[0, len(pids) - 1] for h in hs]).float().cpu().half()); HA.append(torch.stack([h[0, -1] for h in hs]).float().cpu().half())
                if rids: hg = torch.stack([h[0, len(pids):len(pids) + len(rids)].float().mean(0) for h in hs]).cpu().half()
                ans = continuation(model, tok, o.past_key_values, top, n=10); del o, hs
                if rids:                                                                 # H_gen = mean over the forced solution tokens
                    HG.append(hg)
                elif p["family"] == "comp":                                                # H_gen skipped for competition problems (time); zeros
                    HG.append(torch.zeros_like(HL[-1]))
                else:
                    g = model.generate(input_ids=torch.tensor([pids], device=model.device), max_new_tokens=64, do_sample=False, temperature=None, top_p=None, top_k=None)
                    o2 = model(input_ids=g, output_hidden_states=True, use_cache=False, logits_to_keep=1)
                    HG.append(torch.stack([h[0, len(pids):].float().mean(0) for h in o2.hidden_states]).cpu().half()); del o2, g
            a = parse_answer(ans)
            recs.append(dict(key=p["key"], family=p["family"], group=p["group"], kind=p["kind"], gold=p["gold"], answer_raw=ans[:40], answer=a,
                             correct=same(a, p["gold"]) if p["gold"] is not None else None))
        np.savez_compressed(f, H_last=torch.stack(HL).numpy(), H_ans=torch.stack(HA).numpy(), H_gen=torch.stack(HG).numpy(), keys=np.array([r["key"] for r in recs]))
        with open(f.with_suffix(".jsonl"), "w") as fo:
            for r in recs: fo.write(json.dumps(r) + "\n")
        acc = {fam: np.mean([r["correct"] for r in recs if r["family"] == fam and r["correct"] is not None] or [np.nan]) for fam in sorted({r["family"] for r in recs})}
        log(f"  R pass {tag} shard {s0 // SHARD + 1}/{-(-len(rows) // SHARD)}: {len(recs)} prompts in {time.time() - t0:.0f}s | no-reasoning accuracy "
            + ", ".join(f"{k} {v:.2f}" for k, v in acc.items()) + f" | {gpu()}")
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------------------------- D
def auc(s, y):
    p, n = s[y == 1], s[y == 0]
    return float(((p[:, None] > n[None]) + .5 * (p[:, None] == n[None])).mean()) if len(p) and len(n) else float("nan")


def direction_stage(odir):
    sdir = odir / "shards"; recs = []; H = {t: [] for t in TYPES}
    for f in sorted(sdir.glob("shard*.npz")):
        z = np.load(f); recs += [json.loads(l) for l in open(f.with_suffix(".jsonl"))]
        for t in TYPES: H[t].append(z[f"H_{t}"])
    H = {t: np.concatenate(v) for t, v in H.items()}; idx = {r["key"]: i for i, r in enumerate(recs)}
    D, info = {}, {}
    for fam in ("gsm", "comp", "arith", "formula", "forced"):
        R = [r for r in recs if r["family"] == fam]
        if fam in ("gsm", "comp"):
            orig = {r["group"]: r for r in R if r["kind"] == "original"}
            pairs = [(idx[orig[r["group"]]["key"]], idx[r["key"]]) for r in R if r["kind"] == "variant" and r["group"] in orig and orig[r["group"]]["correct"] and not r["correct"]]
            pos = sorted({a for a, _ in pairs}); neg = sorted({b for _, b in pairs}); mode = "filtered"
            if len(pos) < 8:
                pos = [idx[r["key"]] for r in R if r["kind"] == "original"]; neg = [idx[r["key"]] for r in R if r["kind"] == "variant"]; mode = "fallback all"
        else:
            pos = [idx[r["key"]] for r in R if r["kind"] == "recall"]; neg = [idx[r["key"]] for r in R if r["kind"] == "compute"]; mode = "all"
        if len(pos) < 4 or len(neg) < 4:
            log(f"  D {fam}: skipped ({len(pos)} recall / {len(neg)} compute rows)"); continue
        sel = np.array(pos + neg); y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]; grp = np.array([recs[i]["group"] for i in sel])
        ug = np.unique(grp); rng = np.random.default_rng(0); fold = {g: k % 5 for k, g in enumerate(rng.permutation(ug))}; fo = np.array([fold[g] for g in grp])
        info[fam] = dict(mode=mode, n_pos=len(pos), n_neg=len(neg), n_groups=len(ug), cv_auc={}, best_layer={},
                         no_reasoning_acc={k: float(np.mean([r["correct"] for r in R if r["kind"] == k and r["correct"] is not None] or [np.nan]))
                                           for k in sorted({r["kind"] for r in R})})
        for t in TYPES:
            if fam == "comp" and t == "gen": continue                                    # not computed for competition problems
            if fam == "forced" and t == "last": continue                                 # identical prompt on both sides: no contrast
            X = H[t][sel].astype(np.float32); L = X.shape[1]; dirs = np.zeros((L, X.shape[2]), np.float32); cv = []
            for l in range(L):
                Xl = X[:, l]; s = np.full(len(sel), np.nan)
                for k in range(5):
                    tr, te = fo != k, fo == k
                    if not te.any() or y[tr].min() == y[tr].max(): continue
                    d = Xl[tr][y[tr] == 1].mean(0) - Xl[tr][y[tr] == 0].mean(0); s[te] = Xl[te] @ (d / (np.linalg.norm(d) + 1e-9))
                ok = np.isfinite(s); cv.append(auc(s[ok], y[ok]))
                d = Xl[y == 1].mean(0) - Xl[y == 0].mean(0); dirs[l] = d / (np.linalg.norm(d) + 1e-9)
            D[f"{fam}__{t}"] = dirs; b = int(np.nanargmax(np.r_[[-1.0], cv[1:]]))
            info[fam]["cv_auc"][t] = [round(float(v), 3) for v in cv]; info[fam]["best_layer"][t] = b
            log(f"  D {fam:7s} {t:4s}: {mode}, {len(pos)} recall / {len(neg)} compute rows ({len(ug)} groups) | best layer {b} CV AUC {cv[b]:.3f}")
    fams = sorted({k.split("__")[0] for k in D})
    info["cosine"] = {t: {f"{a}|{b}": [round(float(v), 3) for v in (D[f"{a}__{t}"] * D[f"{b}__{t}"]).sum(1)] for i, a in enumerate(fams) for b in fams[i + 1:]
                          if f"{a}__{t}" in D and f"{b}__{t}" in D} for t in TYPES}
    for t in TYPES:
        L = next(D[k].shape[0] for k in D); mid = slice(L // 3, 2 * L // 3)
        log(f"  D cosine between families ({t}, middle-third layers mean): " + ", ".join(f"{k} {np.mean(v[mid]):+.2f}" for k, v in info["cosine"][t].items()))
    np.savez_compressed(odir / "directions.npz", **D); json.dump(info, open(odir / "directions.json", "w"))


# ----------------------------------------------------------------------------------------------------------------- P
def projection_stage(model, tok, ctx, rows, odir):
    out = odir / "projections.jsonl"; done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    Dz = np.load(odir / "directions.npz"); names = sorted(Dz.files); layers = ctx["layers"]; nL = len(layers)
    # hidden_states index l = output of decoder layer l-1 -> directions at index l+1 belong to the output of layers[l]
    U = torch.tensor(np.stack([Dz[n][1:nL + 1] for n in names], 1), device=model.device, dtype=torch.float32)       # [nL, n_dirs, d]
    todo = [r for r in rows if r["id"] not in done]; log(f"  P: {len(rows)} traces, {len(todo)} to do, {len(names)} directions x {nL} layers")
    for n, r in enumerate(todo):
        t0 = time.time(); gen = r["greedy_gen_ids"][:6144]; te = find_think_end(tok, gen, ctx["tid"]); c = min(max(1, te if te is not None else len(gen)), 6144)
        n_p = len(r["prompt_ids"]); acc = {}

        def mk(li):
            def fn(m, a, o):
                h = (o[0] if isinstance(o, tuple) else o)[0, n_p:n_p + c].float()                                 # [c, d] trace positions
                pr = h @ U[li].T                                                                                  # [c, n_dirs]
                q = max(1, c // 4); acc[li] = torch.stack([pr.mean(0), pr[:q].mean(0), pr[-q:].mean(0)]).cpu().numpy()
            return fn
        hooks = [layers[li].register_forward_hook(mk(li)) for li in range(nL)]
        try:
            with torch.no_grad():
                model(input_ids=torch.tensor([r["prompt_ids"] + gen[:c]], device=model.device), use_cache=False, logits_to_keep=1)
        finally:
            for h in hooks: h.remove()
        P = np.stack([acc[li] for li in range(nL)])                                                               # [nL, 3, n_dirs]
        rec = dict(id=r["id"], dataset=r["dataset"], robust=r["robust"], cut=c, think_end=te, names=names,
                   mean=P[:, 0].round(4).tolist(), first_q=P[:, 1].round(4).tolist(), last_q=P[:, 2].round(4).tolist())
        with open(out, "a") as f: f.write(json.dumps(rec) + "\n")
        log(f"    P [{n + 1}/{len(todo)}] {r['id']}: {c} tokens | {time.time() - t0:.1f}s | {gpu()}")
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=""); ap.add_argument("--out", default=str(HERE / "out_recall")); ap.add_argument("--stages", default="R,D,P")
    ap.add_argument("--inputs", default=str(HERE / "recall_inputs.jsonl")); ap.add_argument("--traces", default=str(HERE / "graph_inputs.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="first N prompts only (smoke test)")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.inputs)]
    if a.limit: rows = [r for fam in ("gsm", "comp", "arith", "formula", "forced") for r in [x for x in rows if x["family"] == fam][:a.limit]]
    traces = [json.loads(l) for l in open(a.traces)]
    models = [m for m in ORDER if not a.models or m in a.models.split(",")]; stages = set(a.stages.split(","))
    import transformers
    log(f"models {models} | stages {sorted(stages)} | {len(rows)} prompts, {len(traces)} labelled traces | torch {torch.__version__} "
        f"transformers {transformers.__version__} | {gpu()}")
    for k, m in enumerate(models):
        prefetch(models[k + 1:k + 2])                                                 # the next model downloads while this one runs
        odir = Path(a.out) / m.replace("/", "__"); odir.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import snapshot_download
        snapshot_download(m)
        model, tok = SI.load_model_and_tokenizer(m, local_files_only=True); layers, n_layers = SI.find_decoder_layers(model); tid = think_end_token_id(tok)
        has_think = tid is not None or "<think>" in build_prompt(tok, "x")[1]
        sfx = {kk: tok(v, add_special_tokens=False)["input_ids"] for kk, v in (SUFFIXES_THINK if has_think else SUFFIXES_NOTHINK).items()}
        ctx = dict(layers=layers, n_layers=n_layers, tid=tid, suffixes=sfx, norm=final_norm(model))
        log(f"=== {m}: {n_layers} layers | {gpu()}"); T0 = time.time()
        try:
            if "R" in stages: recall_stage(model, tok, ctx, rows, odir)
            if "D" in stages and not (odir / "directions.npz").exists(): direction_stage(odir)
            if "P" in stages and (odir / "directions.npz").exists():
                projection_stage(model, tok, ctx, [r for r in traces if r["model_id"] == m], odir)
        except Exception:                                                             # noqa: BLE001 - log, go on with the next model
            log(f"FAILED {m}:\n{traceback.format_exc()}")
        log(f"=== {m} done in {(time.time() - T0) / 60:.1f} min")
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
