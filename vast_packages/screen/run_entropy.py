"""Entropy trajectories (user 2026-11-01: "why did entropy at 4096 work - try the whole trajectory"). Per stored greedy trace (first 6144 tokens),
ONE chunked pass with a KV cache over prompt + trace:
  token trajectory   at every generated token: next-token entropy, probability of the token actually generated, top-1 probability
  forced curve       every 256 trace tokens: forced answer (suffix B) on a copy of the cache -> entropy, top-1 prob, top-1 token id
Saves <out>/<tag>/<pid>.npz (tok_ent, tok_lp, tok_p1 float16 [n]; forced [k, 4] = budget, entropy, p1, top1) and <out>/<tag>.jsonl (pid,
n, still_thinking) - analysis on CPU. Resumable, progress line per trace.
  python run_entropy.py --model M --screen S --keep K --out out/entropy"""
import argparse, copy, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "submission_core"))
import numpy as np
import torch

STEP = 256
T = lambda: time.strftime("%H:%M:%S")


@torch.no_grad()
def trace_entropy(model, pids, gen, sfx):
    ids = pids + gen; n_p = len(pids); cache = None; done = 0
    ent, lp_tok, p1 = [], [], []; forced = []
    bounds = sorted(set(list(range(n_p + STEP, n_p + len(gen) + 1, STEP)) + [n_p + len(gen)]))
    for end in bounds:
        o = model(input_ids=torch.tensor([ids[done:end]], device=model.device), past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        lg = o.logits[0].float(); del o
        # logits at position j predict token j+1: keep predictions for generated tokens (positions n_p-1 .. n_p+len(gen)-2)
        lo, hi = max(done, n_p - 1), min(end, n_p + len(gen) - 1)
        if hi > lo:
            L = torch.log_softmax(lg[lo - done:hi - done], -1); P = L.exp()
            ent.append((-(P * L).sum(-1)).cpu()); p1.append(P.max(-1).values.cpu())
            tgt = torch.tensor(ids[lo + 1:hi + 1], device=model.device); lp_tok.append(L.gather(1, tgt[:, None])[:, 0].cpu()); del L, P
        del lg; done = end
        try:
            br = copy.deepcopy(cache); o = model(input_ids=torch.tensor([sfx], device=model.device), past_key_values=br, use_cache=True, logits_to_keep=1); del br
        except (RuntimeError, TypeError, AttributeError) as e:
            if isinstance(e, torch.cuda.OutOfMemoryError): raise
            o = model(input_ids=torch.tensor([ids[:end] + sfx], device=model.device), use_cache=False, logits_to_keep=1)
        l = torch.log_softmax(o.logits[0, -1].float(), -1); p = l.exp(); t = int(l.argmax())
        forced.append([end - n_p, float(-(p * l).sum()), float(p[t]), t]); del o
    del cache; torch.cuda.empty_cache()
    return (torch.cat(ent).numpy(), torch.cat(lp_tok).numpy(), torch.cat(p1).numpy(), np.array(forced, float))


def main():
    ap = argparse.ArgumentParser()
    for k in ("model", "screen", "keep", "out"): ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    import spurious_inference as SI
    from spurious_features import SUFFIXES_THINK
    tag = a.model.replace("/", "__"); out = Path(a.out) / f"{tag}.jsonl"; nd = Path(a.out) / tag; nd.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["pid"] for l in open(out)} if out.exists() else set()
    scr = {json.loads(l)["pid"]: json.loads(l) for l in open(a.screen)}
    keep = [p for p in (l.strip() for l in open(a.keep)) if p and p in scr and p not in done]
    if a.limit: keep = keep[:a.limit]
    print(T(), f"{a.model}: {len(keep)} traces to do ({len(done)} done) | {SI._gpu()}", flush=True)
    if not keep: return
    model, tok = SI.load_model_and_tokenizer(a.model, local_files_only=False); model.eval()
    sfx = tok(SUFFIXES_THINK["B"], add_special_tokens=False)["input_ids"]; t0 = time.time()
    for k, p in enumerate(keep):
        r = scr[p]; t1 = time.time(); gen = r["gen_ids"][:6144]
        try:
            e, lp, p1, fc = trace_entropy(model, r["prompt_ids"], gen, sfx)
        except Exception as ex:                                          # noqa: BLE001 - record and continue
            import traceback; print(T(), f"  {p} FAILED: {ex}\n{traceback.format_exc()}", flush=True); torch.cuda.empty_cache(); continue
        np.savez_compressed(nd / f"{p}.npz", tok_ent=e.astype(np.float16), tok_lp=lp.astype(np.float16), tok_p1=p1.astype(np.float16), forced=fc)
        with open(out, "a") as fo: fo.write(json.dumps(dict(pid=p, model=a.model, n=len(gen), still_thinking=r.get("still_thinking"), n_forced=len(fc))) + "\n")
        el = time.time() - t0
        print(T(), f"[{k + 1}/{len(keep)}] {p} {time.time() - t1:.1f}s ({len(e)} tokens, {len(fc)} forced points, mean H {e.mean():.3f}) | ETA {el / (k + 1) * (len(keep) - k - 1) / 60:.0f} min | {SI._gpu()}", flush=True)
    print(T(), f"done: {len(keep)} traces in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
