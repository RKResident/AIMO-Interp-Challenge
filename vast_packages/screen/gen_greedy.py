"""Greedy traces exactly as the submission generates them (spurious_inference.generate_batch, HF, cap 6144), in the screen format the
feature scripts read: pid, prompt_ids, gen_ids, still_thinking, stop, problem. Resumable; one progress line per batch (user 2026-11-01:
train-main-v2 features + screening easier problems).
  python gen_greedy.py --model M --pool pool.jsonl --keep keep.txt --out out.jsonl [--batch 16]"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "submission_core"))


def main():
    ap = argparse.ArgumentParser()
    for k in ("model", "pool", "keep", "out"): ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    import spurious_inference as SI
    from spurious_features import build_prompt, find_think_end, think_end_token_id
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["pid"] for l in open(out)} if out.exists() else set()
    pool = {}
    for l in open(a.pool): d = json.loads(l); pool.setdefault(d["pid"], d["problem"])
    keep = [p for p in dict.fromkeys(l.strip() for l in open(a.keep)) if p and p in pool and p not in done]
    if a.limit: keep = keep[:a.limit]
    print(time.strftime("%H:%M:%S"), f"{a.model}: {len(keep)} problems to generate ({len(done)} done) | {SI._gpu()}", flush=True)
    if not keep: return
    model, tok = SI.load_model_and_tokenizer(a.model, local_files_only=False); tid = think_end_token_id(tok); t0 = time.time()
    keep.sort(key=lambda p: len(pool[p]))                                # similar prompt lengths per batch: less padding
    for s in range(0, len(keep), a.batch):
        ch = keep[s:s + a.batch]; prompts = [build_prompt(tok, pool[p])[0] for p in ch]; t1 = time.time()
        try:
            res = SI.generate_batch(model, tok, prompts, 6144, 3600)
        except Exception as e:                                           # noqa: BLE001 - halve the batch once on failure (e.g. OOM)
            print(time.strftime("%H:%M:%S"), f"  batch failed ({e}); retrying one by one", flush=True); import torch; torch.cuda.empty_cache()
            res = [SI.generate_batch(model, tok, [pp], 6144, 3600)[0] for pp in prompts]
        with open(out, "a") as fo:
            for p, pr, (g, why) in zip(ch, prompts, res):
                fo.write(json.dumps(dict(pid=p, model=a.model, prompt_ids=pr, gen_ids=g, still_thinking=find_think_end(tok, g, tid) is None, stop=why, problem=pool[p])) + "\n")
        n = s + len(ch); el = time.time() - t0
        print(time.strftime("%H:%M:%S"), f"  [{n}/{len(keep)}] batch {len(ch)} in {time.time() - t1:.0f}s | ETA {el / n * (len(keep) - n) / 60:.0f} min | {SI._gpu()}", flush=True)
    print(time.strftime("%H:%M:%S"), f"done: {len(keep)} traces in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
