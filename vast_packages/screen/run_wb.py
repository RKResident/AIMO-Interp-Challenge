"""White-box battery extraction (user 2026-10-31): submission_core/wb_battery.battery on stored greedy traces (first 6144 tokens).
Writes <out>/<tag>.jsonl (pid + every scalar, one line per trace, resumable) and <out>/<tag>/<pid>.npz (per-layer / per-head arrays incl. the
64-token window means of every layer). Progress line every trace with timings and an ETA.
  python run_wb.py --model Qwen/Qwen3.5-4B --screen <screen.jsonl> --keep set100.txt --out out/wb"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "submission_core"))
import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--screen", required=True); ap.add_argument("--keep", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    import spurious_inference as SI, wb_battery as WB
    from spurious_features import SUFFIXES_THINK, build_prompt
    tag = a.model.replace("/", "__"); out = Path(a.out) / f"{tag}.jsonl"; nd = Path(a.out) / tag; nd.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["pid"] for l in open(out)} if out.exists() else set()
    scr = {json.loads(l)["pid"]: json.loads(l) for l in open(a.screen)}
    keep = [p for p in (l.strip() for l in open(a.keep)) if p and p in scr and p not in done]
    if a.limit: keep = keep[:a.limit]
    print(time.strftime("%H:%M:%S"), f"{a.model}: {len(keep)} traces to do ({len(done)} done) | {SI._gpu()}", flush=True)
    if not keep: return
    model, tok = SI.load_model_and_tokenizer(a.model, local_files_only=False); layers, L = SI.find_decoder_layers(model)
    sfx = tok(SUFFIXES_THINK["B"], add_special_tokens=False)["input_ids"]; t0 = time.time()
    for k, p in enumerate(keep):
        r = scr[p]; t1 = time.time()
        pids = r.get("prompt_ids") or build_prompt(tok, r["problem"])[0]
        try:
            S, A = WB.battery(model, tok, layers, pids, r["gen_ids"][:6144], sfx, log=lambda m: print(time.strftime("%H:%M:%S"), m, flush=True))
        except Exception as e:                                       # noqa: BLE001 - record and continue
            import traceback; print(time.strftime("%H:%M:%S"), f"  {p} FAILED: {e}\n{traceback.format_exc()}", flush=True); continue
        np.savez_compressed(nd / f"{p}.npz", **A)
        with open(out, "a") as fo: fo.write(json.dumps(dict(pid=p, model=a.model, n_gen=len(r["gen_ids"][:6144]), still_thinking=r.get("still_thinking"), **S)) + "\n")
        el = time.time() - t0
        print(time.strftime("%H:%M:%S"), f"[{k + 1}/{len(keep)}] {p} {time.time() - t1:.0f}s | ETA {el / (k + 1) * (len(keep) - k - 1) / 60:.0f} min | {SI._gpu()}", flush=True)
    print(time.strftime("%H:%M:%S"), f"done: {len(keep)} traces in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
