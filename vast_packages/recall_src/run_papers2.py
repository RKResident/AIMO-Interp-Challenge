"""Faithful paper features (submission_core/spurious_papers.py, the SAME code the submission runs) on our labelled traces
(graph_inputs.jsonl: 118) and the set100 screening traces (traces_set100.jsonl: 150). One model per process (run_papers2.sh).
LID needs a reference set, so it is NOT computed here: every trace's final forced-answer-position state at every layer is saved
(states_<model>.npz) and LID is computed locally (leave-problem-out) and its references shipped with the submission.
Recall directions: every family x position at its recall-chosen layer (out_recall/<model>/directions.*).
Output <out>/<model>/papers2.jsonl (resumable) + states.npz."""
from __future__ import annotations

import argparse, json, sys, time, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "submission_core"))
import numpy as np
import torch

import spurious_inference as SI
import spurious_papers as SP
from spurious_features import SUFFIXES_NOTHINK, SUFFIXES_THINK, auto_variants, build_prompt, find_think_end, think_end_token_id
from run_t1 import gpu, log, final_norm


def recall_dirs(m):
    d = HERE / "out_recall" / m.replace("/", "__"); z = np.load(d / "directions.npz"); info = json.load(open(d / "directions.json"))
    names, layers, U = [], [], []
    for nm in sorted(z.files):
        fam, t = nm.split("__"); b = info[fam]["best_layer"][t]                       # hidden-state index b = output of decoder layer b - 1
        names.append(nm); layers.append(b - 1); U.append(z[nm][b])
    return dict(names=names, layers=layers, U=torch.tensor(np.stack(U)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", required=True); ap.add_argument("--out", default=str(HERE / "out_papers2"))
    ap.add_argument("--traces", default="graph_inputs.jsonl,traces_set100.jsonl"); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    rows = [json.loads(l) for f in a.traces.split(",") for l in open(HERE / f)]
    for m in a.models.split(","):
        R = [r for r in rows if r["model_id"] == m]
        if a.limit: R = R[:a.limit]
        odir = Path(a.out) / m.replace("/", "__"); odir.mkdir(parents=True, exist_ok=True); out = odir / "papers2.jsonl"
        done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set(); todo = [r for r in R if r["id"] not in done]
        log(f"{m}: {len(R)} traces, {len(todo)} to do | {gpu()}")
        if not todo: continue
        model, tok = SI.load_model_and_tokenizer(m, local_files_only=True); layers, nL = SI.find_decoder_layers(model); tid = think_end_token_id(tok)
        has_think = tid is not None or "<think>" in build_prompt(tok, "x")[1]
        sfx = tok((SUFFIXES_THINK if has_think else SUFFIXES_NOTHINK)["B"], add_special_tokens=False)["input_ids"]
        norm_m = final_norm(model); rec = recall_dirs(m); SI._install_attention_hook()
        log(f"=== {m}: {nL} layers | {len(rec['names'])} recall directions | {gpu()}")
        sp = odir / "states"; sp.mkdir(exist_ok=True)
        for k, r in enumerate(todo):
            t0 = time.time(); gen = r["greedy_gen_ids"][:6144]; te = find_think_end(tok, gen, tid); c = min(max(1, te if te is not None else len(gen)), 6144)
            try:
                f, st = SP.paper_features(model, tok, layers, norm_m, build_prompt, auto_variants, r["prompt_ids"], gen, c, r["problem"], sfx,
                                          lid_ref=None, recall=rec, want_states=True)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); log(f"    OOM on {r['id']} - skipped | {gpu()}"); continue
            except Exception:                                                         # noqa: BLE001
                log(f"    FAILED {r['id']}:\n{traceback.format_exc()}"); continue
            np.save(sp / f"{r['id']}.npy", st.numpy())
            rec_ = dict(id=r["id"], dataset=r["dataset"], robust=r["robust"], cut=c, think_end=te, **f)
            with open(out, "a") as fo: fo.write(json.dumps(rec_) + "\n")
            log(f"    [{k + 1}/{len(todo)}] {r['id']}: {c} tok {f.get('n_sent')} sent | rhythm {f.get('rhythm_cooc_excess', float('nan')):+.3f} | "
                f"supKL {f.get('sup_logkl_top_minus_rnd', float('nan')):+.3f} | minK++20 {f.get('minkpp_20', float('nan')):+.2f} | LI {f.get('li_sum', float('nan')):+.1f} | "
                f"DC {f.get('lara_dc_mid', float('nan')):+.3f} | GW {f.get('cie_gw', float('nan')):.4f} | {time.time() - t0:.1f}s | {gpu()}")
            torch.cuda.empty_cache()
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
