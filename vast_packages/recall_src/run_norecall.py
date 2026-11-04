"""Memorization test (user, 2026-10-29): each labelled (model, problem) row - the model answers with NO reasoning (the submission's prompt +
the forced-answer suffix straight away, 16 greedy tokens). On AIME/HMMT-style problems a correct no-reasoning answer is near-impossible by
chance, so it marks a problem the model has seen; 2026 problems (after every model's release) give the chance rate. Output norecall.jsonl."""
import json, re, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE / "submission_core"))
import torch
import spurious_inference as SI
from spurious_features import SUFFIXES_NOTHINK, SUFFIXES_THINK, build_prompt, think_end_token_id
rows = [json.loads(l) for l in open(HERE / "norecall_inputs.jsonl")]
out = HERE / "norecall.jsonl"; done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
for m in sorted({r["model_id"] for r in rows}):
    R = [r for r in rows if r["model_id"] == m and r["id"] not in done]
    if not R: continue
    model, tok = SI.load_model_and_tokenizer(m, local_files_only=True); tid = think_end_token_id(tok)
    sfx = tok((SUFFIXES_THINK if (tid is not None or "<think>" in build_prompt(tok, "x")[1]) else SUFFIXES_NOTHINK)["B"], add_special_tokens=False)["input_ids"]
    t0 = time.time(); n_ok = 0
    for r in R:
        ids = build_prompt(tok, r["problem"])[0] + sfx
        with torch.no_grad():
            g = model.generate(input_ids=torch.tensor([ids], device=model.device), max_new_tokens=16, do_sample=False, temperature=None, top_p=None, top_k=None)
        txt = tok.decode(g[0, len(ids):], skip_special_tokens=True); a = txt.split("}")[0].replace(",", "").replace("$", "").strip()
        mm = re.search(r"-?\d+", a); ans = str(int(mm.group(0))) if mm else a
        ok = ans == str(int(float(r["gold"]))) if re.fullmatch(r"-?\d+(\.0+)?", r["gold"]) else ans == r["gold"]; n_ok += ok
        with open(out, "a") as fo: fo.write(json.dumps(dict(id=r["id"], model_id=m, answer=ans, raw=txt[:30], correct=bool(ok))) + "\n")
    print(f"{m}: {len(R)} rows, {n_ok} correct with no reasoning, {time.time() - t0:.0f}s", flush=True)
    del model; torch.cuda.empty_cache()
