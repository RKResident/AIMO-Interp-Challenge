"""Submission features for the set100 screening traces (user, 2026-10-29: evaluate recipes on the new labels), computed by the SUBMISSION's
own code path: submission_core.spurious_inference.predict_robustness, one problem per call as at test time, with two substitutions -
  * generation is replaced by the stored greedy screening trace (the 6144-token trace the submission would generate; vLLM greedy, the same
    prompt via the same build_prompt), and
  * the model is loaded once per model instead of once per call (load_model_and_tokenizer cached).
Recipe = recipe_allfeats.json (every feature we use + log_extra), so every value is logged exactly as a submission logs it. The raw values
are parsed from the call-summary log line. Output out_feats/<model>.jsonl (resumable by pid).
  python run_feats.py --model Qwen/Qwen3.5-4B --screen scr/Qwen__Qwen3.5-4B.jsonl --keep set100.txt"""
import argparse, json, re, shutil, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--screen", required=True); ap.add_argument("--keep", required=True)
ap.add_argument("--out", default=str(HERE / "out_feats")); ap.add_argument("--zip", required=True, help="a submitted build: its recipe.json + artifacts are used (exp120 replay)"); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--crop-cache", action="store_true", help="forced_answers without the cache deepcopy (Olmo on the A10G)")
ap.add_argument("--check-crop", action="store_true", help="with --crop-cache: compare crop vs deepcopy on short cuts, then exit")
ap.add_argument("--pool", default=str(HERE / "pool.jsonl"), help="problem texts by pid (olympiad: oly_pool.jsonl)"); a = ap.parse_args()
import zipfile                                                                          # exp120: one build's recipe + artifacts, current submission code
bname = Path(a.zip).name.replace("-small.zip", ""); sol = HERE / f"feat_solution_{bname}_{a.model.replace('/', '__')}"; art = sol / "spurious_artifacts"
shutil.rmtree(sol, ignore_errors=True); art.mkdir(parents=True, exist_ok=True)
for f in ("spurious_inference.py", "spurious_features.py", "spurious_papers.py"): shutil.copy(HERE / "submission_core" / f, sol / f)
with zipfile.ZipFile(a.zip) as z:
    for n in z.namelist():
        if n.startswith("spurious_artifacts/") and not n.endswith("/"): (art / Path(n).name).write_bytes(z.read(n))
sys.path.insert(0, str(sol))
import spurious_inference as SI                                                        # noqa: E402

_cache = {}
_load = SI.load_model_and_tokenizer
SI.load_model_and_tokenizer = lambda m, local_files_only=True: _cache[m] if m in _cache else _cache.setdefault(m, _load(m, local_files_only))  # (setdefault alone reloaded every call)
import torch                                                                          # noqa: E402


from transformers.cache_utils import DynamicLayer, DynamicSlidingWindowLayer                       # noqa: E402
_orig_forced = SI.forced_answers


def _peek_update(layer):
    """layer.update for the suffix pass: return exactly the states the real update returns, without storing them."""
    def upd(key_states, value_states, *args, **kwargs):
        return torch.cat([layer.keys, key_states], dim=-2), torch.cat([layer.values, value_states], dim=-2)
    return upd


def forced_answers_crop(model, prompt_ids, gen_ids, cuts, suffixes, keep=()):
    """SI.forced_answers without the per-suffix deepcopy of the prefix cache: the suffix pass runs with every layer's update replaced by
    a non-storing one, so the prefix cache is left untouched (same states, same attention inputs; peak extra = one layer's concat).
    For Olmo's 6.3k-token MHA cache (~3.3 GB per copy), which OOMs on the A10G. Only for plain / sliding-window DynamicCache layers
    (both update by torch.cat and return the concatenation; a sliding layer that is not recording returns the full states too);
    anything else (Qwen3.5's linear-attention layers) keeps the original deepcopy path."""
    full = prompt_ids + gen_ids; n_p = len(prompt_ids); cache, pos, results = None, 0, {}
    with torch.no_grad():
        for c in sorted(set(cuts)):
            target = n_p + c
            for s in range(pos, target, 4096):
                out = SI.forward_last_logits(model, input_ids=torch.tensor([full[s:min(target, s + 4096)]], device=model.device),
                                             past_key_values=cache, use_cache=True)
                cache = out.past_key_values
            pos = target
            L = getattr(cache, "layers", None)
            if not L or any(type(l) not in (DynamicLayer, DynamicSlidingWindowLayer) or getattr(l, "record_past", False) for l in L):
                del cache, out; torch.cuda.empty_cache()
                return _orig_forced(model, prompt_ids, gen_ids, cuts, suffixes, keep)
            for name, sids in suffixes.items():
                for l in L: l.update = _peek_update(l)
                try:
                    out = SI.forward_last_logits(model, input_ids=torch.tensor([sids], device=model.device), past_key_values=cache, use_cache=True)
                finally:
                    for l in L: del l.update
                lp = torch.log_softmax(out.logits[0, -1].float(), -1); p = lp.exp()
                results[(c, name)] = (float(p.max()), float(-(p * lp).sum()), int(lp.argmax()), lp.cpu() if c in keep else None)
                del out
    del cache
    torch.cuda.empty_cache()
    return results


if a.crop_cache:
    if a.check_crop:                                                                   # crop vs deepcopy on short cuts (both fit)
        m, tk = SI.load_model_and_tokenizer(a.model)
        r = next(json.loads(l) for l in open(a.screen) if json.loads(l)["pid"] in set(l.strip() for l in open(a.keep)))
        pr = r["prompt_ids"]
        sfx = {"A": tk(" The answer is")["input_ids"], "B": tk("\n</think>\n\nFinal answer: \\boxed{")["input_ids"]}
        cuts = [256, 4200, 4900]
        A = SI.forced_answers(m, pr, r["gen_ids"][:4900], cuts, sfx, keep=(1024,)); B = forced_answers_crop(m, pr, r["gen_ids"][:4900], cuts, sfx, keep=(1024,))
        for k in A:
            d = float((A[k][3] - B[k][3]).abs().max()) if A[k][3] is not None else 0.0
            print("CHECK", k, "deepcopy", [round(x, 6) for x in A[k][:2]], A[k][2], "| crop", [round(x, 6) for x in B[k][:2]], B[k][2], "| max |dlp|", d, flush=True)
        sys.exit(0)
    SI.forced_answers = forced_answers_crop
CUR = {}
SI.generate_batch = lambda model, tok, prompts, cap, share: [(CUR["gen"], CUR["reason"]) for _ in prompts]
LOG = []
_log = SI._log
SI._log = lambda msg: (LOG.append(msg), _log(msg))
keep = [l.strip() for l in open(a.keep) if l.strip()]
scr = {json.loads(l)["pid"]: json.loads(l) for l in open(a.screen)}
pool = {json.loads(l)["pid"]: json.loads(l) for l in open(a.pool)}
out = Path(a.out) / f"{a.model.replace('/', '__')}.jsonl"; out.parent.mkdir(parents=True, exist_ok=True)
done = {json.loads(l)["pid"] for l in open(out)} if out.exists() else set()
todo = [p for p in keep if p in scr and p not in done]
if a.limit: todo = todo[:a.limit]
print(f"{a.model}: {len(todo)} traces to do ({len(done)} done)", flush=True)
for k, p in enumerate(todo):
    t0 = time.time(); r = scr[p]; g = r["gen_ids"]
    CUR.update(gen=g, reason="cap" if r["still_thinking"] and len(g) >= 6144 else "eos")
    LOG.clear()
    pred = SI.predict_robustness(a.model, [pool[p]["problem"]])
    summ = next((m for m in LOG if m.startswith("call summary")), "")
    raw = re.search(r"raw features (\{.*?\}) \|", summ)
    vals = {f: v[0] for f, v in eval(raw.group(1).replace("nan", "float('nan')")).items()} if raw else {}
    extra = {m.split(":")[0].strip(): m for m in LOG if "logged only" in m or "TOHA" in m or "reasoning graph" in m or "lookback" in m}
    rec = dict(pid=p, model=a.model, still_thinking=r["still_thinking"], pred=pred[0], **vals,
               log_extra=[m[:300] for m in LOG if "logged only" in m or "TOHA" in m or "reasoning graph:" in m])
    with open(out, "a") as fo: fo.write(json.dumps(rec) + "\n")
    print(f"[{k + 1}/{len(todo)}] {p}: " + " ".join(f"{f}={v:.4g}" for f, v in vals.items()) + f" | {time.time() - t0:.0f}s", flush=True)
