"""Perturbation CKA (user 2026-10-31, idea 2): does the model's representation of ITS OWN reasoning change when the problem is perturbed?
Per stored greedy trace: teacher-force the first 2048 trace tokens after the clean prompt and after each perturbed prompt
  sonnet_<ptype>  the labelling perturbations (rephrase / rename / distract / domain / typos)   -> upper bound, not deployable
  code_<kind>     deterministic code perturbations (typos / rename / distract, as direct_test.py) -> deployable
and compare the hidden states over those trace tokens, every layer: linear CKA (tokens subsampled every 4th -> 512 rows, Gram form),
CKA on 64-token window means (32 rows), and mean cosine drift. Writes <out>/<tag>.jsonl (pid + scalars, resumable); progress per trace.
  python run_cka.py --model M --screen S --keep K --pool P --perts PERTS --out out/cka"""
import argparse, json, random, sys, time, zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "submission_core"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch

N_TF, STRIDE = 2048, 4
T = lambda: time.strftime("%H:%M:%S")


def cka_gram(X, Y):
    X = X - X.mean(0, keepdim=True); Y = Y - Y.mean(0, keepdim=True)
    K, L = X @ X.T, Y @ Y.T
    return float((K * L).sum() / (K.norm() * L.norm() + 1e-12))


@torch.no_grad()
def states(model, p_ids, g):
    o = model(input_ids=torch.tensor([p_ids + g], device=model.device), output_hidden_states=True, use_cache=False, logits_to_keep=1)
    n_p = len(p_ids); out = [h[0, n_p:n_p + len(g)].float() for h in o.hidden_states[1:]]; del o
    return out


def main():
    ap = argparse.ArgumentParser()
    for k in ("model", "screen", "keep", "pool", "perts", "out"): ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    import spurious_inference as SI
    from spurious_features import build_prompt
    from direct_test import code_typos, code_rename, code_distract
    tag = a.model.replace("/", "__"); out = Path(a.out) / f"{tag}.jsonl"; out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["pid"] for l in open(out)} if out.exists() else set()
    scr = {json.loads(l)["pid"]: json.loads(l) for l in open(a.screen)}
    prob = {json.loads(l)["pid"]: json.loads(l)["problem"] for l in open(a.pool)}
    perts = {}
    for l in open(a.perts): d = json.loads(l); perts.setdefault(d["pid"], {})[d["ptype"]] = d["text"]
    keep = [p for p in (l.strip() for l in open(a.keep)) if p and p in scr and p in prob and p not in done]
    if a.limit: keep = keep[:a.limit]
    print(T(), f"{a.model}: {len(keep)} traces to do ({len(done)} done) | {SI._gpu()}", flush=True)
    if not keep: return
    model, tok = SI.load_model_and_tokenizer(a.model, local_files_only=False); model.eval(); t0 = time.time()
    for k, p in enumerate(keep):
        t1 = time.time(); g = scr[p]["gen_ids"][:N_TF]; q = prob[p]; rng = random.Random(zlib.crc32(p.encode()))
        V = {f"sonnet_{t}": x for t, x in perts.get(p, {}).items()}
        V.update(code_typos=code_typos(q, rng), code_rename=code_rename(q, rng), code_distract=code_distract(q, rng))
        S = {}
        try:
            C = states(model, build_prompt(tok, q)[0], g); L = len(C); nw = len(g) // 64
            Cs = [c[::STRIDE] for c in C]; Cw = [c[:nw * 64].view(nw, 64, -1).mean(1) for c in C]
            for v, text in V.items():
                P = states(model, build_prompt(tok, text)[0], g)
                for l in range(L):
                    S[f"cka_{v}__l{l}"] = cka_gram(Cs[l], P[l][::STRIDE])
                    S[f"ckaw_{v}__l{l}"] = cka_gram(Cw[l], P[l][:nw * 64].view(nw, 64, -1).mean(1)) if nw >= 4 else float("nan")
                    S[f"drift_{v}__l{l}"] = float(1 - torch.nn.functional.cosine_similarity(C[l], P[l], dim=-1).mean())
                del P
            del C, Cs, Cw; torch.cuda.empty_cache()
        except Exception as e:                                           # noqa: BLE001 - record and continue
            import traceback; print(T(), f"  {p} FAILED: {e}\n{traceback.format_exc()}", flush=True); torch.cuda.empty_cache(); continue
        with open(out, "a") as fo: fo.write(json.dumps(dict(pid=p, model=a.model, n_tf=len(g), versions=sorted(V), **S)) + "\n")
        el = time.time() - t0
        print(T(), f"[{k + 1}/{len(keep)}] {p} {time.time() - t1:.1f}s ({len(V)} versions, {len(S)} scalars) | ETA {el / (k + 1) * (len(keep) - k - 1) / 60:.0f} min | {SI._gpu()}", flush=True)
    print(T(), f"done: {len(keep)} traces in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
