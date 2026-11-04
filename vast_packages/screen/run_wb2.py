"""White-box retry, part 2 (user 2026-10-31: rerun the OLD white-box approaches the battery did not cover, on the new labelled traces).
Per stored greedy trace (first 6144 tokens), one model load, three passes:
  1. hidden pass (all layers): anchoring cosine of every 64-token window to the prompt (mean / last-token state) per layer; saved float16
     arrays for CPU analysis: prompt last-token + prompt mean + trace mean + 256-token segment means at 25/50/75% (every 2nd layer), and the
     64-token window means of 3 layers (25/50/75% depth) for K-means state graphs.
  2. answer-commitment curve: forced answer (suffix B) at budgets 512..6144 from ONE incremental prefix cache: entropy / p1 / top-1 per
     budget, KL between consecutive budgets, and a logit lens over every layer at each forced position (MemLens-style emergence depth).
  3. PEARL: letter typos in the prose (rates 0.10 / 0.25, never inside math) -> teacher-force the first 2048 trace tokens: mean KL(clean||noisy),
     top-1 flip rate, NLL change of the trace tokens, hidden-state drift at mid and last layer.
Writes <out>/<tag>.jsonl (pid + scalars, resumable) and <out>/<tag>/<pid>.npz. Progress line per trace with timings and ETA.
  python run_wb2.py --model Qwen/Qwen3.5-4B --screen <screen.jsonl> --keep set100.txt --pool pool.jsonl --out out/wb2"""
import argparse, copy, json, random, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "submission_core"))
import numpy as np
import torch

BUDGETS = (512, 1024, 2048, 3072, 4096, 6144)
MATH = re.compile(r"(\$\$.*?\$\$|\$[^$]*\$|\\\(.*?\\\)|\\\[.*?\\\])", re.S)
T = lambda: time.strftime("%H:%M:%S")


def typos(text, rate, seed):
    """swap two inner letters in `rate` of plain words of length >= 4, outside math (PEARL-style prose noise)"""
    rng = random.Random(seed); parts = MATH.split(text)
    def sw(w):
        if len(w) < 4 or not w.isalpha() or rng.random() > rate: return w
        j = rng.randrange(1, len(w) - 2); return w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return "".join(p if MATH.fullmatch(p) else re.sub(r"(?<!\\)\b[A-Za-z]+", lambda m: sw(m.group(0)), p) for p in parts)


def final_norm(model):
    best = None
    for name, mod in model.named_modules():
        if name.endswith("norm") and ".layers." not in name and "layers" not in name.split(".")[-1]: best = mod
    return best


def unit(x): return x / (x.norm(dim=-1, keepdim=True) + 1e-8)


@torch.no_grad()
def hidden_pass(model, pids, gen):
    n_p, n = len(pids), len(gen)
    out = model(input_ids=torch.tensor([pids + gen], device=model.device), output_hidden_states=True, use_cache=False, logits_to_keep=1)
    hs = out.hidden_states[1:]; L = len(hs); S, A = {}, {}
    nw = n // 64; keep3 = sorted({int(round(f * (L - 1))) for f in (.25, .5, .75)})
    PL, PM, TM, SEG, W3 = [], [], [], [], []
    for l, h in enumerate(hs):
        g = h[0, n_p:n_p + nw * 64].float(); W = g.view(nw, 64, -1).mean(1); pm = h[0, :n_p].float().mean(0); pl = h[0, n_p - 1].float()
        for nm, ref in (("pm", pm), ("pl", pl)):
            c = (unit(W) @ unit(ref)).cpu().numpy(); q = max(1, nw // 4)
            S[f"anchor_{nm}_mean__l{l}"] = float(c.mean()); S[f"anchor_{nm}_lastq__l{l}"] = float(c[-q:].mean())
            S[f"anchor_{nm}_slope__l{l}"] = float(np.polyfit(np.arange(nw), c, 1)[0]) if nw > 2 else np.nan
            S[f"anchor_{nm}_drop__l{l}"] = float(c[:q].mean() - c[-q:].mean())
        if l % 2 == 0 or l == L - 1:
            PL.append(pl); PM.append(pm); TM.append(g.mean(0))
            SEG.append(torch.stack([g[max(0, int(f * len(g)) - 128):int(f * len(g)) + 128].mean(0) for f in (.25, .5, .75)]))
        if l in keep3: W3.append(W)
    A["prompt_last"] = torch.stack(PL).half().cpu().numpy(); A["prompt_mean"] = torch.stack(PM).half().cpu().numpy()
    A["trace_mean"] = torch.stack(TM).half().cpu().numpy(); A["seg_means"] = torch.stack(SEG).half().cpu().numpy()   # [layers/2, 3, d]
    A["w64_3layers"] = torch.stack(W3).half().cpu().numpy(); A["w64_layers"] = np.array(keep3)
    del out, hs; torch.cuda.empty_cache()
    return S, A


@torch.no_grad()
def commitment(model, pids, gen, sfx, norm, lm_head):
    """forced answer at each budget from one incremental prefix cache + logit lens over layers at each forced position"""
    S, A = {}, {}; cache, done = None, 0; ids = pids + gen; recs = []
    for b in [b for b in BUDGETS if b <= len(gen)] or [len(gen)]:
        end = len(pids) + b
        for s in range(done, end, 1024):
            o = model(input_ids=torch.tensor([ids[s:min(s + 1024, end)]], device=model.device), past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = o.past_key_values; del o
        done = end
        try:
            br = copy.deepcopy(cache); o = model(input_ids=torch.tensor([sfx], device=model.device), past_key_values=br, use_cache=True, output_hidden_states=True, logits_to_keep=1); del br
        except (RuntimeError, TypeError, AttributeError) as e:
            if isinstance(e, torch.cuda.OutOfMemoryError): raise
            o = model(input_ids=torch.tensor([ids[:end] + sfx], device=model.device), use_cache=False, output_hidden_states=True, logits_to_keep=1)
        lp = torch.log_softmax(o.logits[0, -1].float(), -1); p = lp.exp(); top = int(lp.argmax())
        hl = torch.stack([h[0, -1] for h in o.hidden_states[1:]])                     # [L, d]
        lens = torch.log_softmax(lm_head(norm(hl[:-1])).float(), -1)                  # layers before the last (last is already normed)
        lens = torch.cat([lens, lp[None]]); recs.append(dict(b=b, lp=lp.cpu(), top=top, p1=float(p[top]), ent=float(-(p * lp).sum()), lens=lens.cpu()))
        del o, hl
    fin = recs[-1]; L = fin["lens"].shape[0]
    for r in recs:
        S[f"force_ent__b{r['b']}"] = r["ent"]; S[f"force_p1__b{r['b']}"] = r["p1"]; S[f"force_same_as_final__b{r['b']}"] = float(r["top"] == fin["top"])
        lt = r["lens"].argmax(-1).numpy(); stay = np.where(lt != r["top"])[0]
        S[f"lens_emerge__b{r['b']}"] = float((stay.max() + 1) / L if len(stay) else 0.0)          # first depth from which the final top-1 stays top-1
        pe = r["lens"].exp(); S[f"lens_ent_late__b{r['b']}"] = float(-(pe * r["lens"]).sum(-1)[L // 2:].mean())
        S[f"lens_ptop_mid__b{r['b']}"] = float(pe[L // 2, r["top"]])
    tops = [r["top"] for r in recs]; same = [t == fin["top"] for t in tops]
    k = len(same) - 1
    while k > 0 and same[k - 1]: k -= 1
    S["commit_budget"] = float(recs[k]["b"]); S["commit_frac"] = float(k / max(1, len(recs) - 1)); S["n_answer_changes"] = float(sum(a != b for a, b in zip(tops, tops[1:])))
    kls = [float((recs[i]["lp"].exp() * (recs[i]["lp"] - recs[i + 1]["lp"])).sum()) for i in range(len(recs) - 1)]
    S["force_kl_mean"] = float(np.mean(kls)) if kls else np.nan; S["force_kl_last"] = kls[-1] if kls else np.nan
    S["force_ent_slope"] = float(np.polyfit([r["b"] for r in recs], [r["ent"] for r in recs], 1)[0]) if len(recs) > 2 else np.nan
    A["force_curve"] = np.array([[r["b"], r["ent"], r["p1"], r["top"]] for r in recs], float)
    del cache; torch.cuda.empty_cache()
    return S, A


@torch.no_grad()
def pearl(model, tok, problem, pids, gen, norm, lm_head, build_prompt, n_tf=2048):
    S = {}; g = gen[:n_tf]; n = len(g)
    def run(p_ids):
        o = model(input_ids=torch.tensor([p_ids + g], device=model.device), output_hidden_states=True, use_cache=False, logits_to_keep=1)
        hs = o.hidden_states; L = len(hs) - 1; n_p = len(p_ids)
        last = hs[-1][0, n_p - 1:n_p + n - 1]; mid = hs[1 + L // 2][0, n_p:n_p + n].float(); fin = hs[-1][0, n_p:n_p + n].float()
        del o; return last, mid, fin
    pids = build_prompt(tok, problem)[0]                                # clean prompt through the same tokenizer as the noisy ones
    c_last, c_mid, c_fin = run(pids); tgt = torch.tensor(g, device=model.device)
    for rate in (0.10, 0.25):
        nids = build_prompt(tok, typos(problem, rate, 1234))[0]
        n_last, n_mid, n_fin = run(nids); kl, flip, dnll = [], [], []
        for s in range(0, n, 256):
            lc = torch.log_softmax(lm_head(c_last[s:s + 256]).float(), -1); ln = torch.log_softmax(lm_head(n_last[s:s + 256]).float(), -1)
            kl.append((lc.exp() * (lc - ln)).sum(-1)); flip.append((lc.argmax(-1) != ln.argmax(-1)).float())
            t = tgt[s:s + 256]; dnll.append(lc.gather(1, t[:, None])[:, 0] - ln.gather(1, t[:, None])[:, 0]); del lc, ln
        kl, flip, dnll = torch.cat(kl), torch.cat(flip), torch.cat(dnll); q = max(1, n // 4)
        S[f"pearl_kl__r{rate}"] = float(kl.mean()); S[f"pearl_kl_lastq__r{rate}"] = float(kl[-q:].mean()); S[f"pearl_flip__r{rate}"] = float(flip.mean())
        S[f"pearl_dnll__r{rate}"] = float(dnll.mean())
        S[f"pearl_drift_mid__r{rate}"] = float(1 - (unit(c_mid) * unit(n_mid)).sum(-1).mean()); S[f"pearl_drift_last__r{rate}"] = float(1 - (unit(c_fin) * unit(n_fin)).sum(-1).mean())
        S[f"pearl_changed_tokens__r{rate}"] = float(len(nids) - sum(a == b for a, b in zip(nids, pids)))
        del n_last, n_mid, n_fin
    torch.cuda.empty_cache()
    return S


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--screen", required=True); ap.add_argument("--keep", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pool", required=True, help="jsonl with pid + problem text (for PEARL)"); a = ap.parse_args()
    import spurious_inference as SI
    from spurious_features import SUFFIXES_THINK, build_prompt
    tag = a.model.replace("/", "__"); out = Path(a.out) / f"{tag}.jsonl"; nd = Path(a.out) / tag; nd.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["pid"] for l in open(out)} if out.exists() else set()
    scr = {json.loads(l)["pid"]: json.loads(l) for l in open(a.screen)}
    prob = {json.loads(l)["pid"]: json.loads(l)["problem"] for l in open(a.pool)}
    keep = [p for p in (l.strip() for l in open(a.keep)) if p and p in scr and p in prob and p not in done]
    if a.limit: keep = keep[:a.limit]
    print(T(), f"{a.model}: {len(keep)} traces to do ({len(done)} done) | {SI._gpu()}", flush=True)
    if not keep: return
    model, tok = SI.load_model_and_tokenizer(a.model, local_files_only=False); model.eval()
    norm, lm_head = final_norm(model), model.get_output_embeddings()
    print(T(), f"final norm {type(norm).__name__}, lm_head {tuple(lm_head.weight.shape)}", flush=True)
    sfx = tok(SUFFIXES_THINK["B"], add_special_tokens=False)["input_ids"]; t0 = time.time()
    for k, p in enumerate(keep):
        r = scr[p]; t1 = time.time(); gen = r["gen_ids"][:6144]
        pids = r.get("prompt_ids") or build_prompt(tok, prob[p])[0]
        S, A, tm = {}, {}, []
        for nm, fn in (("hidden", lambda: hidden_pass(model, pids, gen)), ("commit", lambda: commitment(model, pids, gen, sfx, norm, lm_head)),
                       ("pearl", lambda: (pearl(model, tok, prob[p], pids, gen, norm, lm_head, build_prompt), {}))):
            ts = time.time()
            try:
                s, arr = fn(); S.update(s); A.update(arr)
            except Exception as e:                                       # noqa: BLE001 - record and continue
                import traceback; print(T(), f"  {p} {nm} FAILED: {e}\n{traceback.format_exc()}", flush=True); torch.cuda.empty_cache()
            tm.append(f"{nm} {time.time() - ts:.1f}s")
        np.savez_compressed(nd / f"{p}.npz", **A)
        with open(out, "a") as fo: fo.write(json.dumps(dict(pid=p, model=a.model, n_gen=len(gen), still_thinking=r.get("still_thinking"), **S)) + "\n")
        el = time.time() - t0
        print(T(), f"[{k + 1}/{len(keep)}] {p} {time.time() - t1:.0f}s ({', '.join(tm)}; {len(S)} scalars) | ETA {el / (k + 1) * (len(keep) - k - 1) / 60:.0f} min | {SI._gpu()}", flush=True)
    print(T(), f"done: {len(keep)} traces in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
