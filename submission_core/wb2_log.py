"""LOGGING ONLY (user 2026-10-31: put every pending white-box lead on the leaderboard). Never affects a call.
Per problem, on the submission's own greedy trace (<= 6144 tokens), mirrors vast_packages/screen/run_wb2.py + analysis/exp103 / exp104:
  anchoring cosine per layer; answer-commitment curve + logit lens (budgets 512..6144); PEARL prose noise (rates 0.10 / 0.25);
  K-means state graphs (centroids shipped, fitted label-free on our traces); difference-of-means direction projections (directions shipped,
  fitted on our labels; the leaderboard is only read, never fitted); exp104 delta-hyperbolicity / compression / fixed Poincare map;
  self-perturbation (code typos / rename / distractor, 3 extra greedy traces) for the models in SELF_PERTURB.
log_all(...) -> dict of scalars (the caller logs it as one 'WB2 {json}' line); self_perturb(...) -> dict."""
from __future__ import annotations

import copy, random, re, time, zlib
from pathlib import Path

import numpy as np
import torch

BUDGETS = (512, 1024, 2048, 3072, 4096, 6144)
MATH = re.compile(r"(\$\$.*?\$\$|\$[^$]*\$|\\\(.*?\\\)|\\\[.*?\\\])", re.S)
SELF_PERTURB = ("allenai/Olmo-3-7B-Think",)
DISTRACT = ["Note: this problem was posed on a rainy Tuesday afternoon.", "A student's younger brother is learning to play the violin.",
            "The library where this problem was found closes at 6 pm.", "Fun fact: honey never spoils if it is stored properly."]


def _unit(x): return x / (x.norm(dim=-1, keepdim=True) + 1e-8)
def _unitn(X): return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)


def typos(text, rate, seed):
    rng = random.Random(seed); parts = MATH.split(text)
    def sw(w):
        if len(w) < 4 or not w.isalpha() or rng.random() > rate: return w
        j = rng.randrange(1, len(w) - 2); return w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return "".join(p if MATH.fullmatch(p) else re.sub(r"(?<!\\)\b[A-Za-z]+", lambda m: sw(m.group(0)), p) for p in parts)


def final_norm(model):
    best = None
    for name, mod in model.named_modules():
        if name.endswith("norm") and "layers" not in name: best = mod
    return best


@torch.no_grad()
def hidden_pass(model, pids, gen):
    n_p, n = len(pids), len(gen)
    out = model(input_ids=torch.tensor([pids + gen], device=model.device), output_hidden_states=True, use_cache=False, logits_to_keep=1)
    hs = out.hidden_states[1:]; L = len(hs); S = {}
    nw = n // 64; keep3 = sorted({int(round(f * (L - 1))) for f in (.25, .5, .75)})
    PL, TM, SEG, W3 = [], [], [], []
    for l, h in enumerate(hs):
        g = h[0, n_p:n_p + nw * 64].float(); W = g.view(nw, 64, -1).mean(1); pm = h[0, :n_p].float().mean(0); pl = h[0, n_p - 1].float()
        for nm, ref in (("pm", pm), ("pl", pl)):
            c = (_unit(W) @ _unit(ref)).cpu().numpy(); q = max(1, nw // 4)
            S[f"anchor_{nm}_mean__l{l}"] = float(c.mean()); S[f"anchor_{nm}_lastq__l{l}"] = float(c[-q:].mean())
            S[f"anchor_{nm}_slope__l{l}"] = float(np.polyfit(np.arange(nw), c, 1)[0]) if nw > 2 else float("nan")
            S[f"anchor_{nm}_drop__l{l}"] = float(c[:q].mean() - c[-q:].mean())
        if l % 2 == 0 or l == L - 1:
            PL.append(pl); TM.append(g.mean(0))
            SEG.append(torch.stack([g[max(0, int(f * len(g)) - 128):int(f * len(g)) + 128].mean(0) for f in (.25, .5, .75)]))
        if l in keep3: W3.append(W)
    A = dict(prompt_last=torch.stack(PL).cpu().numpy(), trace_mean=torch.stack(TM).cpu().numpy(), seg_means=torch.stack(SEG).cpu().numpy(),
             w64_3layers=torch.stack(W3).cpu().numpy())
    del out, hs; torch.cuda.empty_cache()
    return S, A


@torch.no_grad()
def commitment(model, pids, gen, sfx, norm, lm_head):
    S = {}; cache, done = None, 0; ids = pids + gen; recs = []
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
        hl = torch.stack([h[0, -1] for h in o.hidden_states[1:]])
        lens = torch.cat([torch.log_softmax(lm_head(norm(hl[:-1])).float(), -1), lp[None]])
        recs.append(dict(b=b, lp=lp.cpu(), top=top, p1=float(p[top]), ent=float(-(p * lp).sum()), lens=lens.cpu())); del o, hl
    fin = recs[-1]; L = fin["lens"].shape[0]
    for r in recs:
        S[f"force_ent__b{r['b']}"] = r["ent"]; S[f"force_p1__b{r['b']}"] = r["p1"]; S[f"force_same_as_final__b{r['b']}"] = float(r["top"] == fin["top"])
        lt = r["lens"].argmax(-1).numpy(); stay = np.where(lt != r["top"])[0]
        S[f"lens_emerge__b{r['b']}"] = float((stay.max() + 1) / L if len(stay) else 0.0)
        pe = r["lens"].exp(); S[f"lens_ent_late__b{r['b']}"] = float(-(pe * r["lens"]).sum(-1)[L // 2:].mean()); S[f"lens_ptop_mid__b{r['b']}"] = float(pe[L // 2, r["top"]])
    tops = [r["top"] for r in recs]; same = [t == fin["top"] for t in tops]; k = len(same) - 1
    while k > 0 and same[k - 1]: k -= 1
    S["commit_budget"] = float(recs[k]["b"]); S["commit_frac"] = float(k / max(1, len(recs) - 1)); S["n_answer_changes"] = float(sum(a != b for a, b in zip(tops, tops[1:])))
    kls = [float((recs[i]["lp"].exp() * (recs[i]["lp"] - recs[i + 1]["lp"])).sum()) for i in range(len(recs) - 1)]
    S["force_kl_mean"] = float(np.mean(kls)) if kls else float("nan"); S["force_kl_last"] = kls[-1] if kls else float("nan")
    S["force_ent_slope"] = float(np.polyfit([r["b"] for r in recs], [r["ent"] for r in recs], 1)[0]) if len(recs) > 2 else float("nan")
    del cache; torch.cuda.empty_cache()
    return S


@torch.no_grad()
def pearl(model, tok, problem, gen, norm, lm_head, build_prompt, n_tf=2048):
    S = {}; g = gen[:n_tf]; n = len(g)
    def run(p_ids):
        o = model(input_ids=torch.tensor([p_ids + g], device=model.device), output_hidden_states=True, use_cache=False, logits_to_keep=1)
        hs = o.hidden_states; L = len(hs) - 1; n_p = len(p_ids)
        r = (hs[-1][0, n_p - 1:n_p + n - 1], hs[1 + L // 2][0, n_p:n_p + n].float(), hs[-1][0, n_p:n_p + n].float()); del o; return r
    pids = build_prompt(tok, problem)[0]; c_last, c_mid, c_fin = run(pids); tgt = torch.tensor(g, device=model.device)
    for rate in (0.10, 0.25):
        nids = build_prompt(tok, typos(problem, rate, 1234))[0]; n_last, n_mid, n_fin = run(nids); kl, flip, dnll = [], [], []
        for s in range(0, n, 256):
            lc = torch.log_softmax(lm_head(c_last[s:s + 256]).float(), -1); ln = torch.log_softmax(lm_head(n_last[s:s + 256]).float(), -1)
            kl.append((lc.exp() * (lc - ln)).sum(-1)); flip.append((lc.argmax(-1) != ln.argmax(-1)).float())
            t = tgt[s:s + 256]; dnll.append(lc.gather(1, t[:, None])[:, 0] - ln.gather(1, t[:, None])[:, 0]); del lc, ln
        kl, flip, dnll = torch.cat(kl), torch.cat(flip), torch.cat(dnll); q = max(1, n // 4)
        S[f"pearl_kl__r{rate}"] = float(kl.mean()); S[f"pearl_kl_lastq__r{rate}"] = float(kl[-q:].mean()); S[f"pearl_flip__r{rate}"] = float(flip.mean())
        S[f"pearl_dnll__r{rate}"] = float(dnll.mean())
        S[f"pearl_drift_mid__r{rate}"] = float(1 - (_unit(c_mid) * _unit(n_mid)).sum(-1).mean()); S[f"pearl_drift_last__r{rate}"] = float(1 - (_unit(c_fin) * _unit(n_fin)).sum(-1).mean())
        S[f"pearl_changed_tokens__r{rate}"] = float(len(nids) - sum(a == b for a, b in zip(nids, pids))); del n_last, n_mid, n_fin
    torch.cuda.empty_cache()
    return S


# ---- exp103 / exp104 features from the window trajectories (numpy) ----
def _walk(seq, k):
    n = len(seq); T = np.zeros((k, k))
    for a, b in zip(seq, seq[1:]): T[a, b] += 1
    E = (T + T.T > 0) & ~np.eye(k, dtype=bool); u = np.unique(seq); V = u.size; ne = E.sum() / 2
    P = T / np.maximum(T.sum(1, keepdims=True), 1); pi = np.bincount(seq, minlength=k) / n
    with np.errstate(all="ignore"): er = float(-np.nansum(pi[:, None] * P * np.log(np.where(P > 0, P, 1))))
    Lp = np.diag(E.sum(1)) - E; ev = np.sort(np.linalg.eigvalsh(Lp[np.ix_(u, u)].astype(float)))
    seen, ret = set(), 0
    for i, s in enumerate(seq):
        if s in seen and seq[i - 1] != s: ret += 1
        seen.add(s)
    return dict(cycle_rank=float(ne - V + 1), entropy_rate=er, self_loops=float(np.trace(T) / max(1, n - 1)), distinct=float(V / n),
                returns=float(ret / n), spectral_gap=float(ev[1]) if len(ev) > 1 else 0.0)


def _delta(D, bases):
    out = 0.0
    for w in bases:
        G = .5 * (D[w][:, None] + D[w][None, :] - D); M = np.max(np.minimum(G[:, :, None], G.T[None, :, :]), axis=1); out = max(out, float((M - G).max()))
    return out


def _dists(X, kind):
    if kind == "cos": U = _unitn(X); return np.clip(1 - U @ U.T, 0, None)
    s = (X ** 2).sum(1); return np.sqrt(np.clip(s[:, None] + s[None, :] - 2 * X @ X.T, 0, None))


def _poincare(Z, c):
    n2 = (Z ** 2).sum(1); d2 = np.clip(n2[:, None] + n2[None, :] - 2 * Z @ Z.T, 0, None)
    return np.arccosh(1 + 2 * c * d2 / np.outer(1 - c * n2, 1 - c * n2)) / np.sqrt(c)


def geometry(A, ship):
    """exp103 K-means graphs + directions, exp104 delta / compression / Poincare features. ship: npz dict or None."""
    S = {}
    for di in range(A["w64_3layers"].shape[0]):
        W = A["w64_3layers"][di].astype(np.float64); n = len(W); h = n // 2
        if n < 8: continue
        bases = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
        for kind in ("cos", "l2"):
            D = _dists(W, kind); dl = _delta(D, bases); S[f"delta_{kind}__d{di}"] = dl; S[f"delta_rel_{kind}__d{di}"] = 2 * dl / (D.max() + 1e-12)
            d1 = _delta(D[:h, :h], [0, h // 2, h - 1]); d2 = _delta(D[h:, h:], [0, (n - h) // 2, n - h - 1])
            S[f"delta_rel_late_minus_early_{kind}__d{di}"] = 2 * d2 / (D[h:, h:].max() + 1e-12) - 2 * d1 / (D[:h, :h].max() + 1e-12)
        Wc = W - W.mean(0); ev = np.clip(np.linalg.eigvalsh(Wc @ Wc.T / n), 0, None)[::-1]
        S[f"pr__d{di}"] = float(ev.sum() ** 2 / ((ev ** 2).sum() + 1e-12)); S[f"top1_share__d{di}"] = float(ev[0] / (ev.sum() + 1e-12)); S[f"top5_share__d{di}"] = float(ev[:5].sum() / (ev.sum() + 1e-12))
        nr = np.linalg.norm(W, axis=1); S[f"norm_mean__d{di}"] = float(nr.mean()); S[f"norm_cv__d{di}"] = float(nr.std() / nr.mean())
        if ship is not None:
            T = (W - ship[f"pca_mu_d{di}"]) @ ship[f"pca_V_d{di}"].T * float(ship[f"pca_s_d{di}"]); iu = np.triu_indices(n, 1)
            for c in (0.5, 1.0):
                nv = np.linalg.norm(T, axis=1, keepdims=True) + 1e-12; Zb = np.tanh(np.sqrt(c) * nv) * T / (np.sqrt(c) * nv); Dp = _poincare(Zb, c); De = _dists(T, "l2")
                S[f"hyp_path__c{c}_d{di}"] = float(np.mean(np.diag(Dp, 1))); S[f"hyp_spread__c{c}_d{di}"] = float(Dp[iu].mean()); S[f"hyp_over_euc__c{c}_d{di}"] = float(Dp[iu].mean() / (De[iu].mean() + 1e-12))
            U = _unitn(W.astype(np.float32))
            for k in (16, 32):
                seq = np.argmax(U @ ship[f"km{k}_d{di}"].astype(np.float32).T, 1)
                for nm, v in _walk(seq, k).items(): S[f"kg_{nm}__k{k}_d{di}"] = v
    if "norm_mean__d0" in S:
        for k in ("pr", "top1_share", "norm_mean", "norm_cv"): S[f"{k}_ratio_deep_shallow"] = S[f"{k}__d2"] / (S[f"{k}__d0"] + 1e-12)
    tn = np.linalg.norm(A["trace_mean"].astype(np.float64), axis=1); S["norm_last_over_prev"] = float(tn[-1] / tn[-2]); S["norm_slope_layers"] = float(np.polyfit(np.arange(len(tn)), tn / tn.mean(), 1)[0])
    if ship is not None:                                                 # projections on the shipped difference-of-means directions, every stored layer
        for kind, arr in (("prompt_last", A["prompt_last"]), ("trace_mean", A["trace_mean"]), ("seg25", A["seg_means"][:, 0]), ("seg50", A["seg_means"][:, 1]), ("seg75", A["seg_means"][:, 2])):
            Dk = ship.get(f"dir_{kind}")
            if Dk is not None and Dk.shape[0] == arr.shape[0]:
                for l, v in enumerate((arr.astype(np.float32) * Dk.astype(np.float32)).sum(1)): S[f"dirproj_{kind}__s{l}"] = float(v)
    return S


def log_all(model, tok, prompt_ids, gen, sfx, problem, model_id, art_dir, build_prompt, log=print):
    t0 = time.time(); S, tm = {}, []
    ship_f = Path(art_dir) / f"wb2_ship_{model_id.replace('/', '__')}.npz"; ship = dict(np.load(ship_f)) if ship_f.exists() else None
    norm, lm_head = final_norm(model), model.get_output_embeddings(); A = None
    for nm, fn in (("hidden", lambda: hidden_pass(model, prompt_ids, gen)), ("commit", lambda: (commitment(model, prompt_ids, gen, sfx, norm, lm_head), None)),
                   ("pearl", lambda: (pearl(model, tok, problem, gen, norm, lm_head, build_prompt), None))):
        ts = time.time()
        try:
            s, a = fn(); S.update(s); A = a if a is not None else A
        except Exception as e:                                           # noqa: BLE001 - logging only
            log(f"  wb2 {nm} FAILED (logging only): {e}"); torch.cuda.empty_cache()
        tm.append(f"{nm} {time.time() - ts:.1f}s")
    if A is not None:
        ts = time.time()
        try: S.update(geometry(A, ship))
        except Exception as e: log(f"  wb2 geometry FAILED (logging only): {e}")   # noqa: BLE001
        tm.append(f"geometry {time.time() - ts:.1f}s")
    log(f"  wb2: {len(S)} scalars in {time.time() - t0:.1f}s ({', '.join(tm)}) | ship {'yes' if ship is not None else 'MISSING'}")
    return S


# ---- self-perturbation (exp93): does the answer survive code-made perturbations? ----
def _boxed(text):
    i = text.rfind("\\boxed{")
    if i < 0: return None
    j, depth = i + 7, 1
    while j < len(text) and depth: depth += {"{": 1, "}": -1}.get(text[j], 0); j += 1
    return text[i + 7:j - 1].strip() if depth == 0 else None


def _norm(s):
    if s is None: return None
    s = s.replace(",", "").replace("$", "").replace("\\!", "").replace(" ", ""); return str(int(s)) if re.fullmatch(r"-?\d+", s) else s


def _code_typos(text, rng):
    parts = MATH.split(text)
    def sw(w):
        if len(w) < 5 or not w.isalpha() or rng.random() > 0.12: return w
        j = rng.randrange(1, len(w) - 2); return w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return "".join(p if MATH.fullmatch(p) else re.sub(r"(?<!\\)\b[A-Za-z]+", lambda m: sw(m.group(0)), p) for p in parts)


def _code_rename(text, rng):
    parts = MATH.split(text); pat = re.compile(r"(?<![\\A-Za-z])([A-Za-z])(?![A-Za-z])"); ism = lambda p: bool(MATH.fullmatch(p))
    used = {m for p in parts if ism(p) for m in pat.findall(p)}
    fl = [c for c in "abcghjkmnpqrstuvwxyz" if c not in used]; fu = [c for c in "ABCDFGHJKLMNPQRSTUVWXYZ" if c not in used]; rng.shuffle(fl); rng.shuffle(fu); mp = {}
    for v in sorted(used):
        if v in set("eidEI"): continue
        pool = fl if v.islower() else fu
        if pool: mp[v] = pool.pop()
    return "".join(pat.sub(lambda m: mp.get(m.group(1), m.group(1)), p) if ism(p) else p for p in parts)


def self_perturb(model, tok, problem, prompt_ids, gen, sfx, SI, find_think_end, tid, build_prompt, max_time, log=print):
    t0 = time.time(); rng = random.Random(zlib.crc32(problem.encode()))
    V = {"code_typos": _code_typos(problem, rng), "code_rename": _code_rename(problem, rng), "code_distract": rng.choice(DISTRACT) + " " + problem}
    prompts = {v: build_prompt(tok, t)[0] for v, t in V.items()}
    outs = SI.generate_batch(model, tok, list(prompts.values()), 6144, max_time)
    vers = {"original": (prompt_ids, gen)} | {v: (prompts[v], g) for v, (g, _) in zip(prompts, outs)}
    res = {}
    for v, (p, g) in vers.items():
        e = find_think_end(tok, g, tid)
        if e is not None: ans = _norm(_boxed(tok.decode(g[e:], skip_special_tokens=True)))
        else:
            f = SI.generate_batch(model, tok, [p + g + sfx], 16, 60)[0][0]; ans = _norm(_boxed("\\boxed{" + tok.decode(f, skip_special_tokens=True)))
        res[v] = dict(answer=ans, ended=e is not None, n=len(g))
    o = res["original"]["answer"]; agree = {v: bool(d["answer"] == o) for v, d in res.items() if v != "original"}
    out = dict(versions=res, agree=agree, agree_code=float(np.mean(list(agree.values()))), seconds=round(time.time() - t0, 1))
    log(f"  self-perturbation: agreement {out['agree_code']:.2f} ({sum(agree.values())}/{len(agree)}) in {out['seconds']}s")
    return out
