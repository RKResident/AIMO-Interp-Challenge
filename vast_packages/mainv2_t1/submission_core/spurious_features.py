"""Prompt construction and feature formulas for the spurious-reasoning submissions.

Everything here mirrors our extraction / analysis code (extract_features.py, build_features.py, exp8, exp18, exp19, exp28, exp32)
and is checked against the values those scripts saved for the 43 main_v2 rows (test_against_saved.py). Pure Python / NumPy.
"""

from __future__ import annotations

import random
import re

import numpy as np

PROMPT_TEMPLATE = (
    "Solve the following mathematical problem step by step. "
    "Put your final answer within \\boxed{{}}.\n\n"
    "{problem}"
)
SUFFIXES_THINK = {
    "A": "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking "
         "directly now.\n</think>\n\n**Final Answer**\n\\boxed{",
    "B": "\n</think>\n\nThe final answer is $\\boxed{",
}
SUFFIXES_NOTHINK = {
    "A": "\n\nI need to stop here and give my final answer now.\n\n**Final Answer**\n\\boxed{",
    "B": "\n\nThe final answer is $\\boxed{",
}
TRAJ_LAYERS = {
    "Skywork/Skywork-OR1-Math-7B": [10, 13, 16, 19, 22, 25],
    "Qwen/Qwen3.5-4B": [10, 14, 18, 22, 26, 30],
    "allenai/Olmo-3-7B-Think": [10, 14, 18, 22, 26, 30],
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": [12, 16, 20, 24, 28, 32],
}
SEL_LAYER_FRACS = (0.25, 0.40, 0.55, 0.70, 0.85, 1.00)
BUDGETS = (512, 1024, 2048, 3072, 4096, 6144, 8192)
FRACS = (0.25, 0.5, 0.75, 0.9, 1.0)
PEARL_RATES = (0.02, 0.05, 0.1, 0.2, 0.3)
PEARL_SEEDS = 4
STRIDE = 16
LOOP_WINDOW, LOOP_REPEATS = 32, 3


def build_prompt(tokenizer, problem: str):
    """(token ids, rendered text, (start, end) of the problem tokens or None)."""
    user = PROMPT_TEMPLATE.format(problem=problem)
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    else:
        text = user
        enc = tokenizer(text, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    span = None
    c0 = text.rfind(problem)
    if c0 >= 0:
        c1 = c0 + len(problem)
        toks = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if s >= c0 and e <= c1 and e > s]
        if toks:
            span = (toks[0], toks[-1] + 1)
    return ids, text, span


def think_end_token_id(tokenizer):
    try:
        tid = tokenizer.convert_tokens_to_ids("</think>")
    except Exception:
        return None
    if tid is None or tid == getattr(tokenizer, "unk_token_id", None) or tokenizer.convert_ids_to_tokens(tid) != "</think>":
        return None
    return int(tid)


def find_think_end(tokenizer, gen, tid):
    """Index of the first generated token of '</think>' (None if thinking never ended)."""
    if tid is not None:
        return gen.index(tid) if tid in gen else None
    if "</think>" not in tokenizer.decode(gen):
        return None
    lo, hi = 0, len(gen)
    while lo < hi:
        mid = (lo + hi) // 2
        if "</think>" in tokenizer.decode(gen[:mid]):
            hi = mid
        else:
            lo = mid + 1
    j = max(lo - 1, 0)
    partial = ("<", "</", "</t", "</th", "</thi", "</thin", "</think")
    while j > 0 and tokenizer.decode(gen[:j]).endswith(partial):
        j -= 1
    return j


def ends_in_loop(tokens) -> bool:
    """True if the sequence ends with LOOP_REPEATS identical LOOP_WINDOW-token blocks."""
    n = LOOP_WINDOW * LOOP_REPEATS
    if len(tokens) < n:
        return False
    recent = tokens[-LOOP_WINDOW:]
    return all(tokens[len(tokens) - (r + 1) * LOOP_WINDOW:len(tokens) - r * LOOP_WINDOW] == recent for r in range(1, LOOP_REPEATS))


def find_loop_end(tokens):
    """First t such that tokens[:t] ends in a loop (the cut our extraction applied after generation)."""
    w = LOOP_WINDOW
    for t in range(w * LOOP_REPEATS, len(tokens) + 1):
        recent = tokens[t - w:t]
        if all(tokens[t - (r + 1) * w:t - r * w] == recent for r in range(1, LOOP_REPEATS)):
            return t
    return None


def cutoff_labels(L: int) -> dict:
    lab = {f"b{b}": max(1, min(b, L)) for b in BUDGETS}
    lab.update({f"f{int(round(f * 100))}": max(1, min(L, int(round(f * L)))) for f in FRACS})
    return lab


NUMBER_WORDS = set(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen "
    "nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion first second third fourth "
    "fifth sixth seventh eighth ninth tenth half twice once double triple".split()
)
_MATH = re.compile(r"(\$\$.*?\$\$|\$.*?\$|\\\[.*?\\\]|\\\(.*?\\\)|\\begin\{(\w+\*?)\}.*?\\end\{\2\}|\\[a-zA-Z]+|\{[^{}]*\}|https?://\S+)", re.S)


def prose_noise(text: str, rate: float, seed: int):
    """One random character edit in a fraction `rate` of the prose words (outside math, never number words)."""
    rng = random.Random(seed)
    pieces, last = [], 0
    for m in _MATH.finditer(text):
        pieces.append((text[last:m.start()], True))
        pieces.append((m.group(0), False))
        last = m.end()
    pieces.append((text[last:], True))
    cand = [(pi, w.start(), w.group(0)) for pi, (s, prose) in enumerate(pieces) if prose
            for w in re.finditer(r"[A-Za-z]{3,}", s) if w.group(0).lower() not in NUMBER_WORDS]
    if not cand or rate <= 0:
        return text, 0
    k = min(len(cand), max(1, round(rate * len(cand))))
    pick = sorted(rng.sample(cand, k), key=lambda t: (t[0], -t[1]))
    segs = [s for s, _ in pieces]

    def edit(w):
        j = rng.randrange(len(w))
        op = rng.choice(("swap", "delete", "dup", "sub"))
        if op == "swap" and len(w) >= 4:
            j = rng.randrange(1, len(w) - 2)
            return w[:j] + w[j + 1] + w[j] + w[j + 2:]
        if op == "delete":
            return w[:j] + w[j + 1:]
        if op == "dup":
            return w[:j] + w[j] + w[j:]
        c = rng.choice("abcdefghijklmnopqrstuvwxyz")
        return w[:j] + (c.upper() if w[j].isupper() else c) + w[j + 1:]

    for pi, st, w in pick:
        for _ in range(20):
            nw = edit(w)
            if nw != w and nw.lower() not in NUMBER_WORDS:
                break
        else:
            continue
        segs[pi] = segs[pi][:st] + nw + segs[pi][st + len(w):]
    return "".join(segs), k


def pearl_texts(problem: str) -> list[str]:
    """The original problem, then PEARL_SEEDS prose-typo copies at each rate in PEARL_RATES."""
    return [problem] + [prose_noise(problem, r, seed=1000 * s + int(r * 1000))[0] for r in PEARL_RATES for s in range(PEARL_SEEDS)]


def f_pearl_area(S) -> float:
    """S: [1 + 20, 6, d] mean problem-token states (original first) -> area under the cosine-distance curve over typo rates."""
    o = S[0]
    cosd = lambda A, B: 1 - (A * B).sum(-1) / (np.linalg.norm(A, axis=-1) * np.linalg.norm(B, axis=-1) + 1e-9)   # noqa: E731
    curve = [cosd(S[1 + k * PEARL_SEEDS:1 + (k + 1) * PEARL_SEEDS], o[None]).mean() for k in range(len(PEARL_RATES))]
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(curve, PEARL_RATES))


def f_seg25_pc1(ws, L: int) -> float:
    """First-PC variance share of the 64 window means in the 1024-token segment at 25% of thinking, mean over layers."""
    nw = min(len(ws), max(1, L // 16))
    if nw < 64:
        return float("nan")
    st = int(np.clip(0.25 * nw - 32, 0, nw - 64))
    vals = []
    for l in range(ws.shape[1]):
        H = ws[st:st + 64, l].astype(np.float32)
        Hc = H - H.mean(0)
        ev = np.clip(np.linalg.eigvalsh(Hc @ Hc.T), 0, None)
        vals.append(ev[-1] / (ev.sum() + 1e-12))
    return float(np.mean(vals))


def f_late_vs_early(ws) -> float:
    """Dot product of the mean unit direction of the first and last quarter of the window means, mean over layers."""
    vals = []
    for l in range(ws.shape[1]):
        H = ws[:, l].astype(np.float32)
        Wn = len(H)
        if Wn < 6:
            return float("nan")
        Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-9)
        vals.append(float(Hn[: Wn // 4].mean(0) @ Hn[-Wn // 4:].mean(0)))
    return float(np.mean(vals))


def f_global_eff(ws3, mu, L: int, b: int = 4096, k: int = 5) -> float:
    """Global efficiency of the cosine k-NN graph over 64-token mean states (model-mean centred) of the first b tokens."""
    if L < b:
        return float("nan")
    W = ws3.astype(np.float32) - mu
    nw = min(b // 16, len(W))
    S = W[: nw // 4 * 4].reshape(-1, 4, W.shape[1]).mean(1)
    S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-6)
    n = len(S)
    if n < 3 * k:
        return float("nan")
    C = S @ S.T
    np.fill_diagonal(C, -np.inf)
    nb = np.argsort(-C, axis=1)[:, :k]
    A = np.zeros((n, n), bool)
    A[np.repeat(np.arange(n), k), nb.ravel()] = True
    A |= A.T
    tot = 0.0
    for s in range(n):
        dist = np.full(n, -1)
        dist[s] = 0
        frontier, d = [s], 0
        while frontier:
            d += 1
            nxt = np.unique(np.nonzero(A[frontier].any(0) & (dist < 0))[0])
            dist[nxt] = d
            frontier = list(nxt)
        tot += (1.0 / dist[dist > 0]).sum()
    return float(tot / (n * (n - 1)))


def percentile(x: float, ref) -> float:
    ref = np.asarray(ref, float)
    return float(((ref < x).sum() + 0.5 * (ref == x).sum()) / len(ref))


def ensemble_score(values: dict, recipe: dict, model_id: str) -> float:
    """Mean over available features of (percentile among this model's main_v2 values, flipped if lower = robust) - 0.5."""
    u = []
    for f, x in values.items():
        if np.isfinite(x):
            refs = recipe["ref"][f]
            q = percentile(x, refs["*"] if recipe.get("pooled_refs") else refs[model_id])     # pooled: one scale for every model
            u.append((q if recipe["signs"][f] > 0 else 1 - q) - 0.5)
    return float(np.mean(u)) if u else 0.0


def rips_persistence(D):
    """Vietoris-Rips persistence diagrams (H0, H1) of a distance matrix, in NumPy only (no ripser / gudhi in the evaluation image).

    H0: union-find over edges in order of length (= minimum spanning tree); every merge is a bar [0, edge length).
    H1: Z/2 column reduction of the triangle boundaries, triangles generated per edge in filtration order (a triangle's filtration
    value = its longest edge). Only edges up to the enclosing radius are used (beyond it the complex is a cone, so every loop is dead),
    and an edge's triangles are skipped whenever no loop is alive (their columns would reduce to zero). Zero-length bars dropped.
    Returns (H0 [n-1, 2] finite bars + one infinite, H1 [m, 2]) like ripser's 'dgms'."""
    D = np.asarray(D, dtype=np.float64)
    n = len(D)
    iu, ju = np.triu_indices(n, 1)
    w = D[iu, ju]
    r_enc = float(np.min(np.max(D, axis=1)))
    order = np.argsort(w, kind="stable")
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    rank = np.full((n, n), np.iinfo(np.int64).max, dtype=np.int64)      # rank of edge (a, b) in the filtration, if added
    h0, h1 = [], []
    pivot_col = {}                     # edge rank -> reduced triangle column (set of edge ranks) whose lowest entry it is
    alive = 0                          # loops born and not yet dead
    lengths = w[order]
    for k, e in enumerate(order):
        a, b = int(iu[e]), int(ju[e])
        if lengths[k] > r_enc and not alive and len(h0) == n - 1:
            break
        rank[a, b] = rank[b, a] = k
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            h0.append((0.0, lengths[k]))
        else:
            alive += 1                 # edge k closes a cycle: a loop is born
        if not alive:
            continue
        ws = np.nonzero((rank[a] < k) & (rank[b] < k))[0]                 # triangles whose longest edge is k
        for c in ws:
            col = {k, int(rank[a, c]), int(rank[b, c])}
            low = k
            while low in pivot_col:
                col ^= pivot_col[low]
                if not col:
                    break
                low = max(col)
            if col:
                pivot_col[low] = col
                birth, death = lengths[low], lengths[k]
                if death > birth:
                    h1.append((birth, death))
                alive -= 1
                if not alive:
                    break
    h0.append((0.0, np.inf))
    return [np.array(h0, dtype=np.float64), np.array(h1, dtype=np.float64).reshape(-1, 2)]


def mst_edge_lengths(D):
    """Edge lengths of a minimum spanning tree of a full distance matrix (Prim, O(n^2)) = the finite H0 death times of its
    Vietoris-Rips filtration."""
    n = len(D)
    in_tree = np.zeros(n, bool); in_tree[0] = True
    best = D[0].astype(np.float64).copy(); best[0] = np.inf
    out = []
    for _ in range(n - 1):
        j = int(np.argmin(np.where(in_tree, np.inf, best)))
        out.append(best[j]); in_tree[j] = True
        best = np.minimum(best, D[j]); best[in_tree] = np.inf
    return np.array(out)


def f_tda_h0_spread(ws3, L: int) -> float:
    """H0 Betti-curve spread ("The Shape of Reasoning", arXiv 2510.20665; exp35 / exp38) of the raw 64-token mean states of the
    thinking part at the 4th trajectory layer: cosine-distance Vietoris-Rips persistence, Betti curve on a 200-point grid over
    [0, largest finite death in H0 or H1]; spread = normalised standard deviation of the curve about its centroid. Higher -> robust."""
    nw = min(len(ws3), max(1, L // 16))
    X = ws3[: nw // 4 * 4].astype(np.float64)
    if len(X) < 40:                                      # fewer than 10 nodes
        return float("nan")
    S = X.reshape(-1, 4, X.shape[1]).mean(1)
    S /= np.linalg.norm(S, axis=1, keepdims=True) + 1e-9
    D = np.clip(1 - S @ S.T, 0, None); np.fill_diagonal(D, 0)
    h0, h1 = rips_persistence(D)
    h0 = h0[np.isfinite(h0[:, 1])]; h1 = h1[h1[:, 1] > h1[:, 0]]
    eps = max(h0[:, 1].max() if len(h0) else 0.0, h1[:, 1].max() if len(h1) else 0.0)
    if eps <= 0:
        return float("nan")
    grid = np.linspace(0, eps, 200); x = grid / eps
    beta = ((h0[:, :1] <= grid[None]) & (grid[None] < h0[:, 1:2])).sum(0).astype(float)
    c = (x * beta).sum() / beta.sum()
    return float(np.sqrt(((x - c) ** 2 * beta).sum() / beta.sum()))


def cascade_score(values: dict, recipe: dict, model_id: str) -> float:
    """Hierarchical score: the primary feature decides; values at or beyond its saturation point count as tied (on the worker their
    differences are at numeric-noise level), and the secondary features order those ties.
    score = percentile(primary clamped at saturation) + tie_weight * (mean secondary percentile - 0.5), all signed as in ensemble_score."""
    c = recipe["cascade"]
    p = values.get(c["primary"], np.nan)
    if np.isfinite(p):
        sat = c.get("saturate_at")
        if sat is not None:
            p = min(p, sat) if recipe["signs"][c["primary"]] > 0 else max(p, sat)
            ref = [min(v, sat) if recipe["signs"][c["primary"]] > 0 else max(v, sat) for v in recipe["ref"][c["primary"]][model_id]]
        else:
            ref = recipe["ref"][c["primary"]][model_id]
        q = percentile(p, ref)
        prim = (q if recipe["signs"][c["primary"]] > 0 else 1 - q) - 0.5
    else:
        prim = 0.0
    sec = ensemble_score({f: values.get(f, np.nan) for f in c["secondary"]}, recipe, model_id)
    return float(prim + c.get("tie_weight", 0.01) * sec)


def score_trace(values: dict, recipe: dict, model_id: str) -> float:
    return cascade_score(values, recipe, model_id) if recipe.get("cascade") else ensemble_score(values, recipe, model_id)


def decide_batch(scores, threshold: float, per_model_split: bool) -> list[bool]:
    """robust = score > threshold; with per_model_split the batch (one model's problems) is split in half by rank instead, the middle
    problem of an odd batch falling back to the threshold, so no model can come out all-True or all-False."""
    s = np.asarray(scores, float)
    if not per_model_split or len(s) < 2:
        return [bool(v > threshold) for v in s]
    order = np.argsort(-s, kind="stable"); n = len(s); out = [False] * n
    for rank, j in enumerate(order):
        if rank < n // 2:
            out[j] = True
        elif n % 2 == 1 and rank == n // 2:
            out[j] = bool(s[j] > threshold)
    return out


def decide_absolute(values: dict, recipe: dict) -> bool:
    """Fixed absolute cut (hand-set from main_v2): robust = value above the cut (below it, for a feature where lower = robust);
    a missing value (e.g. kNN efficiency when thinking ended before 4096 tokens) counts as robust."""
    f, cut = recipe["abs_feature"], recipe["abs_cut"]
    x = values.get(f, np.nan)
    if not np.isfinite(x):
        return True
    return bool(x > cut) if recipe["signs"][f] > 0 else bool(x < cut)


def kmeans_nodes(ws3, mu, L: int, b: int = 4096):
    """exp32 / exp33 nodes: exact 64-token mean states of the first b tokens, model-mean centred, unit norm (None if thinking < b)"""
    if L < b:
        return None
    W = ws3.astype(np.float32) - mu
    nw = min(b // 16, len(W))
    S = W[: nw // 4 * 4].reshape(-1, 4, W.shape[1]).mean(1)
    return S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-6)


def f_cluster_cycle_rank(ws3, mu, centroids, L: int) -> float:
    """exp33: walk of the 64-token states (first 4096 tokens) over this model's K-means clusters (centroids fitted on main_v2);
    cycle rank of the visited undirected graph = (distinct edges - visited clusters + 1) / steps. Higher -> robust."""
    S = kmeans_nodes(ws3, mu, L)
    if S is None:
        return float("nan")
    C = np.asarray(centroids, np.float32)
    lab = np.argmin(((S[:, None, :] - C[None]) ** 2).sum(-1), axis=1).tolist()
    if len(lab) < 10:
        return float("nan")
    moves = [(a, c) for a, c in zip(lab[:-1], lab[1:]) if a != c]
    E = {tuple(sorted(e)) for e in moves}
    return float((len(E) - len(set(lab)) + 1) / len(lab))


_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_CHECKW = re.compile(r"\b(check|verif|confirm|correct|indeed|consistent|double)", re.I)


def f_check_share(text: str) -> float:
    """exp29: share of the thinking's sentence units (>= 4 words) that check / verify / confirm something. Higher -> robust.
    text = the trace decoded with special tokens skipped (as our extraction stored it), cut at the first '</think>'."""
    units = [u.strip().lower() for u in _SPLIT.split(text.split("</think>")[0]) if len(u.split()) >= 4]
    if len(units) < 10:
        return float("nan")
    return float(np.mean([bool(_CHECKW.search(u)) for u in units]))


SHIFT_BUDGETS = (512, 1024, 2048, 3072, 4096, 6144)


def f_forced_slope(p1s) -> float:
    """exp44: slope of the forced-answer top-1 probability against log budget (512 ... 6144 tokens, suffix B). Higher -> robust."""
    return float(np.polyfit(np.log(SHIFT_BUDGETS), np.asarray(p1s, float), 1)[0])


def f_forced_saturated_share(p1s) -> float:
    """exp44: share of the budgets (512 ... 6144) at which the forced answer is already >= 0.99 confident. Higher -> robust."""
    return float(np.mean(np.asarray(p1s, float) >= .99))


def f_pc1_change(ws) -> float:
    """exp44: log ratio of the first-PC variance share of the 64-window segment at 75% vs at 25% of the window states (first 6144
    tokens, all generated windows), mean over the saved layers. Higher -> robust."""
    nw = len(ws)
    if nw < 128:
        return float("nan")
    def share(H):
        Hc = H - H.mean(0); ev = np.clip(np.linalg.eigvalsh(Hc @ Hc.T), 0, None); return ev[-1] / (ev.sum() + 1e-12)
    a, b = int(.25 * nw - 32), int(.75 * nw - 32)
    late = np.mean([share(ws[b:b + 64, l].astype(np.float64)) for l in range(ws.shape[1])])
    early = np.mean([share(ws[max(0, a):max(0, a) + 64, l].astype(np.float64)) for l in range(ws.shape[1])])
    return float(np.log(late / early))
