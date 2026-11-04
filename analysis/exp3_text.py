"""Exp 3: reasoning-behaviour features decoded from the traces (hesitation / verification / restart markers, answer mentions) + entropy-burst
statistics from the teacher-forced pass. Marker counts are per 1000 words so they are not simply trace length."""
import os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from common import *
sys.path.insert(0, str(ROOT))
import extract_features as ef
allm, _ = load_all()
MARK = {
    "wait": r"\bwait\b", "alternatively": r"\balternatively\b", "hmm": r"\bhmm+\b", "but": r"\bbut\b", "actually": r"\bactually\b",
    "maybe": r"\b(maybe|perhaps)\b", "check": r"\b(check|verify|double-check|recheck|confirm)\w*", "mistake": r"\b(mistake|error|wrong|incorrect|oops)\b",
    "restart": r"(start over|from scratch|different approach|another approach|let me try|try again|reconsider|rethink)",
    "so": r"\bso\b", "therefore": r"\b(therefore|thus|hence)\b", "assume": r"\b(assume|suppose|let's say|let us)\b",
    "answer_is": r"(the answer is|final answer|answer would be|so the answer)", "boxed": r"\\boxed", "equals": r"=", "digits": r"\d+",
    "question": r"\?", "newpar": r"\n\n", "latex": r"\\[a-zA-Z]+", "verify_numeric": r"(plug|substitut|test|example|sanity)", "guess": r"(guess|probably|likely|seems|looks like)",
}
tok_cache = {}
rows = []
gens = {}
for m in MODELS:
    for l in open(OUT / m / "generations.jsonl"):
        r = json.loads(l); gens[r["id"]] = r["gen_ids"]
for k, r in enumerate(allm.itertuples()):
    if r.model not in tok_cache:
        tok_cache[r.model] = ef.load_tokenizer(HF[r.model])
    tk = tok_cache[r.model]
    fj = force_json(r.model, r.id)
    gen = gens[r.id]
    te = fj["think_end"]
    think = gen[:te] if te is not None else gen
    text = tk.decode(think, skip_special_tokens=True).lower()
    words = max(len(text.split()), 1)
    d = {f"t_{n}": len(re.findall(p, text)) / words * 1000 for n, p in MARK.items()}
    d["t_words"] = words
    d["t_tok_per_word"] = len(think) / words
    # position of first "answer is"/boxed mention (fraction of the trace) and how much trace comes after it
    ms = [mm.start() for mm in re.finditer(MARK["answer_is"] + "|" + MARK["boxed"], text)]
    d["t_first_answer_pos"] = ms[0] / max(len(text), 1) if ms else 1.0
    d["t_n_answer_mentions"] = len(ms)
    nums = re.findall(r"\b\d+(?:\.\d+)?\b", text)
    d["t_distinct_numbers_per_1k"] = len(set(nums)) / words * 1000
    d["t_repeat_frac"] = 1 - len(set(text.split())) / words
    d["t_max_para_len"] = max((len(p.split()) for p in text.split("\n\n")), default=0)
    # entropy bursts from the teacher-forced pass
    tf = np.load(OUT / r.model / "tf" / f"{r.id}.npz")
    ent = tf["gen_entropy"].astype(np.float32)[: len(think)]
    n = max(len(ent), 1)
    d.update(e_gt1=(ent > 1).mean(), e_gt2=(ent > 2).mean(), e_gt3=(ent > 3).mean(), e_lt01=(ent < .1).mean(), e_p99=np.quantile(ent, .99) if len(ent) else np.nan)
    hi = (ent > 2).astype(int)
    d["e_bursts_per_1k"] = (np.diff(np.concatenate([[0], hi])) == 1).sum() / n * 1000
    run = best = 0
    for v in (ent < .1):
        run = run + 1 if v else 0; best = max(best, run)
    d["e_longest_lowrun_frac"] = best / n
    q = np.array_split(ent, 4) if len(ent) >= 4 else [ent] * 4
    qm = [x.mean() if len(x) else np.nan for x in q]
    d.update(e_q1=qm[0], e_q4=qm[3], e_q4_minus_q1=qm[3] - qm[0])
    rows.append(d)
    if (k + 1) % 60 == 0: print("decoded", k + 1, "/", len(allm), flush=True)
T = pd.DataFrame(rows); T.to_pickle(A / "exp3_features.pkl")
print("=== Exp 3: trace-text behaviour features ===")
single_auc(T, allm, top=14)
mk_cols = [c for c in T if c.startswith("t_") and c not in ("t_words",)]
e_cols = [c for c in T if c.startswith("e_")]
for nm, cols in (("markers (per 1k words)", mk_cols), ("entropy bursts", e_cols), ("markers + entropy", mk_cols + e_cols)):
    evaluate(nm, T[cols].values, allm, C=0.03)
