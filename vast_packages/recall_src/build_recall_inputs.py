"""Inputs for the recall-direction run (A10G). One row per prompt: key, family, group (matched pair / problem id), kind, question, gold.
Families (user, 2026-10-28: short-sentence contrasts AND competition-problem recall):
  gsm      GSM-Symbolic: GSM8K original (kind original) + 3 templated variants (kind variant, new numbers, new gold). Existing pairs.
  comp     competition problems the models have probably seen (AIME 1983-2022, MATH level 5; integer answers, no figures): the original and
           a scripted typo variant with the SAME answer. recall = right with no reasoning on the original but wrong on the variant.
           (Reworded variants by Sonnet come later, only for originals answered right with no reasoning.)
  arith    memorized vs computed arithmetic, same template and same number of digits in the answer where possible: kind recall = facts every
           model has seen countless times (times tables, small squares, powers of 2, factorials), kind compute = same operation on numbers
           that must be worked out.
  formula  recall a standard formula / constant (kind recall) vs compute a specific instance of the same formula (kind compute, numeric gold).
Writes recall_inputs.jsonl."""
import json, random, re, sys
from pathlib import Path
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "olympiad"))
rng = random.Random(0); rows = []
FIG = re.compile(r"\[asy\]|\\includegraphics|figure below|shown below|diagram below|pictured|in the figure|shown in the figure|as shown", re.I)


def as_int(a):
    s = str(a).strip().replace(",", "").replace("$", "").replace("\\boxed{", "").rstrip("}").strip()
    return str(int(s)) if re.fullmatch(r"-?\d+", s) else None


def typos(text, seed, rate=0.06, min_swaps=2):
    """the olympiad set's scripted typo perturbation (olympiad/assemble_perturbations.py): swap two interior letters in ~6% of prose words"""
    r = random.Random(seed)
    parts = re.split(r"(\$\$.*?\$\$|\$.*?\$|\\\[.*?\\\]|\\\(.*?\\\)|\\[a-zA-Z]+)", text, flags=re.S)
    cand = [(i, m.start(), m.group(0)) for i, p in enumerate(parts) if i % 2 == 0 for m in re.finditer(r"[A-Za-z]{5,}", p)]
    k = min(len(cand), max(min_swaps, round(rate * len(cand))))
    for i, st, w in sorted(r.sample(cand, k), key=lambda t: (t[0], -t[1])):
        j = r.randrange(1, len(w) - 2); parts[i] = parts[i][:st] + w[:j] + w[j + 1] + w[j] + w[j + 2:] + parts[i][st + len(w):]
    return "".join(parts)


# ---- gsm: the existing GSM-Symbolic pairs
for l in open(ROOT / "vast_packages/recall_graphs_src/recall_pairs.jsonl"):
    p = json.loads(l)
    rows.append(dict(key=f"gsm_{p['key']}", family="gsm", group=f"gsm{p['pid']}", kind=p["kind"], question=p["question"], gold=p["gold"],
                     original_gold=p.get("original_gold")))

# ---- comp: AIME 1983-2022 + MATH level 5 (probably in every model's training data)
api = HfApi()
def parquet(ds):
    """all parquet files of a dataset, else its csv / jsonl files"""
    fs = api.list_repo_files(ds, repo_type="dataset"); get = lambda f: hf_hub_download(ds, f, repo_type="dataset", cache_dir=str(ROOT / "hard_pool/hf"))
    pq_ = [f for f in fs if f.endswith(".parquet")]
    if pq_: return pd.concat([pd.read_parquet(get(f)) for f in pq_], ignore_index=True)
    return pd.concat([pd.read_csv(get(f)) if f.endswith(".csv") else pd.read_json(get(f), lines=True) for f in fs if f.endswith((".csv", ".jsonl"))],
                     ignore_index=True)
A = parquet("gneubig/aime-1983-2024"); print("AIME columns", list(A.columns))
yc = next(c for c in A.columns if c.lower() == "year"); qc = next(c for c in A.columns if c.lower() in ("question", "problem")); ac = next(c for c in A.columns if c.lower() == "answer")
A = A[A[yc].astype(int) <= 2022]
comp = [("aime" + str(i), q, as_int(a)) for i, (q, a) in enumerate(zip(A[qc], A[ac]))]
M = parquet("nlile/hendrycks-MATH-benchmark"); print("MATH columns", list(M.columns))
M = M[M["level"].astype(str).str.contains("5")]
comp += [("math" + str(i), q, as_int(a)) for i, (q, a) in enumerate(zip(M["problem"], M["answer"]))]
comp = [(g, q, a) for g, q, a in comp if a is not None and not FIG.search(q)]
# ---- forced: MATH level 5 with the reference solution teacher-forced as the thinking (compute) vs the same problem with no reasoning
#      (recall attempt). Does not depend on the model solving the problem itself (user, 2026-10-28).
FM = [(i, q, as_int(a), sol) for i, (q, a, sol) in enumerate(zip(M["problem"], M["answer"], M["solution"]))]
FM = [t for t in FM if t[2] is not None and not FIG.search(t[1]) and 200 <= len(t[3]) <= 3000]
for i, q, a, sol in random.Random(1).sample(FM, min(400, len(FM))):
    g = f"forced_math{i}"
    rows.append(dict(key=f"{g}_recall", family="forced", group=g, kind="recall", question=q, gold=a))
    rows.append(dict(key=f"{g}_compute", family="forced", group=g, kind="compute", question=q, gold=a, reasoning=sol))
for n, (g, q, a) in enumerate(comp):
    rows.append(dict(key=f"comp_{g}_o", family="comp", group=g, kind="original", question=q, gold=a))
    rows.append(dict(key=f"comp_{g}_typos", family="comp", group=g, kind="variant", question=typos(q, seed=n), gold=a, original_gold=a))

# ---- arith: memorized facts vs computed, same templates
def add_pair(group, tmpl, rec, comp_):
    for kind, (args, ans) in (("recall", rec), ("compute", comp_)):
        rows.append(dict(key=f"arith_{group}_{kind}", family="arith", group=group, kind=kind, question=tmpl.format(*args), gold=str(ans)))
k = 0
for _ in range(60):                                   # times tables (2-12) vs 2-digit x 1-digit products with the same answer length
    a, b = rng.randint(4, 12), rng.randint(4, 12); c, d = rng.randint(13, 49), rng.randint(3, 9)   # a*b >= 16: always matchable
    while len(str(c * d)) != len(str(a * b)): c, d = rng.randint(11, 33), rng.randint(3, 9)
    add_pair(f"mul{k}", "What is {} \\times {}?", ((a, b), a * b), ((c, d), c * d)); k += 1
for n in range(4, 21):                                # squares up to 20 vs squares of 2-digit numbers beyond 30 (same template)
    m = rng.randint(31, 99); add_pair(f"sq{n}", "What is ${}^2$?", ((n,), n * n), ((m,), m * m))
for e in range(5, 17):                                # powers of 2 vs powers of 3 / 7 with a similar size
    b = 3 if e <= 10 else 7; f = max(2, round(e * 0.6931 / (1.0986 if b == 3 else 1.9459)))
    add_pair(f"pow{e}", "What is ${}^{{{}}}$?", ((2, e), 2 ** e), ((b, f), b ** f))
import math
for n in range(4, 11):                                # factorials vs a product of consecutive numbers that is not a factorial
    s = rng.randint(3, 6); prod = math.prod(range(s, s + n - 2))
    add_pair(f"fact{n}", "What is {}?", ((f"{n}!",), math.factorial(n)), ((f"the product {s} \\times {s + 1} \\times \\cdots \\times {s + n - 3}",), prod))

# ---- formula: recall a formula / fact vs compute an instance (compute side has a numeric gold; recall side is graded later by hand / not at all)
F = [("the sum of the first n positive integers", "the sum 1 + 2 + \\cdots + {n}", lambda n: n * (n + 1) // 2),
     ("the sum of the squares of the first n positive integers", "1^2 + 2^2 + \\cdots + {n}^2", lambda n: n * (n + 1) * (2 * n + 1) // 6),
     ("the sum of the cubes of the first n positive integers", "1^3 + 2^3 + \\cdots + {n}^3", lambda n: (n * (n + 1) // 2) ** 2),
     ("the number of diagonals of a convex polygon with n sides", "the number of diagonals of a convex polygon with {n} sides", lambda n: n * (n - 3) // 2),
     ("the sum of the interior angles of a convex polygon with n sides, in degrees", "the sum of the interior angles, in degrees, of a convex polygon with {n} sides", lambda n: 180 * (n - 2)),
     ("the number of ways to choose 2 objects from n", "the number of ways to choose 2 objects from {n}", lambda n: n * (n - 1) // 2),
     ("the number of subsets of a set with n elements", "the number of subsets of a set with {n} elements", lambda n: 2 ** n),
     ("the n-th triangular number", "the {n}-th triangular number", lambda n: n * (n + 1) // 2),
     ("the number of edges of the complete graph on n vertices", "the number of edges of the complete graph on {n} vertices", lambda n: n * (n - 1) // 2),
     ("the number of positive divisors of p^k for a prime p", "the number of positive divisors of $2^{{{n}}}$", lambda n: n + 1),
     ("the sum of the first n odd positive integers", "1 + 3 + 5 + \\cdots + {m}", None),
     ("the n-th term of an arithmetic sequence with first term a and common difference d", "the {n}-th term of the arithmetic sequence 3, 7, 11, \\ldots", lambda n: 3 + 4 * (n - 1)),
     ("the number of regions n lines in general position divide the plane into", "the number of regions {n} lines in general position divide the plane into", lambda n: n * (n + 1) // 2 + 1),
     ("the number of handshakes when each of n people shakes hands once with every other person", "the number of handshakes when each of {n} people shakes hands once with every other person", lambda n: n * (n - 1) // 2),
     ("the number of permutations of n distinct objects", "the number of ways to arrange {n} distinct books in a row", lambda n: math.factorial(n)),
     ("the n-th Catalan number", "the {n}-th Catalan number (with $C_0 = 1$)", lambda n: math.comb(2 * n, n) // (n + 1)),
     ("Euler's totient of a prime p", "Euler's totient $\\varphi({p})$", None),
     ("the number of trailing zeros of n!", "the number of trailing zeros of ${n}!$", lambda n: sum(n // 5 ** i for i in range(1, 6))),
     ("the area of a triangle with base b and height h", "the area of a triangle with base {n} and height 10", lambda n: 5 * n),
     ("the n-th Fibonacci number", "the {n}-th Fibonacci number (with $F_1 = F_2 = 1$)", None)]
def fib(n):
    a, b = 1, 1
    for _ in range(n - 2): a, b = b, a + b
    return a if n <= 2 else b
PR = [53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
for fi, (rec_txt, comp_tmpl, fn) in enumerate(F):
    for v in range(3):
        n = rng.randint(8, 30) if fi not in (9, 15) else rng.randint(5, 12)
        if fn is None and "odd" in rec_txt: m = 2 * n - 1; q2, ans, q1 = comp_tmpl.format(m=m), n * n, comp_tmpl.format(m="(2n-1)")
        elif fn is None and "totient" in rec_txt: p = PR[(fi + v) % len(PR)]; q2, ans, q1 = comp_tmpl.format(p=p), p - 1, comp_tmpl.format(p="p") + " for a prime $p$"
        elif fn is None: q2, ans, q1 = comp_tmpl.format(n=n), fib(n), comp_tmpl.format(n="n")
        else: q2, ans, q1 = comp_tmpl.format(n=n), fn(n), comp_tmpl.format(n="n")
        # matched wording: the same sentence, symbolic n (answer = a formula, recalled) vs a number (answer = a value, computed)
        g = f"f{fi}v{v}"
        rows.append(dict(key=f"formula_{g}_recall", family="formula", group=g, kind="recall", question=f"Find {q1}.", gold=None))
        rows.append(dict(key=f"formula_{g}_compute", family="formula", group=g, kind="compute", question=f"Find {q2}.", gold=str(ans)))

with open(HERE / "recall_inputs.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
D = pd.DataFrame(rows); print(D.groupby(["family", "kind"]).size().to_string()); print(len(D), "prompts")
