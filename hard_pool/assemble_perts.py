"""Merge the Sonnet perturbation batches (hard_pool/perts/batch*.json: {pid: {rephrase, rename, distract, domain}}), add the scripted
'typos' version (same function as the olympiad set), validate, and write ../vast_packages/screen/perturbations.jsonl (pid, ptype, text).
Checks per version (as olympiad/assemble_perturbations.py): every distinct number of the original still appears; $ delimiters balanced;
non-empty and different from the original; plus: all 4 written types present. Failing versions are listed and left out (re-request them)."""
import glob, json, random, re
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
P = pd.read_csv(HERE / "pool_screen.csv").set_index("pid")
nums = lambda s: set(re.findall(r"\d+(?:\.\d+)?", s))
TYPES = ["rephrase", "rename", "distract", "domain"]


def typos(text, seed, rate=0.06, min_swaps=2):
    r = random.Random(seed)
    parts = re.split(r"(\$\$.*?\$\$|\$.*?\$|\\\[.*?\\\]|\\\(.*?\\\)|\\[a-zA-Z]+)", text, flags=re.S)
    cand = [(i, m.start(), m.group(0)) for i, p in enumerate(parts) if i % 2 == 0 for m in re.finditer(r"[A-Za-z]{5,}", p)]
    k = min(len(cand), max(min_swaps, round(rate * len(cand))))
    for i, st, w in sorted(r.sample(cand, k), key=lambda t: (t[0], -t[1])):
        j = r.randrange(1, len(w) - 2); parts[i] = parts[i][:st] + w[:j] + w[j + 1] + w[j] + w[j + 2:] + parts[i][st + len(w):]
    return "".join(parts)


rows, issues, seen = [], [], set()
for f in sorted(glob.glob(str(HERE / "perts" / "batch*.json"))):
    for pid, d in json.load(open(f)).items():
        if pid not in P.index: issues.append(f"{pid}: unknown pid ({f})"); continue
        orig = P.loc[pid, "problem"]; seen.add(pid)
        miss_types = [t for t in TYPES if t not in d]
        if miss_types: issues.append(f"{pid}: missing types {miss_types}")
        d = {t: d[t] for t in TYPES if t in d}; d["typos"] = typos(orig, seed=int(pid[2:]))
        for t, txt in d.items():
            bad = []
            miss = nums(orig) - nums(txt)
            if miss: bad.append(f"missing numbers {sorted(miss)}")
            if txt.count("$") % 2: bad.append("unbalanced $")
            if not txt.strip() or txt.strip() == orig.strip(): bad.append("empty or unchanged")
            if bad: issues.append(f"{pid} {t}: " + "; ".join(bad)); continue
            rows.append(dict(pid=pid, ptype=t, text=txt))
out = HERE.parent / "vast_packages/screen/perturbations.jsonl"
with open(out, "w") as fo:
    for r in rows: fo.write(json.dumps(r) + "\n")
D = pd.DataFrame(rows)
print(f"{len(seen)} problems, {len(D)} valid versions -> {out}")
if len(D): print("versions per problem:", D.groupby("pid").size().value_counts().sort_index().to_dict())
print(f"{len(issues)} issues" + (":" if issues else "")); [print("  " + x) for x in issues[:60]]
