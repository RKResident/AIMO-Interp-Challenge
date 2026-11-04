"""Assemble the Sonnet chat replies for the confidence-matched problems (user 2026-11-01). Reads sonnet_prompts/reply_*.json (a raw reply
pasted into a file is fine: the first {...} JSON object, or a ```json block, is extracted). Adds the scripted 'typos' version (same function as
assemble_perts.py), validates every version as assemble_perts.py does (all numbers of the original kept, $ balanced, non-empty, differs from the
original, all 4 written types present), and writes:
  vast_packages/screen/new_perts.jsonl   (pid, ptype, text)
  vast_packages/screen/new_pool.jsonl    (pid, source, problem, answer: originals of the selected problems)
  vast_packages/screen/newsel_<tag>.txt  (per-model keep lists from sonnet_prompts/selection.json, only pids with all 5 versions valid)
  python hard_pool/assemble_new.py"""
import glob, json, random, re, zlib
from pathlib import Path

A = Path(__file__).resolve().parent.parent; S = A / "vast_packages/screen"
TYPES = ["rephrase", "rename", "distract", "domain"]; nums = lambda s: set(re.findall(r"\d+(?:\.\d+)?", s))
TAG = {"Olmo": "allenai__Olmo-3-7B-Think", "Qwen": "Qwen__Qwen3.5-4B", "Skywork": "Skywork__Skywork-OR1-Math-7B"}


def typos(text, seed, rate=0.06, min_swaps=2):
    r = random.Random(seed)
    parts = re.split(r"(\$\$.*?\$\$|\$.*?\$|\\\[.*?\\\]|\\\(.*?\\\)|\\[a-zA-Z]+)", text, flags=re.S)
    cand = [(i, m.start(), m.group(0)) for i, p in enumerate(parts) if i % 2 == 0 for m in re.finditer(r"[A-Za-z]{5,}", p)]
    k = min(len(cand), max(min_swaps, round(rate * len(cand))))
    for i, st, w in sorted(r.sample(cand, k), key=lambda t: (t[0], -t[1])):
        j = r.randrange(1, len(w) - 2); parts[i] = parts[i][:st] + w[:j] + w[j + 1] + w[j] + w[j + 2:] + parts[i][st + len(w):]
    return "".join(parts)


def parse(text):
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
    if m: return json.loads(m.group(1))
    i = text.index("{"); return json.loads(text[i:text.rindex("}") + 1])


POOL = {}
for f in ("pool.jsonl", "easy_pool.jsonl"):
    for l in open(S / f): d = json.loads(l); POOL[d["pid"]] = d
sel = json.load(open(A / "sonnet_prompts/selection.json")); want = sorted(set().union(*map(set, sel.values())))
got, issues = {}, []
for f in sorted(glob.glob(str(A / "sonnet_prompts/reply_*"))):
    try: got.update(parse(open(f).read()))
    except Exception as e: issues.append(f"{f}: could not parse ({e})")
rows, ok = [], set()
for p in want:
    if p not in got: issues.append(f"{p}: no reply"); continue
    orig = POOL[p]["problem"]; d = got[p]; bad = [t for t in TYPES if t not in d]
    vers = {t: d[t] for t in TYPES if t in d}; vers["typos"] = typos(orig, zlib.crc32(p.encode()))
    for t, v in vers.items():
        why = [w for w, c in (("empty", not v.strip()), ("same as original", v.strip() == orig.strip()), ("numbers lost", not nums(orig) <= nums(v)),
                               ("unbalanced $", v.count("$") % 2 == 1)) if c]
        if why: bad.append(f"{t} ({', '.join(why)})")
    if bad: issues.append(f"{p}: " + "; ".join(bad)); continue
    ok.add(p); rows += [dict(pid=p, ptype=t, text=v) for t, v in vers.items()]
with open(S / "new_perts.jsonl", "w") as fo:
    for r in rows: fo.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(S / "new_pool.jsonl", "w") as fo:
    for p in sorted(ok): fo.write(json.dumps(dict(pid=p, source=POOL[p]["source"], problem=POOL[p]["problem"], answer=str(POOL[p]["answer"])), ensure_ascii=False) + "\n")
for m, ps in sel.items(): (S / f"newsel_{TAG[m]}.txt").write_text("\n".join(p for p in ps if p in ok) + "\n")
print(f"{len(ok)}/{len(want)} problems with all 5 valid versions | per model: " + ", ".join(f"{m} {sum(p in ok for p in ps)}/{len(ps)}" for m, ps in sel.items()))
for i in issues: print("  ISSUE", i)
print("next: bash vast_packages/screen/launch_label_new.sh   (copies to both boxes and starts the labelling queues)")
