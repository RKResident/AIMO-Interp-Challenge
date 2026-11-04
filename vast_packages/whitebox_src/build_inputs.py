"""Local: writes whitebox_inputs.jsonl = t1_run/inputs.jsonl (118 labelled rows: main_v2 43 + olympiad 75, each with its prompt ids and the
greedy trace capped at 6144) + per row the semantic variants of its problem:
  friend's five LLM rewrites from olympiad/perturbations.csv (rephrase, rename, distract, domain, typos), when the row's problem is one of
  the 80 olympiad problems verbatim (all olympiad rows, 16 of main_v2's 22 problems);
  three automatic ones that can also be made at test time: prose typos (10% of prose words, numbers never touched), a neutral distractor
  sentence, and a consistent renaming of the single-letter math variables."""
import csv, json, re, sys
sys.path.insert(0, "/home/sabyasachi19/PycharmProjects/AimoAGAIN/submission_core")
from spurious_features import prose_noise

ROOT = "/home/sabyasachi19/PycharmProjects/AimoAGAIN/"
norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()                                    # noqa: E731
sel = {norm(r["problem"]): r["pid"] for r in csv.DictReader(open(ROOT + "olympiad_run_package/olympiad/selected_problems.csv"))}
var = {}
for r in csv.DictReader(open(ROOT + "olympiad_run_package/olympiad/perturbations.csv")):
    var.setdefault(r["pid"], []).append((r["ptype"], r["text"]))
DISTRACT = "Note that this problem was part of a practice set."


def rename_vars(text: str) -> str:
    """Consistently map the single-letter variables inside $...$ math to other letters (LaTeX commands untouched)."""
    segs = re.split(r"(\$\$?[^$]+\$\$?)", text)
    used = sorted({c for s in segs if s.startswith("$") for c in re.findall(r"(?<![\\a-zA-Z])([a-zA-Z])(?![a-zA-Z])", s)})
    pool = [c for c in "pqrsuvwxyzabcdefghjkmn" if c not in used]
    mp = {c: pool[i] for i, c in enumerate(used) if i < len(pool)}
    out = [re.sub(r"(?<![\\a-zA-Z])([a-zA-Z])(?![a-zA-Z])", lambda m: mp.get(m.group(1), m.group(1)), s) if s.startswith("$") else s for s in segs]
    return "".join(out)


def distract(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
    return parts[0] + " " + DISTRACT + (" " + parts[1] if len(parts) > 1 else "")


n_fr = 0
with open(ROOT + "vast_packages/whitebox_src/whitebox_inputs.jsonl", "w") as f:
    for l in open(ROOT + "t1_run/inputs.jsonl"):
        r = json.loads(l); p = r["problem"]; pid = sel.get(norm(p))
        v = [("friend:" + t, x) for t, x in var.get(pid, [])] if pid else []
        n_fr += bool(v)
        v += [("auto:typos", prose_noise(p, 0.10, 7)[0]), ("auto:distract", distract(p)), ("auto:rename_vars", rename_vars(p))]
        r["variants"] = [dict(kind=k, text=x) for k, x in v if x.strip() and x != p]
        f.write(json.dumps(r) + "\n")
print(f"rows with the friend's variants: {n_fr}")
