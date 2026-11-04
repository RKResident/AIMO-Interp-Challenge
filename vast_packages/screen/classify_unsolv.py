"""exp112d: classify responses to unsolvable variants (run on the big box: tokenizers are cached there).
Per sample: decode the generated ids; 'final' = text after </think> (or the last 1500 chars if thinking never ended).
  flagged : the final text says the problem is impossible / contradictory / underspecified (keyword match, shown for review)
  boxed   : the \\boxed{} answer, if any;  orig : boxed == the ORIGINAL problem's answer (26244 / 70)
  doubt   : keyword hits anywhere in the reasoning (did it ever question the premises?)
  python classify_unsolv.py [out_dir]   -> prints a table + writes <out_dir>/classified.jsonl"""
import glob, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/submission_core"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from screen import boxed                                                          # noqa: E402
from spurious_inference import load_tokenizer                                     # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/nvme/sabya/out/label_unsolv"
ORIG = {"7ee9": "26244", "26a2": "70"}
KW = re.compile(r"no solution|not possible|impossible|contradict|inconsisten|cannot be|can't be|ill[- ]posed|invalid|not valid|undefined|"
                r"does not exist|doesn't exist|no such|not well[- ]defined|insufficient|not enough information|missing|not specified|"
                r"unspecified|not a prime|isn't prime|is not prime|false premise|incorrect premise|typo|mistake in the problem|error in the problem", re.I)
rows = []
for f in sorted(glob.glob(f"{OUT}/*.jsonl")):
    if f.endswith("classified.jsonl"): continue
    tag = os.path.basename(f)[:-6]; tok = load_tokenizer(tag.replace("__", "/"), local_files_only=True)
    for l in open(f):
        r = json.loads(l); z = np.load(f"{OUT}/{r['ids_file']}")
        for j, s in enumerate(r["samples"]):
            txt = tok.decode(z[f"s{j}"].tolist(), skip_special_tokens=False)
            k = txt.rfind("</think>"); final = txt[k + 8:] if k >= 0 else txt[-1500:]
            b = boxed(final); fl = KW.findall(final); dt = KW.findall(txt[:k] if k >= 0 else txt)
            rows.append(dict(model=tag.split("__")[1][:10], pid=r["pid"], j=j, n_gen=s["n_gen"], finished=s["finish"] != "length", boxed=b,
                             orig=b is not None and b.strip() == ORIG.get(r["pid"].split("_")[1], "\x00"), flagged=bool(fl), flag_words=sorted(set(w.lower() for w in fl))[:4],
                             doubt_hits=len(dt), final_tail=final[-300:].replace("\n", " ")))
with open(f"{OUT}/classified.jsonl", "w") as fo:
    for x in rows: fo.write(json.dumps(x) + "\n")
import collections
G = collections.defaultdict(list)
for x in rows: G[(x["model"], x["pid"])].append(x)
print(f"{'model':10s} {'variant':20s} flagged  boxed-orig  boxed-other  no-box  doubt(median hits)  boxed values")
for (m, p), xs in sorted(G.items()):
    print(f"{m:10s} {p:20s} {sum(x['flagged'] for x in xs)}/{len(xs)}      {sum(x['orig'] for x in xs)}/{len(xs)}         "
          f"{sum(x['boxed'] is not None and not x['orig'] for x in xs)}/{len(xs)}         {sum(x['boxed'] is None for x in xs)}/{len(xs)}    "
          f"{int(np.median([x['doubt_hits'] for x in xs]))}                  {[x['boxed'] for x in xs]}")
