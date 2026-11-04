import re, glob, ast, sys, json
D = "/home/sabyasachi19/PycharmProjects/AimoAGAIN/submission_logs/"
SCORE = {"abs-knn-small.zip": .43, "confidence-cascade-small.zip|bc1fb42efc49": .50, "shift-robust-small.zip": .43,
         "confidence-knn-small.zip": .50, "tda-small.zip": .57, "top5-knn-small.zip": .35, "abs-forced-confidence-small.zip": .28}
runs = {}
for f in sorted(glob.glob(D + "*")):
    t = open(f, errors="ignore").read()
    m = re.search(r"SUBMISSION (\S+) .*?build ([0-9a-f]+)", t); name, build = m.groups()
    calls = []
    for blk in t.split("recipe spurious-")[1:]:
        mid = re.search(r"resolved id (\S+)", blk).group(1)
        p = re.search(r"prompt (\d+) tokens \| generated (\d+) tokens \| stop: (\w+)", blk)
        raw = re.search(r"raw features (\{.*?\}) \|", blk); rob = re.search(r"robust (\d)/1", blk)
        fp = re.search(r"forced p1 at budgets (\{.*?\})", blk)
        calls.append(dict(model=mid.split("/")[0], prompt=int(p.group(1)), gen=int(p.group(2)), stop=p.group(3),
                          raw={k: v[0] for k, v in eval(raw.group(1), {"nan": float("nan")}).items()} if raw else {}, robust=int(rob.group(1)) if rob else None,
                          fp=eval(fp.group(1), {"nan": float("nan")}) if fp else None))
    runs[(name, build)] = calls
json.dump({f"{k[0]}|{k[1]}": v for k, v in runs.items()}, open(sys.argv[1], "w"))
keys = sorted({(c["model"], c["prompt"]) for v in runs.values() for c in v if len(v) == 14})
print("distinct (model, prompt len) over 14-call runs:", len(keys))
for k, v in runs.items():
    if len(v) != 14: continue
    order = [(c["model"][:4], c["prompt"]) for c in v]
    print(f"{k[0]:34s} {k[1]} preds {''.join(str(c['robust']) for c in v)} gens {[c['gen'] for c in v]}")
print("order of first run:", [(c["model"][:4], c["prompt"]) for c in list(runs.values())[0]])
