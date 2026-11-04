"""exp75: difficulty of the friend's olympiad set vs main_v2 vs the 14 leaderboard problems, per model, from what we can measure on all three:
  - still thinking at 6144 tokens (greedy) and, where the full trace exists (main_v2 16k greedy, friend's 16k greedy), the median length of
    thinking when it ends
  - forced confidence at 6144 (p of the forced answer's first token) - higher = the model is surer at the cap
  - prompt length (tokens)
  - robust share among labelled rows; greedy solve rate of the originals (olympiad only: all 80 selected problems, solved or not)
The leaderboard only gives what our logs record (6144-capped traces): unfinished share, forced confidence, prompt length."""
import csv, glob, json, os, re, sys
import numpy as np
import pandas as pd
sys.path.insert(0, "/home/sabyasachi19/PycharmProjects/AimoAGAIN/submission_core")
from spurious_inference import load_tokenizer                                            # noqa: E402
from spurious_features import find_think_end, think_end_token_id                          # noqa: E402

ROOT = "/home/sabyasachi19/PycharmProjects/AimoAGAIN/"; A = ROOT + "analysis/"
D = pd.read_csv(A + "exp56_rows.csv")
SHORT = {"Qwen/Qwen3.5-4B": "Qwen", "Skywork/Skywork-OR1-Math-7B": "Skywork", "allenai/Olmo-3-7B-Think": "Olmo", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": "DeepSeek"}
TOK = {m: load_tokenizer(m) for m in SHORT}; TID = {m: think_end_token_id(TOK[m]) for m in SHORT}
inp = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT + "t1_run/inputs.jsonl")}

rows = []
# main_v2 + olympiad labelled rows (exp56 frame: finished at 6144, forced confidence at 6144)
for r in D.itertuples():
    full = None
    if r.dataset == "main_v2" and r.id in inp:                                            # 16k greedy trace
        g = inp[r.id]["greedy_gen_ids"]; te = find_think_end(TOK[r.model_id], g, TID[r.model_id]); full = te if te is not None else np.nan
    rows.append(dict(set=r.dataset, model=SHORT[r.model_id], unfinished=not bool(r.finished), fa=r.fa, prompt=len(inp[r.id]["prompt_ids"]) if r.id in inp else np.nan,
                     think_full=full, robust=r.robust))
X = pd.DataFrame(rows)
# olympiad full completion length (thinking + answer) + solve rate over all 80 selected problems (friend's greedy originals, up to 16k)
OLY = ROOT + "extra_olympiad_problems_dataset/olympiad_labels_greedy/"
olyfull, solve = {}, {}
for mdir in glob.glob(OLY + "*__*/"):                                                    # completions carry n_tokens / finish / correct
    mid = mdir.rstrip("/").split("/")[-1].replace("__", "/"); m = SHORT[mid]
    for f in glob.glob(mdir + "completions_shard*.jsonl"):
        for l in open(f):
            d = json.loads(l)
            if d["ptype"] != "original": continue
            olyfull.setdefault(m, []).append(d["n_tokens"] if d["finish"] == "stop" else np.nan)
            solve.setdefault(m, [0, 0]); solve[m][0] += int(d["correct"]); solve[m][1] += 1
LBP = [("Qwen", 152, .9940), ("Qwen", 140, .4450), ("Qwen", 85, .9998), ("Qwen", 692, .9987), ("Qwen", 114, .5948), ("Skywork", 135, 1.0), ("Skywork", 149, .9984),
       ("Skywork", 172, 1.0), ("Olmo", 210, .9967), ("Olmo", 175, .9999), ("Olmo", 219, .4553), ("DeepSeek", 109, .5877), ("DeepSeek", 78, .9174), ("DeepSeek", 147, .3820)]
LBU = [1] * 7 + [0] + [1] * 6

print("per model: n | still thinking at 6144 | forced confidence @6144 median [share >= 0.99] | prompt tokens median | robust share | full-trace thinking median (finished share at 16k)")
for m in ["Qwen", "Skywork", "Olmo", "DeepSeek"]:
    print(f"\n{m}")
    for s in ("main_v2", "olympiad"):
        g = X[(X.set == s) & (X.model == m)]
        if g.empty: print(f"   {s:9s} -"); continue
        if s == "main_v2":
            tf = g.think_full.astype(float); fulls = "- (stored traces stop at 6144)"
        else:
            tf = np.array(olyfull.get(m, []), float); fulls = f"{np.nanmedian(tf):.0f} tokens incl. answer ({np.isfinite(tf).mean():.0%} finish within 16k; all originals)" if len(tf) else "-"
        extra = f" | greedy solve rate of all originals {solve[m][0]}/{solve[m][1]}" if s == "olympiad" and m in solve else ""
        print(f"   {s:9s} n={len(g):2d} | {g.unfinished.mean():.0%} | {g.fa.median():.3f} [{(g.fa >= .99).mean():.0%}] | {g.prompt.median():.0f} | "
              f"{g.robust.mean():.0%} | {fulls}{extra}")
    lb = [(p, f, u) for (mm, p, f), u in zip(LBP, LBU) if mm == m]
    print(f"   leaderbd  n={len(lb):2d} | {np.mean([u for _, _, u in lb]):.0%} | {np.median([f for _, f, _ in lb]):.3f} [{np.mean([f >= .99 for _, f, _ in lb]):.0%}] | "
          f"{np.median([p for p, _, _ in lb]):.0f} | 6/13 unfinished robust overall | -")
print("\nall models pooled:")
for s in ("main_v2", "olympiad"):
    g = X[X.set == s]; print(f"   {s:9s} n={len(g)} | still thinking at 6144 {g.unfinished.mean():.0%} | forced confidence median {g.fa.median():.3f} | robust {g.robust.mean():.0%}")
print(f"   leaderbd  n=14 | still thinking at 6144 {np.mean(LBU):.0%} | forced confidence median {np.median([f for _, _, f in LBP]):.3f} | robust 50% (by design)")
