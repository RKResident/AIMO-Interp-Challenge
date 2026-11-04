import json, re, pandas as pd
rows = [json.loads(l) for l in open("saved_trajectories_v2/allenai__Olmo-3-7B-Think/results.jsonl")]
df = pd.DataFrame(rows)
print(pd.crosstab(df.model, df.stop_reason))
def first_box_frac(t):
    m = re.search(r"\\boxed\{[^}]+\}", t)
    return m.start()/max(len(t),1) if m else None
df["box_pos"] = df.text.map(first_box_frac)
print(df.groupby("model").box_pos.describe())