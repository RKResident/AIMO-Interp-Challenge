import re, pandas as pd
norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())[:160]
P = pd.read_parquet("pool_raw.parquet"); P["key"] = P.problem.map(norm)
seen = set(pd.read_parquet("../train-main-v2.parquet").problem.map(norm)) | set(pd.read_csv("../augmented-sample-math-agg.csv").original_problem.map(norm))
oly = set(pd.read_csv("../olympiad/selected_problems.csv").problem.map(norm))
P["in_ours"] = P.key.isin(seen | oly)
P["chars"] = P.problem.str.len()
P = P[~P.in_ours].drop_duplicates("key")
P = P[P.answer.astype(float).between(0, 1e6)]                     # non-negative integer answers like the competition sets
tier = lambda r: ("difficulty >= 8" if r.difficulty >= 8 else "difficulty 7-8" if r.difficulty >= 7 else "difficulty < 7") if pd.notna(r.difficulty) else "no rating"
P["tier"] = P.apply(tier, axis=1)
print(P.groupby(["source", "tier"]).size().to_string())
H = P[(P.tier != "difficulty < 7")]
print(f"\npool without low-rated rows: {len(H)} problems | median chars {H.chars.median():.0f}")
H.drop(columns=["key"]).to_parquet("pool_hard.parquet")
