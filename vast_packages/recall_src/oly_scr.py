"""Friend's olympiad greedy traces (T=0, 10k cap, our prompt: prompt_matches_rebuild) -> run_feats screen format, cut at 6144 tokens as the
submission generates; still_thinking = no end of thinking within those 6144 tokens (2026-10-30, features for the olympiad redo labels).
  python oly_scr.py <generations.jsonl> <model> <out.jsonl>"""
import json, sys
sys.path.insert(0, "submission_core")
from spurious_features import find_think_end, think_end_token_id
from spurious_inference import load_tokenizer
src, model, out = sys.argv[1:4]; tok = load_tokenizer(model, local_files_only=False); tid = think_end_token_id(tok)
with open(out, "w") as fo:
    for r in map(json.loads, open(src)):
        g = r["gen_ids"][:6144]; te = find_think_end(tok, g, tid)
        fo.write(json.dumps(dict(pid=r["base_id"], gen_ids=g, still_thinking=te is None, prompt_ids=r["prompt_ids"])) + "\n")
print(model, "converted", sum(1 for _ in open(out)), "traces")
