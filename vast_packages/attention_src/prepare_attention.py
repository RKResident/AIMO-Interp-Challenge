"""Writes out_attn/<model>/generations.jsonl (the greedy traces of ../inputs.jsonl, capped at --cap tokens, in extract_features.py's
row format) and local_models/<model> links to the weights already in the Hugging Face cache, so extract_features.py neither regenerates
nor downloads anything again."""
import argparse, json
from pathlib import Path
from huggingface_hub import snapshot_download

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument("--cap", type=int, default=6144); a = ap.parse_args()
rows = [json.loads(l) for l in open(HERE.parent / "inputs.jsonl")]
for m in sorted({r["model_id"] for r in rows}):
    slug = m.replace("/", "__"); d = HERE / "out_attn" / slug; d.mkdir(parents=True, exist_ok=True)
    with open(d / "generations.jsonl", "w") as f:
        for r in rows:
            if r["model_id"] == m:
                f.write(json.dumps(dict(id=r["id"], model_id=m, problem=r["problem"], prompt_ids=r["prompt_ids"],
                                        gen_ids=r["greedy_gen_ids"][: a.cap], stop_reason="given", source=r["dataset"], is_robust=r["robust"])) + "\n")
    link = HERE / "local_models" / slug; link.parent.mkdir(exist_ok=True)
    if not link.exists():
        link.symlink_to(snapshot_download(m))
    print(f"{m}: {sum(r['model_id'] == m for r in rows)} traces (cap {a.cap}) -> {d / 'generations.jsonl'} | weights {link} -> {link.resolve()}")
