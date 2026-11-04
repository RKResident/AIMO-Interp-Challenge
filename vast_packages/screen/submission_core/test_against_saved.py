"""Check the submission's feature code (spurious_features.py) against what our extraction + analysis saved for the 43 main_v2 rows:
prompt token ids, cutoff labels, prose-typo texts, and every feature value the submissions use."""
import sys, json
sys.path.insert(0, "/home/sabyasachi19/PycharmProjects/AimoAGAIN/analysis"); sys.path.insert(0, "/home/sabyasachi19/PycharmProjects/AimoAGAIN/submission_core")
from common import *
import spurious_features as rc
from spurious_inference import load_tokenizer

P2 = ROOT / "extraction_out_generations_p2" / "extraction_out_generations"
R3 = ROOT / "extraction_out_generations_pearl_more" / "extraction_out_generations"
allm, _ = load_all(); rows = datasets(allm)["main_v2"]; model = allm.model.values
F = pickle.load(open(A / "exp28_features_cache.pkl", "rb")); F.update({"e32:" + k: v for k, v in pickle.load(open(A / "exp32_features.pkl", "rb")).items()})
MU = {m: np.concatenate([np.load(P2 / m / "tf" / f"{allm.id.iloc[i]}.npz")["win_states"][:, 3].astype(np.float32) for i in rows if model[i] == m]).mean(0)
      for m in np.unique(model[rows])}
GEN = {}
for m in np.unique(model[rows]):
    for l in open(P2 / m / "generations.jsonl"):
        d = json.loads(l); GEN[d["id"]] = d

res = {k: [] for k in ("traj", "seg", "ge", "pearl", "ab", "fa", "labels", "noise_texts", "loop")}
for i in rows:
    rid, m = allm.id.iloc[i], model[i]
    fj = force_json(m, rid); L = fj["L"]
    ws = np.load(P2 / m / "tf" / f"{rid}.npz")["win_states"]
    res["traj"].append((rc.f_late_vs_early(ws), F["traj:k1|full:late_vs_early_cos"][i]))
    res["seg"].append((rc.f_seg25_pc1(ws, L), F["subsample:seg25:f11"][i]))
    res["ge"].append((rc.f_global_eff(ws[:, 3], MU[m], L), F["e32:step64|k5|4k|global_efficiency"][i]))
    lab = rc.cutoff_labels(L); res["labels"].append(lab == {k: int(v) for k, v in fj["labels"].items()})
    R = fj["results"]
    res["fa"].append((R[f"{lab['b6144']}|B"]["top"][0]["probs"][0], F["meta:fa_b6144_B_p1"][i]))
    res["ab"].append((max(abs(R[f"{c}|A"]["first_entropy"] - R[f"{c}|B"]["first_entropy"]) for c in lab.values()), F["ab:dent_max"][i]))
    g = GEN[rid]["gen_ids"]; res["loop"].append(rc.find_loop_end(g) is None or GEN[rid]["stop_reason"] == "loop")
    p = R3 / m / "pearl" / f"{rid}.npz"
    if p.exists():
        z = np.load(p, allow_pickle=True); keep = np.isin(z["kind"], ["original", "noise"])
        res["pearl"].append((rc.f_pearl_area(z["mean_state"][keep].astype(np.float32)), F["pearl:mean_state:area"][i]))
        prob = GEN[rid]["problem"]
        mine = [rc.prose_noise(prob, r, seed=1000 * s + int(r * 1000))[0] for r in rc.PEARL_RATES for s in range(rc.PEARL_SEEDS)]
        res["noise_texts"].append(mine == list(z["texts"][z["kind"] == "noise"]))

for k in ("traj", "seg", "ge", "pearl", "ab", "fa"):
    a = np.array(res[k], float); ok = np.isfinite(a).all(1); both_nan = np.isnan(a).all(1)
    print(f"{k:6s} rows {len(a)} | both finite {ok.sum()} max |diff| {np.abs(a[ok, 0] - a[ok, 1]).max() if ok.any() else float('nan'):.2e} | "
          f"both NaN {both_nan.sum()} | mismatched NaN {(~ok & ~both_nan).sum()}")
print("cutoff labels identical:", all(res["labels"]), "| prose-typo texts identical:", all(res["noise_texts"]), f"({len(res['noise_texts'])} problems)",
      "| loop rule consistent:", all(res["loop"]))

# prompt ids: our build_prompt + tokenizer fix vs the ids saved with each trace (transformers 5.x, offline cache)
for m in np.unique(model[rows]):
    tok = load_tokenizer(HF[m])
    ids = [i for i in rows if model[i] == m]
    same = [rc.build_prompt(tok, GEN[allm.id.iloc[i]]["problem"])[0] == GEN[allm.id.iloc[i]]["prompt_ids"] for i in ids]
    print(f"prompt ids identical for {m}: {sum(same)}/{len(same)}")
