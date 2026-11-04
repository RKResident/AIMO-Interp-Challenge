"""Pool per-trace extraction files into small per-model feature tables (one trace in memory at a time).
Output: analysis/feats_<slug>.pkl = {'meta': DataFrame, 'F': {family: ndarray[n, ...]}}"""
import json, re, sys, pickle, glob
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "extraction_out"
EXTRA = {"deepseek-ai__DeepSeek-R1-0528-Qwen3-8B": ROOT / "extra_deepsek" / "extra"}
REF = ["b1024", "b4096", "f50", "f100"]          # reference cutoffs used for high-dim features
TOPK = 8


def norm(s):
    if s is None:
        return None
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", str(s))
    s = s.strip().strip("$").replace(",", "").replace("\\!", "").replace("\\,", "").replace("\\dfrac", "\\frac")
    return re.sub(r"\s+", "", s).rstrip(".")


def scal(x):
    x = np.asarray(x, dtype=np.float64)
    return x[np.isfinite(x)]


def traj_scalars(lp, ent, top):
    f = {}
    n = len(lp)
    for nm, a in (("lp", lp), ("ent", ent), ("top1", top)):
        a = scal(a)
        segs = {"all": a, "first512": a[:512], "last25": a[int(.75 * len(a)):], "last512": a[-512:]}
        for sn, s in segs.items():
            f[f"{nm}_{sn}_mean"] = s.mean() if len(s) else np.nan
        f[f"{nm}_std"] = a.std() if len(a) else np.nan
        if nm == "ent":
            f["ent_q90"] = np.quantile(a, .9) if len(a) else np.nan
            f["ent_frac_gt2"] = (a > 2).mean() if len(a) else np.nan
        if len(a) > 20:
            f[f"{nm}_slope"] = np.polyfit(np.linspace(0, 1, len(a)), a, 1)[0]
        else:
            f[f"{nm}_slope"] = np.nan
    return f


def build(slug):
    d = OUT / slug
    plan = pd.read_csv(OUT / "plan.csv").set_index("id")
    gens = {}
    for l in open(d / "generations.jsonl"):
        r = json.loads(l)
        gens[r["id"]] = (len(r["prompt_ids"]), len(r["gen_ids"]), r["stop_reason"])
    rows, F = [], {k: [] for k in ["scal", "prompt_last", "prompt_mean", "cut_last", "lap_no", "lap_force", "lap_mean_no", "lap_mean_force",
                                    "x_fft", "x_look", "x_kl", "x_klhm", "x_res", "x_attn", "x_mlp", "force_curve"]}
    ids = [i for i in gens if i in plan.index and (d / "tf" / f"{i}.npz").exists() and (d / "force" / f"{i}.json").exists()]
    for k, i in enumerate(sorted(ids)):
        p = plan.loc[i]
        n_p, n_g, stop = gens[i]
        fj = json.load(open(d / "force" / f"{i}.json"))
        tf = np.load(d / "tf" / f"{i}.npz")
        labels, L = fj["labels"], fj["L"]
        s = traj_scalars(tf["gen_logprob"], tf["gen_entropy"], tf["gen_top1"])
        pl = scal(tf["problem_logprob"]) if "problem_logprob" in tf.files else np.array([])
        s.update(prob_lp_mean=pl.mean() if len(pl) else np.nan, prob_lp_min=pl.min() if len(pl) else np.nan,
                 prob_ent_mean=scal(tf["problem_entropy"]).mean() if len(pl) else np.nan,
                 n_prompt=n_p, n_gen=n_g, L_think=L, finished=int(stop == "eos"), think_ended=int(fj["think_end"] is not None))
        # forced-answer dynamics
        res = fj["results"]
        fin = norm(fj.get("final_boxed"))
        cuts = {}
        for lab in ["b512", "b1024", "b2048", "b3072", "b4096", "b6144", "b8192", "f25", "f50", "f75", "f90", "f100"]:
            c = labels[lab]
            for v in ("A", "B"):
                r = res.get(f"{c}|{v}")
                if r is None:
                    continue
                s[f"fa_{lab}_{v}_ent"] = r["first_entropy"]
                s[f"fa_{lab}_{v}_mlp"] = r["mean_logprob"] if r["mean_logprob"] is not None else np.nan
                s[f"fa_{lab}_{v}_p1"] = r["top"][0]["probs"][0] if r["top"] else np.nan
                s[f"fa_{lab}_{v}_closed"] = int(r["closed"])
                cuts.setdefault(lab, {})[v] = norm(r["answer"])
        # self-consistency across cutoffs (variant B, f-cutoffs and b-cutoffs separately), agreement with final / last cutoff
        for tag, labs in (("f", ["f25", "f50", "f75", "f90", "f100"]), ("b", ["b512", "b1024", "b2048", "b3072", "b4096"])):
            for v in ("A", "B"):
                ans = [cuts[l][v] for l in labs if l in cuts and v in cuts[l] and cuts[l][v]]
                if ans:
                    vals, cnt = np.unique(ans, return_counts=True)
                    s[f"sc_{tag}_{v}_modal"] = cnt.max() / len(ans)
                    s[f"sc_{tag}_{v}_ndist"] = len(vals)
                last = cuts.get("f100", {}).get(v) if tag == "f" else cuts.get("b4096", {}).get(v)
                if last:
                    s[f"sc_{tag}_{v}_agree_last"] = np.mean([a == last for a in ans[:-1]]) if len(ans) > 1 else np.nan
                if fin:
                    s[f"sc_{tag}_{v}_agree_final"] = np.mean([a == fin for a in ans]) if ans else np.nan
        s["has_final"] = int(bool(fin))
        for ci, lab in enumerate(["f50", "f100"]):
            for v in ("A", "B"):
                s[f"agree_{lab}_{v}"] = float(cuts.get(lab, {}).get(v) == fin) if fin else np.nan
        # hidden states at reference cutoffs
        ce = list(tf["cut_eff"])
        idx = [ce.index(labels[l]) for l in REF]
        F["cut_last"].append(tf["cut_last"][idx].astype(np.float32))              # [4, n_sel, d]
        F["prompt_last"].append(tf["prompt_last_all_layers"].astype(np.float32))  # [n_layers, d]
        F["prompt_mean"].append(tf["problem_mean"].astype(np.float32))            # [n_sel, d]
        # Laplacian
        a = np.load(d / "attn" / f"{i}.npz") if (d / "attn" / f"{i}.npz").exists() else None
        if a is not None:
            ae = list(a["cut_eff"]); ai = [ae.index(labels[l]) for l in REF]
            F["lap_no"].append(a["topk_noforce"][ai][..., :TOPK].astype(np.float32))      # [4,L,H,k]
            F["lap_force"].append(a["topk_force"][ai][..., :TOPK].astype(np.float32))     # [4,2,L,H,k]
            F["lap_mean_no"].append(a["mean_noforce"][ai].astype(np.float32))
            F["lap_mean_force"].append(a["mean_force"][ai].astype(np.float32))
        else:
            for kk in ("lap_no", "lap_force", "lap_mean_no", "lap_mean_force"):
                F[kk].append(None)
        # extras (DeepSeek only)
        ep = EXTRA.get(slug)
        if ep is not None and (ep / f"{i}.npz").exists():
            e = np.load(ep / f"{i}.npz")
            nc = len(e["cut_eff"]); sel = [0, nc - 1]
            nl = e["res_last"].shape[2]; lay = [nl // 2, (2 * nl) // 3, (5 * nl) // 6]
            F["x_fft"].append(e["fft"][sel].mean(1).astype(np.float32))                # [2,L,H,32]
            F["x_look"].append(e["lookback"][sel].mean(1).astype(np.float32))          # [2,L,H]
            F["x_kl"].append(e["kl_adj"][sel].mean(1).astype(np.float32))              # [2,Lkl,H]
            F["x_klhm"].append(e["kl_headmean"][sel].mean(1).astype(np.float32))
            F["x_res"].append(e["res_last"][sel][:, :, lay].mean(1).astype(np.float32))   # [2,3,d]
            F["x_attn"].append(e["attn_last"][sel][:, :, lay].mean(1).astype(np.float32))
            F["x_mlp"].append(e["mlp_last"][sel][:, :, lay].mean(1).astype(np.float32))
            s["x_ncut"] = nc
        else:
            for kk in ("x_fft", "x_look", "x_kl", "x_klhm", "x_res", "x_attn", "x_mlp"):
                F[kk].append(None)
        rows.append({"id": i, "source": p["source"], "is_robust": p["is_robust"], "max_drop": p["max_drop"], "base_id": p["base_id"],
                     "parent_id": p["parent_id"], "perm_idx": p["perm_idx"], "decay_causing": p["decay_causing"], "gold": p["gold"],
                     "stop": stop, **s})
        if (k + 1) % 20 == 0:
            print(slug, k + 1, "/", len(ids), flush=True)
    meta = pd.DataFrame(rows)
    out = {}
    for kk, v in F.items():
        if kk == "force_curve" or not v:
            continue
        ok = [x is not None for x in v]
        if not any(ok):
            continue
        shape = next(x for x in v if x is not None).shape
        out[kk] = np.stack([x if x is not None else np.full(shape, np.nan, np.float32) for x in v])
    pickle.dump({"meta": meta, "F": out}, open(ROOT / "analysis" / f"feats_{slug}.pkl", "wb"), protocol=4)
    print(slug, "done", meta.shape, {k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    for sl in sys.argv[1:]:
        build(sl)
