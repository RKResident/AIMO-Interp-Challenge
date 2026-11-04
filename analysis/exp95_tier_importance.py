"""exp95 (user 2026-10-31): how much do black / grey / white-box components matter to v8's decisions? Per model, drop one tier at a time from
the v8 recipe (equal-weight mean of what is left, threshold = median of the pool's scores as usual; a branch left empty calls the pool's
majority) and report (a) FLIP RATE: share of decisions that change on the model's label-free pool (all reference traces), (b) ACCURACY on
labelled rows (new forced labels for Qwen / Olmo; old rows for Skywork / DeepSeek, in-sample for them). Olmo also: router and judge removed.
Tiers: black = text only (chk); grey = output probabilities (fa, nrk, f1k, cfl, Min-K% router); white = hidden states / attention (cfc, jmp,
lble, pmle; Qwen judge)."""
import json, numpy as np, pandas as pd

src = open("/home/sabyasachi19/PycharmProjects/AimoAGAIN/analysis/exp91_generalization.py").read()
exec(src[:src.index('print("=" * 110 + "\\nS1')])
TIER = {"chk": "black", "fa": "grey", "nrk": "grey", "f1k": "grey", "cfl": "grey", "cfc": "white", "jmp": "white", "lble": "white", "pmle": "white"}


def drop(spec, tier):
    if spec[0] == "router": return ("router", drop(spec[1], tier), drop(spec[2], tier))
    fs = tuple(f for f in spec[0] if TIER[f] != tier); return (fs, {k: v for k, v in spec[1].items() if k in fs})


def calls2(spec, df, ref):
    """calls() that tolerates empty feature sets (-> the pool's majority: robust iff more than half the pool scores above 0 is undefined,
    so an empty branch calls robust, the majority of our labels)"""
    if spec[0] == "router":
        c = np.zeros(len(df), bool); nov = np.nan_to_num(df.mink.values.astype(float), nan=-99) < RT
        for k, br in ((True, spec[1]), (False, spec[2])): c[nov == k] = calls2(br, df, ref)[nov == k]
        return c
    if not spec[0]: return np.ones(len(df), bool)
    return calls(spec, df, ref)[0]


# Olmo v8 = router + Qwen judge: rebuild the judge pieces (as in the v8 validation)
FQ = pd.DataFrame([json.loads(l) for l in open(A + "ec2_results/a10g/feats/Qwen__Qwen3.5-4B.jsonl")]).set_index("pid")
_, qs = calls(BUILDS["v7"]["Qwen"], FQ.assign(mink=np.nan), ref_of("Qwen", "R2")); QS = pd.Series(qs, index=FQ.index)
pct_ = lambda v, r: (np.searchsorted(r, v, "left") + np.searchsorted(r, v, "right")) / (2 * len(r))

print(f"{'model':8s} {'removed':22s} | flips on pool | accuracy on labelled (full recipe -> without)")
for m in MS:
    ref = ref_of(m, REG[m]); spec = BUILDS["v7"][m]; ev = new_df(m, "forced") if m in TAG else OLD[m]
    pool = ref.copy(); pool["mink"] = pool["mink"] if "mink" in pool else np.nan
    full_p, full_e = calls2(spec, pool, ref), calls2(spec, ev, ref); a0 = (full_e == ev.y.values).mean()
    for tier in ("black", "grey", "white"):
        sp = drop(spec, tier)
        if sp == spec: continue
        fp, fe = calls2(sp, pool, ref), calls2(sp, ev, ref)
        print(f"{m:8s} {tier + '-box features':22s} | {(fp != full_p).mean():5.0%}         | {a0:.2f} -> {(fe == ev.y.values).mean():.2f}")
    if spec[0] == "router":
        for nm, sp in (("router (novel br. everywhere)", spec[1]), ("router (familiar everywhere)", spec[2])):
            fp, fe = calls2(sp, pool, ref), calls2(sp, ev, ref)
            print(f"{m:8s} {nm:22s} | {(fp != full_p).mean():5.0%}         | {a0:.2f} -> {(fe == ev.y.values).mean():.2f}")
        # v8: add the Qwen judge on the rows / pool problems that have a Qwen trace
        _, r_pool = calls(spec, ref, ref); r_ref = np.sort(r_pool)
        P50 = [p for p in POOL[m].pid if p in QS.index]; q_ref = np.sort(QS.loc[P50].values)
        _, re_ = calls(spec, ev, ref); judge = np.array([(pct_(a, r_ref) + pct_(QS[p], q_ref)) / 2 > .5 for a, p in zip(re_, ev.pid)])
        pp = POOL[m].assign(mink=[MINK.get(p, np.nan) for p in POOL[m].pid]); _, rp = calls(spec, pp, ref)
        jp = np.array([(pct_(a, r_ref) + pct_(QS[p], q_ref)) / 2 > .5 for a, p in zip(rp, pp.pid)]); router_p = rp > 0
        print(f"{m:8s} {'Qwen judge (v8 -> v7)':22s} | {(jp != router_p).mean():5.0%}         | v8 {(judge == ev.y.values).mean():.2f} -> v7 {a0:.2f}")
        print(f"{m:8s} {'v8 judge alone':22s} | {np.mean([(QS[p] > 0) != j for p, j in zip(pp.pid, jp)]):5.0%}         | {np.mean([(QS[p] > 0) == y for p, y in zip(ev.pid, ev.y.values)]):.2f}")
