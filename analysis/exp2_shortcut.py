"""Exp 2: shortcut / guessability. Answer properties (from the finished trace's own boxed answer, ~= gold on agg where base accuracy ~1)
and early-correctness (is the forced answer already right at 512 tokens, when does it stabilise)."""
from common import *
allm, _ = load_all()
rows = []
for i, r in allm.iterrows():
    fj = force_json(r.model, r.id)
    lab, L = fj["labels"], max(fj["L"], 1)
    cuts = sorted(set(lab.values()))
    res = fj["results"]
    ans = {v: [nrm(res.get(f"{c}|{v}", {}).get("answer")) for c in cuts] for v in "AB"}
    fin = nrm(fj.get("final_boxed"))
    proxy = fin if fin else ans["B"][-1]
    f = {"has_final": int(bool(fin))}
    # answer properties
    a = proxy or ""
    try:
        v = float(a); num = True
    except Exception:
        v, num = np.nan, False
    f.update(ans_numeric=int(num), ans_int=int(num and float(v).is_integer()), ans_log10=np.log10(abs(v) + 1) if num else np.nan,
             ans_ndigits=len(re.sub(r"\D", "", a)), ans_le10=int(num and abs(v) <= 10), ans_le100=int(num and abs(v) <= 100),
             ans_negative=int(num and v < 0), ans_fracdec=int(("frac" in a) or ("/" in a) or ("." in a)), ans_nonnumeric=int(not num),
             ans_len=len(a))
    # early correctness / stabilisation (relative to the model's own final answer)
    for var in "AB":
        s = ans[var]
        ok = [x == proxy for x in s]
        f[f"ok_b512_{var}"] = float(ans[var][0] == proxy and cuts[0] <= 600) if cuts else np.nan
        f[f"ok_first_{var}"] = float(ok[0])
        f[f"frac_ok_{var}"] = float(np.mean(ok))
        stable = next((k for k in range(len(cuts)) if all(ok[k:])), None)
        f[f"stable_frac_{var}"] = cuts[stable] / L if stable is not None else 1.0
        fc = next((k for k, o in enumerate(ok) if o), None)
        f[f"first_correct_frac_{var}"] = cuts[fc] / L if fc is not None else 1.0
        f[f"n_changes_{var}"] = sum(1 for k in range(1, len(s)) if s[k] != s[k - 1])
    f["A_ne_B"] = float(np.mean([x != y for x, y in zip(ans["A"], ans["B"])]))
    f["ok_at_1024"] = float(ans["B"][[k for k, c in enumerate(cuts) if c >= 1000][0]] == proxy) if any(c >= 1000 for c in cuts) else np.nan
    rows.append(f)
Fx = pd.DataFrame(rows)
ans_cols = [c for c in Fx if c.startswith("ans_")]
early_cols = [c for c in Fx if c not in ans_cols and c != "has_final"]
Fx.to_pickle(A / "exp2_features.pkl")
print("=== Exp 2: shortcut / guessability ===")
print("answer proxy = final boxed answer of the trace; for traces that never finished, the forced answer at the last cutoff.")
single_auc(Fx, allm, top=14)
for nm, cols in (("answer properties", ans_cols), ("early-correctness / stabilisation", early_cols), ("both", ans_cols + early_cols + ["has_final"])):
    evaluate(nm, Fx[cols].values, allm, C=0.05)
