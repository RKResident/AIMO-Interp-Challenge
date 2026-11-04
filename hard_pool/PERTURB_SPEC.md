# Writing perturbations of competition math problems

You get a JSON list of problems: `{"pid": ..., "problem": ..., "answer": ...}`. For each problem, write **four** perturbed versions. Each version must keep **exactly the same mathematical problem and the same final answer.** A scripted fifth type (typos) is added later, so don't write it.

| type | what to change | example (original: "An urn contains $n$ marbles, each red or blue ...") |
|---|---|---|
| `rephrase` | Reword the sentences and reorder the clauses. Keep every variable name, object and number. | "There are $n$ marbles in an urn, each colored either red or blue ..." |
| `rename` | Rename variables, people and objects consistently (e.g. $n$→$k$, marbles→beads, red/blue→green/yellow). Keep the wording otherwise. | "A jar contains $k$ beads, each green or yellow ..." |
| `distract` | Add one short, true and irrelevant sentence: a side fact or a reminder of a definition. Don't add numbers it could be confused with. | "... The marbles are all the same size and weight. ..." |
| `domain` | Recast the problem as a real-world scenario (factory, architect, sports, ...) with the same mathematical structure. | "A factory crate holds $n$ light bulbs, each defective or working ..." |

## Hard rules (every version is checked automatically)
1. **Every number in the original appears in the version**, with the same values: integers, decimals, exponents and indices. Don't write numbers out as words.
2. **The answer doesn't change.** Don't add, drop or weaken any condition. Keep "at least", "distinct", "positive" and "integer" exactly.
3. **LaTeX stays valid.** `$` delimiters are balanced, and every original `$...$` expression survives with only variable names changed (in `rename` and `domain`).
4. **Each version differs from the original** and stands on its own (no references to "the original problem").
5. **Keep the original's final request** ("Find ...", "Compute ...", "What is ...").
6. **Variable choice:** in `rename`, pick new variable names that don't clash with names already in the problem.

## Output
Write ONE JSON file (path given in your task) mapping each pid to its four versions:
```json
{"hp0123": {"rephrase": "...", "rename": "...", "distract": "...", "domain": "..."}, "hp0124": {...}}
```
Write valid JSON only. In JSON strings, LaTeX backslashes must be escaped (`\\frac`). Don't add commentary.
