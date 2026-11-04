# Our robustness datasets

Five datasets we built during the AIMO Interpretability Challenge (Small Models track) to calibrate our submissions. Each gives, for one or more of
the four small models, a set of competition math problems, five answer-preserving perturbations of each, sampled answers for every version,
and a robustness label computed with the challenge's labelling rule as we inferred it.

**Read this first.** Section 6 of our report (`report/main.pdf`) shows that these labels do **not** reproduce the official labels well:
- with 4–8 samples per version, many labels for problems of middling accuracy are driven by sampling noise;
- the labels partly track problem difficulty;
- our perturbations fail to reproduce official non-robust labels on a `train-main-v2` problem.

We used them only for low-capacity choices (feature signs, per-model feature subsets, one hidden-state direction) and never to measure
accuracy. Use them with that in mind.

Built by `analysis/build_datasets.py` from our raw labelling outputs. Only answers and correctness are kept, not the generated text.

## The datasets

| Folder | Problems | Models (problems labelled) | Samples per version | Generation cap | What it is |
|---|---|---|---|---|---|
| `set100_hard_32k/` | 100 hard competition problems, mostly post-2023 (BeyondAIME, OlympiadBench, MathArena: SMT / HMMT / AIME 2023–25) | Qwen3.5-4B (100), Olmo-3-7B-Think (50) | original 8; perturbations 4, or 8 for problems relabelled in a second round | 32768 | Our first labelled pool, screened for problems the models solve but do not finish quickly. |
| `olympiad_32k/` | 41 of the 80 olympiad problems (AIME 2025/26, HMMT Feb 2025/26) | Skywork-OR1-Math-7B (35), DeepSeek-R1-0528-Qwen3-8B (15) | 4–8 | 32768 | Redo of the problems whose greedy 16k labels (next row) were damaged by the cap or unlabelable. |
| `olympiad_greedy_16k/` | 80 olympiad problems | Skywork (80), DeepSeek (80) | 1 (greedy) | 16384 | An earlier, different scheme: one greedy generation per version; non-robust if any perturbation fails; `unlabelable` if the original fails. Kept as produced. |
| `confmatched_6144/` | 70 problems (AIME 1983–2025, AMC, MATH, HMMT, BeyondAIME, MathArena) | Qwen (40), Olmo (39), Skywork (30) | 8 | 6144 | **Confidence-matched**: problems still thinking at 6144 tokens whose forced-answer top-1 probability is at least 0.7, like most unfinished `train-main-v2` traces. Report exp110. |
| `confmatched_32k/` | the same 70 problems | Qwen (15), Skywork (70) | 8 | 32768 | Relabel of the confidence-matched set at 32k with natural answers (report exp117). |

`summary.json` has the label counts per dataset, model and scoring rule.

## Files in each folder

| File | Fields |
|---|---|
| `problems.jsonl` | `pid`, `source` (dataset or competition it came from), `problem` (text), `answer` (gold final answer) |
| `perturbations.jsonl` | `pid`, `ptype`, `text`: five versions per problem (below) |
| `samples_<model>.jsonl` | one record per (`pid`, `ptype`): `model`, `cap`, and `samples`, a list of `{n_gen, finish, answer, correct, forced_answer, correct_any}` (generated tokens, stop reason `stop`/`length`, boxed answer if it finished, whether that is right, the answer forced at the cap, whether the natural-or-forced answer is right) |
| `labels_<model>.csv` | per problem: sample counts per version, then for each scoring rule (`natural`, `forced`): accuracy of the original and of each perturbation, `max_drop`, the `worst` perturbation, `unsolved` and `label` |

`set100_hard_32k/` also has `labels_as_used_<model>.csv`, the labels as our calibration used them at the time. They differ only in that problems
never solved were marked `unsolved` rather than robust. `olympiad_greedy_16k/` has the original label files and compact sample records
(`stage`, `answer`, `correct`, `n_tokens`, `finish`).

## Perturbations

Five types, each meant to keep the mathematical problem and its answer unchanged:

| `ptype` | Change |
|---|---|
| `rephrase` | reword and reorder sentences, keeping every variable, object and number |
| `rename` | rename variables, people and objects consistently |
| `distract` | add one short, true, irrelevant sentence |
| `domain` | recast as a real-world scenario with the same structure |
| `typos` | scripted letter swaps in about 6% of prose words (never in math or numbers), seeded per problem |

The first four were written by Claude Sonnet following `PERTURB_SPEC.md`, for every dataset. Every version is checked automatically: all
numbers of the original present, balanced `$`, non-empty, different from the original. `typos` is produced by `hard_pool/assemble_new.py` /
`assemble_perts.py`.

## Sampling and labels

- **Prompt:** the challenge prompt ("Solve the following mathematical problem step by step. Put your final answer within \boxed{}.") with each model's chat template.
- **Sampling:** temperature 1, top-k 40, top-p 0.95, vLLM, up to the cap. When a sample hits the cap, an answer is forced by appending `</think> The final answer is $\boxed{`.
- **Scoring rules:**
  - `natural`: a sample is right only if it finished and boxed the gold answer;
  - `forced`: the forced answer also counts.

  At 32k almost every sample finishes, so the two agree. At 6144 tokens `natural` is nearly always 0 for unfinished problems, so use `forced` there.
- **Label:**
  - `max_drop` = accuracy(original) − min over perturbations of accuracy(perturbation);
  - **robust** if `max_drop` ≤ 0.10 or the original is never solved;
  - **non_robust** if `max_drop` ≥ 0.25;
  - **ambiguous** otherwise.

  No official threshold is published. We inferred these from `train-main-v2`, where every robust row has a drop ≤ 0.10 and every non-robust row ≥ 0.25, and the same rule reproduces the public validation sample exactly.

## Not included

Perturbations and samples we made of the organisers' `train-main-v2` problems (diagnostics in report Section 6) are left out because they
contain the organisers' problem texts. Problem texts here come from public datasets and competitions, credited in `source`; check each
source's terms before reuse.
