# AIMO Interpretability Challenge: per-model white-box recipes (Small Models track)

Code and technical report for our two nominated submissions to the
[AIMO Interpretability Challenge](https://aimo-interp.github.io/) (Small Models track: Qwen3.5-4B, Skywork-OR1-Math-7B, Olmo-3-7B-Think,
DeepSeek-R1-0528-Qwen3-8B). For each (model, problem) pair, the method predicts whether the model's success survives faithful rewordings
of the problem. It reads a single greedy reasoning trace, capped at 6144 tokens, plus a few teacher-forced passes over it.

| Submission | Codabench submission ID | Zip | Public leaderboard | `train-main-v2` (our 4 models, 43 rows) |
|---|---|---|---|---|
| **S1** | 1023618 | `submitted_solutions/permodel-v8-ent4096-small.zip` | 1.00 | 33 / 43 |
| **S2** | 1024956 | `submitted_solutions/permodel-v8-ent4096-olmoseg25-small.zip` | 0.93 | 36 / 43 |
| all-robust baseline | — | — | 0.50 | 31 / 43 |

Each zip also carries a build ID in its `SUBMISSION_INFO.json`. The SHA-256 of each zip lets anyone check that the file in this repo is
byte-identical to the submitted one:

| Submission | Build ID | Built | SHA-256 of the zip |
|---|---|---|---|
| S1 | `c03b14ac1abc` | 2026-10-31 | `a930cb3daea646811394bc7b218fe5bacfeec147a0c2fc307ebe916aecc1478a` |
| S2 | `87f69d278fb3` | 2026-11-01 | `95b0f4962635afa98c164341b1aa06ebbfc16379d773cf7148640deface5a9d0` |

The full description, ablations (including the white-box ablation) and limitations are in the report: [`report/main.pdf`](report/main.pdf).

## Method in one paragraph

A trace that finishes thinking within the cap is called robust. Unfinished traces are scored per model:
- a rank ensemble of label-free-calibrated features (percentile against unlabelled reference traces; threshold = their median), mixing:
  - **text** features: check/verify share;
  - **output-distribution** features: forced-answer confidence, and answer survival when the reasoning is cut;
  - **internal** features: hidden-state sensitivity to automatic rewordings, attention look-back, and 64-token-block attention jumps;
- for Qwen, the entropy of the forced answer after 4096 tokens;
- for Olmo, a Min-K% familiarity router plus a Qwen "judge" of the same problem.

S2 differs from S1 only in Olmo's unfinished calls: it uses a projection of the hidden state at 25% of the trace onto a
robust-minus-non-robust direction.

## Repository layout

| Path | What it is |
|---|---|
| `submission_core/` | Inference code that runs on the evaluator. The entry point is `solution.py`, which calls `predict_robustness()` in `spurious_inference.py`. `spurious_features.py` has the feature formulas; `wb2_log.py` has the hidden-state readouts (incl. the S2 direction) and logging. |
| `submitted_solutions/` | The two nominated zips, exactly as submitted: code + `spurious_artifacts/recipe.json` (features, signs, reference values, thresholds, router/judge settings) + shipped `wb2_ship_*.npz` (PCA bases and directions). |
| `analysis/build_submission_artifacts.py` | Builds every `recipe.json` (calibration: reference distributions, label-free median thresholds, per-model recipes, overrides). |
| `reproduce/` | **Start here to check the report's numbers.** `replay.py` plus `features.csv` (our per-trace feature values) reproduce every Main column on CPU in seconds; see [Reproducing the report](#reproducing-the-report). |
| `analysis/` | Experiment scripts (`expNN_*.py`), numbered as in the report's appendix and our experiment log. They are published for transparency, not as a runnable pipeline: they read intermediate data that is not in this repo, contain absolute paths from our machine, and some load scripts we did not publish. |
| `datasets/` | **Our labelled robustness datasets** (5 sets: problems, five perturbations each, per-sample answers, labels under the inferred official rule), documented in [`datasets/README.md`](datasets/README.md). Built by `analysis/build_datasets.py`. |
| `hard_pool/` | Labelling tooling for our own calibration data: perturbation spec (`PERTURB_SPEC.md`), assembly and label computation. |
| `vast_packages/` | GPU job code: greedy generation, labelling with vLLM, feature extraction (`screen/`, `recall_src/`, …). |
| `extract_*.py` | Earlier feature / trajectory extraction code. |
| `package_submissions.sh` | Builds the Small-track zips from the solution directories and checks each zip's `SUBMISSION_INFO.json`. |
| `report/` | LaTeX source of the technical report (NeurIPS 2026 style), `main.pdf`, and candidate figures in `report/figs_review/`. |

## Running a submission

Each zip is self-contained. Unzip it and call the entry point the challenge defines:

```python
from solution import are_robust            # or: from spurious_inference import predict_robustness
calls = are_robust("Qwen/Qwen3.5-4B", "default", ["<problem text>", ...])   # list[bool], True = robust
```

Requirements match the challenge's evaluation environment: one GPU, PyTorch + Hugging Face Transformers, the four models available locally,
no `accelerate` (models are loaded and then moved with `.to("cuda")`), and one problem per call for the final evaluation.

## Reproducing the report

Every `train-main-v2` (Main) number in the report can be regenerated on a laptop with no GPU and no model downloads:

```bash
pip install -r requirements.txt                      # numpy, pandas, pyarrow are enough for this step
python reproduce/replay.py --main /path/to/train-main-v2.parquet
```

`--main` is the organisers' `train-main-v2` parquet: the 2026-11-02 release on their Hugging Face page has all 43 rows for our four models
(older copies miss some Skywork rows, and the script stops if any label is missing). The script prints each value next to the
number in the report:

| Output block | Report location |
|---|---|
| All robust, finished only, S1, S2, text only | Table 1, Main column |
| v8, v8 without router/judge, S1, v8 + Olmo direction, S2 | Table 2, Main column |
| Text only / no internals / no output features, per model | Table 3, Main columns |
| Drop one feature from S1 | Appendix B, Main column |
| Qwen $H_b$ by truncation, Olmo direction by position | Table 4 |
| Per-feature AUC (our first sets / confidence-matched set / Main) | Appendix C |

How it works:
- It loads both recipes straight from `submitted_solutions/*.zip`. `v8` is S1 without Qwen's $H_{4096}$ readout; the two recipes differ only there.
- It scores each trace with the shipped `submission_core/spurious_features.score_trace`, mirroring the submission's decision: finished → robust; otherwise the per-model rank ensemble, Olmo's Min-K% router, then the override.
- The ablation variants are the same recipe edits described in the report.

What it does not cover:
- **Leaderboard columns.** They are the scores the leaderboard returned, and per-problem leaderboard data is not public.
- **Olmo's Qwen judge.** It needs a Qwen trace of each Olmo problem, so Olmo's own router branch decides on Main, as in the report.
- **Extracting the features from scratch.** That needs the models and a GPU; see `vast_packages/`.

`reproduce/features.csv` contains one row per trace:
- **Main** (`set = main`): our own 6144-token greedy traces of the 43 labelled rows. Problems are identified by `pid = "mv2_" + sha256(problem.strip())[:8]`; there is no problem text and no official label.
- **Our labelled problems** (`set = ours_first`, `ours_matched`): unfinished traces only, with our labels (forced rule at 6144; see `datasets/`).

| Columns | Meaning |
|---|---|
| `finished` | thinking ended within 6144 tokens |
| `chk`, `fa`, `nrk`, `f1k`, `cfl`, `cfc`, `lble`, `pmle`, `jmp` | the rank-ensemble features (definitions in the report's Appendix A) |
| `mink_20` | Olmo's router input (Main rows only) |
| `force_ent__b{512…6144}` | forced-answer entropy $H_b$ after $b$ tokens of reasoning |
| `dirproj_seg{25,50,75}__s2` | Olmo direction projection at 25/50/75% of the trace |

The file was exported by `analysis/build_reproduce.py`, which needs our unpublished intermediate data and is not included.

## Rebuilding the zips

`analysis/build_submission_artifacts.py` writes each solution directory into the organisers' `getting-started` repository, and
`package_submissions.sh` builds and checks the zips with that repo's `scripts/build.py --small`. Both scripts contain absolute paths
from our machine; adjust them first. Recalibrating from scratch also needs our extracted features and labelled data, which are not
included (they are large, and partly derived from competition data we don't redistribute). The shipped `recipe.json` files contain every
value the submissions use.

## Data not included

Competition data (`train-main-v2`, leaderboard problems), our perturbations of the organisers' problems, raw greedy traces and generated
text, the full extracted feature sets, and the papers we read are not in this repository. Our own labelled datasets are, in `datasets/`;
the feature values the report's Main tables need are in `reproduce/features.csv` (no problem text, no official labels).

## Citation

```bibtex
@article{stefanik2026aimointerp,
  author  = {Michal Stef{\'{a}}nik and Philipp Mondorf and Andreas Waldis and Qianying Liu and Chuan Yang and Michal Spiegel and
             Josef Kuchar and Marek Kadlc{\'{\i}}k and Adam Vawda{-}Oomerjee and Chaoran Liu and Simon Frieder and Barbara Plank and
             Fazl Barez and Pontus Stenetorp},
  title   = {{AIMO} Interpretability Challenge},
  journal = {CoRR}, volume = {abs/2607.13899}, year = {2026},
  doi     = {10.48550/ARXIV.2607.13899},
}
```

## AI usage

Much of the code and experimentation was done with Claude Code (mainly Claude Opus 5.5); perturbation texts for our own calibration labels
were written by Claude Sonnet. See the AI usage statement in the report.
