# AIMO Interpretability Challenge: per-model white-box recipes (Small Models track)

Code and technical report for our two nominated submissions to the
[AIMO Interpretability Challenge](https://aimo-interp.github.io/) (Small Models track: Qwen3.5-4B, Skywork-OR1-Math-7B, Olmo-3-7B-Think,
DeepSeek-R1-0528-Qwen3-8B). For each (model, problem) pair, the method predicts whether the model's success survives faithful rewordings
of the problem. It reads a single greedy reasoning trace, capped at 6144 tokens, plus a few teacher-forced passes over it.

| Submission | Zip | Public leaderboard | `train-main-v2` (our 4 models, 43 rows) |
|---|---|---|---|
| **S1** | `submitted_solutions/permodel-v8-ent4096-small.zip` | 1.00 | 33 / 43 |
| **S2** | `submitted_solutions/permodel-v8-ent4096-olmoseg25-small.zip` | 0.93 | 36 / 43 |
| all-robust baseline | — | 0.50 | 31 / 43 |

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
| `analysis/` | Experiment scripts (`expNN_*.py`), numbered as in the report's appendix and our experiment log. They read intermediate data that is not in this repo. |
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

## Rebuilding the zips

`analysis/build_submission_artifacts.py` writes each solution directory into the organisers' `getting-started` repository, and
`package_submissions.sh` builds and checks the zips with that repo's `scripts/build.py --small`. Both scripts contain absolute paths
from our machine; adjust them first. Recalibrating from scratch also needs our extracted features and labelled data, which are not
included (they are large, and partly derived from competition data we don't redistribute). The shipped `recipe.json` files contain every
value the submissions use.

## Data not included

Competition data (`train-main-v2`, leaderboard problems), raw greedy traces and sampled generations, extracted features, our own
labelled sets, and the papers we read are not in this repository.

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
