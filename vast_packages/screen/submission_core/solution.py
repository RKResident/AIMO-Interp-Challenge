"""Codabench entry point for the spurious-reasoning signal submissions.

Predicts whether a model's answer to each problem is robust from interpretable signals of its own reasoning (forced-answer
confidence, hidden-state trajectory geometry, prompt-noise sensitivity, trace length), as configured in spurious_artifacts/recipe.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SOLUTION_DIR = str(Path(__file__).resolve().parent)
if _SOLUTION_DIR not in sys.path:
    sys.path.insert(0, _SOLUTION_DIR)

import json as _json
import traceback as _tb

_INFO_PATH = Path(_SOLUTION_DIR) / "SUBMISSION_INFO.json"
_INFO = _json.loads(_INFO_PATH.read_text()) if _INFO_PATH.is_file() else {}
_banner = (f"[spurious] SUBMISSION {_INFO.get('zip_name', '?')} | cut {_INFO.get('cut', '?')} (q_nonrobust {_INFO.get('q_nonrobust', '?')}) | "
           f"cap {_INFO.get('cap', '?')} | features {_INFO.get('recipe_features', '?')} | build {_INFO.get('build_id', '?')} built {_INFO.get('built', '?')}")
print(_banner, flush=True)
print(_banner, file=sys.stderr, flush=True)
print("[spurious] solution.py imported, importing spurious_inference", flush=True)
print("[spurious] solution.py imported, importing spurious_inference", file=sys.stderr, flush=True)
try:
    from spurious_inference import predict_robustness
except BaseException:   # the ingestion program hides this exception, so print it before re-raising
    print("[spurious] import FAILED:\n" + _tb.format_exc(), file=sys.stderr, flush=True)
    print("[spurious] import FAILED:\n" + _tb.format_exc(), flush=True)
    raise


def are_robust(model_id: str, reasoning_effort: str, problems: list[str]) -> list[bool]:
    """Predict robustness for a batch of problems.

    Args:
        model_id: Hugging Face model identifier cached on the evaluation worker.
        reasoning_effort: Evaluated reasoning effort ("default", "low", "medium").
        problems: Mathematical problem statements in evaluation order.

    Returns:
        One native Python boolean per input problem (True = robust, False = not robust).
    """
    return predict_robustness(model_id=model_id, problem_texts=problems)
