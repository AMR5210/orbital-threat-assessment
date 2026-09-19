"""Shared inference module: load -> validate -> predict -> explain.

Single source of truth for "the trained model," used by:
- FastAPI (src/otda/api.py — local + Docker)
- The LangChain agent's predict_pha tool (Stage 7, src/otda/agent.py)

NOT used by the Stage 9 SageMaker deployment (scripts/deploy_to_sagemaker.py):
that deploys straight from MLflow's own registered model via MLflow's
generic SageMaker deployment tooling (the documented path, per
instruction), which runs MLflow's own scoring server as a separate
process — it does not import this module. See that script's docstring
for exactly what that trades away (hard labels only, no explanation
field) versus going through here.

Serves the Decision Tree from Experiment 2 (MOID-included feature set):
the best-performing (test PR-AUC 0.999, RESULTS.md) AND the
interpretable model that reconstructs NASA's own two-condition PHA
rule (legacy/Project_codeFile.ipynb's Exp2 finding) — which is exactly
why explain() below can state the rule directly instead of trying to
extract it from the model post hoc.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlflow.sklearn
import pandas as pd

from otda.features import EXP2_FEATURES
from otda.logging_backend import log_prediction

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "champion"

# NASA/JPL's own PHA definition (confirmed directly against CNEOS):
# potentially hazardous iff MOID <= 0.05 AU AND H <= 22.0.
MOID_THRESHOLD_AU = 0.05
H_THRESHOLD = 22.0


@dataclass
class PredictionResult:
    is_pha: bool
    probability: float
    explanation: str


def load_model(path: Path = DEFAULT_MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"No exported model at {path}. Run "
            "scripts/export_champion_model.py first."
        )
    return mlflow.sklearn.load_model(str(path))


def explain(moid: float, H: float, is_pha: bool, probability: float) -> str:
    """Deterministic, rule-grounded explanation — never delegates to an
    LLM to characterize *why*, so this string is safe to hand an agent
    (Stage 7) as ground truth rather than something it has to reason
    its way to from raw numbers alone."""
    moid_met = moid <= MOID_THRESHOLD_AU
    h_met = H <= H_THRESHOLD

    conditions = (
        f"MOID = {moid:.4f} AU ({'<=' if moid_met else '>'} {MOID_THRESHOLD_AU} AU, "
        f"condition {'met' if moid_met else 'not met'}) and "
        f"H = {H:.2f} ({'<=' if h_met else '>'} {H_THRESHOLD}, "
        f"condition {'met' if h_met else 'not met'})"
    )

    verdict = "potentially hazardous" if is_pha else "not potentially hazardous"
    rule_agrees = "consistent with" if (moid_met and h_met) == is_pha else "in tension with"

    return (
        f"Model prediction: {verdict} (probability={probability:.3f}). "
        f"NASA's PHA definition requires MOID <= {MOID_THRESHOLD_AU} AU AND "
        f"H <= {H_THRESHOLD}. This object has {conditions}. "
        f"The model's prediction is {rule_agrees} NASA's two-condition rule "
        "applied directly to this object's MOID and H."
    )


def predict(model, features: dict) -> PredictionResult:
    """features must contain every key in EXP2_FEATURES
    (e, q, H, is_neo, class_code, moid) — extra keys are ignored."""
    row = pd.DataFrame([{k: features[k] for k in EXP2_FEATURES}])
    probability = float(model.predict_proba(row)[0, 1])
    is_pha = bool(model.predict(row)[0])
    explanation = explain(features["moid"], features["H"], is_pha, probability)
    result = PredictionResult(is_pha=is_pha, probability=probability, explanation=explanation)

    # No-op unless OTDA_DYNAMODB_TABLE is set (see otda.logging_backend
    # for exactly which callers this does and doesn't reach).
    log_prediction(features, result)

    return result
