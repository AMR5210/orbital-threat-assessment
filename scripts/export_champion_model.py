"""Export the champion model's artifact from MLflow to a local directory.

The champion is fixed: Decision Tree, Experiment 2 (MOID included),
full population — the best-performing (test PR-AUC 0.999, per
RESULTS.md) AND most interpretable model (it's the one that
reconstructs NASA's own two-condition PHA rule; see legacy/
Project_codeFile.ipynb's Exp2 finding).

Exports to models/champion/ as a plain local directory (not a live
`models:/name@alias` registry reference) so the serving layer — FastAPI
here, a SageMaker /invocations adapter later (Stage 9) — never needs a
live connection to mlflow.db/mlruns at runtime. This is also exactly
the artifact shape the documented MLflow -> SageMaker deploy path
expects (Stage 9 decision: SageMaker deploys from MLflow directly, no
Snowflake registry in that path).

models/ is gitignored — regenerate with this script after any
`scripts/run_experiments.py` run.

Known cross-environment gotcha: MLflow's local file-based artifact
store records each run's artifact location as an ABSOLUTE path at
logging time. A run logged from the host (e.g.
`file:C:/Users/.../mlruns/...`) has an artifact URI that doesn't exist
when this script instead runs inside the Airflow container (repo
mounted at /opt/airflow/project) — `download_artifacts` silently
returns an empty directory rather than raising in that case. This
script verifies the export actually produced files (below) for exactly
that reason. Running `scripts/run_experiments.py --dataset full` from
the SAME environment before this script avoids the mismatch entirely,
since that logs a fresh run with a path native to wherever it just ran.

Usage:
    python scripts/export_champion_model.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import mlflow

from otda.tracking import get_tracking_db

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = REPO_ROOT / "models" / "champion"

EXPERIMENT_NAME = "orbital_threat_assessment"
DATASET_NAME = "full_population_932335"
FEATURE_SET_NAME = "exp2_moid_included"
MODEL_NAME = "decision_tree"


def main() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{get_tracking_db().as_posix()}")
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise SystemExit(
            f"No MLflow experiment '{EXPERIMENT_NAME}' — run "
            "scripts/run_experiments.py first."
        )

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=(
            f"params.dataset = '{DATASET_NAME}' and "
            f"params.feature_set = '{FEATURE_SET_NAME}' and "
            f"params.model_name = '{MODEL_NAME}'"
        ),
    )
    if runs.empty:
        raise SystemExit(
            f"No run found for feature_set={FEATURE_SET_NAME}, "
            f"model={MODEL_NAME}, dataset={DATASET_NAME}."
        )
    run_id = runs.sort_values("start_time").iloc[-1]["run_id"]
    test_pr_auc = runs.sort_values("start_time").iloc[-1]["metrics.test_pr_auc"]

    if EXPORT_DIR.exists():
        shutil.rmtree(EXPORT_DIR)
    EXPORT_DIR.parent.mkdir(parents=True, exist_ok=True)

    local_path = mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{run_id}/model", dst_path=str(EXPORT_DIR.parent)
    )
    # download_artifacts preserves the artifact's own subpath ("model");
    # normalize to the fixed EXPORT_DIR name regardless of that subpath.
    downloaded = Path(local_path)
    if downloaded != EXPORT_DIR:
        downloaded.rename(EXPORT_DIR)

    # download_artifacts does NOT raise when the source artifact URI is
    # unreachable (e.g. a host-absolute path from inside a container) —
    # it silently returns an empty directory instead (see module
    # docstring). Verify something actually landed rather than trust a
    # print statement.
    exported_files = list(EXPORT_DIR.rglob("*"))
    if not exported_files:
        raise SystemExit(
            f"Export produced an EMPTY directory at {EXPORT_DIR} — the source "
            f"artifact for run {run_id} was not reachable from this environment "
            "(see this script's docstring for the host-vs-container path gotcha). "
            "Nothing was actually exported; do not treat this as a success."
        )

    print(f"Exported run {run_id} (test_pr_auc={test_pr_auc:.4f}) -> {EXPORT_DIR}")
    print(f"  {len(exported_files)} file(s): {[f.name for f in exported_files]}")


if __name__ == "__main__":
    main()
