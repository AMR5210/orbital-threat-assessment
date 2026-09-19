"""MLflow experiment tracking.

Replaces the original notebook's model-selection workflow: run,
eyeball a printed table, hand-transcribe the winning numbers into the
README (Cells 25/31/32). Every run below is logged with its params,
val/test metrics, and the fitted model artifact — the README's results
table (scripts/generate_results_table.py) is generated FROM this
tracking store, not typed by hand.

Local, SQLite-backed tracking store (./mlflow.db) — no server, no
credentials. MLflow 3.x put the plain filesystem backend (`./mlruns`)
into maintenance mode as of this version and recommends a database
backend instead (mlflow.exceptions.MlflowException raised otherwise,
pointing at `mlflow migrate-filestore`); SQLite keeps this fully local
and credential-free, same as the filesystem store was, just via a
single .db file instead of a directory tree. Artifacts (models) still
land as files under ./mlartifacts.

This is tracking only, not a model registry: nothing here promotes a
model to "production." See project decision on MLflow vs. Snowflake
Model Registry (Stage 9 handles registries).
"""
from __future__ import annotations

import os
from pathlib import Path

import mlflow
import mlflow.sklearn

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MLFLOW_DB = REPO_ROOT / "mlflow.db"


def get_tracking_db() -> Path:
    """OTDA_MLFLOW_DB lets tests/CI point at an isolated file instead of
    the shared local tracking store, without touching the real one."""
    override = os.environ.get("OTDA_MLFLOW_DB")
    return Path(override) if override else DEFAULT_MLFLOW_DB


def get_or_create_experiment(name: str) -> str:
    mlflow.set_tracking_uri(f"sqlite:///{get_tracking_db().as_posix()}")
    exp = mlflow.get_experiment_by_name(name)
    if exp is not None:
        return exp.experiment_id
    return mlflow.create_experiment(name)


def log_experiment_run(
    experiment_name: str,
    run_name: str,
    feature_set_name: str,
    dataset_name: str,
    model_name: str,
    split_sizes: dict,
    result,  # otda.train.ExperimentResult
    n_iter: int,
    random_state: int,
) -> str:
    """Logs one ExperimentResult as one MLflow run. Returns the run_id.

    Params logged: everything needed to reproduce the run (model,
    feature set, dataset, split sizes, search budget, winning
    hyperparameters). Metrics logged: every Metrics field, twice —
    once prefixed val_ (used for model selection) and once prefixed
    test_ (touched once, the number that gets reported)."""
    experiment_id = get_or_create_experiment(experiment_name)

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        mlflow.log_params(
            {
                "model_name": model_name,
                "feature_set": feature_set_name,
                "dataset": dataset_name,
                "n_iter": n_iter,
                "random_state": random_state,
                **{f"n_{k}": v for k, v in split_sizes.items()},
                **{f"best_param__{k}": v for k, v in result.best_params.items()},
            }
        )

        for prefix, metrics in [("val", result.val_metrics), ("test", result.test_metrics)]:
            if metrics is None:
                continue
            for field_name in (
                "recall", "precision", "f1", "pr_auc", "roc_auc", "accuracy",
                "naive_baseline_accuracy", "true_positives", "false_negatives",
                "false_positives", "true_negatives",
            ):
                mlflow.log_metric(f"{prefix}_{field_name}", getattr(metrics, field_name))

        # mlflow.sklearn now serializes via skops (safer than pickle by
        # default) with a strict audit that refuses certain internal
        # sklearn types unless explicitly trusted — a real protection
        # against loading a tampered file from an untrusted source.
        # Safe to trust here: this model was just trained in this same
        # process, not loaded from anywhere external. sklearn.tree._tree.Tree
        # covers DecisionTree/RandomForest; the calibration._* types cover
        # LinearSVC wrapped in CalibratedClassifierCV. Harmless no-op for
        # plain LogisticRegression, which contains neither.
        if result.fitted_model is not None:
            mlflow.sklearn.log_model(
                result.fitted_model, name="model",
                skops_trusted_types=[
                    "sklearn.tree._tree.Tree",
                    "sklearn.calibration._CalibratedClassifier",
                    "sklearn.calibration._SigmoidCalibration",
                ],
            )
        if result.fitted_scaler is not None:
            mlflow.sklearn.log_model(result.fitted_scaler, name="scaler")

        return run.info.run_id
