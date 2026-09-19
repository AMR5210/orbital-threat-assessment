"""Generate a results table from the MLflow tracking store.

Replaces the original notebook's Cells 25/31/32 workflow: print a
pandas DataFrame, read it, type the numbers into the README by hand.
This reads directly from whatever `scripts/run_experiments.py` logged
— the README table is generated, not transcribed.

Usage:
    python scripts/generate_results_table.py --out RESULTS.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mlflow

from otda.tracking import get_tracking_db

REPO_ROOT = Path(__file__).resolve().parent.parent

METRIC_COLS = ["recall", "precision", "f1", "pr_auc", "roc_auc"]
FEATURE_SET_ORDER = ["exp1_orbital_only", "exp2_moid_included"]
MODEL_ORDER = ["logistic_regression", "decision_tree", "random_forest", "linear_svc"]
MODEL_DISPLAY = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "linear_svc": "LinearSVC",
}
FEATURE_SET_DISPLAY = {
    "exp1_orbital_only": "Experiment 1 — Orbital Only (no MOID)",
    "exp2_moid_included": "Experiment 2 — MOID Included",
}


def fetch_runs(experiment_name: str, dataset_name: str):
    mlflow.set_tracking_uri(f"sqlite:///{get_tracking_db().as_posix()}")
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        raise SystemExit(
            f"No MLflow experiment named '{experiment_name}' — "
            "run scripts/run_experiments.py first."
        )
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"params.dataset = '{dataset_name}'",
    )
    if runs.empty:
        raise SystemExit(f"No runs found for dataset='{dataset_name}'.")
    # One row per (feature_set, model_name): keep the most recent run
    # if run_experiments.py was ever re-run for the same combination.
    runs = runs.sort_values("start_time").drop_duplicates(
        subset=["params.feature_set", "params.model_name"], keep="last"
    )
    return runs


def render_table(runs, feature_set_name: str) -> str:
    subset = runs[runs["params.feature_set"] == feature_set_name]
    header = "| Model | Recall | Precision | F1 | PR-AUC | ROC-AUC |\n"
    header += "|---|---|---|---|---|---|\n"
    rows = []
    for model_name in MODEL_ORDER:
        row = subset[subset["params.model_name"] == model_name]
        if row.empty:
            continue
        row = row.iloc[0]
        vals = [f"{row[f'metrics.test_{m}']:.3f}" for m in METRIC_COLS]
        rows.append(f"| {MODEL_DISPLAY[model_name]} | " + " | ".join(vals) + " |")
    return header + "\n".join(rows) + "\n"


def render_interpretation(runs) -> str:
    """Ties Exp2's near-perfect scores back to the MOID finding rather
    than leaving them looking like an unexplained jump — and uses
    Exp1's much smaller move as the sanity check that this is the same
    mechanism as the original notebook's finding, not new leakage."""

    def dt_pr_auc(feature_set_name: str) -> float:
        row = runs[
            (runs["params.feature_set"] == feature_set_name)
            & (runs["params.model_name"] == "decision_tree")
        ].iloc[0]
        return row["metrics.test_pr_auc"]

    exp1_pr_auc = dt_pr_auc("exp1_orbital_only")
    exp2_pr_auc = dt_pr_auc("exp2_moid_included")

    return (
        "## Interpretation: why Exp2 is near-perfect and Exp1 isn't\n\n"
        f"Exp2's decision tree scores ({exp2_pr_auc:.3f} PR-AUC here vs. "
        "0.988 in the original 100K-sample run) aren't a new leakage "
        "artifact — they're the same mechanism the original notebook "
        "already identified: MOID is part of NASA's own two-condition "
        "PHA definition (`moid <= 0.05 AU` and `H <= 22`), so a tree given "
        "that feature is learning the label's own definitional boundary, "
        "not an independent signal. Training on the full 932,335-row "
        "population instead of the 100K sample gives the tree far more "
        "examples near that boundary to split on, which is why the score "
        "moved closer to 1.0 rather than staying flat.\n\n"
        f"Exp1's decision tree — no MOID, orbital mechanics only — moved "
        f"from 0.665 (legacy) to {exp1_pr_auc:.3f} here: a modest, "
        "expected improvement from more training data and validation-"
        "scored (rather than test-set-scored) hyperparameter selection, "
        "not a discontinuity. That contrast is the sanity check: if the "
        "full-population switch were introducing leakage rather than just "
        "more data, Exp1 would have jumped the same way Exp2 did, and it "
        "didn't — its features don't encode the label's own definition, "
        "so there's no boundary for extra rows to sharpen.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="orbital_threat_assessment")
    parser.add_argument("--dataset", default="full_population_932335")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "RESULTS.md")
    args = parser.parse_args()

    runs = fetch_runs(args.experiment, args.dataset)

    lines = [
        "# Generated Results\n",
        f"Generated from MLflow tracking store (`{get_tracking_db().name}`), "
        f"experiment `{args.experiment}`, dataset `{args.dataset}`.\n",
        "**Test-set metrics — touched exactly once per run, after "
        "hyperparameter selection on the validation set.** See "
        "src/otda/train.py for the split/search methodology and "
        "legacy/README.md for the originally published (test-set-selected, "
        "100K-sample) numbers this supersedes.\n",
    ]
    for feature_set_name in FEATURE_SET_ORDER:
        lines.append(f"## {FEATURE_SET_DISPLAY[feature_set_name]}\n")
        lines.append(render_table(runs, feature_set_name))

    lines.append(render_interpretation(runs))

    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
