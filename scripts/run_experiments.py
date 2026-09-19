"""Run every model x feature-set combination and log each to MLflow.

Replaces notebook Cells 15-33: two experiments (orbital-only vs. MOID-
included) x four models, each now split three ways (train/val/test),
hyperparameter-searched against validation, tracked in MLflow instead
of printed and hand-copied.

Usage:
    python scripts/run_experiments.py --dataset full --n-iter 10
    python scripts/run_experiments.py --dataset sample --n-iter 5   # fast smoke test
"""
from __future__ import annotations

import argparse
import time

from otda.data import EXP1_FEATURES, EXP2_FEATURES, load_modeling_data, load_sample_data
from otda.models import ALL_MODELS
from otda.split import three_way_split
from otda.tracking import log_experiment_run
from otda.train import run_experiment

EXPERIMENT_SETS = {
    "exp1_orbital_only": EXP1_FEATURES,
    "exp2_moid_included": EXP2_FEATURES,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["full", "sample"], default="full")
    parser.add_argument("--n-iter", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--mlflow-experiment", default="orbital_threat_assessment",
        help="MLflow experiment name to log all runs under.",
    )
    args = parser.parse_args()

    loader = load_modeling_data if args.dataset == "full" else load_sample_data
    dataset_label = "full_population_932335" if args.dataset == "full" else "sample_100k"

    for feature_set_name, features in EXPERIMENT_SETS.items():
        ds = loader(features)
        split = three_way_split(ds.X, ds.y, random_state=args.random_state)
        split_sizes = {
            "train": len(split.y_train), "val": len(split.y_val), "test": len(split.y_test),
        }
        print(f"\n=== {feature_set_name} ({dataset_label}) ===")
        print(f"  features: {features}")
        print(f"  split sizes: {split_sizes}")

        for model_name in ALL_MODELS:
            t0 = time.time()
            result = run_experiment(
                split, model_name, n_iter=args.n_iter, random_state=args.random_state
            )
            elapsed = time.time() - t0

            run_id = log_experiment_run(
                experiment_name=args.mlflow_experiment,
                run_name=f"{feature_set_name}__{model_name}__{dataset_label}",
                feature_set_name=feature_set_name,
                dataset_name=dataset_label,
                model_name=model_name,
                split_sizes=split_sizes,
                result=result,
                n_iter=args.n_iter,
                random_state=args.random_state,
            )

            print(
                f"  {model_name:22s} val_pr_auc={result.val_metrics.pr_auc:.4f}  "
                f"test_pr_auc={result.test_metrics.pr_auc:.4f}  "
                f"test_recall={result.test_metrics.recall:.3f}  "
                f"test_precision={result.test_metrics.precision:.3f}  "
                f"({elapsed:.1f}s, run_id={run_id[:8]})"
            )


if __name__ == "__main__":
    main()
