"""Orchestrates the full local pipeline: ingest -> dbt build -> train ->
evaluate -> register (export).

schedule=None (manually triggered), deliberately — the source data is a
static one-time Kaggle/JPL snapshot that never changes, so there is
nothing to run this "on a schedule" against; a schedule would be
orchestration theater, not a real requirement. See ../README.md for
the division of labor with CI: GitHub Actions runs tests on every
push; this DAG runs the full data-to-model refresh on demand.

Targets the DuckDB dev dbt target only, matching ../docker-compose.yaml's
scope — this is the one late-stage piece of the project that needs no
cloud credentials to actually execute, so unlike Stages 7-9 this DAG
has been run for real (see the repo README's verification note).

Uses airflow.providers.standard.operators.bash.BashOperator — the
current Airflow 3.x import path (BashOperator moved out of airflow-core
into the standard provider package; the old airflow.operators.bash
path is deprecated).
"""
from __future__ import annotations

import datetime

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, Param

PROJECT_DIR = "/opt/airflow/project"

with DAG(
    dag_id="otda_pipeline",
    description="Ingest -> dbt build -> train -> evaluate -> export (DuckDB dev target)",
    schedule=None,
    start_date=datetime.datetime(2026, 1, 1),
    catchup=False,
    tags=["otda", "stage10"],
    params={
        "dataset": Param("full", enum=["full", "sample"], description="Modeling population for run_experiments.py"),
        "n_iter": Param(8, type="integer", minimum=1, description="Hyperparameter search budget per model"),
    },
) as dag:
    ingest = BashOperator(
        task_id="ingest",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "python scripts/ingest_csv_to_duckdb.py --csv data/raw/dataset.csv"
        ),
    )

    dbt_build = BashOperator(
        task_id="dbt_build",
        bash_command=f"cd {PROJECT_DIR}/dbt && dbt build --profiles-dir . --project-dir .",
    )

    train = BashOperator(
        task_id="train",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "python scripts/run_experiments.py "
            "--dataset {{ params.dataset }} --n-iter {{ params.n_iter }}"
        ),
    )

    evaluate = BashOperator(
        task_id="evaluate",
        bash_command=f"cd {PROJECT_DIR} && python scripts/generate_results_table.py",
    )

    register = BashOperator(
        task_id="register",
        bash_command=f"cd {PROJECT_DIR} && python scripts/export_champion_model.py",
    )

    ingest >> dbt_build >> train >> evaluate >> register
