# Extends the official Airflow image with this project's core (Stage
# 0-4) dependencies. Deliberately does NOT COPY or pip-install the
# `otda` package or project source here — docker-compose.yaml
# bind-mounts the live repo instead (see PYTHONPATH there), so a code
# change doesn't require rebuilding this image.
#
# Only the Stage 0-2/4 dependency group is installed (ingest -> dbt
# build -> train -> export), matching this DAG's scope: DuckDB dev
# target only, no Snowflake/AWS/LangChain packages, since this stage
# is explicitly the one that needs no cloud credentials to run.
FROM apache/airflow:3.3.2-python3.13

USER airflow
RUN pip install --no-cache-dir \
    duckdb==1.5.5 \
    dbt-core==1.12.5 \
    dbt-duckdb==1.11.0 \
    pandas==3.0.5 \
    scikit-learn==1.9.1 \
    imbalanced-learn==0.14.2 \
    numpy==2.5.3 \
    scipy==1.18.1 \
    mlflow==3.16.1
