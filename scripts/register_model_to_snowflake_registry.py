"""Register the champion model in Snowflake's native Model Registry.

Per the project's registry decision: MLflow is the single artifact
SOURCE for both serving paths (FastAPI loads a local export;
SageMaker deploys straight from `runs:/<id>/model` — see
scripts/deploy_to_sagemaker.py). This script registers the SAME
already-trained model into Snowflake's registry too, but nothing reads
from it — it is a documented, parallel artifact, not a load-bearing
part of any serving path. Framing this any other way ("Snowflake
serves the model") would be inaccurate; the honest line is "registered
in both, deployed from MLflow."

snowflake.ml.registry.Registry's API (Registry(session,
database_name=, schema_name=).log_model(...)) verified against
snowflake-ml-python 2.1.0 via `inspect.signature`.

Key-pair auth: Session.builder.configs() passes straight through to
snowflake.connector.connect(**options), so private_key_file/
private_key_file_pwd work identically to the raw connector (per
snowpark's server_connection.py).

Usage (after terraform apply + the champion model is exported):
    python scripts/register_model_to_snowflake_registry.py
"""
from __future__ import annotations

import os

import pandas as pd
from dotenv import load_dotenv
from snowflake.ml.registry import Registry
from snowflake.snowpark import Session

from otda.features import EXP2_FEATURES
from otda.inference import load_model

load_dotenv()


def _connection_params() -> dict:
    """Same SNOWFLAKE_* env vars as dbt/profiles.yml's prod target."""
    return {
        "account": os.environ.get("SNOWFLAKE_ACCOUNT", "myorg-myaccount"),
        "user": os.environ.get("SNOWFLAKE_USER", "your_snowflake_user"),
        "private_key_file": os.environ.get(
            "SNOWFLAKE_PRIVATE_KEY_PATH", "~/.snowflake/keys/rsa_key.p8"
        ),
        "private_key_file_pwd": os.environ.get("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE") or None,
        "role": os.environ.get("SNOWFLAKE_ROLE", "TRANSFORMER"),
        "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE", "OTDA_WH"),
        "database": os.environ.get("SNOWFLAKE_DATABASE", "ORBITAL_THREAT_ASSESSMENT"),
        "schema": os.environ.get("SNOWFLAKE_SCHEMA", "ANALYTICS"),
    }


def register() -> None:
    model = load_model()
    session = Session.builder.configs(_connection_params()).create()
    try:
        registry = Registry(
            session,
            database_name=os.environ.get("SNOWFLAKE_DATABASE", "ORBITAL_THREAT_ASSESSMENT"),
            schema_name=os.environ.get("SNOWFLAKE_SCHEMA", "ANALYTICS"),
        )

        # A few representative rows (not the champion's real training
        # data) purely so the registry can infer an input/output
        # signature — this is metadata for the registry entry, not
        # anything served.
        sample_input = [
            {"e": 0.19, "q": 0.75, "H": 19.7, "is_neo": True, "class_code": 3, "moid": 0.0003},
            {"e": 0.22, "q": 1.13, "H": 10.4, "is_neo": True, "class_code": 0, "moid": 0.15},
        ]
        sample_df = pd.DataFrame(sample_input)[EXP2_FEATURES]

        model_version = registry.log_model(
            model,
            model_name="OTDA_PHA_CLASSIFIER",
            version_name="V1",
            comment=(
                "Decision Tree, Experiment 2 (MOID-included) — reconstructs "
                "NASA's own two-condition PHA rule. Registered here as a "
                "parallel, documented artifact; MLflow (not this registry) "
                "is the source both FastAPI and SageMaker actually serve "
                "from. See RESULTS.md for test-set metrics (PR-AUC 0.999)."
            ),
            sample_input_data=sample_df,
        )
        print(f"Registered: {model_version.model_name} version {model_version.version_name}")
    finally:
        session.close()


if __name__ == "__main__":
    register()
