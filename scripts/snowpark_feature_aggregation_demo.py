"""Snowpark demo: one narrow server-side aggregation, never the
training path.

Per the project's Snowpark scoping decision: training stays local
sklearn always — the risk of pulling scikit-learn/imbalanced-learn
into Snowflake's Anaconda-channel-constrained runtime isn't worth it
for a model this cheap to train locally. What Snowpark IS a good fit
for: pushing an aggregation down to warehouse compute and only pulling
back the small aggregated result, instead of dragging 932K rows to the
client to aggregate in pandas. That's what this script demonstrates —
class-level summary stats (count, mean MOID, mean H, PHA count) over
fct_asteroids_modeling, computed entirely in Snowflake.

Assumes dbt's `prod` target has already been run (`dbt build -t prod`)
so fct_asteroids_modeling exists in Snowflake.

Key-pair auth: Session.builder.configs() is a thin pass-through to
snowflake.connector.connect(**options) (confirmed by reading
snowpark's server_connection.py directly), so it accepts the exact
same private_key_file/private_key_file_pwd parameters as the raw
connector — not a differently-named Snowpark-specific config key.

Usage (after terraform apply + dbt build -t prod):
    python scripts/snowpark_feature_aggregation_demo.py
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from snowflake.snowpark import Session
from snowflake.snowpark import functions as F

load_dotenv()


def _connection_params() -> dict:
    """Same SNOWFLAKE_* env vars as dbt/profiles.yml's prod target —
    this demo reads what dbt already built, so it authenticates the
    same way dbt does."""
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


def run_class_aggregation_demo() -> None:
    session = Session.builder.configs(_connection_params()).create()
    try:
        df = session.table("FCT_ASTEROIDS_MODELING")

        agg = (
            df.group_by("CLASS_CODE")
            .agg(
                F.count("*").alias("N"),
                F.avg("MOID").alias("AVG_MOID"),
                F.avg("H").alias("AVG_H"),
                F.sum(F.iff(F.col("IS_PHA"), 1, 0)).alias("N_PHA"),
            )
            .sort("CLASS_CODE")
        )

        # .collect() is the only point data leaves Snowflake's compute:
        # 13 aggregated rows come back, not the underlying ~932K rows
        # that produced them.
        rows = agg.collect()

        print(f"{'class_code':>10} {'n':>10} {'avg_moid':>10} {'avg_H':>8} {'n_pha':>8}")
        for r in rows:
            print(
                f"{r['CLASS_CODE']:>10} {r['N']:>10} "
                f"{r['AVG_MOID']:>10.4f} {r['AVG_H']:>8.2f} {r['N_PHA']:>8}"
            )
    finally:
        session.close()


if __name__ == "__main__":
    run_class_aggregation_demo()
