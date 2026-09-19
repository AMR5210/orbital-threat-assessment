"""Load the raw asteroid CSV into Snowflake's RAW schema.

Snowflake counterpart to scripts/ingest_csv_to_duckdb.py — same source
file, same target shape (a raw.asteroids table dbt's sources.yml can
point at), different warehouse. Uses INFER_SCHEMA + CREATE TABLE
USING TEMPLATE (Snowflake's schema-inference path — the same "infer
types from the file" approach as DuckDB's read_csv, for behavioral
parity between the dev and prod targets) rather than hand-writing 45
column definitions to keep in sync by hand.

Assumes terraform/main.tf has already been applied. SQL syntax matches
Snowflake's own INFER_SCHEMA / CREATE TABLE USING TEMPLATE
documentation.

Key-pair auth (private_key_file/private_key_file_pwd), not password —
these are the connector's actual accepted parameter names, per
snowflake.connector.connection's DEFAULT_CONFIGURATION (dbt's
differently-named private_key_path is dbt's own convention, not the
connector's).

The mid-field quote character that broke DuckDB's default strict CSV
parser (scripts/ingest_csv_to_duckdb.py, fixed there with
strict_mode=false) does not affect Snowflake's COPY INTO — this file
loads cleanly with the FILE FORMAT below (958,524 rows, exact match to
DatasetLink.txt).

Usage:
    python scripts/load_raw_to_snowflake.py --csv path/to/dataset.csv
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import snowflake.connector
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "data" / "raw" / "dataset.csv"

STAGE_NAME = "asteroid_stage"
FILE_FORMAT_NAME = "asteroid_csv_format"
TABLE_NAME = "ASTEROIDS"


def _connect():
    """Same SNOWFLAKE_* env vars as dbt/profiles.yml's prod target,
    plus SNOWFLAKE_RAW_SCHEMA (distinct from dbt's SNOWFLAKE_SCHEMA,
    which points at the ANALYTICS/marts schema, not RAW) —
    matches terraform/variables.tf's raw_schema_name default."""
    return snowflake.connector.connect(
        account=os.environ.get("SNOWFLAKE_ACCOUNT", "myorg-myaccount"),
        user=os.environ.get("SNOWFLAKE_USER", "your_snowflake_user"),
        private_key_file=os.environ.get(
            "SNOWFLAKE_PRIVATE_KEY_PATH", "~/.snowflake/keys/rsa_key.p8"
        ),
        private_key_file_pwd=os.environ.get("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE") or None,
        role=os.environ.get("SNOWFLAKE_ROLE", "TRANSFORMER"),
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "OTDA_WH"),
        database=os.environ.get("SNOWFLAKE_DATABASE", "ORBITAL_THREAT_ASSESSMENT"),
        schema=os.environ.get("SNOWFLAKE_RAW_SCHEMA", "RAW"),
    )


def load(csv_path: Path) -> None:
    if not csv_path.exists():
        raise SystemExit(f"CSV not found at {csv_path} (see DatasetLink.txt)")

    con = _connect()
    cur = con.cursor()
    try:
        cur.execute(
            f"""
            CREATE FILE FORMAT IF NOT EXISTS {FILE_FORMAT_NAME}
                TYPE = CSV
                PARSE_HEADER = TRUE
                FIELD_OPTIONALLY_ENCLOSED_BY = '"'
                COMPRESSION = AUTO
            """
        )
        cur.execute(f"CREATE STAGE IF NOT EXISTS {STAGE_NAME}")

        # PUT wants forward slashes and a file:// URI, even on Windows.
        local_uri = "file://" + csv_path.as_posix()
        cur.execute(f"PUT '{local_uri}' @{STAGE_NAME} AUTO_COMPRESS=TRUE OVERWRITE=TRUE")

        # PARSE_HEADER=TRUE makes INFER_SCHEMA preserve the CSV header's
        # exact casing (mostly lowercase, e.g. "spkid") as a QUOTED,
        # case-sensitive identifier. dbt's sources.yml declares columns
        # like `spkid` unquoted, which Snowflake folds to `SPKID` --
        # a mismatch against the literal quoted-lowercase column, which
        # surfaced as "invalid identifier 'SPKID'" the first time
        # `dbt build -t prod` ran (2026-09-18). Forcing COLUMN_NAME to
        # UPPER() here makes the created table use Snowflake's native
        # unquoted convention, matching what dbt's unquoted source
        # references resolve to.
        #
        # Separately, INFER_SCHEMA also guessed BOOLEAN for the `pha`
        # and `neo` columns, since they only ever contain 'Y'/'N'/empty
        # -- Snowflake's implicit string->boolean conversion recognizes
        # single-letter Y/N. DuckDB's read_csv makes no such guess and
        # keeps them VARCHAR. stg_asteroids.sql's `pha != ''` filter
        # (needed to drop missing-target rows) then fails there with
        # "Boolean value '' is not recognized", since '' isn't a valid
        # boolean literal. Forcing these two back to VARCHAR keeps the
        # raw layer's types identical across both warehouse targets, so
        # the same dbt SQL works unmodified against either.
        cur.execute(
            f"""
            CREATE OR REPLACE TABLE {TABLE_NAME} USING TEMPLATE (
                SELECT ARRAY_AGG(
                    OBJECT_INSERT(
                        OBJECT_INSERT(OBJECT_CONSTRUCT(*), 'COLUMN_NAME', UPPER(COLUMN_NAME), TRUE),
                        'TYPE',
                        IFF(UPPER(COLUMN_NAME) IN ('PHA', 'NEO'), 'VARCHAR', TYPE),
                        TRUE
                    )
                )
                FROM TABLE (
                    INFER_SCHEMA(
                        LOCATION => '@{STAGE_NAME}',
                        FILE_FORMAT => '{FILE_FORMAT_NAME}'
                    )
                )
            )
            """
        )

        cur.execute(
            f"""
            COPY INTO {TABLE_NAME}
            FROM @{STAGE_NAME}
            FILE_FORMAT = (FORMAT_NAME = '{FILE_FORMAT_NAME}')
            MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE
            """
        )

        row_count = cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
        print(f"Loaded {row_count:,} rows into RAW.{TABLE_NAME}")
        if row_count != 958_524:
            print(
                f"WARNING: expected 958,524 rows (per DatasetLink.txt), got {row_count:,}."
            )
    finally:
        cur.close()
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()
    load(args.csv)


if __name__ == "__main__":
    main()
