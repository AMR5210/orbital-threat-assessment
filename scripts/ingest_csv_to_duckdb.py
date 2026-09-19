"""Load the raw NASA JPL asteroid CSV into a local DuckDB file.

Deliberately does not use pandas: DuckDB reads the 435MB / 958,524-row
CSV natively and faster, and this keeps the ingest step free of any
Python-environment dependency beyond `duckdb` itself. All cleaning,
encoding, and feature engineering happens downstream in dbt models —
this script's only job is "get the raw file into a queryable table,
unmodified, with sane types."

Usage:
    python scripts/ingest_csv_to_duckdb.py [--csv PATH] [--db PATH]

Defaults match the dbt DuckDB profile's `path:` (dbt/profiles.yml) and
the source path documented in DatasetLink.txt / legacy/DatasetLink.txt.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "data" / "raw" / "dataset.csv"
DEFAULT_DB = REPO_ROOT / "data" / "dev.duckdb"

RAW_SCHEMA = "raw"
RAW_TABLE = "asteroids"


def ingest(csv_path: Path, db_path: Path, expected_rows: int = 958_524) -> None:
    if not csv_path.exists():
        print(f"ERROR: CSV not found at {csv_path}", file=sys.stderr)
        print(
            "  Copy the NASA JPL Small-Body Database export there first "
            "(see DatasetLink.txt) or pass --csv.",
            file=sys.stderr,
        )
        sys.exit(1)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Ingesting {csv_path} ({csv_path.stat().st_size / 1e6:.1f} MB) -> {db_path}")

    t0 = time.time()
    con = duckdb.connect(str(db_path))
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {RAW_SCHEMA}")

    # read_csv with explicit sample_size=-1 forces a full-file type scan
    # rather than DuckDB's default 20480-row sniff, since a handful of
    # sigma_* / diameter columns are >85% null and a partial sample can
    # misinfer their type as VARCHAR from stray blank fields.
    #
    # strict_mode=false + a generous max_line_size: DuckDB's strict CSV
    # parser trips on this file (a quote character appears mid-field
    # somewhere, not just at field boundaries), ballooning one "line" to
    # several MB before it finds a matching quote. Python's csv module
    # parses the same file cleanly with the standard row/column counts,
    # and the file's total double-quote byte count is even (globally
    # balanced) — this is DuckDB being stricter than RFC 4180 requires,
    # not corrupt data. Row count is verified below regardless.
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {RAW_SCHEMA}.{RAW_TABLE} AS
        SELECT * FROM read_csv(
            '{csv_path.as_posix()}',
            header = true,
            sample_size = -1,
            all_varchar = false,
            strict_mode = false,
            max_line_size = 10000000
        )
        """
    )

    row_count = con.execute(
        f"SELECT COUNT(*) FROM {RAW_SCHEMA}.{RAW_TABLE}"
    ).fetchone()[0]
    col_count = len(con.execute(f"DESCRIBE {RAW_SCHEMA}.{RAW_TABLE}").fetchall())
    con.close()

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s — {row_count:,} rows x {col_count} columns")
    print(f"  table: {RAW_SCHEMA}.{RAW_TABLE}")

    if row_count != expected_rows:
        print(
            f"WARNING: expected {expected_rows:,} rows, got "
            f"{row_count:,}. Verify this is the same dataset export.",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=958_524,
        help="Row count to sanity-check against (e.g. 5004 for the CI fixture).",
    )
    args = parser.parse_args()
    ingest(args.csv, args.db, args.expected_rows)


if __name__ == "__main__":
    main()
