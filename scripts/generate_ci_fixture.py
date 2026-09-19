"""Generate the small CI fixture from the real ingested data.

Not synthetic: samples real rows out of the already-ingested
raw.asteroids table (data/dev.duckdb), so the fixture reflects real
distributions rather than fabricated ones. Deterministic (fixed seed
via hash() ordering) so re-running this script reproduces the same
fixture byte-for-byte.

The fixture deliberately includes the 4 real HYA (Hyperbolic Asteroid)
rows with a null `neo` flag — see dbt/models/staging/stg_asteroids.sql
— so `dbt build` in CI exercises the neo-null-drop fix on every run,
not just when someone happens to run it against the full dataset.

Usage:
    python scripts/generate_ci_fixture.py
"""
from __future__ import annotations

from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DB = REPO_ROOT / "data" / "dev.duckdb"
FIXTURE_CSV = REPO_ROOT / "tests" / "fixtures" / "asteroids_fixture.csv"

FIXTURE_SIZE = 5000
# Real population ratio (2066 / 932335 = 0.0022167) applied to
# FIXTURE_SIZE, rounded — see dbt/tests/assert_pha_ratio_population.sql
# for the bounds this needs to fall inside after cleaning.
N_PHA = round(FIXTURE_SIZE * 2066 / 932335)  # == 11


def main() -> None:
    if not SOURCE_DB.exists():
        raise SystemExit(
            f"{SOURCE_DB} not found — run scripts/ingest_csv_to_duckdb.py first."
        )

    con = duckdb.connect(str(SOURCE_DB), read_only=True)

    # Deterministic stratified sample from the RAW table (pre-cleaning),
    # restricted to rows that pass the pha/neo not-null checks so the
    # fixture's positive count is controllable and predictable post-
    # cleaning. hash()-based ordering, not random(), for reproducibility.
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE fixture_rows AS
        WITH ranked AS (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY (pha = 'Y')
                    ORDER BY hash(spkid || '_otda_fixture_seed')
                ) AS rn
            FROM raw.asteroids
            WHERE pha IS NOT NULL AND pha != ''
              AND neo IS NOT NULL AND neo != ''
        )
        SELECT * EXCLUDE (rn) FROM ranked
        WHERE
            ((pha = 'Y') AND rn <= {N_PHA})
            OR ((pha != 'Y') AND rn <= {FIXTURE_SIZE - N_PHA})
        """
    )

    # Inject the 4 real HYA / neo-null edge-case rows (pha not null,
    # neo IS null) so CI exercises the corrected drop-on-null-neo path
    # every run. These do not affect the ratio bounds below — they are
    # dropped by stg_asteroids before the ratio is computed.
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE fixture_with_edge_cases AS
        SELECT * FROM fixture_rows
        UNION ALL
        SELECT * FROM raw.asteroids
        WHERE class = 'HYA' AND pha IS NOT NULL AND (neo IS NULL OR neo = '')
        """
    )

    n_rows = con.execute("SELECT COUNT(*) FROM fixture_with_edge_cases").fetchone()[0]
    n_pha = con.execute(
        "SELECT COUNT(*) FROM fixture_with_edge_cases WHERE pha = 'Y'"
    ).fetchone()[0]

    FIXTURE_CSV.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"""
        COPY fixture_with_edge_cases TO '{FIXTURE_CSV.as_posix()}'
        (HEADER, DELIMITER ',')
        """
    )
    con.close()

    print(f"Wrote {n_rows:,} rows ({n_pha} PHA) -> {FIXTURE_CSV}")
    print(
        f"  post-cleaning PHA ratio target: {n_pha}/{n_rows - 4} "
        f"= {n_pha / (n_rows - 4):.4f} (4 HYA edge-case rows excluded by cleaning)"
    )


if __name__ == "__main__":
    main()
