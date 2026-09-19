"""Load cleaned data from the dbt marts.

All cleaning, encoding, and outlier capping happens in dbt (see
dbt/models/). This module's only job is: connect to the DuckDB file,
pull a mart into a pandas DataFrame, and hand back X/y with spkid kept
as a separate identifier column (never as a model feature) so
predictions stay traceable to a specific asteroid.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import NamedTuple

import duckdb
import pandas as pd

from otda.features import CLUSTER_FEATURES, EXP1_FEATURES, EXP2_FEATURES, ID_COL, TARGET_COL

__all__ = [
    "CLUSTER_FEATURES",
    "EXP1_FEATURES",
    "EXP2_FEATURES",
    "ID_COL",
    "TARGET_COL",
    "Dataset",
    "load_class_map",
    "load_modeling_data",
    "load_orbital_elements",
    "load_sample_data",
]

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB = REPO_ROOT / "data" / "dev.duckdb"
CLASS_MAP_SEED = REPO_ROOT / "dbt" / "seeds" / "asteroid_class_map.csv"


class Dataset(NamedTuple):
    """X (features only, no id/target), y (target), and ids (spkid) —
    kept as three aligned objects rather than one DataFrame so it's
    structurally impossible to accidentally feed spkid or is_pha into
    a model as a feature."""

    X: pd.DataFrame
    y: pd.Series
    ids: pd.Series


def load_class_map() -> dict[str, int]:
    """Read the class encoding from the same seed dbt uses — the single
    source of truth. Never hardcode a duplicate copy of this mapping in
    Python; that's exactly the drift risk the seed was introduced to
    eliminate (see dbt/models/staging/stg_asteroids.sql)."""
    with open(CLASS_MAP_SEED, newline="", encoding="utf-8") as f:
        return {row["class_abbr"]: int(row["class_code"]) for row in csv.DictReader(f)}


def _load_mart(db_path: Path, table: str, features: list[str]) -> Dataset:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        cols = ", ".join([ID_COL, TARGET_COL, *features])
        df = con.execute(f"SELECT {cols} FROM main.{table}").df()
    finally:
        con.close()

    ids = df.pop(ID_COL)
    y = df.pop(TARGET_COL).astype(int)
    X = df[features]
    return Dataset(X=X, y=y, ids=ids)


def load_modeling_data(
    features: list[str], db_path: Path = DEFAULT_DB
) -> Dataset:
    """Default modeling population: fct_asteroids_modeling (~932K rows,
    full cleaned population — see project decision on sample vs. full)."""
    return _load_mart(db_path, "fct_asteroids_modeling", features)


def load_sample_data(features: list[str], db_path: Path = DEFAULT_DB) -> Dataset:
    """Legacy-parity path: the 100K stratified sample matching the
    originally published notebook numbers (see fct_asteroids_sample_100k
    for the "same method, not same rows" caveat)."""
    return _load_mart(db_path, "fct_asteroids_sample_100k", features)


def load_orbital_elements(table: str, db_path: Path = DEFAULT_DB) -> pd.DataFrame:
    """Loads CLUSTER_FEATURES plus the ground-truth class label (both
    the string abbreviation and its stable code) from the given mart.
    Unlike load_modeling_data/load_sample_data, this keeps `class` as a
    label to compare cluster assignments against, not a model feature."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        cols = ", ".join([ID_COL, *CLUSTER_FEATURES, "class", "class_code"])
        df = con.execute(f"SELECT {cols} FROM main.{table}").df()
    finally:
        con.close()
    return df
