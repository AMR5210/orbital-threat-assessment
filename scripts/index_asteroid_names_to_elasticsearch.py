"""Index dim_asteroid_names into Elasticsearch for fuzzy name lookup.

Scoped narrowly: this indexes exactly the name-lookup mart
(dim_asteroid_names -- spkid/full_name/name/pdes), not the modeling
data. Elasticsearch here is a search-assistant that resolves a
misspelled or partial identifier to a spkid; otda.agent.fetch_asteroid_impl
still does the actual record lookup against DuckDB afterward, so
DuckDB remains the single source of truth for asteroid data.

Assumes elasticsearch/docker-compose.yaml is up (`docker compose up -d`
from that directory) and the dev DuckDB file already has
dim_asteroid_names built (`dbt build`).

Usage:
    python scripts/index_asteroid_names_to_elasticsearch.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
from elasticsearch.helpers import bulk

from elasticsearch import Elasticsearch
from otda.search import INDEX_NAME

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO_ROOT / "data" / "dev.duckdb"

# text fields for fuzzy/full-text matching, keyword for exact spkid lookup.
_MAPPINGS = {
    "properties": {
        "spkid": {"type": "long"},
        "full_name": {"type": "text"},
        "name": {"type": "text"},
        "pdes": {"type": "text"},
    }
}


def index_names(db_path: Path = DEFAULT_DB, es_url: str = "http://localhost:9200") -> int:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute(
            "select spkid, full_name, name, pdes from main.dim_asteroid_names"
        ).df()
    finally:
        con.close()

    es = Elasticsearch(es_url)

    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)
    es.indices.create(index=INDEX_NAME, mappings=_MAPPINGS)

    actions = (
        {
            "_index": INDEX_NAME,
            "_id": int(row.spkid),
            "_source": {
                "spkid": int(row.spkid),
                "full_name": row.full_name,
                "name": row.name,
                "pdes": row.pdes,
            },
        }
        for row in df.itertuples()
    )
    success, _errors = bulk(es, actions)
    es.indices.refresh(index=INDEX_NAME)
    return success


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--es-url", default="http://localhost:9200")
    args = parser.parse_args()

    count = index_names(args.db, args.es_url)
    print(f"Indexed {count:,} asteroid names into '{INDEX_NAME}'.")


if __name__ == "__main__":
    main()
