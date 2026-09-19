"""Fuzzy asteroid-name search, backed by Elasticsearch.

Scoped narrowly: resolves a misspelled or partial name/designation to
a spkid, over exactly the dim_asteroid_names lookup table (see
scripts/index_asteroid_names_to_elasticsearch.py) -- not a general
search feature over the modeling data. otda.agent.fetch_asteroid_impl
calls this ONLY as a fallback after its own exact/substring DuckDB
strategies all miss; DuckDB stays the source of truth for the actual
returned record.

No-op (returns None) if Elasticsearch isn't reachable -- fuzzy search
is a nice-to-have on top of the agent's exact-match lookups, never a
hard dependency. This mirrors otda.logging_backend's no-op-if-unset
pattern for DynamoDB.
"""
from __future__ import annotations

import os
from difflib import SequenceMatcher

from elasticsearch import Elasticsearch

# Elasticsearch's own relevance score isn't a usable confidence
# threshold here: "the quick brown fox" scored higher (19.4) than a
# genuine typo like "Apohis" (16.2), because multi_match treats each
# query word as an independent fuzzy term and "brown" exactly matches
# an asteroid named Brown. Re-ranking the top candidates by string
# similarity (difflib.SequenceMatcher, stdlib) against the matched
# name/full_name/pdes fixes this: real typos land around 0.55-0.93,
# garbage queries below 0.42.
_CANDIDATE_POOL_SIZE = 5
MIN_MATCH_RATIO = 0.5

# Index name is defined here (the library) and imported by
# scripts/index_asteroid_names_to_elasticsearch.py, not the other way
# around -- scripts/ isn't guaranteed importable everywhere otda is
# (e.g. inside the FastAPI/agent Docker images), so the dependency has
# to point from script to library, like every other script in this repo.
INDEX_NAME = "asteroid_names"

_client_cache: dict = {}


def _get_client() -> Elasticsearch | None:
    if "client" not in _client_cache:
        es_url = os.environ.get("OTDA_ELASTICSEARCH_URL", "http://localhost:9200")
        try:
            client = Elasticsearch(es_url, request_timeout=2)
            if not client.ping():
                _client_cache["client"] = None
            else:
                _client_cache["client"] = client
        except Exception:
            _client_cache["client"] = None
    return _client_cache["client"]


def _best_similarity(query: str, source: dict) -> float:
    """Highest string-similarity ratio between the raw query and any of
    the candidate's name/full_name/pdes fields."""
    query = query.strip().lower()
    fields = (source.get("name"), source.get("full_name"), source.get("pdes"))
    return max(
        (SequenceMatcher(None, query, f.strip().lower()).ratio() for f in fields if f),
        default=0.0,
    )


def search_asteroid_name(query: str) -> int | None:
    """Fuzzy/full-text match against name, full_name, and pdes, then
    re-ranked by string similarity against the raw query (see
    MIN_MATCH_RATIO's comment for why ES's own relevance score isn't
    used directly). Returns the best-matching spkid, or None if
    Elasticsearch is unreachable or nothing clears the similarity
    floor."""
    client = _get_client()
    if client is None:
        return None

    try:
        resp = client.search(
            index=INDEX_NAME,
            size=_CANDIDATE_POOL_SIZE,
            query={
                "multi_match": {
                    "query": query,
                    "fields": ["name^2", "full_name", "pdes"],
                    "fuzziness": "AUTO",
                }
            },
        )
    except Exception:
        return None

    hits = resp["hits"]["hits"]
    if not hits:
        return None

    best_hit = max(hits, key=lambda h: _best_similarity(query, h["_source"]))
    if _best_similarity(query, best_hit["_source"]) < MIN_MATCH_RATIO:
        return None
    return int(best_hit["_source"]["spkid"])
