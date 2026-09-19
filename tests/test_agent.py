"""Tests for the Stage 7 agent.

Split into two tiers:
- Tool-logic tests (fetch_asteroid_impl, predict_pha_impl, the @tool
  wrappers, build_agent's construction): pure Python + DuckDB + the
  exported sklearn model, no LLM call, no API key needed. These run
  wherever the DB and exported model exist.
- The live end-to-end test (test_agent_live_ask) actually invokes the
  agent against the real Anthropic API — skipped unless
  ANTHROPIC_API_KEY is set, since it costs money and needs credentials
  neither CI nor a fresh checkout has.

Whole-module import-skip (langchain/langchain-anthropic aren't in the
CI pytest job's install list) plus a data/model-file check — same
pattern as tests/test_api.py.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("langchain")
pytest.importorskip("langchain_anthropic")

from otda.agent import (
    DEFAULT_DB,
    build_agent,
    fetch_asteroid,
    fetch_asteroid_impl,
    predict_pha,
    predict_pha_impl,
)
from otda.inference import DEFAULT_MODEL_PATH

pytestmark = pytest.mark.skipif(
    not (DEFAULT_DB.exists() and DEFAULT_MODEL_PATH.exists()),
    reason=(
        "Needs data/dev.duckdb (dbt build) and models/champion/ "
        "(scripts/export_champion_model.py)"
    ),
)


class TestFetchAsteroidImpl:
    def test_lookup_by_name(self):
        record = fetch_asteroid_impl("Apophis")
        assert record is not None
        assert record["name"] == "Apophis"
        assert record["is_pha"] is True
        assert record["moid"] < 0.05  # famous, genuinely hazardous NEO

    def test_lookup_by_spkid(self):
        record = fetch_asteroid_impl("2101955")  # Bennu's spkid
        assert record is not None
        assert record["name"] == "Bennu"

    def test_lookup_by_designation(self):
        record = fetch_asteroid_impl("99942")  # Apophis's pdes
        assert record is not None
        assert record["name"] == "Apophis"

    def test_lookup_not_found_returns_none(self):
        assert fetch_asteroid_impl("NotARealAsteroidXYZ123") is None

    def test_eros_correctly_labeled_non_pha(self):
        """Eros is a famous, well-studied NEO but NOT a PHA (MOID too
        large) — a real-world sanity check that feature values and
        labels line up correctly end to end."""
        record = fetch_asteroid_impl("Eros")
        assert record is not None
        assert record["is_pha"] is False
        assert record["moid"] > 0.05


def _elasticsearch_reachable() -> bool:
    from otda.search import _get_client

    return _get_client() is not None


class TestFetchAsteroidElasticsearchFallback:
    """otda.search's fuzzy fallback -- only reached once all of
    fetch_asteroid_impl's own exact/substring DuckDB strategies miss.
    Skipped if Elasticsearch isn't running (elasticsearch/docker-compose.yaml
    up + scripts/index_asteroid_names_to_elasticsearch.py run) — same
    no-op-if-unavailable pattern as otda.logging_backend's DynamoDB hook."""

    pytestmark = pytest.mark.skipif(
        not _elasticsearch_reachable(),
        reason="Needs Elasticsearch running + indexed (elasticsearch/docker-compose.yaml)",
    )

    def test_typo_resolves_via_fuzzy_fallback(self):
        record = fetch_asteroid_impl("Apohis")  # missing the second 'p'
        assert record is not None
        assert record["name"] == "Apophis"

    def test_garbage_query_still_returns_none(self):
        """The critical negative case: fuzzy search must not turn a
        nonsense query into a confident wrong answer (see
        otda/search.py's MIN_MATCH_RATIO comment) — guards the fix,
        not just the happy path."""
        assert fetch_asteroid_impl("the quick brown fox jumps") is None


class TestPredictPhaImpl:
    def test_predict_by_identifier_matches_known_pha(self):
        result = predict_pha_impl(identifier="Apophis")
        assert "error" not in result
        assert result["is_pha"] is True
        assert "0.05 AU" in result["explanation"]

    def test_predict_hypothetical_feature_values(self):
        result = predict_pha_impl(
            e=0.3, q=0.9, H=18.0, is_neo=True, class_code=1, moid=0.02
        )
        assert result["is_pha"] is True

    def test_predict_missing_args_returns_error_not_exception(self):
        result = predict_pha_impl()
        assert "error" in result

    def test_predict_unknown_identifier_returns_error(self):
        result = predict_pha_impl(identifier="NotARealAsteroidXYZ123")
        assert "error" in result


class TestToolWrappers:
    """The @tool-decorated versions, invoked the way LangChain's agent
    loop actually calls them (.invoke(dict)), not as plain functions."""

    def test_fetch_asteroid_tool_invoke(self):
        result = fetch_asteroid.invoke({"identifier": "Bennu"})
        assert result["name"] == "Bennu"

    def test_predict_pha_tool_invoke_by_identifier(self):
        result = predict_pha.invoke({"identifier": "Bennu"})
        assert result["is_pha"] is True

    def test_predict_pha_tool_schema_exposes_identifier(self):
        schema = fetch_asteroid.args_schema.model_json_schema()
        assert "identifier" in schema["properties"]


def test_build_agent_constructs_without_api_call():
    """Construction alone (no .invoke()) doesn't require a live API
    key — credential resolution is deferred to the first real call."""
    agent = build_agent()
    assert agent is not None


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="Live agent test needs ANTHROPIC_API_KEY — costs money, not run in CI",
)
def test_agent_live_ask_about_known_pha():
    from otda.agent import ask

    answer = ask("Is Apophis potentially hazardous? Why or why not?")
    assert isinstance(answer, str)
    assert len(answer) > 0
