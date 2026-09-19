"""LangChain create_agent tool-calling agent for asteroid PHA questions.

A real two-tool agent, not an LLM wrapper: fetch_asteroid queries the
DuckDB mart for an asteroid's actual orbital elements; predict_pha runs
the trained model (otda.inference) and returns a rule-grounded
explanation. The system prompt requires the model to call these tools
for any factual claim rather than answer from its own (unreliable,
possibly out of date) astronomy knowledge.

Uses langchain.agents.create_agent (LangChain 1.x) — NOT the deprecated
AgentExecutor/create_react_agent, and not a hand-authored LangGraph
StateGraph, since this is a plain tool-calling loop with no multi-step
retry logic, human-in-the-loop approval, or multi-agent handoff.

Requires ANTHROPIC_API_KEY (or another credential source the Anthropic
SDK resolves — see langchain_anthropic.ChatAnthropic, which wraps the
same `anthropic` Python SDK client) to actually invoke the agent.
fetch_asteroid_impl / predict_pha_impl below have no such dependency
and are directly testable without one. load_dotenv() below picks up
ANTHROPIC_API_KEY from .env for direct CLI use (`python -m otda.agent`)
— pytest's own credential-gated test relies on tests/conftest.py
instead, which does the same thing for the test-collection process.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic

from otda.features import EXP2_FEATURES
from otda.inference import DEFAULT_MODEL_PATH, load_model, predict
from otda.search import search_asteroid_name

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB = REPO_ROOT / "data" / "dev.duckdb"

# Tried in order; first strategy that returns a row wins. Covers: exact
# spkid, exact common name, exact numeric/provisional designation
# (pdes), then a fuzzy substring match on the full designation string.
_LOOKUP_STRATEGIES = [
    "cast(m.spkid as varchar) = ?",
    "lower(n.name) = lower(?)",
    "n.pdes = ?",
    "lower(n.full_name) like '%' || lower(?) || '%'",
]

_RECORD_COLUMNS = [
    "spkid", "full_name", "name", "pdes",
    "is_pha", "e", "q", "H", "is_neo", "class_code", "moid", "a", "i", "per",
]


def _to_native(value):
    """DuckDB/pandas hand back numpy scalar types (np.int64, np.bool_,
    ...); convert to plain Python types so tool results are cleanly
    JSON-serializable when passed back to the LLM."""
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def _query_by_spkid(con, spkid: int) -> dict | None:
    df = con.execute(
        f"""
        select {", ".join("m." + c if c not in ("full_name", "name", "pdes") else "n." + c
                           for c in _RECORD_COLUMNS)}
        from main.fct_asteroids_modeling m
        join main.dim_asteroid_names n on m.spkid = n.spkid
        where m.spkid = ?
        limit 1
        """,
        [spkid],
    ).df()
    if df.empty:
        return None
    row = df.iloc[0]
    return {col: _to_native(row[col]) for col in _RECORD_COLUMNS}


def fetch_asteroid_impl(identifier: str, db_path: Path = DEFAULT_DB) -> dict | None:
    """Looks up one asteroid's record. Tries exact/substring DuckDB
    strategies first; if all of those miss, falls back to Elasticsearch
    fuzzy search (otda.search) to resolve typos/misspellings to a
    spkid, then re-queries DuckDB by that spkid — DuckDB stays the
    source of truth for the returned record either way. Returns None
    if nothing matches anything — callers must handle that, not assume
    a hit."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        for where_clause in _LOOKUP_STRATEGIES:
            df = con.execute(
                f"""
                select {", ".join("m." + c if c not in ("full_name", "name", "pdes") else "n." + c
                                   for c in _RECORD_COLUMNS)}
                from main.fct_asteroids_modeling m
                join main.dim_asteroid_names n on m.spkid = n.spkid
                where {where_clause}
                limit 1
                """,
                [identifier.strip()],
            ).df()
            if not df.empty:
                row = df.iloc[0]
                return {col: _to_native(row[col]) for col in _RECORD_COLUMNS}

        fuzzy_spkid = search_asteroid_name(identifier.strip())
        if fuzzy_spkid is not None:
            return _query_by_spkid(con, fuzzy_spkid)
        return None
    finally:
        con.close()


_model_cache = {}


def _get_model():
    if "model" not in _model_cache:
        _model_cache["model"] = load_model(DEFAULT_MODEL_PATH)
    return _model_cache["model"]


def predict_pha_impl(
    identifier: str | None = None,
    e: float | None = None,
    q: float | None = None,
    H: float | None = None,
    is_neo: bool | None = None,
    class_code: int | None = None,
    moid: float | None = None,
    db_path: Path = DEFAULT_DB,
) -> dict:
    """Provide EITHER identifier (looks up real values from the
    database — the exact-values path, no transcription risk) OR all
    six explicit feature values (for a hypothetical/what-if scenario
    not tied to a real catalogued object)."""
    if identifier:
        record = fetch_asteroid_impl(identifier, db_path)
        if record is None:
            return {"error": f"No asteroid found matching '{identifier}'."}
        features = {k: record[k] for k in EXP2_FEATURES}
    elif None not in (e, q, H, is_neo, class_code, moid):
        features = {
            "e": e, "q": q, "H": H,
            "is_neo": is_neo, "class_code": class_code, "moid": moid,
        }
    else:
        return {
            "error": "Provide either `identifier`, or all six of "
            "e, q, H, is_neo, class_code, moid."
        }

    result = predict(_get_model(), features)
    return {
        "is_pha": result.is_pha,
        "probability": result.probability,
        "explanation": result.explanation,
    }


@tool
def fetch_asteroid(identifier: str) -> dict:
    """Look up an asteroid's real orbital record by name (e.g.
    "Apophis"), designation (e.g. "99942" or "1998 OR2"), or spkid.
    Tolerant of typos/misspellings (falls back to Elasticsearch fuzzy
    search if no exact/substring match is found). Returns its orbital
    elements (e, q, H, is_neo, class_code, moid, a, i, per) and its
    known PHA label from the training data (ground truth, not a model
    prediction — use predict_pha for that). Returns {"error": ...} if
    no match is found — do not guess which asteroid was meant."""
    record = fetch_asteroid_impl(identifier)
    if record is None:
        return {"error": f"No asteroid found matching '{identifier}'."}
    return record


@tool
def predict_pha(
    identifier: str = "",
    e: float = 0.0,
    q: float = 0.0,
    H: float = 0.0,
    is_neo: bool = False,
    class_code: int = 0,
    moid: float = 0.0,
) -> dict:
    """Predict whether an asteroid is Potentially Hazardous (PHA).

    Preferred usage: pass `identifier` (a real asteroid's name/
    designation/spkid) to predict from its actual database values.
    Alternatively, pass all six of e, q, H, is_neo, class_code, moid
    directly for a hypothetical scenario not tied to a real object.
    Returns is_pha, probability, and a rule-grounded explanation
    checking NASA's actual two-condition PHA definition (MOID <= 0.05
    AU and H <= 22.0) against the values used."""
    kwargs = {"identifier": identifier or None}
    if not identifier:
        kwargs.update(e=e, q=q, H=H, is_neo=is_neo, class_code=class_code, moid=moid)
    return predict_pha_impl(**kwargs)


SYSTEM_PROMPT = (
    "You are an assistant for the Orbital Threat Assessment project. "
    "You answer questions about specific asteroids — whether they are "
    "Potentially Hazardous Asteroids (PHAs) and why — using the "
    "fetch_asteroid and predict_pha tools.\n\n"
    "Rules:\n"
    "1. For any factual claim about a specific real asteroid's orbital "
    "elements, PHA status, or prediction, call a tool first. Do not "
    "state a MOID, H, or PHA status from your own general astronomy "
    "knowledge — you do not reliably know a real object's current "
    "catalogued values, and a wrong guess is worse than saying you "
    "don't know.\n"
    "2. When asked whether a specific real asteroid is hazardous, call "
    "fetch_asteroid first to confirm it exists and show its values, "
    "then call predict_pha with that asteroid's identifier (not by "
    "retyping the numbers) so the prediction uses the exact database "
    "values.\n"
    "3. When explaining a result, relay predict_pha's own explanation "
    "field — it already checks NASA's real two-condition PHA rule "
    "against the object's actual MOID and H — rather than inventing "
    "your own reasoning about the numbers.\n"
    "4. If fetch_asteroid returns an error (no match found), say so "
    "plainly rather than guessing which asteroid was meant.\n"
    "5. For an explicitly hypothetical question (\"what if an asteroid "
    "had MOID=0.01 and H=20\"), call predict_pha with the given values "
    "directly — no identifier needed."
)


def build_agent(model_name: str = "claude-sonnet-5"):
    """No `temperature` param: current-generation Claude models (Sonnet
    5 included) reject it outright with HTTP 400 ("`temperature` is
    deprecated for this model"), not just a deprecation warning."""
    model = ChatAnthropic(model=model_name)
    return create_agent(
        model=model,
        tools=[fetch_asteroid, predict_pha],
        system_prompt=SYSTEM_PROMPT,
    )


def ask(question: str, agent=None) -> str:
    if agent is None:
        agent = build_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "Is 433 Eros potentially hazardous?"
    print(f"Q: {q}\n")
    print(f"A: {ask(q)}")
