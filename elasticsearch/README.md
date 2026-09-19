# Elasticsearch — fuzzy asteroid-name lookup

Scoped to exactly one real use case: the agent's `fetch_asteroid` tool
resolving a misspelled or garbled asteroid name/designation when exact
matching fails. This is not a general search feature over the dataset
— it indexes only `dim_asteroid_names` (spkid/full_name/name/pdes),
nothing else.

## Status

**Verified working, 2026-09-18** — real container, real 932,335-row
index, real fallback resolution through `otda.agent.fetch_asteroid_impl`.

## Why this exists (and what it doesn't fix)

`otda.agent.fetch_asteroid_impl` already resolves exact spkid/name/
designation matches, and substring matches against `full_name` — which
turns out to already cover alternate designations like `"2004 MN4"`
(Apophis's old provisional designation, which appears verbatim inside
its `full_name`). The genuine gap is real typos (`"Apohis"`,
`"Apophiss"`), which no exact/substring strategy can catch.

**A real bug, fixed:** Elasticsearch's own relevance score is not a
safe confidence threshold for "is this a good match." `"the quick
brown fox"` scored *higher* (19.4) than the genuine typo `"Apohis"`
(16.2), because `multi_match` treats each query word as an independent
fuzzy term, and "brown" happens to exactly match a real asteroid
literally named Brown. Fixed by re-ranking the top-5 Elasticsearch
candidates using `difflib.SequenceMatcher` (stdlib, no new dependency)
against the raw query string, rejecting anything below
`MIN_MATCH_RATIO = 0.5` (real typos land at 0.55–0.93, garbage queries
at 0.0–0.42). See `src/otda/search.py`'s module docstring for the full
trail.

**Known, accepted limitation:** fuzzy matching over pure numeric/
designation-style strings (e.g. a bare `"2004 MN4"` with no name) is
inherently ambiguous — many objects share similarly-shaped provisional
designations across different years. This doesn't matter in practice
because the exact-substring strategy already resolves the real case
before ever reaching Elasticsearch.

## Usage

```bash
cd elasticsearch
docker compose up -d
cd ..

python scripts/index_asteroid_names_to_elasticsearch.py
```

No further wiring needed — `otda.agent.fetch_asteroid_impl` calls
`otda.search.search_asteroid_name` automatically as a fallback, and the
whole thing no-ops (returns `None`, same as "not found") if
Elasticsearch isn't running. Point at a non-default host via
`OTDA_ELASTICSEARCH_URL` (defaults to `http://localhost:9200`).

```bash
python -m otda.agent "Is Apohis dangerous?"   # typo, resolves via fallback
```

## Security note

`xpack.security.enabled=false` — no auth, no TLS. Fine for a local,
single-user dev index with no sensitive data (this is a name-lookup
table, not the modeling data); never expose port 9200 outside
localhost.
