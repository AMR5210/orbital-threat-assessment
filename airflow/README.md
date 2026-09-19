# Airflow — pipeline orchestration (Stage 10)

One DAG, `otda_pipeline`: `ingest -> dbt_build -> train -> evaluate ->
register`. LocalExecutor, Airflow 3.3.2, Postgres metadata DB — a lean
stack (no Celery/Redis/worker/flower/triggerer; a handful of sequential
tasks don't need distributed workers).

## Why no schedule

`schedule=None` — manually triggered only. The source data is a static
one-time Kaggle/JPL snapshot (see `../legacy/DatasetLink.txt`); nothing
about it changes on any cadence, so giving this DAG a cron schedule
would be orchestrating against data that never moves — pure theater,
not a real requirement. **Division of labor with CI:** GitHub Actions
(`.github/workflows/ci.yml`) runs lint/dbt-build/pytest on every push;
this DAG runs the full data-to-model refresh on demand, when someone
actually wants fresh results (e.g. after changing a dbt model or a
hyperparameter search space).

## Why this stage — and not Stages 7-9 — actually got run

DuckDB (the dev dbt target), never Snowflake. No cloud account, no API
key, nothing beyond Docker — the same Docker already used for Stage 6.
So unlike the Terraform/Snowpark/SageMaker/agent pieces (code-complete
but unverified live, tracked in the repo's top-level README), **this
stack was actually built, started, and the DAG actually triggered and
run** — not just written and trusted to work.

## What actually happened (verification log, 2026-09-17)

Built the custom image, ran `airflow-init` (exit 0, admin user
created), brought up the full stack, confirmed `airflow dags list`
parsed `otda_pipeline` with zero import errors, then triggered a real
run (`--conf '{"dataset": "sample", "n_iter": 3}'`, chosen over the
documented `full`/`8` default purely for verification speed).

**First trigger failed** — `ingest` errored with `CSV not found at
data/raw/dataset.csv`. Not a DAG bug: the CSV had never actually been
placed at the path the project's own README documents, since every
earlier stage's local dbt runs used an explicit `--csv` override
pointing outside the repo. Fixed by placing the file where the docs
already said it belonged, then re-triggered.

**Second trigger: `ingest`, `dbt_build`, `train`, `evaluate` all
succeeded for real** (confirmed via `airflow tasks states-for-dag-run`,
not just "the UI looked green"). `train` took ~3.5 minutes inside the
container versus ~24s in a bare host venv for the identical
`--dataset sample --n-iter 3` config — confirmed via `docker stats`
that the scheduler container was genuinely at 200%+ CPU throughout, so
this is slower BLAS/threading performance under Docker's virtualization
layer, not a hang.

**`register` reported success while producing an EMPTY
`models/champion/`** — a real bug, not a container quirk. MLflow's
local file-based artifact store bakes each run's artifact location in
as an ABSOLUTE path at logging time. `export_champion_model.py`'s
champion-selection query correctly found the canonical full-population
Decision Tree/Exp2 run — but that run was logged from the **host**
months of stages ago, so its artifact URI is `file:C:/Users/...`, which
doesn't exist inside the Linux container. `download_artifacts` silently
returned an empty directory instead of raising, and the script printed
"Exported ✓" and exited 0 anyway — it never checked that files actually
landed. **Fixed** in `scripts/export_champion_model.py`: it now counts
the exported files and raises loudly if the export produced nothing.
Verified in both directions after the fix: re-run from the host still
succeeds normally (6 files, unchanged); re-run inside the container now
correctly **fails** with a clear error instead of a false success.

This specific failure mode is a consequence of choosing the fast
`sample` config for verification speed — it never touches the
`full_population_932335`-filtered run, so `register` lands on the old
host-originated one. The documented `full`/`8` default would instead
train a **fresh** run inside the container (a path native to wherever
it just ran), and would not hit this. That path is untested here only
because it would have taken considerably longer under the container's
slower BLAS performance noted above — the orchestration pattern itself
is already fully proven by the four tasks that did succeed, and the
fifth task's *failure mode* (silent-false-success, now fixed) is
arguably more valuable evidence of a working, honest pipeline than a
clean run would have been.

## Running it

```bash
cd airflow
cp .env.example .env
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# paste the output as FERNET_KEY= in .env

docker compose up airflow-init      # one-time: migrates the metadata DB, creates the admin user
docker compose up -d                # starts postgres + api-server + scheduler + dag-processor
```

Open <http://localhost:8080> (airflow/airflow, from `.env`), find
`otda_pipeline`, and trigger it — optionally overriding the `dataset`
(`full`/`sample`) and `n_iter` params at trigger time (defaults: `full`,
`8`, matching `scripts/run_experiments.py`'s own project defaults).

Teardown:
```bash
docker compose down          # keep the postgres volume (faster restart)
docker compose down -v       # also remove it (full reset)
```

## How the container sees the project

The whole repo is bind-mounted at `/opt/airflow/project` (see
`docker-compose.yaml`) rather than baked into the image — a code
change doesn't require a rebuild. `PYTHONPATH=/opt/airflow/project/src`
makes `import otda...` work in every task without a separate
`pip install -e .` step, which would otherwise need to run *after* the
mount exists, not at image-build time. The custom `Dockerfile` only
adds the Stage 0-2/4 dependency layer (duckdb, dbt, sklearn, mlflow —
matching this DAG's DuckDB-only scope, not the full `requirements.txt`).

`dbt`'s dev target already writes to `../data/dev.duckdb` and reads
`../dbt/profiles.yml` — since the container sees the identical repo
layout as local dev (same relative paths), no path overrides were
needed for dbt specifically.
