# Terraform — Snowflake infrastructure (Stage 8)

Provisions one warehouse (`OTDA_WH`, XSMALL, auto-suspends after 60s
idle), one database (`ORBITAL_THREAT_ASSESSMENT`), two schemas
(`ANALYTICS` for dbt's prod target, `RAW` for the raw CSV load), and
one role (`TRANSFORMER`) scoped to only those objects — matching
`dbt/profiles.yml`'s `prod` target defaults. Nothing ML-facing; that's
dbt/Snowpark's job once these exist (Snowflake Model Registry's
`CREATE MODEL` grant is the one exception — see below).

## Status

**Applied for real, live, and dbt-verified — CLOSED.**

- `terraform init` — succeeded, provider `snowflakedb/snowflake` v2.21.0
  resolved and locked (`.terraform.lock.hcl`, committed).
- `terraform validate` — passed against the provider's actual current
  docs (fetched 2026-09-17 — this provider moved namespaces from
  `Snowflake-Labs/snowflake` to `snowflakedb/snowflake` at some point,
  and renamed `snowflake_role` -> `snowflake_account_role` along the
  way, so an older cached example would have been wrong).
- `terraform apply` — **run for real by the user on 2026-09-18**: "Apply
  complete! Resources: 10 added, 0 changed, 0 destroyed." Two real
  findings along the way getting to a clean `plan`, not hypothetical:
  (1) a multi-line PEM private key passed via a `TF_VAR_` environment
  variable did not survive the Git-Bash-to-native-Windows-exe handoff
  intact — switched to a file path read via `file()` in `main.tf`
  instead, matching dbt's own pattern; (2) this provider needs an
  explicit `authenticator = "SNOWFLAKE_JWT"` to actually use
  `private_key` — the opposite of dbt-snowflake, which auto-detects
  key-pair auth from `private_key_path` alone and rejects an explicit
  `authenticator: keypair` as unrecognized.
- A follow-up `terraform apply` (also run by the user) added
  `CREATE MODEL` to the `ANALYTICS` schema grant — found missing when
  `scripts/register_model_to_snowflake_registry.py` failed with a real
  SQL access-control error. "Plan: 0 to add, 1 to change, 0 to
  destroy" both times.
- `dbt build -t prod` is fully green against this infra (27/27) and
  `scripts/load_raw_to_snowflake.py` / `snowpark_feature_aggregation_demo.py`
  have both run for real — see the main README's credentials update
  note for the bugs found getting there (none were Terraform's fault;
  all were in the loader script or dbt SQL).

## Prerequisites

1. A Snowflake account (trial: <https://signup.snowflake.com/>). The
   trial's initial user typically has `ACCOUNTADMIN` by default, which
   is what this config uses to provision (see
   `snowflake_provisioning_role` in `variables.tf`).
2. Terraform >= 1.5 (this was run against 1.16.2, installed via
   `winget install Hashicorp.Terraform`).
3. An RSA key pair for key-pair auth (see the main repo README's
   credential setup section) — `snowflake_private_key_path` defaults
   to `~/.snowflake/keys/rsa_key.p8`.

## Usage

`organization_name`/`account_name`/`user`/`private_key_path` all have
real defaults in `variables.tf` for this project — `terraform plan`/
`apply` work with zero env vars. Override via `TF_VAR_snowflake_*`
only if pointing at a different account.

```bash
cd terraform
terraform init      # already run once; re-run if providers/config change
terraform plan       # review before applying -- verified working 2026-09-17
terraform apply
```

Outputs (`warehouse_name`, `database_name`, `schema_name`, `role_name`)
match `dbt/profiles.yml`'s `prod` target env var defaults — no further
wiring needed between the two once applied.

## Cost control / teardown

`initially_suspended = true` and `auto_suspend = 60` on the warehouse
mean near-zero idle cost — this is the whole reason Terraform owns
this instead of clicking through Snowsight by hand: **destroyable on
command**.

```bash
terraform destroy
```

Run this when not actively using the Snowflake stages — a trial
account's credits are finite and this project has no schedule that
needs the warehouse running continuously (see the Airflow stage's own
README section on why a static dataset needs no schedule either).
