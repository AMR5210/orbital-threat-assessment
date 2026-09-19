# Provisions the Snowflake objects dbt's `prod` target and the Stage 8
# Snowpark demo connect to: one warehouse, one database, one schema,
# and one role scoped to exactly those three objects.
#
# Deliberately NOT provisioning anything ML-facing here — this is
# infrastructure only. Data loading (scripts/load_raw_to_snowflake.py)
# and the dbt/Snowpark work that runs against it are separate, later
# steps that assume these objects already exist.
#
# Verified against the current provider docs (snowflakedb/snowflake,
# fetched 2026-09-17 — this provider moved namespaces from
# Snowflake-Labs/snowflake at some point; don't reuse old examples that
# reference the old namespace or the pre-refactor snowflake_role /
# snowflake_database_grant resource names, which no longer exist).
#
# Account created 2026-09-17. Key-pair auth, not password -- see
# variables.tf's snowflake_private_key_path. organization_name/
# account_name/user have no real default in this public repo -- set
# via TF_VAR_snowflake_* or terraform.tfvars (see
# terraform.tfvars.example).

terraform {
  required_version = ">= 1.5"
  required_providers {
    snowflake = {
      source  = "snowflakedb/snowflake"
      version = "~> 2.20"
    }
  }
}

provider "snowflake" {
  organization_name       = var.snowflake_organization_name
  account_name            = var.snowflake_account_name
  user                    = var.snowflake_user
  authenticator           = "SNOWFLAKE_JWT" # required -- unlike dbt-snowflake, this provider does not auto-detect key-pair auth from private_key alone
  private_key             = file(var.snowflake_private_key_path)
  private_key_passphrase  = var.snowflake_private_key_passphrase != "" ? var.snowflake_private_key_passphrase : null
  # The role Terraform itself authenticates as to CREATE these objects —
  # distinct from role_name below, which is the role dbt/Snowpark will
  # USE afterward. Needs privileges to create warehouses/databases/
  # roles/grants; ACCOUNTADMIN is the simplest choice for a fresh trial
  # account, where the initial user typically has it by default.
  role = var.snowflake_provisioning_role
}

resource "snowflake_warehouse" "otda" {
  name                = var.warehouse_name
  warehouse_size      = "XSMALL"
  auto_suspend        = 60 # seconds idle before suspend -- cost control
  auto_resume         = true
  initially_suspended = true
  comment             = "Orbital Threat Assessment -- dbt prod target + Snowpark demo"
}

resource "snowflake_database" "otda" {
  name    = var.database_name
  comment = "Orbital Threat Assessment"
}

resource "snowflake_schema" "otda" {
  name     = var.schema_name
  database = snowflake_database.otda.name
  comment  = "Staging/marts models (dbt prod target)"
}

resource "snowflake_schema" "raw" {
  name     = var.raw_schema_name
  database = snowflake_database.otda.name
  comment  = "Raw asteroid CSV, loaded by scripts/load_raw_to_snowflake.py -- mirrors the dev target's `raw` schema, referenced by dbt/models/staging/sources.yml"
}

resource "snowflake_account_role" "transformer" {
  name    = var.role_name
  comment = "Role used by dbt's prod target and the Snowpark demo -- scoped to exactly the warehouse/database/schema above"
}

resource "snowflake_grant_privileges_to_account_role" "warehouse_usage" {
  privileges        = ["USAGE", "OPERATE"]
  account_role_name = snowflake_account_role.transformer.name
  on_account_object {
    object_type = "WAREHOUSE"
    object_name = snowflake_warehouse.otda.name
  }
}

resource "snowflake_grant_privileges_to_account_role" "database_usage" {
  privileges        = ["USAGE"]
  account_role_name = snowflake_account_role.transformer.name
  on_account_object {
    object_type = "DATABASE"
    object_name = snowflake_database.otda.name
  }
}

resource "snowflake_grant_privileges_to_account_role" "schema_privileges" {
  privileges = [
    "USAGE", "CREATE TABLE", "CREATE VIEW", "CREATE STAGE", "CREATE FILE FORMAT",
    # CREATE MODEL: needed by scripts/register_model_to_snowflake_registry.py
    # (snowflake.ml.registry.Registry.log_model) -- found missing via a
    # real run failing with "must have CREATE MODEL granted on SCHEMA
    # ...ANALYTICS" (2026-09-18). Not needed for dbt build itself.
    "CREATE MODEL",
  ]
  account_role_name = snowflake_account_role.transformer.name
  on_schema {
    schema_name = snowflake_schema.otda.fully_qualified_name
  }
}

resource "snowflake_grant_privileges_to_account_role" "raw_schema_privileges" {
  privileges = [
    "USAGE", "CREATE TABLE", "CREATE STAGE", "CREATE FILE FORMAT",
  ]
  account_role_name = snowflake_account_role.transformer.name
  on_schema {
    schema_name = snowflake_schema.raw.fully_qualified_name
  }
}

resource "snowflake_grant_account_role" "transformer_to_user" {
  role_name = snowflake_account_role.transformer.name
  user_name = var.snowflake_user
}
