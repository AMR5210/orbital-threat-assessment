variable "snowflake_organization_name" {
  description = "Snowflake organization name (Snowsight: Admin -> Accounts, shown at the top). Set via TF_VAR_snowflake_organization_name."
  type        = string
  default     = "myorg" # placeholder -- this identifies a real account, so no real default ships in this public repo
}

variable "snowflake_account_name" {
  description = "Snowflake account name within the organization. Set via TF_VAR_snowflake_account_name."
  type        = string
  # Find yours via CURRENT_ACCOUNT_NAME() -- NOT CURRENT_ACCOUNT(), which
  # returns the legacy account locator (a different identifier).
  default = "myaccount" # placeholder -- see comment above
}

variable "snowflake_user" {
  description = "User Terraform authenticates as, and the user granted role_name afterward. Needs snowflake_provisioning_role's privileges (see main.tf)."
  type        = string
  default     = "your_snowflake_user" # placeholder -- see comment above
}

# Key-pair auth, not password (see dbt/profiles.yml's comment on why,
# and the Aug-Oct 2026 Snowflake deprecation context). A path, read via
# file() in main.tf, not raw PEM content passed as a TF_VAR_ string --
# a multi-line env var value was found (2026-09-17) not to survive the
# Git-Bash-to-native-Windows-exe environment handoff intact, so
# matching dbt's own private_key_path pattern is both more consistent
# AND more reliable here, not just a style preference.
variable "snowflake_private_key_path" {
  description = "Path to the RSA private key (PEM/PKCS8, unencrypted by default). Never put the key CONTENT in a .tfvars file -- only the path is set here."
  type        = string
  default     = "~/.snowflake/keys/rsa_key.p8"
}

variable "snowflake_private_key_passphrase" {
  description = "Passphrase for an encrypted private key. Leave unset for a nocrypt key (this project's default)."
  type        = string
  sensitive   = true
  default     = ""
}

variable "snowflake_provisioning_role" {
  description = "Role Terraform uses to CREATE the objects below (needs CREATE WAREHOUSE/DATABASE/ROLE + MANAGE GRANTS). ACCOUNTADMIN is simplest for a fresh trial account."
  type        = string
  default     = "ACCOUNTADMIN"
}

variable "warehouse_name" {
  description = "Warehouse dbt's prod target and the Snowpark demo will USE (distinct from snowflake_provisioning_role, which only provisions it)."
  type        = string
  default     = "OTDA_WH"
}

variable "database_name" {
  type    = string
  default = "ORBITAL_THREAT_ASSESSMENT"
}

variable "schema_name" {
  description = "Schema dbt's models materialize into -- matches the dev target's `main` schema's role."
  type        = string
  default     = "ANALYTICS"
}

variable "raw_schema_name" {
  description = "Schema the raw CSV is loaded into (scripts/load_raw_to_snowflake.py) -- matches the dev target's `raw` schema, referenced by dbt/models/staging/sources.yml."
  type        = string
  default     = "RAW"
}

variable "role_name" {
  description = "Role dbt's prod target and the Snowpark demo authenticate as day-to-day -- matches dbt/profiles.yml's SNOWFLAKE_ROLE default."
  type        = string
  default     = "TRANSFORMER"
}
