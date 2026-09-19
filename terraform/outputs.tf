output "warehouse_name" {
  value = snowflake_warehouse.otda.name
}

output "database_name" {
  value = snowflake_database.otda.name
}

output "schema_name" {
  value = snowflake_schema.otda.name
}

output "raw_schema_name" {
  value = snowflake_schema.raw.name
}

output "role_name" {
  value = snowflake_account_role.transformer.name
}
