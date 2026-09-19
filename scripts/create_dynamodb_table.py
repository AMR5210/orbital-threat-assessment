"""One-time DynamoDB table setup for otda.logging_backend.

Not Terraform-managed — Terraform in this project is scoped to
Snowflake infra only (terraform/README.md); this is a single table,
on-demand billing (no idle cost, no capacity to size), so a small
script matches its weight better than a second IaC tool/provider.

NOT YET RUN: no AWS credentials available on this machine.

Usage:
    python scripts/create_dynamodb_table.py --table-name otda-predictions
"""
from __future__ import annotations

import argparse
import os

import boto3


def create_table(table_name: str, region: str) -> None:
    dynamodb = boto3.resource("dynamodb", region_name=region)
    table = dynamodb.create_table(
        TableName=table_name,
        KeySchema=[{"AttributeName": "request_id", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "request_id", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",  # on-demand -- no idle cost, no capacity planning
    )
    table.wait_until_exists()
    print(f"Created table '{table_name}' in {region}.")
    print(
        f"Set OTDA_DYNAMODB_TABLE={table_name} to enable logging "
        "(see src/otda/logging_backend.py)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-name", default="otda-predictions")
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    args = parser.parse_args()
    create_table(args.table_name, args.region)


if __name__ == "__main__":
    main()
