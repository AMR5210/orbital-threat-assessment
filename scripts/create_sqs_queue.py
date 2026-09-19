"""One-time SQS queue setup for otda.batch_scoring.

Same weight/rationale as create_dynamodb_table.py: one queue, no idle
cost (SQS bills per-request, not per-hour), so a small script matches
its weight better than adding a second IaC tool for it. Standard queue
(not FIFO) -- batch scoring order doesn't matter, and standard queues
support the higher throughput / simpler create_queue call.

NOT wired into otda.agent in any way -- this is the batch-scoring path
(scripts + otda.batch_scoring), kept deliberately separate from the
LangGraph agent's real-time tool-calling path.

Usage:
    python scripts/create_sqs_queue.py --queue-name otda-batch-scoring
"""
from __future__ import annotations

import argparse
import os

import boto3


def create_queue(queue_name: str, region: str) -> str:
    sqs = boto3.client("sqs", region_name=region)
    resp = sqs.create_queue(QueueName=queue_name)
    queue_url = resp["QueueUrl"]
    print(f"Created queue '{queue_name}' in {region}.")
    print(f"QueueUrl: {queue_url}")
    print(f"Set OTDA_SQS_QUEUE_URL={queue_url} to use it (see src/otda/batch_scoring.py).")
    return queue_url


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-name", default="otda-batch-scoring")
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    args = parser.parse_args()
    create_queue(args.queue_name, args.region)


if __name__ == "__main__":
    main()
