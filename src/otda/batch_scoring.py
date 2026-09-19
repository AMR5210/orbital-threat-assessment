"""Async batch PHA scoring via SQS.

A genuine queue-based pipeline, deliberately separate from the
real-time paths (FastAPI, the LangGraph agent, gRPC): submit_batch()
enqueues identifiers, run_worker() drains the queue and scores each one
through the SAME shared otda.inference.predict() every other serving
path uses (via otda.agent.fetch_asteroid_impl for the identifier ->
features lookup, so this also benefits from the Elasticsearch fuzzy
fallback for free). Results land in a local JSON-lines file, not a new
DuckDB table -- keeps this from touching dbt-managed schema.

Not wired into otda.agent's tool-calling path in any way, and must
stay that way: this is for bulk/offline scoring, not a request-time
dependency for the agent.

Uses real AWS SQS (create_sqs_queue.py sets up the queue) --
pay-per-request, no idle cost, unlike a real-time endpoint.

Usage:
    python -m otda.batch_scoring submit Apophis "433 Eros" "2004 MN4"
    python -m otda.batch_scoring worker
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import boto3

from otda.agent import fetch_asteroid_impl
from otda.features import EXP2_FEATURES
from otda.inference import load_model, predict

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RESULTS_PATH = REPO_ROOT / "data" / "batch_results" / "results.jsonl"

_SEND_BATCH_LIMIT = 10  # SQS's own per-call limit for SendMessageBatch


def _get_queue_url() -> str:
    queue_url = os.environ.get("OTDA_SQS_QUEUE_URL")
    if not queue_url:
        raise RuntimeError(
            "OTDA_SQS_QUEUE_URL is not set. Run scripts/create_sqs_queue.py "
            "first and set the printed QueueUrl in .env."
        )
    return queue_url


def _chunk(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def submit_batch(identifiers: list[str], queue_url: str | None = None) -> list[str]:
    """Enqueues one message per identifier. Returns the SQS message IDs."""
    sqs = boto3.client("sqs", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    queue_url = queue_url or _get_queue_url()

    message_ids = []
    for chunk in _chunk(identifiers, _SEND_BATCH_LIMIT):
        entries = [
            {"Id": str(i), "MessageBody": json.dumps({"identifier": identifier})}
            for i, identifier in enumerate(chunk)
        ]
        resp = sqs.send_message_batch(QueueUrl=queue_url, Entries=entries)
        if resp.get("Failed"):
            raise RuntimeError(f"SQS rejected some messages: {resp['Failed']}")
        message_ids.extend(m["MessageId"] for m in resp["Successful"])
    return message_ids


def _score_one(identifier: str, model) -> dict:
    record = fetch_asteroid_impl(identifier)
    if record is None:
        return {"identifier": identifier, "error": "not found"}

    features = {k: record[k] for k in EXP2_FEATURES}
    result = predict(model, features)
    return {
        "identifier": identifier,
        "spkid": record["spkid"],
        "is_pha": result.is_pha,
        "probability": result.probability,
        "explanation": result.explanation,
    }


def run_worker(
    queue_url: str | None = None,
    results_path: Path = DEFAULT_RESULTS_PATH,
    wait_time_seconds: int = 2,
) -> int:
    """Drains every message currently available on the queue (loops
    receive_message calls until one comes back empty -- a single call
    only returns up to 10 messages regardless of queue depth), scores
    each, appends one JSON line per result, and deletes the message
    only after its result is durably written. Returns the count
    processed."""
    sqs = boto3.client("sqs", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    queue_url = queue_url or _get_queue_url()
    model = load_model()

    results_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    while True:
        resp = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=10,
            WaitTimeSeconds=wait_time_seconds,
        )
        messages = resp.get("Messages", [])
        if not messages:
            break

        with open(results_path, "a", encoding="utf-8") as f:
            for message in messages:
                body = json.loads(message["Body"])
                result = _score_one(body["identifier"], model)
                result["scored_at"] = time.time()
                f.write(json.dumps(result) + "\n")

                sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=message["ReceiptHandle"])
                processed += 1

    return processed


def main() -> None:
    import sys

    if len(sys.argv) < 2 or sys.argv[1] not in ("submit", "worker"):
        print("Usage: python -m otda.batch_scoring submit <identifier> [<identifier> ...]")
        print("       python -m otda.batch_scoring worker")
        raise SystemExit(1)

    if sys.argv[1] == "submit":
        identifiers = sys.argv[2:]
        if not identifiers:
            raise SystemExit("Provide at least one identifier to submit.")
        message_ids = submit_batch(identifiers)
        print(f"Submitted {len(message_ids)} message(s).")
    else:
        count = run_worker()
        print(f"Processed {count} message(s). Results: {DEFAULT_RESULTS_PATH}")


if __name__ == "__main__":
    main()
