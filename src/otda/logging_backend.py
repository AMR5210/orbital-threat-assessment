"""Optional DynamoDB request/response logging for the serving layer.

Attaches to the SHARED otda.inference module, not a per-serving-path
integration — per project decision, every caller of
otda.inference.predict() gets logging for free once DynamoDB is
configured, no extra wiring per endpoint. In practice today that means
otda.api's /predict AND otda.agent's predict_pha both log automatically
(the agent wasn't explicitly named in the original decision, but it
calls the same shared predict() — this is the "free" part of "the
shared module makes logging both essentially free").

No-op if OTDA_DYNAMODB_TABLE is unset (local dev/CI default).

Known gap: this hook only fires for code that calls
otda.inference.predict() in-process. The SageMaker deployment
(scripts/deploy_to_sagemaker.py) uses MLflow's own generic serving
container — a separate process that runs its own scoring server and
never imports this module, so DynamoDB logging doesn't reach that
endpoint. Closing that gap would need a custom SageMaker container
wrapping otda.inference (like the FastAPI Dockerfile, with
/invocations instead of /predict) — out of scope here.
"""
from __future__ import annotations

import os
import time
import uuid

_dynamodb_resource = None


def _get_table():
    table_name = os.environ.get("OTDA_DYNAMODB_TABLE")
    if not table_name:
        return None

    global _dynamodb_resource
    if _dynamodb_resource is None:
        import boto3

        _dynamodb_resource = boto3.resource(
            "dynamodb", region_name=os.environ.get("AWS_REGION", "us-east-1")
        )
    return _dynamodb_resource.Table(table_name)


def log_prediction(features: dict, result) -> None:
    """Fire-and-forget prediction log. Never raises — a logging
    failure must not break a prediction response. `result` is an
    otda.inference.PredictionResult (or anything with the same three
    attributes)."""
    table = _get_table()
    if table is None:
        return
    try:
        table.put_item(
            Item={
                "request_id": str(uuid.uuid4()),
                "timestamp": int(time.time()),
                "features": {k: str(v) for k, v in features.items()},
                "is_pha": result.is_pha,
                "probability": str(result.probability),
            }
        )
    except Exception as exc:  # logging must never break a prediction
        print(f"WARNING: DynamoDB logging failed: {exc}")
