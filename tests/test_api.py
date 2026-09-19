"""API-level smoke tests using FastAPI's TestClient.

Requires the exported model at models/champion/ (run
scripts/export_champion_model.py first) — skipped, not failed, if it's
absent, since the model artifact is a gitignored local build product,
not something a fresh CI checkout has. This is a real end-to-end check
of the request/response contract, distinct from tests/test_invariants.py's
synthetic-data unit tests of the split/SMOTE logic.

Whole-module import-skip (not just the model-file check below): the
CI `pytest` job (.github/workflows/ci.yml) only installs the Stage
0-2 core dependency group, not fastapi/mlflow — this file must not
break test *collection* for that job. Extending CI to cover serving
is a separate, deliberate decision, not a side effect of adding tests.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("mlflow")

from fastapi.testclient import TestClient

from otda.inference import DEFAULT_MODEL_PATH

pytestmark = pytest.mark.skipif(
    not DEFAULT_MODEL_PATH.exists(),
    reason="No exported model at models/champion/ — run scripts/export_champion_model.py",
)


@pytest.fixture
def client():
    from otda.api import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model_loaded": True}


def test_predict_pha_like(client):
    resp = client.post(
        "/predict",
        json={"e": 0.3, "q": 0.9, "H": 18.0, "is_neo": True, "class_code": 1, "moid": 0.02},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_pha"] is True
    assert body["probability"] > 0.5
    assert "0.05 AU" in body["explanation"]
    assert "22.0" in body["explanation"]


def test_predict_non_pha_like(client):
    resp = client.post(
        "/predict",
        json={"e": 0.1, "q": 2.5, "H": 10.0, "is_neo": False, "class_code": 8, "moid": 0.5},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_pha"] is False
    assert body["probability"] < 0.5


def test_predict_rejects_invalid_eccentricity(client):
    """Eccentricity must be in [0, 1) — Pydantic bounds validation,
    not the model, should catch this."""
    resp = client.post(
        "/predict",
        json={"e": 1.5, "q": 0.9, "H": 18.0, "is_neo": True, "class_code": 1, "moid": 0.02},
    )
    assert resp.status_code == 422


def test_predict_rejects_invalid_class_code(client):
    resp = client.post(
        "/predict",
        json={"e": 0.3, "q": 0.9, "H": 18.0, "is_neo": True, "class_code": 99, "moid": 0.02},
    )
    assert resp.status_code == 422
