"""gRPC serving layer smoke tests — real in-process server, real
client stub, real predictions. Same rationale/skip pattern as
test_api.py: skipped (not failed) without an exported model, and
import-skipped as a whole module so CI's pytest job (which doesn't
install grpcio) never breaks at collection time.
"""
from __future__ import annotations

from concurrent import futures

import pytest

pytest.importorskip("grpc")
pytest.importorskip("mlflow")

import grpc

from otda.inference import DEFAULT_MODEL_PATH
from otda.pha_pb2 import AsteroidFeatures, HealthRequest
from otda.pha_pb2_grpc import PhaClassifierStub, add_PhaClassifierServicer_to_server

pytestmark = pytest.mark.skipif(
    not DEFAULT_MODEL_PATH.exists(),
    reason="No exported model at models/champion/ — run scripts/export_champion_model.py",
)


@pytest.fixture
def stub():
    from otda.grpc_server import PhaClassifierServer

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    add_PhaClassifierServicer_to_server(PhaClassifierServer(), server)
    port = server.add_insecure_port("localhost:0")  # OS-assigned free port
    server.start()
    try:
        channel = grpc.insecure_channel(f"localhost:{port}")
        yield PhaClassifierStub(channel)
    finally:
        server.stop(grace=None)


def test_health(stub):
    resp = stub.Health(HealthRequest())
    assert resp.status == "ok"
    assert resp.model_loaded is True


def test_predict_pha_like(stub):
    resp = stub.Predict(
        AsteroidFeatures(e=0.3, q=0.9, h=18.0, is_neo=True, class_code=1, moid=0.02)
    )
    assert resp.is_pha is True
    assert resp.probability > 0.5
    assert "0.05 AU" in resp.explanation
    assert "22.0" in resp.explanation


def test_predict_non_pha_like(stub):
    resp = stub.Predict(
        AsteroidFeatures(e=0.1, q=2.5, h=10.0, is_neo=False, class_code=8, moid=0.5)
    )
    assert resp.is_pha is False
    assert resp.probability < 0.5
