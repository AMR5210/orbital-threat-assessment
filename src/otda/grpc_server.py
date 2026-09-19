"""gRPC serving layer over otda.inference — a third protocol alongside
FastAPI (src/otda/api.py) and SageMaker, over the SAME shared
otda.inference.predict(). See grpc_server/README.md for why this
exists as something distinct from FastAPI's REST API (typed contracts
generated from protos/pha.proto, no JSON schema drift between client
and server, lower per-call overhead, native streaming support if ever
needed) rather than a redundant copy of it.

Stubs (otda/pha_pb2.py, otda/pha_pb2_grpc.py) are generated, checked-in
artifacts -- see grpc_server/README.md for the exact regeneration
command and the grpcio-tools/protobuf version pinning it needs: the
latest grpcio-tools generates gencode newer than this project's
protobuf pin, which dbt/Snowflake/MLflow all constrain to <7.0.
"""
from __future__ import annotations

from concurrent import futures

import grpc

from otda.inference import load_model, predict
from otda.pha_pb2 import HealthResponse, PredictionResponse
from otda.pha_pb2_grpc import PhaClassifierServicer, add_PhaClassifierServicer_to_server


class PhaClassifierServer(PhaClassifierServicer):
    def __init__(self):
        self._model = load_model()

    def Predict(self, request, context):
        features = {
            "e": request.e,
            "q": request.q,
            "H": request.h,
            "is_neo": request.is_neo,
            "class_code": request.class_code,
            "moid": request.moid,
        }
        result = predict(self._model, features)
        return PredictionResponse(
            is_pha=result.is_pha,
            probability=result.probability,
            explanation=result.explanation,
        )

    def Health(self, request, context):
        return HealthResponse(status="ok", model_loaded=self._model is not None)


def serve(port: int = 50051) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_PhaClassifierServicer_to_server(PhaClassifierServer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"gRPC server listening on port {port}")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
