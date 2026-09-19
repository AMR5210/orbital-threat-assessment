# gRPC serving (`src/otda/grpc_server.py`)

A third serving protocol over `otda.inference.predict()`, alongside
FastAPI (`src/otda/api.py`) and SageMaker — not a replacement for
either.

## Why gRPC, distinct from FastAPI's REST API

FastAPI's `/predict` already serves the model over HTTP+JSON. gRPC
earns its place here for reasons that don't apply to that path:

- **A typed contract shared by client and server** (`protos/pha.proto`)
  — the request/response shape is generated code on both ends, not a
  hand-maintained Pydantic model on one side and hope on the other.
  There is no way for a client and server built from the same `.proto`
  to drift out of sync the way a REST client's hand-written JSON
  parsing can.
- **Binary protobuf, not JSON** — smaller payloads, faster
  (de)serialization; matters for high-QPS internal service-to-service
  calls in a way it doesn't for a handful of interactive requests.
- **Streaming is a first-class option** if ever needed (batch-scoring
  many asteroids over one connection) — REST would need a bespoke
  chunked-response or polling design to do the same thing.

None of that makes it a *better* choice than FastAPI for this
project's actual usage (a human hitting `/docs` in a browser, or the
agent's tools) — it's here to demonstrate the protocol and its real
tradeoffs, genuinely working end to end, not as a redundant copy of
the REST API.

## Status

**Verified working, 2026-09-18** — real server, real client call,
correct prediction for Apophis's actual feature values, matching the
FastAPI/agent paths' own results.

## A real version-pinning gotcha

`pip install grpcio-tools` (latest, 1.84.0) pulls in `protobuf>=7.35.1`
— which breaks `dbt-adapters`, `dbt-common`, `sagemaker-core`, and
`snowflake-snowpark-python` in this same venv, all of which pin
`protobuf<7.0`. Worse: **the `protobuf` *Python package* version
installed alongside `grpcio-tools` doesn't control what gencode
version it emits** — that's baked into the compiled `protoc` binary
shipped inside the `grpcio-tools` wheel itself, and protobuf's runtime
refuses to load gencode newer than itself (a hard, documented
constraint, not a warning).

The fix: `grpcio-tools==1.73.0` depends on `protobuf<7.0.0,>=6.30.0` —
compatible with this project's existing `protobuf==6.33.6` pin — so
its bundled `protoc` emits gencode this project's runtime can actually
load. `grpcio-tools` itself is **not** a runtime dependency (only
`grpcio` is — see `requirements-grpc.txt` / the `grpcio` line in
`requirements.txt`), so it never needs to be installed in this
project's actual venv at all; only in an isolated environment for the
one-time (or on-`.proto`-change) codegen step below.

One more generated-code quirk: `pha_pb2_grpc.py`'s default import
(`import pha_pb2 as pha__pb2`) assumes a flat top-level module, not
`otda.pha_pb2` — patched by hand to `from otda import pha_pb2 as
pha__pb2` after generation (see the comment at the top of that file).

## Regenerating the stubs (only if `protos/pha.proto` changes)

`otda/pha_pb2.py`, `otda/pha_pb2.pyi`, and `otda/pha_pb2_grpc.py` are
committed, generated artifacts — not built at install time — because
the generation step needs a specific `grpcio-tools` version this
project's main venv can't hold alongside `grpcio-tools`' own latest
protobuf requirement (see above). To regenerate, in a **separate,
throwaway virtualenv**:

```bash
python -m venv /tmp/grpc-codegen
/tmp/grpc-codegen/bin/pip install "grpcio-tools==1.73.0"

/tmp/grpc-codegen/bin/python -m grpc_tools.protoc \
    --proto_path=protos \
    --python_out=src/otda \
    --grpc_python_out=src/otda \
    --pyi_out=src/otda \
    protos/pha.proto
```

Then re-apply the import fix above to the regenerated
`pha_pb2_grpc.py` before committing.

## Usage

```bash
python -m otda.grpc_server   # listens on :50051
```

```python
import grpc
from otda.pha_pb2 import AsteroidFeatures
from otda.pha_pb2_grpc import PhaClassifierStub

stub = PhaClassifierStub(grpc.insecure_channel("localhost:50051"))
stub.Predict(AsteroidFeatures(e=0.19, q=0.75, h=19.7, is_neo=True, class_code=3, moid=0.0003))
```

### Docker

```bash
python scripts/export_champion_model.py   # bakes the model in at build time
docker build -f Dockerfile.grpc -t otda-grpc .
docker run -p 50051:50051 otda-grpc
```
