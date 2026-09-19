"""Deploy the champion model straight from MLflow to a SageMaker endpoint.

Per instruction: deploy directly from the MLflow-registered model,
skipping the Snowflake registry for this path (see
scripts/register_model_to_snowflake_registry.py for that separate,
parallel artifact — nothing here reads from it).

This uses MLflow's own documented SageMaker deployment tooling
(`mlflow.deployments`, target "sagemaker") rather than a hand-built
container — that IS "the documented MLflow -> SageMaker path." Two
real consequences of using MLflow's generic path worth being explicit
about, not glossing over:

1. The endpoint runs MLflow's own generic pyfunc scoring server, not
   otda.inference / otda.api. It returns whatever the sklearn model's
   plain `.predict()` produces — hard PHA labels (0/1) — not the rich
   `{is_pha, probability, explanation}` shape the FastAPI /predict
   endpoint returns. NASA-rule-grounded explanations are a FastAPI/
   agent-only feature with this deployment approach.
2. Request/response format is MLflow's own scoring contract
   (pandas-split JSON in, `{"predictions": [...]}` out), not this
   project's Pydantic schema.

If explanation-rich SageMaker responses are wanted later, that needs a
custom container wrapping otda.inference (like the FastAPI Dockerfile,
with /invocations + /ping instead of /predict + /health) — out of
scope here.

CLI shape verified against the installed mlflow 3.16.1's own `--help`
output and its `SageMakerDeploymentClient.create_deployment` source —
this API has changed across mlflow major versions (the old top-level
`mlflow.sagemaker.deploy()` function is gone; the current path is the
generic `mlflow deployments` plugin CLI).

Two-step process, both requiring AWS credentials + Docker + an ECR
repo + a SageMaker execution IAM role:

    python scripts/deploy_to_sagemaker.py build-and-push --image-name otda-pha-classifier
    python scripts/deploy_to_sagemaker.py deploy --run-id <mlflow_run_id> \\
        --image-url <ecr-image-uri> \\
        --execution-role-arn arn:aws:iam::ACCOUNT:role/OTDA-SageMaker-ExecutionRole \\
        --region us-east-2

Teardown:
    python scripts/deploy_to_sagemaker.py delete --endpoint-name otda-pha-classifier

## Serverless Inference, not a real-time instance

Deploys via SageMaker Serverless Inference (memory size + max
concurrency) rather than a real-time endpoint pinned to an instance
type, for two reasons:

1. **Sidesteps the instance quota wall entirely.** New AWS accounts
   default to a 0-instance CreateEndpoint quota for most instance
   types (fraud-prevention default) -- ml.c5.xlarge hit this. That
   quota applies to *instance-based* endpoints only;
   ServerlessInferenceConfig has no `InstanceType`/
   `InitialInstanceCount` at all, so there's nothing for the quota to
   apply to (`mlflow.sagemaker._create_sagemaker_endpoint` puts
   `ServerlessConfig` on the ProductionVariant and skips those fields
   entirely when `serverless_config` is set).
2. **Fits a demo/verification endpoint better anyway.** Serverless
   scales to zero when idle and bills per-inference, not per-hour --
   no idle-endpoint billing risk the way a real-time endpoint has.

Two non-obvious blockers, in the order CreateEndpoint's validation
surfaces them:

1. Docker Desktop's containerd-backed image store pushes images with
   an OCI manifest by default (`docker build`, buildx, doesn't matter
   -- even DOCKER_BUILDKIT=0's classic builder still pushes via the
   containerd store). SageMaker's CreateModel only accepts the Docker
   V2 manifest media type and rejects OCI outright. Fix without
   touching Docker Desktop's global settings: re-export the already-
   built image through buildx with explicit Docker media types --
   `docker buildx build --provenance=false --output
   type=image,name=<ecr-uri>,oci-mediatypes=false,push=true .` against
   a trivial `FROM <local-image>` Dockerfile. Verify with
   `aws ecr batch-get-image ... --query 'images[0].imageManifestMediaType'`
   before retrying the deploy -- should read
   `application/vnd.docker.distribution.manifest.v2+json`. This
   affects the image itself, independent of serverless vs. real-time.
2. Serverless endpoints reject images over 10 GiB (a hard SageMaker
   limit, backed by Lambda under the hood) and cap runtime memory
   around 6 GiB regardless of the container's own size. This project's
   image is ~1.09 GiB (`aws ecr describe-images ... imageSizeInBytes`)
   -- comfortable margin.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys


def build_and_push(image_name: str) -> None:
    """Builds MLflow's generic serving image locally and pushes to ECR
    under the current active AWS account/region. Needs Docker running
    locally and AWS credentials with ECR push permissions.

    --env-manager local is required, not optional: the default
    (virtualenv) makes the container try to build a per-model
    virtualenv at startup via pyenv, which this image doesn't have --
    every real invocation fails with "Could not find the pyenv
    binary". `local` serves directly from the image's own
    already-installed environment, which already matches what the
    model needs (baked in from requirements-serving.txt at image build
    time) -- no per-model env to construct."""
    cmd = [
        sys.executable, "-m", "mlflow", "sagemaker", "build-and-push-container",
        "--container", image_name,
        "--env-manager", "local",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def deploy(
    run_id: str,
    image_url: str,
    execution_role_arn: str,
    region: str,
    endpoint_name: str,
    memory_size_in_mb: int,
    max_concurrency: int,
) -> None:
    """Serverless Inference, not a real-time instance -- see this
    module's docstring for why. `serverless_config` is a dict-typed
    `-C` value: mlflow's CLI parses it with json.loads since it isn't
    already a dict (confirmed by reading
    SageMakerDeploymentClient._apply_custom_config's source), so a
    JSON string is the correct way to pass it here, not a flat
    key=value pair the way instance_type/instance_count were.

    MLFLOW_DISABLE_ENV_CREATION=true is required, not optional --
    --env-manager local at build time only changes HOW the container
    would install the model's dependencies, not WHETHER it does; by
    default it always tries (mlflow.models.container: disabled "only
    for SageMaker deployment", i.e. deliberately on for this exact
    path), and the model's own auto-captured requirements.txt pins
    numpy==2.5.3, which has no wheel for the container's Python 3.11
    (needs >=3.12) -- fails with "model process exited" during the pip
    install. Safe to disable entirely: `docker run --entrypoint python
    <image> -c "import sklearn; print(sklearn.__version__)"` shows the
    image already has sklearn==1.9.1, an exact match to what trained
    this model, plus very close numpy/pandas minor versions -- the
    container's own pre-installed environment already covers what the
    model needs."""
    serverless_config = json.dumps(
        {"MemorySizeInMB": memory_size_in_mb, "MaxConcurrency": max_concurrency}
    )
    env_override = json.dumps({"MLFLOW_DISABLE_ENV_CREATION": "true"})
    cmd = [
        sys.executable, "-m", "mlflow", "deployments", "create",
        "-t", "sagemaker",
        "-m", f"runs:/{run_id}/model",
        "--name", endpoint_name,
        "-C", f"region_name={region}",
        "-C", f"serverless_config={serverless_config}",
        "-C", f"env={env_override}",
        "-C", f"execution_role_arn={execution_role_arn}",
        "-C", f"image_url={image_url}",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def delete(endpoint_name: str) -> None:
    """Tear down. Serverless endpoints don't bill while idle the way a
    real-time endpoint does, but there's no reason to leave a demo
    endpoint provisioned once you're done verifying it."""
    cmd = [
        sys.executable, "-m", "mlflow", "deployments", "delete",
        "-t", "sagemaker", "-n", endpoint_name,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser(
        "build-and-push", help="Build + push MLflow's serving container to ECR"
    )
    p_build.add_argument("--image-name", default="otda-pha-classifier")

    p_deploy = sub.add_parser("deploy", help="Create the Serverless Inference endpoint")
    p_deploy.add_argument("--run-id", required=True, help="MLflow run_id of the champion model")
    p_deploy.add_argument("--image-url", required=True, help="ECR image URI from build-and-push")
    p_deploy.add_argument("--execution-role-arn", required=True)
    p_deploy.add_argument("--region", default="us-east-1")
    p_deploy.add_argument("--endpoint-name", default="otda-pha-classifier")
    p_deploy.add_argument(
        "--memory-size-in-mb", type=int, default=2048,
        help="Serverless memory allocation (1024-6144); more memory also means more CPU",
    )
    p_deploy.add_argument(
        "--max-concurrency", type=int, default=5,
        help="Max concurrent invocations before requests queue/throttle (1-200)",
    )

    p_delete = sub.add_parser("delete", help="Tear down the endpoint")
    p_delete.add_argument("--endpoint-name", default="otda-pha-classifier")

    args = parser.parse_args()

    if args.command == "build-and-push":
        build_and_push(args.image_name)
    elif args.command == "deploy":
        deploy(
            args.run_id, args.image_url, args.execution_role_arn,
            args.region, args.endpoint_name,
            args.memory_size_in_mb, args.max_concurrency,
        )
    elif args.command == "delete":
        delete(args.endpoint_name)


if __name__ == "__main__":
    main()
