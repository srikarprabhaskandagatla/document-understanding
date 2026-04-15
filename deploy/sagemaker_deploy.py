import argparse
import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel


def package_and_upload_model(
    local_model_path: str,
    s3_bucket: str,
    s3_prefix: str,
    region: str,
) -> str:
    print(f"Packaging model from {local_model_path}...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tar_path = os.path.join(tmp_dir, "model.tar.gz")

        with tarfile.open(tar_path, "w:gz") as tar:
            model_dir = Path(local_model_path)
            for file_path in model_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(model_dir)
                    tar.add(file_path, arcname=arcname)

            # Add inference script - SageMaker looks for code/inference.py
            inference_script = Path(__file__).parent / "inference.py"
            tar.add(inference_script, arcname="code/inference.py")

        # Upload to S3
        s3_client = boto3.client("s3", region_name=region)
        s3_key = f"{s3_prefix}/model.tar.gz"

        print(f"Uploading model.tar.gz to s3://{s3_bucket}/{s3_key}...")
        s3_client.upload_file(tar_path, s3_bucket, s3_key)

    s3_uri = f"s3://{s3_bucket}/{s3_key}"
    print(f"Model uploaded to {s3_uri}")
    return s3_uri


def deploy_endpoint(
    model_s3_uri: str,
    endpoint_name: str,
    region: str = "us-east-1",
    instance_type: str = "ml.g5.2xlarge",
    instance_count: int = 1,
):
    sess = sagemaker.Session(boto3.Session(region_name=region))
    role = sagemaker.get_execution_role()

    print(f"Creating HuggingFaceModel for endpoint: {endpoint_name}")

    # WHY transformers==4.40.0:
    #   This is the version that introduced stable LLaVA-1.6 (LlavaNext) support.
    #   Earlier versions have a bug where image tiles are duplicated in attention masks.
    hub_config = {
        "HF_MODEL_ID": "__CUSTOM__",  # Using our own model, not Hub model
        "HF_TASK": "image-text-to-text",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",  # Prevents CUDA fragmentation
    }

    llava_model = HuggingFaceModel(
        model_data=model_s3_uri,
        role=role,
        transformers_version="4.40",
        pytorch_version="2.2",
        py_version="py311",
        env=hub_config,
        sagemaker_session=sess,
    )

    print(f"Deploying to {instance_type} x {instance_count}...")
    predictor = llava_model.deploy(
        initial_instance_count=instance_count,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
        # WHY container_startup_health_check_timeout=600:
        #   LLaVA-1.6 takes ~4 min to load on cold start (14GB weights from S3).
        #   Default timeout is 60s, which causes premature health check failures.
        container_startup_health_check_timeout=600,
    )

    print(f"Endpoint deployed: {endpoint_name}")
    print(f"Invoke with: aws sagemaker-runtime invoke-endpoint --endpoint-name {endpoint_name}")
    return predictor


def test_endpoint(endpoint_name: str, region: str, test_image_path: str, question: str):
    import base64

    runtime = boto3.client("sagemaker-runtime", region_name=region)

    with open(test_image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = json.dumps({"image": image_b64, "question": question})

    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=payload,
    )

    result = json.loads(response["Body"].read())
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_s3_uri", required=True, help="S3 URI of model.tar.gz OR local model dir to package")
    parser.add_argument("--endpoint_name", default="llava-docvqa-v1")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--instance_type", default="ml.g5.2xlarge")
    parser.add_argument("--s3_bucket", help="S3 bucket (required if packaging local model)")
    parser.add_argument("--local_model_path", help="Local merged model path to package")
    args = parser.parse_args()

    # If local path provided, package and upload first
    if args.local_model_path:
        assert args.s3_bucket, "Must provide --s3_bucket when packaging local model"
        model_uri = package_and_upload_model(
            args.local_model_path,
            args.s3_bucket,
            "llava-docvqa",
            args.region,
        )
    else:
        model_uri = args.model_s3_uri

    deploy_endpoint(model_uri, args.endpoint_name, args.region, args.instance_type)