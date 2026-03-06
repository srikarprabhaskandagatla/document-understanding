# Base: AWS SageMaker PyTorch DLC with CUDA 12.1 + Python 3.11
# WHY THIS BASE IMAGE:
#   - Pre-installed: torch 2.2, transformers 4.40, cuda 12.1 runtime
#   - Certified by AWS for SageMaker — passes security scans out of the box
#   - ~6GB base image size vs ~12GB for building from scratch with nvidia/cuda
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.2.0-gpu-py311-cu121-ubuntu20.04-sagemaker

LABEL maintainer="your_email@umass.edu"
LABEL project="llava-docvqa"
LABEL version="1.0"

# Install system dependencies
# libGL required for OpenCV image processing (PIL uses it for JPEG decode)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# WHY UPGRADE pip:
#   pip <23.0 has a resolver bug that fails on complex multi-version constraints
#   in the transformers + peft + bitsandbytes dependency tree
RUN pip install --upgrade pip

# Pin exact versions — production models must be reproducible
# Any version drift in transformers can break LLaVA's chat template application
RUN pip install \
    transformers==4.40.0 \
    peft==0.10.0 \
    accelerate==0.28.0 \
    bitsandbytes==0.43.0 \
    python-Levenshtein==0.25.0 \
    pillow==10.3.0 \
    safetensors==0.4.2

# Copy inference handler
# SageMaker's PyTorch container loads from /opt/ml/code/inference.py by default
# when entry_point is set to "inference.py" in HuggingFaceModel
COPY inference.py /opt/ml/code/inference.py

# SageMaker health check endpoint — container must respond to GET /ping
# The base DLC handles this automatically; no custom HTTP server needed
EXPOSE 8080

# Entry point is managed by SageMaker DLC — do not override CMD
# The DLC starts the multi-model server which calls model_fn/predict_fn/etc.