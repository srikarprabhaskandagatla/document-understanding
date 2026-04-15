FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.2.0-gpu-py311-cu121-ubuntu20.04-sagemaker

LABEL maintainer="your_email@umass.edu"
LABEL project="llava-docvqa"
LABEL version="1.0"

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install \
    transformers==4.40.0 \
    peft==0.10.0 \
    accelerate==0.28.0 \
    bitsandbytes==0.43.0 \
    python-Levenshtein==0.25.0 \
    pillow==10.3.0 \
    safetensors==0.4.2

COPY inference.py /opt/ml/code/inference.py

EXPOSE 8080
