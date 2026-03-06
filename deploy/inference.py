"""
inference.py
------------
SageMaker inference handler for the fine-tuned LLaVA-1.6 model.

SageMaker's PyTorchModel container calls these four functions in order:
  1. model_fn()   — Load model from /opt/ml/model (once, at container startup)
  2. input_fn()   — Deserialize incoming HTTP request body
  3. predict_fn() — Run inference
  4. output_fn()  — Serialize predictions back to HTTP response

WHY THIS STRUCTURE OVER A FLASK/FASTAPI SERVER:
  SageMaker manages the HTTP server, TLS, auto-scaling, and health checks.
  These four functions are the ONLY integration points needed.
  Building a custom server would duplicate infrastructure SageMaker provides
  and remove access to SageMaker's built-in multi-model serving and canary deployments.

ENDPOINT INPUT FORMAT (JSON):
  {
    "image": "<base64-encoded-image>",
    "question": "What is the total amount on this invoice?"
  }

ENDPOINT OUTPUT FORMAT (JSON):
  {
    "answer": "USD 4,250.00",
    "confidence": null   # LLaVA doesn't expose token probabilities by default
  }
"""

import base64
import io
import json
import os

import torch
from PIL import Image
from peft import PeftModel
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor


def model_fn(model_dir: str):
    """
    Load the merged (LoRA-integrated) model from SageMaker model directory.
    
    SageMaker extracts model.tar.gz to /opt/ml/model before calling this.
    model_dir = "/opt/ml/model"
    
    WHY MERGED MODEL (not base + adapter):
      SageMaker containers don't have PEFT installed by default.
      Merging LoRA weights into the base model before deployment
      produces a standard HuggingFace model requiring only `transformers`.
      Simpler dependency tree = faster container startup = lower cold start latency.
    """
    print(f"Loading model from {model_dir}")

    processor = LlavaNextProcessor.from_pretrained(model_dir)

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    print("Model loaded successfully.")
    return {"model": model, "processor": processor}


def input_fn(request_body: str, content_type: str = "application/json") -> dict:
    """
    Deserialize incoming request.
    
    WHY BASE64 FOR IMAGES:
      HTTP/JSON doesn't support binary payloads natively.
      Base64 encoding increases payload size by ~33% but is universally
      supported by all API clients without multipart form setup.
      For document images (<2MB), this overhead is negligible.
    """
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}. Use application/json.")

    data = json.loads(request_body)

    if "image" not in data or "question" not in data:
        raise ValueError("Request must contain 'image' (base64) and 'question' fields.")

    # Decode base64 image
    image_bytes = base64.b64decode(data["image"])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    return {
        "image": image,
        "question": data["question"].strip(),
    }


def predict_fn(input_data: dict, model_dict: dict) -> dict:
    """
    Run VQA inference.
    
    WHY max_new_tokens=128:
      DocVQA answers are typically 1-10 words. Capping at 128 prevents
      runaway generation (which adds latency) while covering edge cases
      like dates, addresses, and multi-word entity names.
    
    WHY do_sample=False (greedy decoding):
      Document VQA is deterministic — "What is the invoice number?" has
      one correct answer. Sampling introduces randomness that lowers accuracy.
      Greedy decoding is also faster (no beam search overhead).
    """
    model = model_dict["model"]
    processor = model_dict["processor"]
    device = next(model.parameters()).device

    image = input_data["image"]
    question = input_data["question"]

    # Build conversation in LLaVA format
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    ).to(device, torch.bfloat16)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    # Decode only newly generated tokens
    new_tokens = generated_ids[:, inputs["input_ids"].shape[1]:]
    answer = processor.tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()

    return {"answer": answer}


def output_fn(prediction: dict, accept: str = "application/json") -> str:
    """Serialize prediction to JSON response."""
    if accept != "application/json":
        raise ValueError(f"Unsupported accept type: {accept}")
    return json.dumps(prediction)