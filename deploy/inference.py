import base64
import io
import json
import os

import torch
from PIL import Image
from peft import PeftModel
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor


def model_fn(model_dir: str):
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
    if accept != "application/json":
        raise ValueError(f"Unsupported accept type: {accept}")
    return json.dumps(prediction)