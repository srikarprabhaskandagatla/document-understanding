"""
prepare_dataset.py
------------------
Converts raw DocVQA-style annotations into LLaVA-1.6 instruction format.

WHY THIS FORMAT:
  LLaVA-1.6 was trained on ShareGPT4V conversation format. The model's tokenizer
  and chat template EXPECT this exact structure. Any deviation causes the model
  to hallucinate or produce garbage outputs regardless of LoRA quality.

INPUT FORMAT (raw JSON):
  {"image": "doc_001.png", "question": "What is the invoice number?", "answer": "INV-2024-0042"}

OUTPUT FORMAT (LLaVA instruction tuning):
  {
    "id": "doc_001",
    "image": "doc_001.png",
    "conversations": [
      {"from": "human", "value": "<image>\nWhat is the invoice number?"},
      {"from": "gpt", "value": "INV-2024-0042"}
    ]
  }
"""

import json
import os
import random
from pathlib import Path
from PIL import Image
import argparse


def validate_image(image_path: str) -> bool:
    """Verify image is readable and non-corrupt before training."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def convert_to_llava_format(sample: dict, image_root: str) -> dict | None:
    """
    Convert a single raw sample to LLaVA conversation format.
    Returns None if image is missing or corrupt (hard filter, not soft).
    """
    image_path = os.path.join(image_root, sample["image"])

    if not os.path.exists(image_path):
        print(f"[WARN] Missing image: {image_path}")
        return None

    if not validate_image(image_path):
        print(f"[WARN] Corrupt image: {image_path}")
        return None

    # LLaVA-1.6 REQUIRES <image> token at the start of the human turn
    # This tells the vision encoder WHERE to inject visual features
    human_value = f"<image>\n{sample['question'].strip()}"

    # Normalize answer — strip whitespace, handle list answers (DocVQA has multi-answer)
    answer = sample["answer"]
    if isinstance(answer, list):
        answer = answer[0]  # Take first acceptable answer for training
    answer = str(answer).strip()

    return {
        "id": Path(sample["image"]).stem,
        "image": sample["image"],
        "conversations": [
            {"from": "human", "value": human_value},
            {"from": "gpt", "value": answer},
        ],
    }


def prepare_dataset(
    input_path: str,
    output_path: str,
    image_root: str,
    val_split: float = 0.1,
    seed: int = 42,
):
    random.seed(seed)

    print(f"Loading raw data from {input_path}...")
    with open(input_path) as f:
        raw_data = json.load(f)

    print(f"Converting {len(raw_data)} samples...")
    converted = []
    skipped = 0

    for sample in raw_data:
        result = convert_to_llava_format(sample, image_root)
        if result is not None:
            converted.append(result)
        else:
            skipped += 1

    print(f"Converted: {len(converted)} | Skipped (bad images): {skipped}")

    # Shuffle before split to avoid ordering bias
    random.shuffle(converted)
    split_idx = int(len(converted) * (1 - val_split))
    train_data = converted[:split_idx]
    val_data = converted[split_idx:]

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_out = output_dir / "train.json"
    val_out = output_dir / "val.json"

    with open(train_out, "w") as f:
        json.dump(train_data, f, indent=2)

    with open(val_out, "w") as f:
        json.dump(val_data, f, indent=2)

    print(f"Train: {len(train_data)} samples -> {train_out}")
    print(f"Val:   {len(val_data)} samples -> {val_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw annotations JSON")
    parser.add_argument("--output", required=True, help="Output directory for processed splits")
    parser.add_argument("--image_root", required=True, help="Root directory containing images")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_dataset(args.input, args.output, args.image_root, args.val_split, args.seed)