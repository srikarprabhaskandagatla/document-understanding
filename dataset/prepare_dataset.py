import json
import os
import random
from pathlib import Path
from PIL import Image
import argparse


def validate_image(image_path: str) -> bool:
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def convert_to_llava_format(sample: dict, image_root: str) -> dict | None:
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

    # Normalize answer - strip whitespace, handle list answers (DocVQA has multi-answer)
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
    val_split: float = 0.05,
    test_split: float = 0.05,
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

    random.shuffle(converted)
    n = len(converted)
    n_test = int(n * test_split)
    n_val  = int(n * val_split)

    test_data  = converted[:n_test]
    val_data   = converted[n_test:n_test + n_val]
    train_data = converted[n_test + n_val:]

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        out = output_dir / f"{split}.json"
        with open(out, "w") as f:
            json.dump(data, f, indent=2)
        print(f"{split.capitalize():5s}: {len(data)} samples -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw annotations JSON")
    parser.add_argument("--output", required=True, help="Output directory for processed splits")
    parser.add_argument("--image_root", required=True, help="Root directory containing images")
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--test_split", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_dataset(args.input, args.output, args.image_root, args.val_split, args.test_split, args.seed)