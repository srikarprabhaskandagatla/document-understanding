import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def download(output_dir: str, max_samples: int, split: str = "train"):
    output_dir = Path(output_dir)
    image_dir = output_dir / "raw" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading DocVQA ({split} split, up to {max_samples} samples)...")
    print("This may take a while — images are ~10GB for 15K samples.")

    ds = load_dataset("HuggingFaceM4/DocumentVQA", split=split, streaming=True)

    annotations = []
    for i, sample in enumerate(tqdm(ds, total=max_samples, desc="Downloading")):
        if i >= max_samples:
            break

        img: Image.Image = sample["image"]
        img_filename = f"doc_{i:06d}.png"
        img_path = image_dir / img_filename
        img.save(img_path, format="PNG")

        answers = sample["answers"]
        answer = answers[0] if isinstance(answers, list) else answers

        annotations.append({
            "image": img_filename,
            "question": sample["question"],
            "answer": answer,
        })

    annotations_path = output_dir / "raw_annotations.json"
    with open(annotations_path, "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"\nDone!")
    print(f"  Images saved to : {image_dir}  ({len(annotations)} files)")
    print(f"  Annotations     : {annotations_path}")
    print()
    print("Next — run prepare-data:")
    print(f"  python main.py prepare-data \\")
    print(f"      --input {annotations_path} \\")
    print(f"      --output {output_dir}/processed \\")
    print(f"      --image_root {image_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download DocVQA from HuggingFace")
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Root output directory (e.g. /scratch/YOUR_USERNAME/data)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=15000,
        help="Max number of training samples to download (default: 15000)",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "validation", "test"],
    )
    args = parser.parse_args()
    download(args.output_dir, args.max_samples, args.split)
