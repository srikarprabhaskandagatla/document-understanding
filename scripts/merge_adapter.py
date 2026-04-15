import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from peft import PeftModel
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor


def merge(adapter_path: str, output_dir: str, cache_dir: str, base_model_name: str):
    adapter_path = Path(adapter_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate adapter directory
    if not adapter_path.exists():
        print(f"ERROR: adapter_path does not exist: {adapter_path}")
        sys.exit(1)

    adapter_config = adapter_path / "adapter_config.json"
    if not adapter_config.exists():
        print(f"ERROR: No adapter_config.json found in {adapter_path}")
        print("Make sure training completed successfully and save_lora_adapter() was called.")
        sys.exit(1)

    print(f"Adapter path : {adapter_path}")
    print(f"Output dir   : {output_dir}")
    print(f"Base model   : {base_model_name}")
    print(f"Cache dir    : {cache_dir}")
    print()

    # Step 1: Load processor
    print("Loading processor...")
    processor = LlavaNextProcessor.from_pretrained(
        base_model_name,
        cache_dir=cache_dir,
    )

    # Restore the pad token set during training
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    # Step 2: Load base model in bf16 on CPU
    # WHY CPU + bf16 (not 4-bit):
    #   merge_and_unload() performs float arithmetic on base weights.
    #   NF4-quantized weights cannot be merged directly - bitsandbytes
    #   does not support dequant-then-merge in a single pass.
    #   We load the full bf16 base model (~28GB RAM), merge, then save.
    #   This is a one-time offline step, so the extra RAM cost is fine.
    print("Loading base model in bf16 (CPU) - this takes ~2-3 min and ~28GB RAM...")
    base_model = LlavaNextForConditionalGeneration.from_pretrained(
        base_model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        device_map="cpu",          # CPU-only: no GPU required for merging
        low_cpu_mem_usage=True,
    )

    # Step 3: Load LoRA adapter on top of base model
    print("Loading LoRA adapter...")
    peft_model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        torch_dtype=torch.bfloat16,
    )

    # Step 4: Merge adapter weights into base model and unload PEFT
    print("Merging LoRA weights into base model (W = W_base + A @ B * alpha/r)...")
    merged_model = peft_model.merge_and_unload()

    # Step 5: Save merged model + processor to output_dir
    print(f"Saving merged model to {output_dir} ...")
    merged_model.save_pretrained(
        str(output_dir),
        safe_serialization=True,   # .safetensors format
    )
    processor.save_pretrained(str(output_dir))

    # Step 6: Sanity check - verify the saved model loads cleanly
    print("Verifying saved model loads correctly...")
    _ = LlavaNextForConditionalGeneration.from_pretrained(
        str(output_dir),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    _ = LlavaNextProcessor.from_pretrained(str(output_dir))
    print("Verification passed.")

    # Print output contents for reference
    saved_files = sorted(output_dir.iterdir())
    total_size_gb = sum(f.stat().st_size for f in saved_files if f.is_file()) / 1e9
    print()
    print(f"Merged model saved to: {output_dir}")
    print(f"Total size: {total_size_gb:.2f} GB")
    print("Files:")
    for f in saved_files:
        if f.is_file():
            print(f"  {f.name:50s}  {f.stat().st_size / 1e6:.1f} MB")
    print()
    print("Next step - upload to S3 for SageMaker deployment:")
    print(f"  aws s3 cp --recursive {output_dir}/ s3://YOUR_BUCKET/llava-docvqa/merged_model/")
    print()
    print("Then deploy:")
    print("  python deploy/sagemaker_deploy.py \\")
    print(f"      --model_s3_uri s3://YOUR_BUCKET/llava-docvqa/merged_model/ \\")
    print("      --endpoint_name llava-docvqa-v1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into base LLaVA-1.6 model for SageMaker deployment."
    )
    parser.add_argument(
        "--adapter_path",
        required=True,
        help="Path to saved LoRA adapter directory (contains adapter_config.json)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Where to save the merged full model",
    )
    parser.add_argument(
        "--cache_dir",
        default="/scratch/YOUR_USERNAME/hf_cache",
        help="HuggingFace cache directory (where base model weights are cached)",
    )
    parser.add_argument(
        "--base_model",
        default="llava-hf/llava-v1.6-mistral-7b-hf",
        help="Base model name (must match what was used during training)",
    )
    args = parser.parse_args()

    merge(args.adapter_path, args.output_dir, args.cache_dir, args.base_model)
