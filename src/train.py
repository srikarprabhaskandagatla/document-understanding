"""
train.py
--------
Main training entry point for LLaVA-1.6 LoRA fine-tuning.

Run locally (single GPU):
  python src/train.py --config configs/train_config.yaml

Run on Unity (via SLURM — see scripts/submit_train.sh):
  sbatch scripts/submit_train.sh
"""

import argparse
import os
import functools
from pathlib import Path

import torch
import wandb
import yaml
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader

from dataset import DocVQADataset, collate_fn
from model import load_model_with_lora, load_processor, save_lora_adapter
from evaluate import compute_vqa_accuracy


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main(config_path: str):
    cfg = load_config(config_path)

    # --- W&B Init ---
    # WHY W&B OVER TENSORBOARD:
    #   Unity jobs are headless — no browser on compute nodes.
    #   W&B streams metrics to cloud in real-time, viewable from your laptop.
    #   It also auto-logs GPU utilization, memory, and system metrics,
    #   critical for debugging OOM issues on cluster jobs you can't interactively monitor.
    wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"]["entity"],
        name=cfg["wandb"]["run_name"],
        config=cfg,
    )

    # --- Load Processor & Model ---
    processor = load_processor(
        cfg["model"]["name"],
        cfg["model"]["cache_dir"],
    )

    model = load_model_with_lora(
        cfg["model"]["name"],
        cfg["model"]["cache_dir"],
        cfg["lora"],
        use_4bit=True,
    )

    # --- Datasets ---
    train_dataset = DocVQADataset(
        data_path=cfg["data"]["train_path"],
        image_root=cfg["data"]["image_root"],
        processor=processor,
        max_length=cfg["data"]["max_length"],
        split="train",
    )

    val_dataset = DocVQADataset(
        data_path=cfg["data"]["val_path"],
        image_root=cfg["data"]["image_root"],
        processor=processor,
        max_length=cfg["data"]["max_length"],
        split="val",
    )

    # Partial collate_fn with pad_token_id bound
    data_collator = functools.partial(
        collate_fn,
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    # --- Training Arguments ---
    # WHY TrainingArguments OVER MANUAL OPTIMIZER SETUP:
    #   Handles bf16 scaler, gradient clipping, LR scheduling, and FSDP/DDP
    #   configuration from environment variables set by SLURM automatically.
    training_args = TrainingArguments(
        output_dir=cfg["training"]["output_dir"],
        num_train_epochs=cfg["training"]["num_epochs"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        weight_decay=cfg["training"]["weight_decay"],
        bf16=cfg["training"]["bf16"],
        fp16=cfg["training"]["fp16"],
        dataloader_num_workers=cfg["training"]["dataloader_num_workers"],
        save_strategy=cfg["training"]["save_strategy"],
        evaluation_strategy=cfg["training"]["eval_strategy"],
        load_best_model_at_end=cfg["training"]["load_best_model_at_end"],
        metric_for_best_model=cfg["training"]["metric_for_best_model"],
        logging_steps=cfg["training"]["logging_steps"],
        report_to=cfg["training"]["report_to"],
        remove_unused_columns=False,  # CRITICAL: must keep pixel_values column
        ddp_find_unused_parameters=False,  # LoRA leaves some base params unused; set False to avoid hang
        dataloader_pin_memory=True,  # Speeds up CPU->GPU transfer for image batches
        gradient_checkpointing=True,  # Recompute activations during backward; saves ~30% VRAM
    )

    # --- Custom compute_metrics for VQA accuracy ---
    def compute_metrics(eval_pred):
        """
        HuggingFace Trainer calls this after each eval epoch.
        Returns dict of metric_name -> value for W&B logging.
        """
        predictions, labels = eval_pred

        # Decode predicted token ids to strings
        pred_texts = processor.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )

        # Decode label token ids (replace IGNORE_INDEX with pad before decoding)
        labels[labels == -100] = processor.tokenizer.pad_token_id
        label_texts = processor.tokenizer.batch_decode(
            labels, skip_special_tokens=True
        )

        accuracy = compute_vqa_accuracy(pred_texts, label_texts)
        wandb.log({"vqa_accuracy": accuracy})
        return {"vqa_accuracy": accuracy}

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # --- Train ---
    print("Starting training...")
    trainer.train()

    # --- Save Adapters ---
    final_adapter_path = os.path.join(cfg["training"]["output_dir"], "final_adapter")
    save_lora_adapter(model, final_adapter_path)

    wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()
    main(args.config)