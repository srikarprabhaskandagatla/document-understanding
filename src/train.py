import functools
import os

import torch
import wandb
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments

from dataset.dataset import DocVQADataset, collate_fn
from evaluate import compute_vqa_accuracy, evaluate_full_dataset
from model import load_model_with_lora, load_processor, save_lora_adapter


def main(config_path: str = None):
    from config import CFG as cfg

    wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"]["entity"],
        name=cfg["wandb"]["run_name"],
        config=cfg,
    )

    # Load Processor & Model 
    processor = load_processor(cfg["model"]["name"], cfg["model"]["cache_dir"])

    model = load_model_with_lora(
        cfg["model"]["name"],
        cfg["model"]["cache_dir"],
        cfg["lora"],
        use_4bit=True,
    )

    # Datasets
    train_dataset = DocVQADataset(
        data_path=cfg["data"]["train_path"],
        image_root=cfg["data"]["image_root"],
        processor=processor,
        max_length=cfg["data"]["max_length"],
        split="train",
    )

    # Val: used every epoch for checkpoint selection (load_best_model_at_end)
    val_dataset = DocVQADataset(
        data_path=cfg["data"]["val_path"],
        image_root=cfg["data"]["image_root"],
        processor=processor,
        max_length=cfg["data"]["max_length"],
        split="val",
    )

    # Test: held out — only evaluated once after training is done
    test_dataset = DocVQADataset(
        data_path=cfg["data"]["test_path"],
        image_root=cfg["data"]["image_root"],
        processor=processor,
        max_length=cfg["data"]["max_length"],
        split="test",
    )

    data_collator = functools.partial(
        collate_fn,
        pad_token_id=processor.tokenizer.pad_token_id,
    )

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
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        pred_texts = processor.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels[labels == -100] = processor.tokenizer.pad_token_id
        label_texts = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)

        accuracy = compute_vqa_accuracy(pred_texts, label_texts)
        wandb.log({"vqa_accuracy": accuracy})
        return {"vqa_accuracy": accuracy}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    final_adapter_path = os.path.join(cfg["training"]["output_dir"], "final_adapter")
    save_lora_adapter(model, final_adapter_path)

    # Final Test Set Evaluation 
    print("\nRunning final evaluation on test set...")
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["training"]["per_device_eval_batch_size"],
        collate_fn=functools.partial(collate_fn, pad_token_id=processor.tokenizer.pad_token_id),
        num_workers=cfg["training"]["dataloader_num_workers"],
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_results = evaluate_full_dataset(model, processor, test_loader, device=device)

    print("\n=== Final Test Set Results ===")
    print(f"  ANLS        : {test_results['anls']:.4f}")
    print(f"  Exact Match : {test_results['exact_match']:.4f}")
    print(f"  Samples     : {test_results['num_samples']}")

    wandb.log({
        "test/anls":        test_results["anls"],
        "test/exact_match": test_results["exact_match"],
        "test/num_samples": test_results["num_samples"],
    })

    wandb.finish()
    print(f"\nTraining complete. Adapters saved to {final_adapter_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()
    main(args.config)
