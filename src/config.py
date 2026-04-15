import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

def _require(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(f"Missing required .env variable: {key}")
    return val

CFG = {
    "model": {
        "name": "llava-hf/llava-v1.6-mistral-7b-hf",
        "cache_dir": _require("HF_CACHE_DIR"),
    },
    "lora": {
        "r":             64,
        "lora_alpha":    128,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "lora_dropout":  0.05,
        "bias":          "none",
    },
    "data": {
        "train_path": _require("TRAIN_PATH"),
        "val_path":   _require("VAL_PATH"),
        "test_path":  _require("TEST_PATH"),
        "image_root": _require("IMAGE_ROOT"),
        "max_length": 2048,
    },
    "training": {
        "output_dir":                    _require("CHECKPOINT_DIR"),
        "num_epochs":                    3,
        "per_device_train_batch_size":   4,
        "per_device_eval_batch_size":    4,
        "gradient_accumulation_steps":   8,
        "learning_rate":                 2e-4,
        "lr_scheduler_type":             "cosine",
        "warmup_ratio":                  0.03,
        "weight_decay":                  0.0,
        "fp16":                          False,
        "bf16":                          True,
        "dataloader_num_workers":        4,
        "save_strategy":                 "epoch",
        "eval_strategy":                 "epoch",
        "load_best_model_at_end":        True,
        "metric_for_best_model":         "vqa_accuracy",
        "logging_steps":                 10,
        "report_to":                     "wandb",
    },
    "wandb": {
        "project":  "llava-docvqa",
        "entity":   _require("WANDB_ENTITY"),
        "run_name": "lora-r64-bs32-3ep",
    },
}
