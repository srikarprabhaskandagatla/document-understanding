"""
model.py
--------
Loads LLaVA-1.6 and applies LoRA via PEFT.

WHY PEFT (Parameter-Efficient Fine-Tuning) LIBRARY:
  PEFT is the ONLY library that correctly handles LoRA for multi-modal models.
  It knows which modules are attention projections vs vision encoder vs MLP,
  and applies adapters only where specified. Implementing this manually
  would require patching PyTorch module trees — fragile and unmaintainable.

WHY FREEZE THE VISION ENCODER:
  CLIP's vision encoder (used in LLaVA) is already excellent at document
  feature extraction. Fine-tuning it on 15K samples risks catastrophic
  forgetting of its visual priors. The language model backbone is where
  document-specific reasoning (layout understanding, OCR comprehension)
  needs to improve — so LoRA targets only those layers.

WHY 4-BIT QUANTIZATION FOR LOADING (QLoRA):
  Even with LoRA, the base model weights must be loaded into VRAM.
  LLaVA-1.6 in bf16 = ~14GB. With activations + optimizer states for LoRA
  adapters, total VRAM requirement reaches ~35GB on an 80GB A100, leaving
  comfortable headroom for batch size 4 with gradient accumulation.
  We use bitsandbytes NF4 quantization ONLY for the frozen base weights.
  LoRA adapters themselves train in full bf16 precision.
"""

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, LlavaNextForConditionalGeneration, LlavaNextProcessor


def load_processor(model_name: str, cache_dir: str) -> LlavaNextProcessor:
    """
    Load the LLaVA processor (combines CLIP image processor + Mistral tokenizer).
    
    WHY SET pad_token = eos_token:
      Mistral tokenizer has no dedicated pad token. Using EOS as pad with
      attention_mask ensures padded positions are masked out — the model
      never sees or attends to them. This is standard for Mistral-based models.
    """
    processor = LlavaNextProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir,
    )

    # Mistral has no pad token by default
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    return processor


def load_model_with_lora(
    model_name: str,
    cache_dir: str,
    lora_config_dict: dict,
    use_4bit: bool = True,
) -> LlavaNextForConditionalGeneration:
    """
    Load LLaVA-1.6 with QLoRA (4-bit base + bf16 LoRA adapters).
    """

    # --- Quantization Config ---
    # NF4 (Normal Float 4) is superior to standard INT4 for LLM weights
    # because LLM weight distributions are approximately normal/Gaussian.
    # double_quant further compresses quantization constants (saves ~0.4GB).
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # A100 bf16 tensor cores
        bnb_4bit_use_double_quant=True,
    ) if use_4bit else None

    print(f"Loading base model: {model_name}")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Distributes across available GPUs automatically
        low_cpu_mem_usage=True,  # Stream weights from disk instead of full RAM load
    )

    # Required step for QLoRA: prepares quantized layers for gradient computation
    # Sets requires_grad=False for all base weights, enables gradient checkpointing
    if use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )

    # --- LoRA Config ---
    lora_config = LoraConfig(
        r=lora_config_dict["r"],
        lora_alpha=lora_config_dict["lora_alpha"],
        target_modules=lora_config_dict["target_modules"],
        lora_dropout=lora_config_dict["lora_dropout"],
        bias=lora_config_dict["bias"],
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    # Print trainable parameter summary
    model.print_trainable_parameters()

    return model


def save_lora_adapter(model, output_dir: str):
    """
    Save ONLY the LoRA adapter weights (not the full model).
    
    WHY SAVE ONLY ADAPTERS:
      The full model is 14GB. LoRA adapters are ~168MB for r=64.
      During inference, adapters are merged back into the base model.
      This makes checkpoint storage 83x smaller, critical on Unity's
      scratch filesystem which has quota limits.
    """
    model.save_pretrained(output_dir)
    print(f"LoRA adapters saved to {output_dir}")


def merge_and_save_full_model(model, processor, output_dir: str):
    """
    For final deployment: merge LoRA weights into base model and save.
    This produces a standard HuggingFace model loadable without PEFT.
    Required for SageMaker deployment (SageMaker doesn't support PEFT natively).
    """
    print("Merging LoRA adapters into base model...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    processor.save_pretrained(output_dir)
    print(f"Full merged model saved to {output_dir}")