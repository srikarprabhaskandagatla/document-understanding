import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, LlavaNextForConditionalGeneration, LlavaNextProcessor


def load_processor(model_name: str, cache_dir: str) -> LlavaNextProcessor:
    processor = LlavaNextProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir,
    )

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
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ) if use_4bit else None

    print(f"Loading base model: {model_name}")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    if use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )

    lora_config = LoraConfig(
        r=lora_config_dict["r"],
        lora_alpha=lora_config_dict["lora_alpha"],
        target_modules=lora_config_dict["target_modules"],
        lora_dropout=lora_config_dict["lora_dropout"],
        bias=lora_config_dict["bias"],
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def save_lora_adapter(model, output_dir: str):
    model.save_pretrained(output_dir)
    print(f"LoRA adapters saved to {output_dir}")


def merge_and_save_full_model(model, processor, output_dir: str):
    print("Merging LoRA adapters into base model...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    processor.save_pretrained(output_dir)
    print(f"Full merged model saved to {output_dir}")
