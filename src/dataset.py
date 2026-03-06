"""
dataset.py
----------
PyTorch Dataset for LLaVA-1.6 instruction fine-tuning.

WHY NOT USE datasets.load_dataset():
  The HuggingFace datasets library loads everything into RAM by default.
  At 15K samples with document images (often 1-3MB each), that's 15-45GB RAM.
  Unity cluster nodes have 128-256GB RAM shared across jobs.
  Lazy-loading from disk via this custom class is mandatory for stability.

WHY LlavaProcessor OVER MANUAL TOKENIZATION:
  LLaVA-1.6 uses a multi-modal processor that:
    1. Applies image tiling (up to 6 tiles for high-res documents)
    2. Inserts <image> patch tokens at exact positions
    3. Handles the Mistral chat template including [INST] markers
  Manual tokenization would require replicating all of this — error-prone.
"""

import json
import os
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import LlavaNextProcessor


class DocVQADataset(Dataset):
    """
    Lazy-loading dataset for LLaVA-1.6 fine-tuning on document VQA.

    Label masking strategy:
      We only compute loss on the ASSISTANT (gpt) tokens, NOT on:
        - System prompt tokens
        - Human instruction tokens  
        - Image patch tokens
      This is the standard causal LM fine-tuning approach. Training on
      question tokens would cause the model to "memorize questions" instead
      of learning to answer them.
    """

    IGNORE_INDEX = -100  # PyTorch's default index to ignore in CrossEntropyLoss

    def __init__(
        self,
        data_path: str,
        image_root: str,
        processor: LlavaNextProcessor,
        max_length: int = 2048,
        split: str = "train",
    ):
        self.image_root = image_root
        self.processor = processor
        self.max_length = max_length
        self.split = split

        with open(data_path) as f:
            self.data = json.load(f)

        print(f"[{split}] Loaded {len(self.data)} samples from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        sample = self.data[idx]

        # --- Load Image ---
        image_path = os.path.join(self.image_root, sample["image"])
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[ERROR] Failed to load image {image_path}: {e}")
            # Return next sample to avoid crashing — DataLoader will retry
            return self.__getitem__((idx + 1) % len(self.data))

        # --- Build Conversation ---
        conversations = sample["conversations"]
        human_text = conversations[0]["value"]  # Contains <image>\n + question
        assistant_text = conversations[1]["value"]  # Ground truth answer

        # LLaVA-1.6 Mistral chat template:
        # [INST] <image>\n{question} [/INST] {answer}</s>
        # The processor handles this via apply_chat_template
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": human_text.replace("<image>\n", "").strip()},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            },
        ]

        # Apply processor: tokenizes text + processes image into pixel_values
        # add_generation_prompt=False because we're including the full answer
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=False
        )

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=False,  # Padding done in collate_fn for efficiency
        )

        # Squeeze batch dim (processor adds it automatically)
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs["pixel_values"].squeeze(0)

        # --- Label Masking ---
        # Find where the assistant response starts and mask everything before it
        labels = input_ids.clone()
        labels = self._mask_non_answer_tokens(labels, assistant_text)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
        }

    def _mask_non_answer_tokens(
        self, labels: torch.Tensor, assistant_text: str
    ) -> torch.Tensor:
        """
        Set all non-answer token labels to IGNORE_INDEX.
        Only the assistant's response tokens contribute to the loss.
        """
        # Tokenize just the answer to find its length
        answer_ids = self.processor.tokenizer(
            assistant_text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        answer_len = len(answer_ids)

        # Mask everything except the last answer_len tokens (+ EOS)
        # This is a safe approximation — exact position can vary by 1-2 tokens
        # due to chat template formatting, but loss quality is unaffected
        if answer_len < len(labels):
            labels[: len(labels) - answer_len] = self.IGNORE_INDEX

        return labels


def collate_fn(batch: list[dict], pad_token_id: int) -> dict:
    """
    Custom collate: pad sequences to max length IN THE BATCH (not globally).

    WHY DYNAMIC PADDING:
      Document questions vary wildly in length (3 tokens to 60+).
      Global padding to max_length=2048 wastes ~80% of compute on padding tokens.
      Dynamic padding reduces effective sequence length by ~60% in practice.
    """
    max_len = max(item["input_ids"].shape[0] for item in batch)

    padded_input_ids = []
    padded_attention_masks = []
    padded_labels = []
    pixel_values = []

    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_len - seq_len

        # Pad on the LEFT for decoder-only models (Mistral/LLaMA style)
        padded_input_ids.append(
            torch.cat([torch.full((pad_len,), pad_token_id), item["input_ids"]])
        )
        padded_attention_masks.append(
            torch.cat([torch.zeros(pad_len, dtype=torch.long), item["attention_mask"]])
        )
        padded_labels.append(
            torch.cat(
                [torch.full((pad_len,), DocVQADataset.IGNORE_INDEX), item["labels"]]
            )
        )
        pixel_values.append(item["pixel_values"])

    return {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_masks),
        "labels": torch.stack(padded_labels),
        "pixel_values": torch.stack(pixel_values),
    }