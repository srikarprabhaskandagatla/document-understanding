import json
import os
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import LlavaNextProcessor


class DocVQADataset(Dataset):
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

        image_path = os.path.join(self.image_root, sample["image"])
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[ERROR] Failed to load image {image_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.data))

        conversations = sample["conversations"]
        human_text = conversations[0]["value"]
        assistant_text = conversations[1]["value"]

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

        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=False
        )

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs["pixel_values"]  # keep (num_tiles, C, H, W) — do NOT squeeze
        image_sizes = inputs["image_sizes"]    # (1, 2) original (H, W) — required by LlavaNext

        labels = input_ids.clone()
        labels = self._mask_non_answer_tokens(labels, assistant_text)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_sizes": image_sizes,
            "labels": labels,
        }

    def _mask_non_answer_tokens(
        self, labels: torch.Tensor, assistant_text: str
    ) -> torch.Tensor:
        answer_ids = self.processor.tokenizer(
            assistant_text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        answer_len = len(answer_ids)

        if answer_len < len(labels):
            labels[: len(labels) - answer_len] = self.IGNORE_INDEX

        return labels


def collate_fn(batch: list[dict], pad_token_id: int) -> dict:
    max_len = max(item["input_ids"].shape[0] for item in batch)

    padded_input_ids = []
    padded_attention_masks = []
    padded_labels = []
    pixel_values = []
    image_sizes = []

    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_len - seq_len

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
        pixel_values.append(item["pixel_values"].squeeze(0))  # (num_tiles, C, H, W)
        image_sizes.append(item["image_sizes"].squeeze(0))    # (2,)

    # Pad all images to the same number of tiles so we can stack into a tensor
    max_tiles = max(pv.shape[0] for pv in pixel_values)
    C, H, W = pixel_values[0].shape[1], pixel_values[0].shape[2], pixel_values[0].shape[3]
    padded_pixel_values = []
    for pv in pixel_values:
        num_tiles = pv.shape[0]
        if num_tiles < max_tiles:
            pad = torch.zeros(max_tiles - num_tiles, C, H, W, dtype=pv.dtype)
            pv = torch.cat([pv, pad], dim=0)
        padded_pixel_values.append(pv)

    return {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_masks),
        "labels": torch.stack(padded_labels),
        "pixel_values": torch.stack(padded_pixel_values),  # (B, max_tiles, C, H, W)
        "image_sizes": torch.stack(image_sizes),            # (B, 2)
    }
