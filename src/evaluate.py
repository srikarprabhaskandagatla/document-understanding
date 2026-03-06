"""
evaluate.py
-----------
VQA accuracy and ANLS metric computation.

WHY ANLS (Average Normalized Levenshtein Similarity):
  Official metric from the DocVQA benchmark paper (Mathew et al., 2021).
  Unlike exact match, ANLS:
    1. Is robust to minor OCR errors ("lnvoice" vs "Invoice")
    2. Handles formatting differences ("$1,000" vs "1000")
    3. Uses threshold τ=0.5: similarity below 0.5 counts as 0 (wrong)
  Exact match is also reported because it's interpretable and comparable
  to non-document VQA benchmarks (VQAv2, GQA) which use it exclusively.

WHY NOT USE sacrebleu/ROUGE:
  BLEU and ROUGE are n-gram overlap metrics designed for long-form generation.
  Document VQA answers are 1-5 words. BLEU scores would be near-zero and
  uninformative. ANLS was purpose-built for short, extractive VQA answers.
"""

import re
from Levenshtein import distance as levenshtein_distance


def normalize_answer(answer: str) -> str:
    """
    Normalize answer string for comparison.
    Matches DocVQA benchmark normalization.
    """
    answer = str(answer).lower().strip()
    # Remove articles
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)
    # Normalize whitespace
    answer = " ".join(answer.split())
    # Remove punctuation except hyphen (important for dates/IDs)
    answer = re.sub(r"[^\w\s\-]", "", answer)
    return answer


def compute_anls(prediction: str, ground_truth: str, threshold: float = 0.5) -> float:
    """
    Compute ANLS between a single prediction and ground truth.
    
    ANLS formula:
      1 - (edit_distance / max_length)  if similarity >= threshold
      0                                  otherwise
    
    Threshold τ=0.5 from original DocVQA paper.
    """
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)

    if len(gt_norm) == 0 and len(pred_norm) == 0:
        return 1.0
    if len(gt_norm) == 0 or len(pred_norm) == 0:
        return 0.0

    max_len = max(len(pred_norm), len(gt_norm))
    edit_dist = levenshtein_distance(pred_norm, gt_norm)
    similarity = 1.0 - (edit_dist / max_len)

    return similarity if similarity >= threshold else 0.0


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Binary exact match after normalization."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_vqa_accuracy(
    predictions: list[str],
    ground_truths: list[str],
    metric: str = "anls",
) -> float:
    """
    Compute mean VQA accuracy over a batch.
    
    Args:
        predictions: List of model-generated answer strings
        ground_truths: List of ground truth answer strings
        metric: "anls" (default) or "exact_match"
    
    Returns:
        Float in [0, 1] — the VQA accuracy score
    """
    assert len(predictions) == len(ground_truths), (
        f"Mismatch: {len(predictions)} predictions vs {len(ground_truths)} ground truths"
    )

    if metric == "anls":
        scores = [
            compute_anls(pred, gt)
            for pred, gt in zip(predictions, ground_truths)
        ]
    elif metric == "exact_match":
        scores = [
            compute_exact_match(pred, gt)
            for pred, gt in zip(predictions, ground_truths)
        ]
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'anls' or 'exact_match'.")

    return sum(scores) / len(scores) if scores else 0.0


def evaluate_full_dataset(
    model,
    processor,
    dataloader,
    device: str = "cuda",
    max_new_tokens: int = 64,
) -> dict:
    """
    Run full evaluation loop — used in standalone eval, not during training.
    Training uses compute_metrics() callback instead for efficiency.
    """
    import torch

    model.eval()
    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"]

            # Generate predictions autoregressively
            # WHY NOT TEACHER-FORCING FOR EVAL:
            #   Teacher-forcing (feeding ground truth tokens) measures memorization,
            #   not true generative accuracy. For VQA, we need the model to generate
            #   the answer from scratch given only the question and image.
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=max_new_tokens,
                do_sample=False,       # Greedy decoding — deterministic for benchmarking
                temperature=1.0,
                repetition_penalty=1.1,  # Prevents degenerate repetition loops
            )

            # Decode only the newly generated tokens (not the prompt)
            pred_texts = processor.tokenizer.batch_decode(
                generated_ids[:, input_ids.shape[1]:],
                skip_special_tokens=True,
            )

            # Decode ground truth labels
            labels[labels == -100] = processor.tokenizer.pad_token_id
            gt_texts = processor.tokenizer.batch_decode(
                labels, skip_special_tokens=True
            )

            all_predictions.extend(pred_texts)
            all_ground_truths.extend(gt_texts)

            if (batch_idx + 1) % 50 == 0:
                running_anls = compute_vqa_accuracy(
                    all_predictions, all_ground_truths, metric="anls"
                )
                print(f"  Batch {batch_idx+1}: Running ANLS = {running_anls:.4f}")

    return {
        "anls": compute_vqa_accuracy(all_predictions, all_ground_truths, "anls"),
        "exact_match": compute_vqa_accuracy(all_predictions, all_ground_truths, "exact_match"),
        "num_samples": len(all_predictions),
    }