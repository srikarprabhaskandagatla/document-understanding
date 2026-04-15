import re
from Levenshtein import distance as levenshtein_distance


def normalize_answer(answer: str) -> str:
    answer = str(answer).lower().strip()
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)
    answer = " ".join(answer.split())
    answer = re.sub(r"[^\w\s\-]", "", answer)
    return answer


def compute_anls(prediction: str, ground_truth: str, threshold: float = 0.5) -> float:
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
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_vqa_accuracy(
    predictions: list[str],
    ground_truths: list[str],
    metric: str = "anls",
) -> float:
    assert len(predictions) == len(ground_truths), (
        f"Mismatch: {len(predictions)} predictions vs {len(ground_truths)} ground truths"
    )

    if metric == "anls":
        scores = [compute_anls(pred, gt) for pred, gt in zip(predictions, ground_truths)]
    elif metric == "exact_match":
        scores = [compute_exact_match(pred, gt) for pred, gt in zip(predictions, ground_truths)]
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
    Training uses compute_metrics() callback in train.py for efficiency.
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

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                repetition_penalty=1.1,
            )

            pred_texts = processor.tokenizer.batch_decode(
                generated_ids[:, input_ids.shape[1]:],
                skip_special_tokens=True,
            )

            labels[labels == -100] = processor.tokenizer.pad_token_id
            gt_texts = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)

            all_predictions.extend(pred_texts)
            all_ground_truths.extend(gt_texts)

            if (batch_idx + 1) % 50 == 0:
                running_anls = compute_vqa_accuracy(all_predictions, all_ground_truths, metric="anls")
                print(f"  Batch {batch_idx+1}: Running ANLS = {running_anls:.4f}")

    return {
        "anls": compute_vqa_accuracy(all_predictions, all_ground_truths, "anls"),
        "exact_match": compute_vqa_accuracy(all_predictions, all_ground_truths, "exact_match"),
        "num_samples": len(all_predictions),
    }
