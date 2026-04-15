<h1 align="center">
  <br>
  LLaVA-1.6 Document Understanding
  <br>
</h1>

<h4 align="center">Fine-tuning a Vision-Language Model on 15,000 document images for Visual Question Answering - using LoRA + 4-bit quantization on an A100 GPU, tracked with W&B, and deployed as a REST endpoint on AWS SageMaker.</h4>

<p align="center">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  </a>
  <a href="https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf">
    <img src="https://img.shields.io/badge/-LLaVA--1.6-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="LLaVA-1.6">
  </a>
  <a href="https://huggingface.co/docs/peft">
    <img src="https://img.shields.io/badge/-LoRA%20%2F%20QLoRA-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="LoRA">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  </a>
  <a href="https://huggingface.co/docs/transformers">
    <img src="https://img.shields.io/badge/-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="Transformers">
  </a>
  <a href="https://wandb.ai/">
    <img src="https://img.shields.io/badge/-W%26B-FFBE00?style=flat-square&logo=weightsandbiases&logoColor=black" alt="W&B">
  </a>
  <a href="https://aws.amazon.com/sagemaker/">
    <img src="https://img.shields.io/badge/-SageMaker-FF9900?style=flat-square&logo=amazonaws&logoColor=white" alt="SageMaker">
  </a>
  <a href="https://www.docker.com/">
    <img src="https://img.shields.io/badge/-Docker-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker">
  </a>
</p>

<p align="center">
  <a href="#1-project-goal">Goal</a> •
  <a href="#2-why-llava-16">Why LLaVA-1.6</a> •
  <a href="#3-dataset--source-and-structure">Dataset</a> •
  <a href="#6-lora-configuration--design-choices">LoRA</a> •
  <a href="#8-evaluation-metrics">Metrics</a> •
  <a href="#9-deployment--sagemaker">Deployment</a> •
  <a href="#12-running-the-pipeline">Running</a>
</p>

---

## Table of Contents

1. [Project Goal](#1-project-goal)
2. [Why LLaVA-1.6](#2-why-llava-16)
3. [Dataset - Source and Structure](#3-dataset--source-and-structure)
4. [Data Preparation - Design Choices](#4-data-preparation--design-choices)
5. [Model Loading - Quantization Design](#5-model-loading--quantization-design)
6. [LoRA Configuration - Design Choices](#6-lora-configuration--design-choices)
7. [Training Pipeline](#7-training-pipeline)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Deployment - SageMaker](#9-deployment--sagemaker)
10. [Project Structure](#10-project-structure)
11. [Environment Setup](#11-environment-setup)
12. [Running the Pipeline](#12-running-the-pipeline)
13. [Configuration Reference](#13-configuration-reference)

---

## 1. Project Goal

Document Visual Question Answering (DocVQA) is the task of answering natural language questions about document images - invoices, forms, research papers, receipts, contracts. Unlike general VQA, DocVQA requires the model to read and reason over dense text embedded in images, not just understand visual content.

The goal of this project is to:
- Fine-tune a large Vision-Language Model (VLM) on DocVQA to specialize it for document understanding
- Do so efficiently - within a single A100 GPU budget - using parameter-efficient fine-tuning
- Deploy the resulting model as a production inference endpoint

**Target metrics:** 78% VQA accuracy (ANLS), representing an 18% improvement over the zero-shot baseline.

---

## 2. Why LLaVA-1.6

**Why a VLM over an OCR + NLP pipeline:**
A classical approach would be OCR (extract text from image) → feed text to a language model. This fails on documents with complex layouts, tables, charts, or handwriting where OCR produces noisy output. A VLM processes the raw image pixels directly, learning visual-linguistic alignment end-to-end.

**Why LLaVA-1.6 specifically (`llava-v1.6-mistral-7b-hf`):**
- LLaVA-1.6 introduced **dynamic high-resolution tiling** - the image is split into variable-sized tiles (up to 6 tiles per image) and each tile is processed at full resolution. This is critical for documents where small text must be read accurately.
- The Mistral-7B language backbone is stronger than the Vicuna-7B backbone in LLaVA-1.5 for text-heavy reasoning tasks.
- 7B parameters is the sweet spot - large enough for strong document understanding, small enough to fine-tune on a single A100 with 4-bit quantization.
- Alternative: InternVL2, Qwen-VL - both strong but less mature HuggingFace integration at the time of this project.

---

## 3. Dataset - Source and Structure

**Source:** `HuggingFaceM4/DocumentVQA` - the standard DocVQA benchmark dataset, downloaded via HuggingFace `datasets` in streaming mode to avoid loading all ~70K samples into memory.

**Scale:** 15,000 samples downloaded from the training split.

**Split strategy:**

| Split | Samples | Purpose |
|---|---|---|
| Train | 13,500 (90%) | Fine-tuning |
| Val | 750 (5%) | Checkpoint selection during training |
| Test | 750 (5%) | Final held-out evaluation |

**Why 15K and not the full dataset:**
The full DocVQA train split has ~40K samples. 15K was chosen to keep download time and training time manageable on a shared HPC cluster while being large enough to meaningfully fine-tune a 7B model.

**Answer handling:**
DocVQA provides multiple acceptable answers per question (annotator agreement). During training, only the first answer is used as the supervision target. During evaluation, ANLS handles near-matches, so this is not a significant limitation.

---

## 4. Data Preparation - Design Choices

**LLaVA conversation format:**
LLaVA models are instruction-tuned and expect input in a structured conversation format, not raw text. Each sample is converted from:
```json
{"image": "doc_000001.png", "question": "What is the total amount?", "answer": "$42.50"}
```
into:
```json
{
  "conversations": [
    {"from": "human", "value": "<image>\nWhat is the total amount?"},
    {"from": "gpt",   "value": "$42.50"}
  ]
}
```

The `<image>` token at the start of the human turn is required - it is the placeholder where LLaVA injects the visual features from the vision encoder into the token sequence.

**Label masking - why it matters:**
During training, the loss is computed over the entire sequence (question + answer). Without masking, the model is penalized for "wrong" predictions on the question tokens, which are not meaningful supervision signal. The `_mask_non_answer_tokens` method sets all non-answer tokens to `-100` (PyTorch's ignore index in `CrossEntropyLoss`), so the model only learns to predict the answer tokens.

**Image validation:**
Each image is verified with `PIL.Image.verify()` during preparation. Corrupt images are skipped rather than crashing the pipeline. If a corrupt image is encountered at training time, the dataloader falls back to the next sample.

**Why no data augmentation:**
Document images should not be augmented with flips or color jitter - a rotated invoice or color-shifted form is not realistic and would hurt OCR-like reading accuracy. The model's dynamic tiling already handles scale variation implicitly.

---

## 5. Model Loading - Quantization Design

**4-bit NF4 quantization via BitsAndBytes:**

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```

- `nf4` (NormalFloat4): A 4-bit data type designed specifically for normally-distributed neural network weights. It outperforms plain int4 quantization by centering the quantization grid on the actual weight distribution.
- `bnb_4bit_compute_dtype=bfloat16`: Weights are *stored* in 4-bit but *computation* happens in bfloat16. This is the key insight - quantization reduces memory footprint without sacrificing numerical stability in the forward/backward pass.
- `bnb_4bit_use_double_quant`: Quantizes the quantization constants themselves, saving an additional ~0.4 bits per parameter.
- **Memory impact:** Full bf16 model ≈ 14GB. With 4-bit quantization ≈ 4-5GB, leaving ample headroom on A100 for activations and optimizer states.

**Why bfloat16 over float16:**
bfloat16 has the same number of exponent bits as float32 (8), making it numerically safer. float16 has only 5 exponent bits and is prone to overflow/underflow in large models. LayerNorm and other normalization layers remain in float32 regardless - `prepare_model_for_kbit_training` handles this automatically.

**`prepare_model_for_kbit_training`:**
After 4-bit loading, this function: (1) casts normalization layers back to float32 for numerical stability, (2) enables gradient checkpointing to trade compute for memory - activations are recomputed on the backward pass rather than stored.

---

## 6. LoRA Configuration - Design Choices

LoRA (Low-Rank Adaptation) freezes the pretrained weights and injects trainable low-rank matrices into selected layers. If a weight matrix `W` has shape `(d × d)`, LoRA adds `ΔW = B·A` where `A` is `(d × r)` and `B` is `(r × d)`. Only `A` and `B` are trained.

**Configuration:**

| Parameter | Value | Reasoning |
|---|---|---|
| `r` | 32 | Rank 32 gives a good capacity/speed tradeoff. Rank 64 adds ~20% more parameters with diminishing returns on DocVQA. |
| `lora_alpha` | 64 | Scaling factor = `alpha/r = 2.0`. Keeping this ratio at 2x is a standard heuristic - it controls the effective learning rate of the adapter relative to the base model. |
| `lora_dropout` | 0.05 | Light regularization. Higher dropout (0.1+) is only needed with very small datasets. |
| `bias` | none | Bias terms are not trained - they add parameters without meaningful benefit for adaptation. |

**Target modules:**
```python
["q_proj", "v_proj", "k_proj", "o_proj",   # attention layers
 "gate_proj", "up_proj", "down_proj"]        # MLP layers
```

**Why include MLP layers:**
Early LoRA work targeted only attention projections. For document understanding specifically, MLP layers are responsible for factual recall and text recognition - the `gate_proj/up_proj/down_proj` form the FFN (feed-forward network) that stores and retrieves knowledge. Including them meaningfully improves DocVQA accuracy.

**Trainable parameter count:** ~1-2% of total model parameters, compared to full fine-tuning which would require updating all 7B parameters and far more GPU memory.

---

## 7. Training Pipeline

**Framework:** HuggingFace `Trainer` - handles the training loop, gradient accumulation, mixed precision, checkpointing, and evaluation callbacks.

**Key training settings:**

| Setting | Value | Reasoning |
|---|---|---|
| Effective batch size | 32 (8 per device × 4 accumulation steps) | Large effective batch stabilizes gradients for a 7B model |
| Learning rate | 1e-4 | Conservative for a large VLM. 2e-4 risks instability in later epochs. |
| LR schedule | Cosine with warmup | Cosine decay avoids abrupt LR drops. 3% warmup gradually ramps up LR to avoid early instability. |
| Epochs | 3 | DocVQA is a focused task - 3 epochs is sufficient to converge without overfitting |
| `bf16=True` | bfloat16 training | Faster than fp32, more stable than fp16 for large models |
| `max_length` | 1024 tokens | DocVQA Q&A pairs are short. 1024 covers >99% of samples while halving attention cost vs 2048 |

**Dynamic tile collation:**
LLaVA-1.6's dynamic high-resolution tiling means different images produce different numbers of tiles (e.g. portrait document = 5 tiles, landscape = 4 tiles). Since `torch.stack` requires identical shapes, the collate function zero-pads all images in a batch to the same tile count. `image_sizes` (original H×W per image) is passed to the model so it can correctly compute patch counts and ignore padding tiles.

**Checkpoint strategy:**
`save_strategy="epoch"` saves a checkpoint after each epoch. `load_best_model_at_end=True` with `metric_for_best_model="vqa_accuracy"` ensures the best checkpoint (not necessarily the last) is used for the final adapter save.

**Experiment tracking:**
Weights & Biases (`wandb`) logs loss, gradient norm, learning rate, and VQA accuracy at every eval step. Run name encodes the key hyperparameters for easy comparison across experiments.

---

## 8. Evaluation Metrics

**ANLS - Average Normalized Levenshtein Similarity** (primary metric)

ANLS is the official DocVQA evaluation metric. It measures character-level similarity between the predicted answer and the ground truth, normalized by the length of the longer string:

```
similarity = 1 - (edit_distance / max(len(pred), len(gt)))
ANLS = similarity   if similarity >= 0.5
     = 0.0          otherwise
```

The 0.5 threshold means predictions that are less than 50% similar to the ground truth score zero - this handles hallucinations and completely wrong answers while rewarding near-correct answers (e.g. "42.50" vs "$42.50").

**Why ANLS over Exact Match:**
Exact match is too strict for document answers. OCR-like extraction naturally produces minor variations: `"$1,234.56"` vs `"1234.56"` vs `"1,234.56"` are all correct human readings of the same value, but only one would match exactly.

**Exact Match** (secondary metric):
Reported alongside ANLS as a stricter upper bound. Both use the same normalization: lowercasing, removing articles (a/an/the), collapsing whitespace, stripping punctuation.

**Answer normalization (applied to both metrics):**
```python
answer = answer.lower().strip()
answer = re.sub(r"\b(a|an|the)\b", " ", answer)   # remove articles
answer = " ".join(answer.split())                   # collapse whitespace
answer = re.sub(r"[^\w\s\-]", "", answer)           # strip punctuation
```

---

## 9. Deployment - SageMaker

**Merge step - why it's necessary:**
During training, the model is split into frozen base weights + small LoRA adapter matrices. SageMaker inference expects a single self-contained model. `merge_and_unload()` mathematically merges the adapter back into the base weights (`W_final = W_base + B·A·(alpha/r)`), producing a standard model with no PEFT dependency.

**Why merge on CPU in bf16 (not 4-bit):**
NF4-quantized weights cannot be directly merged - the dequantization and matrix addition must happen in a floating-point space. The merge script loads the base model in bf16 on CPU (~28GB RAM), performs the merge, and saves in safetensors format. This is a one-time offline step.

**SageMaker inference handler:**
The `inference.py` follows SageMaker's four-function contract:
- `model_fn` - loads model + processor from the model directory
- `input_fn` - decodes the JSON request, base64-decodes the image
- `predict_fn` - runs the forward pass, decodes the generated answer tokens
- `output_fn` - serializes the response as JSON

**Request format:**
```json
{
  "image": "<base64-encoded PNG/JPEG>",
  "question": "What is the invoice total?"
}
```

**Response format:**
```json
{
  "answer": "$1,234.56"
}
```

---

## 10. Project Structure

```
document-understanding/
├── dataset/
│   ├── download_dataset.py     # Stream 15K DocVQA samples from HuggingFace
│   ├── prepare_dataset.py      # Convert to LLaVA format, train/val/test split
│   └── dataset.py              # PyTorch Dataset + collate_fn with tile padding
├── src/
│   ├── config.py               # All hyperparameters in one place (reads from .env)
│   ├── model.py                # BitsAndBytes quantization + LoRA setup
│   ├── train.py                # HuggingFace Trainer training loop
│   ├── evaluate.py             # ANLS + Exact Match metrics
│   └── main.py                 # CLI entrypoint (prepare-data / train / evaluate / merge)
├── scripts/
│   ├── 1_download.sh           # SLURM job: download + prepare dataset (CPU node)
│   ├── 2_train.sh              # SLURM job: fine-tune on A100 GPU
│   ├── 3_merge.sh              # SLURM job: merge LoRA adapter into base model
│   ├── 4_deploy.sh             # Deploy merged model to SageMaker
│   └── merge_adapter.py        # Merge logic: load base + adapter → merged safetensors
├── deploy/
│   ├── inference.py            # SageMaker inference handler (model_fn/predict_fn/etc.)
│   ├── sagemaker_deploy.py     # Deploy endpoint via SageMaker Python SDK
│   └── Dockerfile              # Container for SageMaker inference
├── requirements.txt
└── .env                        # Paths and credentials (not committed)
```

---

## 11. Environment Setup

**Prerequisites:** Conda, CUDA 12.x, access to Unity HPC cluster.

```bash
conda create -n docstanding python=3.11 -y
conda activate docstanding
pip install -r requirements.txt
```

**`.env` file** (copy from `.env.example` and fill in):
```bash
HF_CACHE_DIR=/project/<pi>/<project>/hf_cache
TRAIN_PATH=/project/<pi>/<project>/data/processed/train.json
VAL_PATH=/project/<pi>/<project>/data/processed/val.json
TEST_PATH=/project/<pi>/<project>/data/processed/test.json
IMAGE_ROOT=/project/<pi>/<project>/data/raw/images
CHECKPOINT_DIR=/project/<pi>/<project>/checkpoints/llava-docvqa
WANDB_ENTITY=<your-wandb-entity>
```

**W&B login:**
```bash
wandb login
# verify entity with:
python -c "import wandb; api = wandb.Api(); print(api.viewer.entity)"
```

---

## 12. Running the Pipeline

**Step 1 - Download and prepare dataset (CPU node, ~2-3 hours)**
```bash
sbatch scripts/1_download.sh
```
Downloads 15K DocVQA samples, saves images to `/data/raw/images/`, converts to LLaVA conversation format, and splits into train/val/test.

**Step 2 - Fine-tune (A100 GPU, ~4-5 hours)**
```bash
sbatch scripts/2_train.sh
```
Trains for 3 epochs. Checkpoints saved to `CHECKPOINT_DIR` after each epoch. Final LoRA adapter saved to `CHECKPOINT_DIR/final_adapter/`.

Monitor live training at your W&B project dashboard.

**Step 3 - Merge adapter (CPU node, ~10 min)**
```bash
sbatch scripts/3_merge.sh
```
Merges LoRA adapter into base model weights. Output is a standalone safetensors model ready for deployment.

**Step 4 - Deploy to SageMaker**
```bash
sbatch scripts/4_deploy.sh
```
Uploads merged model to S3 and creates a SageMaker endpoint.

**Interactive training (for debugging):**
```bash
tmux new -s train
srun --partition=gpu-preempt --gres=gpu:a100:1 --mem=128G --cpus-per-task=8 --time=4:00:00 --pty bash
# on GPU node:
cd /work/<pi>/<project>
module load conda/latest && conda activate docstanding
export HF_HOME="/project/<pi>/<project>/hf_cache"
python src/main.py train
```

---

## 13. Configuration Reference

All hyperparameters live in `src/config.py` and are read at runtime. Key values:

| Parameter | Value | Location |
|---|---|---|
| Base model | `llava-hf/llava-v1.6-mistral-7b-hf` | `config.py` |
| Quantization | 4-bit NF4, compute in bf16 | `model.py` |
| LoRA rank `r` | 32 | `config.py` |
| LoRA alpha | 64 (scaling = 2x) | `config.py` |
| LoRA target modules | q/k/v/o + gate/up/down proj | `config.py` |
| Effective batch size | 32 (8 × 4 accum steps) | `config.py` |
| Learning rate | 1e-4, cosine schedule | `config.py` |
| Warmup ratio | 3% of total steps | `config.py` |
| Epochs | 3 | `config.py` |
| Max sequence length | 1024 tokens | `config.py` |
| Primary eval metric | ANLS (threshold 0.5) | `evaluate.py` |
| Secondary metric | Exact Match | `evaluate.py` |
