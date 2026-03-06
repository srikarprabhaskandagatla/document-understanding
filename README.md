This project fine-tunes LLaVA-1.6 (a Vision-Language Model) on a custom 15,000-sample document understanding dataset. The goal is to teach the model to answer questions about document images — think invoices, forms, research papers — with high accuracy.

Here's the pipeline at a high level:
1. Dataset prep — 15K image-question-answer triples formatted for LLaVA instruction tuning
2. LoRA fine-tuning — Parameter-efficient training (only ~1% of weights trained) on A100 GPU
3. W&B tracking — Experiment logging, loss curves, VQA eval metrics
4. SageMaker deployment — Model served as a REST endpoint behind a CI/CD pipeline