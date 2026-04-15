import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))


def cmd_prepare(args):
    from dataset.prepare_dataset import prepare_dataset
    prepare_dataset(
        input_path=args.input,
        output_path=args.output,
        image_root=args.image_root,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
    )


def cmd_train(args):
    from train import main as train_main
    train_main()


def cmd_evaluate(args):
    import functools
    import torch
    from torch.utils.data import DataLoader
    from model import load_model_with_lora, load_processor
    from dataset.dataset import DocVQADataset, collate_fn
    from evaluate import evaluate_full_dataset
    from config import CFG as cfg

    processor = load_processor(cfg["model"]["name"], cfg["model"]["cache_dir"])
    model = load_model_with_lora(
        cfg["model"]["name"],
        cfg["model"]["cache_dir"],
        cfg["lora"],
        use_4bit=True,
    )

    test_dataset = DocVQADataset(
        data_path=cfg["data"]["test_path"],
        image_root=cfg["data"]["image_root"],
        processor=processor,
        max_length=cfg["data"]["max_length"],
        split="test",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["training"]["per_device_eval_batch_size"],
        collate_fn=functools.partial(collate_fn, pad_token_id=processor.tokenizer.pad_token_id),
        num_workers=2,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = evaluate_full_dataset(model, processor, test_loader, device=device)

    print("\n=== Test Set Evaluation Results ===")
    print(f"  ANLS        : {results['anls']:.4f}")
    print(f"  Exact Match : {results['exact_match']:.4f}")
    print(f"  Samples     : {results['num_samples']}")


def cmd_merge(args):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from merge_adapter import merge
    merge(args.adapter_path, args.output_dir, args.cache_dir, args.base_model)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="LLaVA-1.6 DocVQA fine-tuning pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # prepare-data
    p = sub.add_parser("prepare-data", help="Convert raw annotations to LLaVA format")
    p.add_argument("--input", required=True, help="Path to raw annotations JSON")
    p.add_argument("--output", required=True, help="Output directory for processed splits")
    p.add_argument("--image_root", required=True, help="Root directory of document images")
    p.add_argument("--val_split", type=float, default=0.05)
    p.add_argument("--test_split", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)

    # train
    sub.add_parser("train", help="Fine-tune LLaVA-1.6 with LoRA")

    # evaluate
    p = sub.add_parser("evaluate", help="Evaluate a trained checkpoint on the val set")
    p.add_argument("--checkpoint", required=True, help="Path to saved adapter checkpoint")

    # merge 
    p = sub.add_parser("merge", help="Merge LoRA adapter into base model for deployment")
    p.add_argument("--adapter_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--cache_dir", default="/scratch/YOUR_USERNAME/hf_cache")
    p.add_argument("--base_model", default="llava-hf/llava-v1.6-mistral-7b-hf")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    commands = {
        "prepare-data": cmd_prepare,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "merge": cmd_merge,
    }
    commands[args.command](args)
