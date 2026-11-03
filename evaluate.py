"""Evaluate a locally trained checkpoint without retraining or selecting on test."""

import argparse
import json
from pathlib import Path

from train import DATASET, evaluate, load_project
from audit_checkpoint import digest
import torch
from torch import nn


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("dataset") != DATASET:
        raise ValueError(f"Expected a {DATASET} checkpoint, got {checkpoint.get('dataset')}")
    if not all(key in checkpoint for key in ("model_state_dict", "epoch", "seed")):
        raise ValueError("Checkpoint is missing model state, epoch or seed")
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    if args.output and args.output.exists():
        parser.error("output file exists; choose a new file")
    checkpoint = load_checkpoint(args.checkpoint)
    factory, get_loaders = load_project()
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = factory()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    _, validation, test = get_loaders(
        root=str(args.data_root), batch_size=args.batch_size,
        seed=checkpoint["seed"], download=not args.no_download,
    )
    report = {
        "dataset": DATASET, "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_sha256": digest(args.checkpoint), "seed": checkpoint["seed"],
        "batch_size": args.batch_size, "device": str(device),
        "validation": evaluate(validation, model, nn.CrossEntropyLoss(), device),
        "test": evaluate(test, model, nn.CrossEntropyLoss(), device),
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
