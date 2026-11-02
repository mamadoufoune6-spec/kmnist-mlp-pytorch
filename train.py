"""Train and evaluate the kmnist learning classifier."""

import argparse
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn


DATASET = 'kmnist'


def train_loop(dataloader, model, loss_fn, optimizer, device):
    """Adapted from Mamadou's MLP loop; return example-weighted metrics."""
    model.train()
    loss_sum, correct, count = 0.0, 0, 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss = loss_fn(predictions, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        count += labels.size(0)
        loss_sum += loss.item() * labels.size(0)
        correct += (predictions.argmax(1) == labels).sum().item()
    if not count:
        raise ValueError("Training loader is empty")
    return {"loss": loss_sum / count, "accuracy_percent": 100 * correct / count,
            "examples": count}


def evaluate(dataloader, model, loss_fn, device):
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            predictions = model(inputs)
            loss = loss_fn(predictions, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite evaluation loss")
            count += labels.size(0)
            loss_sum += loss.item() * labels.size(0)
            correct += (predictions.argmax(1) == labels).sum().item()
    if not count:
        raise ValueError("Evaluation loader is empty")
    return {"loss": loss_sum / count, "accuracy_percent": 100 * correct / count,
            "examples": count}


def load_project():
    from kmnist_mlp import KMNISTMLP, get_data_loaders
    return KMNISTMLP, get_data_loaders


def plot_history(history, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for key in ("train", "validation"):
        axes[0].plot(epochs, [r[key]["loss"] for r in history], marker="o", label=key)
        axes[1].plot(epochs, [r[key]["accuracy_percent"] for r in history], marker="o", label=key)
    axes[0].set_ylabel("Cross-entropy per example")
    axes[1].set_ylabel("Accuracy (%)")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_xticks(epochs)
        axis.legend()
        axis.grid(alpha=0.25)
    figure.savefig(output, dpi=140)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")
    output = args.output_dir or Path("outputs") / f"{DATASET}-seed{args.seed}"
    if output.exists() and any(output.iterdir()):
        parser.error("output directory is not empty; choose a new directory")
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else args.device
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model_class, get_loaders = load_project()
    torch.manual_seed(args.seed)
    model = model_class().to(device)
    train, validation, test = get_loaders(
        root=str(args.data_root), batch_size=args.batch_size,
        seed=args.seed, download=not args.no_download,
    )
    settings = {"lr": 0.001}
    optimizer = torch.optim.SGD(model.parameters(), **settings)
    loss_fn = nn.CrossEntropyLoss()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": DATASET, "seed": args.seed, "split_seed": 42,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "split_sizes": {k: len(v.dataset) for k, v in
                        (("train", train), ("validation", validation), ("test", test))},
        "optimizer": {"name": "SGD", **settings},
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "torch_version": str(torch.__version__), "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "deterministic_algorithms": True, "cpu_threads": 4,
        "selection": "lowest validation cross-entropy; first epoch wins exact ties",
        "history": [],
    }
    best_loss = float("inf")
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        training = train_loop(train, model, loss_fn, optimizer, device)
        metrics = evaluate(validation, model, loss_fn, device)
        report["history"].append({"epoch": epoch, "train": training, "validation": metrics})
        checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "dataset": DATASET, "seed": args.seed}
        torch.save(checkpoint, output / "last.pt")
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            report["best_epoch"] = epoch
            torch.save(checkpoint, output / "best.pt")
        (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Epoch {epoch}/{args.epochs}: train loss {training['loss']:.4f}; "
              f"validation loss {metrics['loss']:.4f}; "
              f"validation accuracy {metrics['accuracy_percent']:.2f}%", flush=True)
    best = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best["model_state_dict"])
    report["test"] = evaluate(test, model, loss_fn, device)
    report["elapsed_seconds"] = time.perf_counter() - started
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_history(report["history"], output / "curves.png")
    print(f"Best epoch: {report['best_epoch']}; final test accuracy: "
          f"{report['test']['accuracy_percent']:.2f}%", flush=True)


if __name__ == "__main__":
    main()
