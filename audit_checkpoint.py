"""Read-only audit of existing local checkpoints and official cached datasets."""

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import struct

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_idx_archive(path):
    """Decode original uint8 IDX images/labels, rejecting malformed payloads."""
    with gzip.open(path, "rb") as stream:
        payload = stream.read()
    if len(payload) < 8:
        raise ValueError("Truncated IDX header")
    magic = struct.unpack(">I", payload[:4])[0]
    dimensions = {2049: 1, 2051: 3}.get(magic)
    if dimensions is None:
        raise ValueError("Unsupported IDX image/label format")
    offset = 4 * (dimensions + 1)
    if len(payload) < offset:
        raise ValueError("Truncated IDX shape")
    shape = struct.unpack(">" + "I" * dimensions, payload[4:offset])
    values = np.frombuffer(payload[offset:], dtype=np.uint8)
    if values.size != int(np.prod(shape)):
        raise ValueError("IDX payload length does not match its shape")
    return values.reshape(shape)


def verify_loaded_kmnist_data(base, test, train_indices, validation_indices):
    """Check actual loaded arrays, not just the compressed files beside them."""
    raw = Path(base.raw_folder)
    arrays = [
        (base.data, "train-images-idx3-ubyte.gz"),
        (base.targets, "train-labels-idx1-ubyte.gz"),
        (test.data, "t10k-images-idx3-ubyte.gz"),
        (test.targets, "t10k-labels-idx1-ubyte.gz"),
    ]
    for loaded, filename in arrays:
        official = read_idx_archive(raw / filename)
        actual = np.asarray(loaded)
        if not np.array_equal(actual, official):
            raise ValueError(f"Loaded dataset differs from official archive: {filename}")
    for labels in [base.targets, test.targets]:
        values = np.asarray(labels)
        if not np.all((values >= 0) & (values <= 9)):
            raise ValueError("Labels outside the expected 0..9 classes")
    hashes = [hashlib.sha256(image.tobytes()).digest() for image in np.asarray(base.data)]
    test_hashes = [hashlib.sha256(image.tobytes()).digest() for image in np.asarray(test.data)]
    train_hashes = {hashes[i] for i in train_indices}
    validation_hashes = {hashes[i] for i in validation_indices}
    full_hashes = set(hashes)
    return {
        "all_loaded_images_and_labels_match_archives": True,
        "loaded_examples_checked": len(base) + len(test),
        "labels_in_range": True,
        "shared_unique_train_validation_image_hashes": len(train_hashes & validation_hashes),
        "validation_examples_matching_training_image": sum(hashes[i] in train_hashes for i in validation_indices),
        "test_examples_matching_official_training_image": sum(h in full_hashes for h in test_hashes),
        "exact_image_hash_algorithm": "sha256",
        "near_duplicate_or_writer_overlap_checked": False,
    }


def independent_evaluation(dataset, model, device, batch_size):
    model.to(device).eval()
    confusion = torch.zeros((10, 10), dtype=torch.int64)
    loss_sum = 0.0
    with torch.no_grad():
        for images, labels in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            labels = labels.to(device)
            logits = model(images.to(device))
            loss_sum += F.cross_entropy(logits, labels, reduction="none").double().sum().item()
            ids = (labels * 10 + logits.argmax(1)).cpu()
            confusion += torch.bincount(ids, minlength=100).reshape(10, 10)
    counts = confusion.sum(1)
    correct = confusion.diagonal()
    n = int(counts.sum())
    return {
        "loss": loss_sum / n, "accuracy_percent": 100 * int(correct.sum()) / n,
        "examples": n, "correct": int(correct.sum()),
        "confusion_matrix": confusion.tolist(), "class_counts": counts.tolist(),
        "class_accuracy_percent": (100 * correct.double() / counts).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(args.repo))
    spec = importlib.util.spec_from_file_location("audited_training", args.repo / "train.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    factory, get_loaders = trainer.load_project()
    train, validation, test = get_loaders(str(args.data_root), batch_size=64, seed=42, download=False)
    base = train.dataset.dataset
    assert validation.dataset.dataset is base
    a, b = set(train.dataset.indices), set(validation.dataset.indices)
    assert not a.intersection(b)
    assert a.union(b) == set(range(len(base)))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    published = json.loads((args.repo / "results/metrics.json").read_text())
    selected = min(published["history"], key=lambda row: row["validation"]["loss"])
    assert checkpoint["dataset"] == trainer.DATASET == published["dataset"]
    assert checkpoint["epoch"] == selected["epoch"] == published["best_epoch"]
    assert checkpoint["seed"] == published["seed"] == 42
    model = factory()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    for dataset in [train.dataset, validation.dataset, test.dataset]:
        for index in [0, len(dataset) // 2, len(dataset) - 1]:
            image, label = dataset[index]
            underlying = dataset.dataset if hasattr(dataset, "indices") else dataset
            original_index = dataset.indices[index] if hasattr(dataset, "indices") else index
            raw = torch.as_tensor(underlying.data[original_index]).float()
            expected = (raw.unsqueeze(0) / 255 if trainer.DATASET == "kmnist"
                        else (raw.permute(2, 0, 1) / 255 - 0.5) / 0.5)
            torch.testing.assert_close(image, expected, rtol=0, atol=1e-7)
            assert int(label) == int(underlying.targets[original_index])
    assert repr(base.transform) == repr(test.dataset.transform)
    integrity = []
    if trainer.DATASET == "kmnist":
        resources = [(args.data_root / "KMNIST/raw" / name, md5) for name, md5 in base.resources]
    else:
        resources = [(args.data_root / base.base_folder / name, md5)
                     for name, md5 in base.train_list + base.test_list]
    for path, expected in resources:
        actual = digest(path, "md5")
        assert actual == expected, path.name
        integrity.append({"filename": path.name, "md5": actual, "matches_torchvision_resource": True})
    loaded_data_check = verify_loaded_kmnist_data(
        base, test.dataset, train.dataset.indices, validation.dataset.indices,
    ) if trainer.DATASET == "kmnist" else None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = {
        "dataset": trainer.DATASET, "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_sha256": digest(args.checkpoint), "seed": 42, "split_seed": 42,
        "split_sizes": {"train": len(a), "validation": len(b), "test": len(test.dataset)},
        "train_validation_index_overlap": 0, "split_complete": True,
        "train_indices_sha256": hashlib.sha256(np.asarray(train.dataset.indices, dtype="<i8").tobytes()).hexdigest(),
        "validation_indices_sha256": hashlib.sha256(np.asarray(validation.dataset.indices, dtype="<i8").tobytes()).hexdigest(),
        "preprocessing_sample_checks": 9, "preprocessing_absolute_tolerance": 1e-7,
        "train_test_transform_match": True,
        "dataset_integrity": integrity, "evaluations": {},
        "loaded_data_check": loaded_data_check,
    }
    for name, loader in [("validation", validation), ("test", test)]:
        model.to(device)
        stored = trainer.evaluate(loader, model, nn.CrossEntropyLoss(), device)
        expected = selected["validation"] if name == "validation" else published["test"]
        assert abs(stored["accuracy_percent"] - expected["accuracy_percent"]) < 1e-9
        assert abs(stored["loss"] - expected["loss"]) < 1e-6
        independent = independent_evaluation(loader.dataset, model, "cpu", 257)
        assert independent["accuracy_percent"] == stored["accuracy_percent"]
        assert abs(independent["loss"] - stored["loss"]) < 1e-6
        report["evaluations"][name] = {
            "published": expected, "original_evaluator_device": device,
            "original_evaluator_batch64": stored,
            "independent_cpu_evaluator_batch257": independent,
        }
        print(name, stored, "independent CPU/batch257 agrees", flush=True)
    for key, value in checkpoint["model_state_dict"].items():
        torch.testing.assert_close(model.state_dict()[key].cpu(), value, rtol=0, atol=0)
    report["evaluation_did_not_change_weights"] = True
    report["gap_percentage_points"] = round(
        report["evaluations"]["validation"]["published"]["accuracy_percent"]
        - report["evaluations"]["test"]["published"]["accuracy_percent"], 8)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("Audit passed:", trainer.DATASET, flush=True)


if __name__ == "__main__":
    main()
