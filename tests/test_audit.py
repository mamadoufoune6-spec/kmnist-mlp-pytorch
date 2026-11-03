"""Ensure the audit checks loaded caches and detects exact image overlap."""

import gzip
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from audit_checkpoint import verify_loaded_kmnist_data


class CachedDataset:
    def __init__(self, images, labels, folder):
        self.data = images.copy()
        self.targets = labels.copy()
        self.raw_folder = folder

    def __len__(self):
        return len(self.data)


def write_archive(path, values):
    magic = 2049 if values.ndim == 1 else 2051
    header = struct.pack(">" + "I" * (values.ndim + 1), magic, *values.shape)
    with gzip.open(path, "wb") as stream:
        stream.write(header + values.astype(np.uint8).tobytes())


class AuditTests(unittest.TestCase):
    def fixture(self, folder, duplicate_test=False):
        images = np.arange(12, dtype=np.uint8).reshape(3, 2, 2)
        labels = np.array([0, 1, 2], dtype=np.uint8)
        test_images = images[:1].copy() if duplicate_test else images[:1] + 50
        test_labels = np.array([0], dtype=np.uint8)
        for name, values in [
            ("train-images-idx3-ubyte.gz", images),
            ("train-labels-idx1-ubyte.gz", labels),
            ("t10k-images-idx3-ubyte.gz", test_images),
            ("t10k-labels-idx1-ubyte.gz", test_labels),
        ]:
            write_archive(folder / name, values)
        return (CachedDataset(images, labels, str(folder)),
                CachedDataset(test_images, test_labels, str(folder)))

    def test_loaded_cache_changes_fail_despite_unchanged_archives(self):
        with tempfile.TemporaryDirectory() as directory:
            base, test = self.fixture(Path(directory))
            result = verify_loaded_kmnist_data(base, test, [0, 1], [2])
            self.assertTrue(result["all_loaded_images_and_labels_match_archives"])
            base.targets[0] = 9
            with self.assertRaisesRegex(ValueError, "Loaded dataset differs"):
                verify_loaded_kmnist_data(base, test, [0, 1], [2])

    def test_exact_test_overlap_is_reported_as_data_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            base, test = self.fixture(Path(directory), duplicate_test=True)
            result = verify_loaded_kmnist_data(base, test, [0, 1], [2])
            self.assertEqual(result["test_examples_matching_official_training_image"], 1)
            self.assertEqual(result["validation_examples_matching_training_image"], 0)


if __name__ == "__main__":
    unittest.main()
