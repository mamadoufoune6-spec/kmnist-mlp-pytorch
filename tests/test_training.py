"""Dataset-free checks of training, evaluation and checkpoint semantics."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

spec = importlib.util.spec_from_file_location(
    "training", Path(__file__).resolve().parents[1] / "train.py",
)
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = nn.Linear(2, 2)
        self.inputs = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
        self.labels = torch.tensor([0, 1, 0])
        self.loader = DataLoader(TensorDataset(self.inputs, self.labels), batch_size=2)
        self.loss = nn.CrossEntropyLoss()

    def test_training_updates_parameters_and_can_restore_checkpoint(self):
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        result = training.train_loop(self.loader, self.model, self.loss, optimizer, "cpu")
        self.assertEqual(result["examples"], 3)
        self.assertTrue(any(not torch.equal(before[k], v)
                            for k, v in self.model.state_dict().items()))
        expected = self.model(self.inputs).detach()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            torch.save({"model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()}, path)
            restored = nn.Linear(2, 2)
            restored.load_state_dict(torch.load(path, weights_only=True)["model_state_dict"])
        torch.testing.assert_close(restored(self.inputs), expected)

    def test_evaluation_weights_short_batch_and_does_not_train(self):
        expected_loss = self.loss(self.model(self.inputs), self.labels).item()
        expected_correct = (self.model(self.inputs).argmax(1) == self.labels).sum().item()
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        result = training.evaluate(self.loader, self.model, self.loss, "cpu")
        self.assertAlmostEqual(result["loss"], expected_loss, places=6)
        self.assertAlmostEqual(result["accuracy_percent"], 100 * expected_correct / 3)
        self.assertFalse(self.model.training)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.assertTrue(all(torch.equal(before[k], v)
                            for k, v in self.model.state_dict().items()))

    def test_empty_evaluation_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            training.evaluate([], self.model, self.loss, "cpu")

    def test_nonfinite_training_loss_fails_before_update(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        with self.assertRaises(FloatingPointError):
            training.train_loop([(self.inputs * float("nan"), self.labels)],
                                self.model, self.loss, optimizer, "cpu")
        self.assertTrue(all(torch.equal(before[k], v)
                            for k, v in self.model.state_dict().items()))


if __name__ == "__main__":
    unittest.main()
