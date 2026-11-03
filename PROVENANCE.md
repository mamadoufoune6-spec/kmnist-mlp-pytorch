# Provenance


- Existing work: the MLP architecture, KMNIST tensor preprocessing,
  cross-entropy objective, SGD learning rate 0.001, and the core training loop.
- Architecture check: the original class was extracted with Python's AST,
  without executing its downloads/training. Original and prepared classes had
  identical seeded state dictionaries and identical forward outputs.

These results use a changed validation protocol, so they should not be compared
directly with the original script's test-after-every-epoch output.

## Standalone extraction

Extracted on 2 November 2026 from portfolio commit `6ad431f`.
Model definitions, data splits, optimisation and metric calculations are preserved. Codex changed file paths and the CLI to run this dataset directly with `train.py`, removed the dependency on the other project and set the default budget to 20 epochs.
Published metrics and curves were copied byte-for-byte from the recorded run.

Training-loop reference and retained upstream BSD notice: see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The MIT licence does not relicense upstream code or the downloaded datasets.

## Evaluation audit follow-up

Codex added read-only checkpoint evaluation and independent audit commands,
a regression test for validation-based checkpoint selection and a dataset-identity
check. The README and audit artifacts report observed checks and their limits.
The model architecture, preprocessing, training split and optimiser were not changed.
The independent preprocessing check allows 1e-7 float32 rounding; an initial
bit-exact scaling assertion was too strict for equivalent multiply/divide conversion.

## Focused KMNIST repository review

Codex reviewed the existing, unpushed diagnostic changes before extending the
audit to compare every loaded raw image/label against the official IDX archives
and check exact image overlap. Codex authored the README findings and separate `data_integrity_review.json` and `reproduction_review.json` verification records.

The original architecture, preprocessing, split, loss, SGD settings and training
loop were preserved. A fresh 20-epoch seed-42 run reproduced the original history,
selected model tensors and test metrics exactly in the installed environment.
The original experiment JSON, plot and checkpoint were not replaced. No model
or evaluation bug causing the validation/test discrepancy was confirmed. These
are verification and documentation changes, not a new scientific method or an
accuracy improvement. No hyperparameter optimisation was performed.
