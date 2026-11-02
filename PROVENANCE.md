# Provenance and changes

Prepared from Mamadou's supplied `kminst_mlp.py` on 2 November 2026.
SHA-256 of that original source:
`ec9d3776ffb4f813c458149baa98da26f10b811a45ceafa6447a3df0e23e86e3`.
The original file was preserved outside the repository.

- Existing work: the MLP architecture, KMNIST tensor preprocessing,
  cross-entropy objective, SGD learning rate 0.001, and the core training loop.
- Codex changes: extracted the model into an importable module; corrected its
  naming; adapted the loop into the trainer; added a seeded 55,000/5,000
  training/validation split; moved test evaluation to after checkpoint selection;
  added weighted metrics, deterministic settings, checkpoints, plots, CLI and
  dataset-free tests. The 20-epoch run budget is taken from the original script.
- Architecture check: the original class was extracted with Python's AST,
  without executing its downloads/training. Original and prepared classes had
  identical seeded state dictionaries and identical forward outputs.
- Earlier tutorial sources and independent authorship remain unverified. This
  record distinguishes supplied implementation from Codex-authored support;
  it does not assert that the full exercise was independently authored.

These results use a changed validation protocol, so they should not be compared
directly with the original script's test-after-every-epoch output.

## Standalone extraction

Extracted on 2 November 2026 from portfolio commit `6ad431f`.
Model definitions, data splits, optimisation and metric calculations are preserved.
Codex changed file paths and the CLI to run this dataset directly with `train.py`,
removed the dependency on the other project and set the default budget to 20 epochs.
Published metrics and curves were copied byte-for-byte from the recorded run.
The MIT licence was added at Mamadou's explicit request, with upstream PyTorch
BSD notices retained separately. Dataset rights remain with their providers.

Training-loop reference and retained upstream BSD notice: see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The MIT licence does not
relicense upstream code or the downloaded datasets.
