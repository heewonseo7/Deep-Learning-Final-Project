# Modern Hopfield Networks — Replication

A from-scratch PyTorch replication of **"Hopfield Networks is All You Need"** (Ramsauer et al., 2020).
The paper shows that modern (continuous) Hopfield networks with an exponential interaction function
are equivalent to transformer attention, achieve exponentially large storage capacity, and can replace
attention in practical architectures. This repo implements the core Hopfield layer, the
`HopfieldPooling` module, and reproduces the multiple-instance learning (MIL) experiments on
MUSK1/2, Fox, Tiger, Elephant, and MNIST-Bags benchmarks reported in Tables 1 & 2 of the paper.

## Setup

```bash
# 1. Install uv (if not already present)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and enter the repo
git clone <repo-url>
cd Deep-Learning-Final-Project

# 3. Install all dependencies (creates .venv automatically)
uv sync --dev

# 4. Verify the install
uv run pytest tests/
```

## Prepare data

```bash
# Download MUSK1/2 from UCI and generate synthetic Fox/Tiger/Elephant stand-ins
uv run python scripts/download_mil.py

# MNIST is downloaded automatically on first run — no manual step needed
```

Fox, Tiger, and Elephant are not freely redistributable. Replace the synthetic `.mat` files in
`data/mil/` with the real ones when available; the loader accepts them as-is.

## Running experiments

```bash
# β-sweep (saves plots to results/beta_sweep/)
uv run python -m experiments.beta_sweep.experiment

# MIL benchmark — MUSK1, 10-fold CV
uv run python -m experiments.mil.train --config configs/mil_musk1.yaml

# MIL benchmark — MNIST-Bags
uv run python -m experiments.mnist_bags.train --config configs/mil_mnist.yaml
```

## Repository structure

```
.
├── hopfield/
│   ├── energy.py        # log-sum-exp energy E(ξ) — Theorem 2 Lyapunov function
│   ├── attention.py     # update rule: ξ_new = X·softmax(β·Xᵀ·ξ)
│   ├── pooling.py       # HopfieldPooling — learned static query for set aggregation
│   └── network.py       # HopfieldLayer — full Q/K/V drop-in attention replacement
├── experiments/
│   ├── beta_sweep/      # retrieval error vs β; storage capacity curve
│   ├── mil/             # MUSK1/2, Fox, Tiger, Elephant — 10-fold CV
│   └── mnist_bags/      # CNN + HopfieldPooling on MNIST-Bags MIL benchmark
├── baselines/
│   ├── attention_mil.py        # Ilse et al. (2018) soft-attention baseline
│   └── gated_attention_mil.py  # Ilse et al. (2018) gated-attention baseline
├── scripts/
│   └── download_mil.py  # fetches MUSK1/2 from UCI; generates synthetic Corel stand-ins
├── configs/
│   ├── mil_musk1.yaml
│   └── mil_mnist.yaml
├── tests/
│   └── test_hopfield_core.py
└── results/             # gitignored — plots and result CSVs land here
```

## Reference

> Ramsauer, H., Schäfl, B., Lehner, J., Seidl, P., Widrich, M., Adler, T., ... & Hochreiter, S.
> *Hopfield Networks is All You Need.* ICLR 2021.
> https://arxiv.org/abs/2008.02217
