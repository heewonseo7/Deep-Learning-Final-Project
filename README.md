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
cd cs1470-final-project

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

Hyperparameters for capacity, β sweep, and attention swap live under `configs/`. Each script accepts `--config path/to.yaml` (defaults point at the matching file in `configs/`).

```bash
# Capacity (writes results/capacity_seed*.npz)
# Quick curve (fewer N): configs/capacity_quick.yaml — full sweep: configs/capacity.yaml
uv run python -m experiments.capacity.experiment --config configs/capacity_quick.yaml --seed 0
uv run python -m experiments.capacity.plot --seed 0

# One-step retrieval grid (constants still in experiments/one_step/experiment.py)
uv run python -m experiments.one_step.experiment --seed 0
uv run python -m experiments.one_step.plot --seed 0

# β sweep (writes results/beta_sweep_seed*.npz; first CIFAR branch may build embeddings cache)
uv run python -m experiments.beta_sweep.experiment --config configs/beta_sweep.yaml --seed 0
uv run python -m experiments.beta_sweep.plot --seed 0

# Attention swap on MNIST (slow; uses configs/attn_swap.yaml)
uv run python -m experiments.attn_swap.experiment --config configs/attn_swap.yaml --seed 0
uv run python -m experiments.attn_swap.plot --seed 0

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
│   ├── capacity/        # modern vs classical capacity (.npz + plot)
│   ├── one_step/        # iterations × noise heatmap
│   ├── beta_sweep/      # accuracy & entropy vs β
│   ├── attn_swap/       # standard attention vs Hopfield on MNIST
│   ├── mil/             # MUSK1/2, Fox, Tiger, Elephant — 10-fold CV
│   └── mnist_bags/      # CNN + HopfieldPooling on MNIST-Bags MIL benchmark
├── baselines/
│   ├── attention_mil.py        # Ilse et al. (2018) soft-attention baseline
│   └── gated_attention_mil.py  # Ilse et al. (2018) gated-attention baseline
├── scripts/
│   └── download_mil.py  # fetches MUSK1/2 from UCI; generates synthetic Corel stand-ins
├── configs/
│   ├── capacity.yaml
│   ├── beta_sweep.yaml
│   ├── attn_swap.yaml
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
