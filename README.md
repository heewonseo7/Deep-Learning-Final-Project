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
cd modern-hopfield

# 3. Install all dependencies (creates .venv automatically)
uv sync

# 4. Verify PyTorch is available
uv run python -c "import torch; print(torch.__version__)"
```

## Running experiments

```bash
# MIL benchmark (MUSK1)
uv run python -m experiments.mil.train --config configs/mil_musk1.yaml

# MIL benchmark (MNIST-Bags)
uv run python -m experiments.mnist_bags.train --config configs/mil_mnist.yaml

# beta sweep (saves plots to outputs/beta_sweep/)
uv run python -m experiments.beta_sweep.experiment
uv run python -m experiments.beta_sweep.plot
```

Plots and results CSVs are written automatically to `outputs/` (gitignored; only `outputs/.gitkeep` is tracked).

## Repository structure

```
modern-hopfield/
├── hopfield/
│   ├── __init__.py
│   ├── energy.py        # log-sum-exp energy function E(xi)
│   ├── attention.py     # update rule: xi_new = X·softmax(beta·X^T·xi)
│   ├── pooling.py       # HopfieldPooling with learned static query
│   └── network.py       # full Hopfield layer with Q/K/V projections
├── experiments/
│   ├── mil/
│   │   ├── dataset.py   # MUSK1/2, Fox, Tiger, Elephant loaders
│   │   ├── model.py     # MIL classifier using HopfieldPooling
│   │   └── train.py     # 10-fold cross-validation loop
│   ├── beta_sweep/
│   │   ├── experiment.py
│   │   └── plot.py
│   └── mnist_bags/
│       ├── dataset.py
│       ├── model.py
│       └── train.py
├── baselines/
│   ├── attention_mil.py        # Ilse et al. (2018) soft-attention
│   └── gated_attention_mil.py  # Ilse et al. (2018) gated attention
├── outputs/                    # gitignored — plots + results CSVs land here
├── configs/
│   ├── mil_musk1.yaml
│   └── mil_mnist.yaml
└── README.md
```

## Reference

> Ramsauer, H., Schäfl, B., Lehner, J., Seidl, P., Widrich, M., Adler, T., ... & Hochreiter, S.
> *Hopfield Networks is All You Need.* ICLR 2021.
> https://arxiv.org/abs/2008.02217
