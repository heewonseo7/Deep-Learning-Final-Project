"""Capacity comparison: modern vs classical Hopfield network."""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from hopfield.attention import hopfield_update
from baselines.classical import ClassicalHopfield
from experiments.runner import save_results

# ── modern Hopfield settings ────────────────────────────────────────────────
D_MODERN = 64
BETA = 10.0
NUM_ITER = 1
N_MODERN = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]

# ── classical Hopfield settings ─────────────────────────────────────────────
D_CLASSICAL = 100
N_CLASSICAL = [5, 10, 14, 20, 30, 50, 75, 100, 150, 200]

SIGMA = 0.1   # Gaussian noise std for modern
TRIALS = 50   # retrieval trials per N


def modern_accuracy(N: int, d: int, beta: float, seed: int) -> float:
    rng = torch.Generator()
    rng.manual_seed(seed)
    X = torch.randn(d, N, generator=rng)          # (d, N)
    correct = 0
    for i in range(min(N, TRIALS)):
        target = X[:, i % N]
        noisy = target + SIGMA * torch.randn(d, generator=rng)
        retrieved = hopfield_update(noisy, X, beta=beta, num_iter=NUM_ITER)
        sim = F.cosine_similarity(retrieved.unsqueeze(0), target.unsqueeze(0)).item()
        correct += float(sim > 0.9)
    return correct / min(N, TRIALS)


def classical_accuracy(N: int, d: int, seed: int) -> float:
    torch.manual_seed(seed)
    net = ClassicalHopfield(d)
    patterns = torch.sign(torch.randn(N, d))
    net.store(patterns)
    correct = 0
    for i in range(min(N, TRIALS)):
        noisy = patterns[i].clone()
        flip = torch.randperm(d)[:int(0.15 * d)]
        noisy[flip] *= -1
        retrieved = net.retrieve(noisy, num_iter=20)
        sim = F.cosine_similarity(
            retrieved.float().unsqueeze(0), patterns[i].float().unsqueeze(0)
        ).item()
        correct += float(sim > 0.9)
    return correct / min(N, TRIALS)


def main(seed: int = 0) -> None:
    print("=== Capacity experiment ===")

    print(f"\nModern Hopfield (d={D_MODERN}, β={BETA})")
    acc_modern = []
    for N in tqdm(N_MODERN, desc="modern"):
        acc_modern.append(modern_accuracy(N, D_MODERN, BETA, seed))
    print(f"  N:   {N_MODERN}")
    print(f"  acc: {[f'{a:.3f}' for a in acc_modern]}")

    print(f"\nClassical Hopfield (d={D_CLASSICAL})")
    acc_classical = []
    for N in tqdm(N_CLASSICAL, desc="classical"):
        acc_classical.append(classical_accuracy(N, D_CLASSICAL, seed))
    print(f"  N:   {N_CLASSICAL}")
    print(f"  acc: {[f'{a:.3f}' for a in acc_classical]}")

    save_results(
        name="capacity",
        data_dict={
            "N_modern": N_MODERN,
            "acc_modern": acc_modern,
            "N_classical": N_CLASSICAL,
            "acc_classical": acc_classical,
        },
        config_dict={"d_modern": D_MODERN, "d_classical": D_CLASSICAL, "beta": BETA},
        seed=seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.seed)
