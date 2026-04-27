"""Capacity comparison: modern vs classical Hopfield network."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from baselines.classical import ClassicalHopfield
from experiments.config_utils import load_yaml_config
from experiments.runner import save_results
from hopfield.attention import hopfield_update


def modern_accuracy(
    N: int,
    d: int,
    beta: float,
    seed: int,
    *,
    sigma: float,
    trials: int,
    num_iter: int,
) -> float:
    rng = torch.Generator()
    rng.manual_seed(seed)
    X = torch.randn(d, N, generator=rng)          # (d, N)
    correct = 0
    for i in range(min(N, trials)):
        target = X[:, i % N]
        noisy = target + sigma * torch.randn(d, generator=rng)
        retrieved = hopfield_update(noisy, X, beta=beta, num_iter=num_iter)
        sim = F.cosine_similarity(retrieved.unsqueeze(0), target.unsqueeze(0)).item()
        correct += float(sim > 0.9)
    return correct / min(N, trials)


def classical_accuracy(
    N: int,
    d: int,
    seed: int,
    *,
    trials: int,
    retrieve_iters: int,
    flip_fraction: float,
) -> float:
    torch.manual_seed(seed)
    net = ClassicalHopfield(d)
    patterns = torch.sign(torch.randn(N, d))
    net.store(patterns)
    correct = 0
    n_flip = max(1, int(flip_fraction * d))
    for i in range(min(N, trials)):
        noisy = patterns[i].clone()
        flip = torch.randperm(d)[:n_flip]
        noisy[flip] *= -1
        retrieved = net.retrieve(noisy, num_iter=retrieve_iters)
        sim = F.cosine_similarity(
            retrieved.float().unsqueeze(0), patterns[i].float().unsqueeze(0)
        ).item()
        correct += float(sim > 0.9)
    return correct / min(N, trials)


def main(seed: int, config_path: Path) -> None:
    cfg = load_yaml_config(config_path)

    d_modern = int(cfg["d_modern"])
    d_classical = int(cfg["d_classical"])
    beta = float(cfg["beta"])
    num_iter = int(cfg["num_iter"])
    sigma = float(cfg["sigma"])
    trials = int(cfg["trials"])
    n_modern = list(cfg["N_modern"])
    n_classical = list(cfg["N_classical"])
    classical_retrieve_iters = int(cfg.get("classical_retrieve_iters", 20))
    classical_flip_fraction = float(cfg.get("classical_flip_fraction", 0.15))

    print("=== Capacity experiment ===")
    print(f"Config: {config_path}")

    print(f"\nModern Hopfield (d={d_modern}, β={beta})")
    acc_modern = []
    for N in tqdm(n_modern, desc="modern"):
        acc_modern.append(
            modern_accuracy(
                N, d_modern, beta, seed,
                sigma=sigma, trials=trials, num_iter=num_iter,
            )
        )
    print(f"  N:   {n_modern}")
    print(f"  acc: {[f'{a:.3f}' for a in acc_modern]}")

    print(f"\nClassical Hopfield (d={d_classical})")
    acc_classical = []
    for N in tqdm(n_classical, desc="classical"):
        acc_classical.append(
            classical_accuracy(
                N, d_classical, seed,
                trials=trials,
                retrieve_iters=classical_retrieve_iters,
                flip_fraction=classical_flip_fraction,
            )
        )
    print(f"  N:   {n_classical}")
    print(f"  acc: {[f'{a:.3f}' for a in acc_classical]}")

    save_results(
        name="capacity",
        data_dict={
            "N_modern": n_modern,
            "acc_modern": acc_modern,
            "N_classical": n_classical,
            "acc_classical": acc_classical,
        },
        config_dict={
            "d_modern": d_modern,
            "d_classical": d_classical,
            "beta": beta,
            "num_iter": num_iter,
            "sigma": sigma,
            "trials": trials,
            "classical_retrieve_iters": classical_retrieve_iters,
            "classical_flip_fraction": classical_flip_fraction,
            "config_path": str(config_path),
        },
        seed=seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capacity: modern vs classical Hopfield.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/capacity.yaml"),
        help="YAML hyperparameters (default: configs/capacity.yaml)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.seed, args.config)
