"""Beta sweep: retrieval accuracy and softmax entropy vs inverse temperature β."""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from hopfield.attention import hopfield_update
from experiments.runner import save_results

N = 50
D = 64
SIGMA = 0.1
STEPS = 20
TRIALS = 200

BETAS = [0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]


def _run(X: torch.Tensor, seed: int) -> tuple[list, list]:
    """Sweep β and return (accuracies, mean_entropies)."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    n_patterns = X.shape[1]

    accs, entropies = [], []
    for beta in BETAS:
        correct, ent_sum = 0, 0.0
        for t in range(TRIALS):
            target_idx = t % n_patterns
            target = X[:, target_idx]
            noisy = target + SIGMA * torch.randn(D, generator=rng)
            retrieved = hopfield_update(noisy, X, beta=beta, num_iter=STEPS)
            sim = F.cosine_similarity(retrieved.unsqueeze(0), target.unsqueeze(0)).item()
            correct += float(sim > 0.9)
            # Softmax entropy of attention weights
            logits = beta * (retrieved @ X)    # (N,)
            log_p = F.log_softmax(logits, dim=-1)
            p = torch.exp(log_p)
            ent_sum += -(p * log_p).sum().item()
        accs.append(correct / TRIALS)
        entropies.append(ent_sum / TRIALS)
    return accs, entropies


def main(seed: int = 0) -> None:
    print(f"=== Beta sweep (N={N}, d={D}, σ={SIGMA}) ===")
    rng = torch.Generator()
    rng.manual_seed(seed)

    # Random Gaussian patterns
    X_random = torch.randn(D, N, generator=rng)
    print("\n[Random patterns]")
    acc_random, ent_random = _run(X_random, seed)

    # CIFAR-10 class prototype embeddings
    print("\n[CIFAR-10 embeddings]")
    try:
        from data.loaders import get_cifar10_embeddings
        emb, labels = get_cifar10_embeddings()         # (50000, 512)
        num_classes = 10
        prototypes = torch.stack([
            emb[labels == c].mean(0) for c in range(num_classes)
        ])                                              # (10, 512)
        # Project to D dimensions with a fixed random matrix
        torch.manual_seed(42)
        proj = torch.randn(512, D) / (512 ** 0.5)
        X_cifar = (prototypes @ proj).T                # (D, 10)
        N_cifar = X_cifar.shape[1]
        # Use only 10 patterns (one per class)
        acc_cifar, ent_cifar = _run(X_cifar, seed)
        cifar_ok = True
    except Exception as exc:
        print(f"  CIFAR-10 unavailable ({exc}); using random stand-in")
        X_cifar_fallback = torch.randn(D, 10, generator=rng)
        acc_cifar, ent_cifar = _run(X_cifar_fallback, seed)
        cifar_ok = False

    print(f"\nβ values: {BETAS}")
    print(f"Random acc:   {[f'{a:.3f}' for a in acc_random]}")
    print(f"CIFAR acc:    {[f'{a:.3f}' for a in acc_cifar]}")

    save_results(
        name="beta_sweep",
        data_dict={
            "betas": BETAS,
            "acc_random": acc_random,
            "entropy_random": ent_random,
            "acc_cifar": acc_cifar,
            "entropy_cifar": ent_cifar,
        },
        config_dict={"N": N, "d": D, "sigma": SIGMA, "cifar_ok": cifar_ok},
        seed=seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.seed)
