"""Beta sweep: retrieval accuracy and softmax entropy vs inverse temperature β."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from experiments.config_utils import load_yaml_config
from experiments.runner import save_results
from hopfield.attention import hopfield_update


def _run(
    X: torch.Tensor,
    seed: int,
    *,
    d: int,
    betas: list,
    sigma: float,
    steps: int,
    trials: int,
) -> tuple[list, list]:
    """Sweep β and return (accuracies, mean_entropies)."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    n_patterns = X.shape[1]

    accs, entropies = [], []
    for beta in betas:
        correct, ent_sum = 0, 0.0
        for t in range(trials):
            target_idx = t % n_patterns
            target = X[:, target_idx]
            noisy = target + sigma * torch.randn(d, generator=rng)
            retrieved = hopfield_update(noisy, X, beta=beta, num_iter=steps)
            sim = F.cosine_similarity(retrieved.unsqueeze(0), target.unsqueeze(0)).item()
            correct += float(sim > 0.9)
            logits = beta * (retrieved @ X)    # (N,)
            log_p = F.log_softmax(logits, dim=-1)
            p = torch.exp(log_p)
            ent_sum += -(p * log_p).sum().item()
        accs.append(correct / trials)
        entropies.append(ent_sum / trials)
    return accs, entropies


def main(seed: int, config_path: Path) -> None:
    cfg = load_yaml_config(config_path)

    n_pat = int(cfg["N"])
    d = int(cfg["d"])
    sigma = float(cfg["sigma"])
    steps = int(cfg["steps"])
    trials = int(cfg["trials"])
    betas = list(cfg["betas"])

    print(f"=== Beta sweep (N={n_pat}, d={d}, σ={sigma}) ===")
    print(f"Config: {config_path}")

    rng = torch.Generator()
    rng.manual_seed(seed)

    X_random = torch.randn(d, n_pat, generator=rng)
    print("\n[Random patterns]")
    acc_random, ent_random = _run(
        X_random, seed, d=d, betas=betas, sigma=sigma, steps=steps, trials=trials,
    )

    print("\n[CIFAR-10 embeddings]")
    try:
        from data.loaders import get_cifar10_embeddings
        emb, labels = get_cifar10_embeddings()
        num_classes = 10
        prototypes = torch.stack([
            emb[labels == c].mean(0) for c in range(num_classes)
        ])
        torch.manual_seed(42)
        proj = torch.randn(512, d) / (512 ** 0.5)
        X_cifar = (prototypes @ proj).T
        acc_cifar, ent_cifar = _run(
            X_cifar, seed, d=d, betas=betas, sigma=sigma, steps=steps, trials=trials,
        )
        cifar_ok = True
    except Exception as exc:
        print(f"  CIFAR-10 unavailable ({exc}); using random stand-in")
        X_cifar_fallback = torch.randn(d, 10, generator=rng)
        acc_cifar, ent_cifar = _run(
            X_cifar_fallback, seed, d=d, betas=betas, sigma=sigma, steps=steps, trials=trials,
        )
        cifar_ok = False

    print(f"\nβ values: {betas}")
    print(f"Random acc:   {[f'{a:.3f}' for a in acc_random]}")
    print(f"CIFAR acc:    {[f'{a:.3f}' for a in acc_cifar]}")

    save_results(
        name="beta_sweep",
        data_dict={
            "betas": betas,
            "acc_random": acc_random,
            "entropy_random": ent_random,
            "acc_cifar": acc_cifar,
            "entropy_cifar": ent_cifar,
        },
        config_dict={
            "N": n_pat,
            "d": d,
            "sigma": sigma,
            "steps": steps,
            "trials": trials,
            "cifar_ok": cifar_ok,
            "config_path": str(config_path),
        },
        seed=seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="β sweep: accuracy and entropy vs β.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/beta_sweep.yaml"),
        help="YAML hyperparameters (default: configs/beta_sweep.yaml)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.seed, args.config)
