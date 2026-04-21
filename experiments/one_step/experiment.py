"""One-step retrieval heatmap: accuracy vs (num_iter × noise_sigma)."""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from hopfield.attention import hopfield_update
from experiments.runner import save_results

N = 50
D = 64
BETA = 10.0
TRIALS = 100

ITER_VALUES = [1, 2, 3, 5, 10, 20]
SIGMA_VALUES = [0.05, 0.1, 0.2, 0.3, 0.5]


def run_grid(seed: int) -> np.ndarray:
    """Return (len(ITER_VALUES), len(SIGMA_VALUES)) accuracy grid."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    X = torch.randn(D, N, generator=rng)   # shared pattern set

    grid = np.zeros((len(ITER_VALUES), len(SIGMA_VALUES)))

    for i, num_iter in enumerate(tqdm(ITER_VALUES, desc="iterations")):
        for j, sigma in enumerate(SIGMA_VALUES):
            correct = 0
            for t in range(TRIALS):
                target_idx = t % N
                target = X[:, target_idx]
                noisy = target + sigma * torch.randn(D, generator=rng)
                retrieved = hopfield_update(noisy, X, beta=BETA, num_iter=num_iter)
                sim = F.cosine_similarity(retrieved.unsqueeze(0), target.unsqueeze(0)).item()
                correct += float(sim > 0.9)
            grid[i, j] = correct / TRIALS
    return grid


def main(seed: int = 0) -> None:
    print(f"=== One-step heatmap (N={N}, d={D}, β={BETA}) ===")
    grid = run_grid(seed)

    print("\nRetrieval accuracy grid (rows=iterations, cols=sigma):")
    print(f"{'':>8}", "  ".join(f"σ={s}" for s in SIGMA_VALUES))
    for i, it in enumerate(ITER_VALUES):
        print(f"iter={it:>2}", "  ".join(f"{grid[i,j]:.3f}" for j in range(len(SIGMA_VALUES))))

    save_results(
        name="onestep_heatmap",
        data_dict={
            "accuracy": grid,
            "iter_values": ITER_VALUES,
            "sigma_values": SIGMA_VALUES,
        },
        config_dict={"N": N, "d": D, "beta": BETA, "trials": TRIALS},
        seed=seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.seed)
