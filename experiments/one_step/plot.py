"""Plot one-step heatmap: retrieval accuracy vs (iterations × noise sigma)."""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")


def main(seed: int = 0) -> None:
    path = RESULTS_DIR / f"onestep_heatmap_seed{seed}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Run experiment first: {path}")
    data = np.load(path)

    grid = data["accuracy"]
    iter_vals = data["iter_values"].tolist()
    sigma_vals = data["sigma_values"].tolist()

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid, aspect="auto", vmin=0, vmax=1, cmap="viridis", origin="lower")
    plt.colorbar(im, ax=ax, label="Retrieval accuracy")

    ax.set_xticks(range(len(sigma_vals)))
    ax.set_xticklabels([f"{s}" for s in sigma_vals])
    ax.set_yticks(range(len(iter_vals)))
    ax.set_yticklabels([f"{it}" for it in iter_vals])
    ax.set_xlabel("Noise σ", fontsize=12)
    ax.set_ylabel("Update iterations", fontsize=12)
    ax.set_title("Retrieval Accuracy vs Iterations & Noise", fontsize=13)

    for i in range(len(iter_vals)):
        for j in range(len(sigma_vals)):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                    color="white" if grid[i, j] < 0.5 else "black", fontsize=9)

    fig.tight_layout()
    out = RESULTS_DIR / "onestep_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.seed)
