"""Plot attention swap results: few-shot accuracy and OOD entropy."""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")


def main(seed: int = 0) -> None:
    path = RESULTS_DIR / f"attn_swap_seed{seed}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Run experiment first: {path}")
    data = np.load(path)

    shots = data["shots"].tolist()
    acc_std = data["acc_standard"].tolist()
    acc_hop = data["acc_hopfield"].tolist()
    std_std = data["std_standard"].tolist()
    std_hop = data["std_hopfield"].tolist()
    ent_std = float(data["ood_entropy_standard"])
    ent_hop = float(data["ood_entropy_hopfield"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Subplot 1: few-shot accuracy
    ax1.errorbar(shots, acc_std, yerr=std_std, marker="o", label="Standard attention", capsize=4)
    ax1.errorbar(shots, acc_hop, yerr=std_hop, marker="s", label="Hopfield attention", capsize=4)

    # Mark crossover
    crossover = None
    for i in range(len(shots) - 1):
        if (acc_hop[i] - acc_std[i]) * (acc_hop[i + 1] - acc_std[i + 1]) <= 0:
            crossover = (shots[i] + shots[i + 1]) / 2
            break
    if crossover:
        ax1.axvline(crossover, color="gray", linestyle=":", alpha=0.7, label=f"crossover ~{crossover:.0f}")

    ax1.set_xscale("log")
    ax1.set_xlabel("Shots per class", fontsize=12)
    ax1.set_ylabel("Test accuracy", fontsize=12)
    ax1.set_title("Few-shot MNIST: Standard vs Hopfield Attention", fontsize=11)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: OOD entropy bar chart
    ax2.bar(["Standard", "Hopfield"], [ent_std, ent_hop], color=["steelblue", "darkorange"])
    ax2.set_ylabel("Mean softmax entropy (nats)", fontsize=12)
    ax2.set_title("OOD Entropy on Fashion-MNIST", fontsize=11)
    ax2.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate([ent_std, ent_hop]):
        ax2.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)

    fig.suptitle("Attention Swap Experiment", fontsize=13)
    fig.tight_layout()

    out = RESULTS_DIR / "attn_swap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.seed)
