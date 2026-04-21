"""Plot β-sweep: accuracy and entropy vs inverse temperature."""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")


def _find_phase_transition(betas, accs):
    """Return β at the largest accuracy jump."""
    diffs = [accs[i + 1] - accs[i] for i in range(len(accs) - 1)]
    idx = int(np.argmax(diffs))
    return betas[idx + 1]


def _plot_panel(ax_acc, ax_ent, betas, accs, entropies, title):
    color_acc, color_ent = "steelblue", "darkorange"

    ax_acc.plot(betas, accs, color=color_acc, marker="o", label="Accuracy")
    ax_acc.set_xscale("log")
    ax_acc.set_ylabel("Retrieval accuracy", color=color_acc, fontsize=10)
    ax_acc.tick_params(axis="y", labelcolor=color_acc)
    ax_acc.set_ylim(0, 1.05)

    ax_ent.plot(betas, entropies, color=color_ent, marker="s", linestyle="--", label="Entropy")
    ax_ent.set_ylabel("Softmax entropy (nats)", color=color_ent, fontsize=10)
    ax_ent.tick_params(axis="y", labelcolor=color_ent)

    beta_star = _find_phase_transition(betas, accs)
    ax_acc.axvline(beta_star, color="gray", linestyle=":", alpha=0.8, label=f"β*={beta_star}")
    ax_acc.set_title(title, fontsize=11)
    ax_acc.set_xlabel("β (inverse temperature)")
    ax_acc.legend(loc="lower right", fontsize=8)


def main(seed: int = 0) -> None:
    path = RESULTS_DIR / f"beta_sweep_seed{seed}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Run experiment first: {path}")
    data = np.load(path)

    betas = data["betas"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax1, ax2 = axes

    ax1_r = ax1.twinx()
    _plot_panel(ax1, ax1_r, betas, data["acc_random"].tolist(),
                data["entropy_random"].tolist(), "Random Gaussian patterns")

    ax2_r = ax2.twinx()
    _plot_panel(ax2, ax2_r, betas, data["acc_cifar"].tolist(),
                data["entropy_cifar"].tolist(), "CIFAR-10 class embeddings")

    fig.suptitle("β sweep — Accuracy & Entropy vs Inverse Temperature", fontsize=13)
    fig.tight_layout()

    out = RESULTS_DIR / "beta_sweep.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.seed)
