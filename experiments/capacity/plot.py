"""Plot capacity curves for modern vs classical Hopfield."""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")


def _cfg_int(data, key: str, default: int) -> int:
    k = f"cfg_{key}"
    if k not in data.files:
        return default
    return int(float(np.asarray(data[k]).item()))


def main(seed: int = 0) -> None:
    path = RESULTS_DIR / f"capacity_seed{seed}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Run experiment first: {path}")
    data = np.load(path)

    d_modern = _cfg_int(data, "d_modern", 64)
    d_classical = _cfg_int(data, "d_classical", 100)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        data["N_modern"],
        data["acc_modern"],
        "b-o",
        label=f"Modern Hopfield (d={d_modern})",
    )
    ax.plot(
        data["N_classical"],
        data["acc_classical"],
        "r-s",
        label=f"Classical Hopfield (d={d_classical})",
    )

    classical_limit = 0.138 * d_classical
    ax.axvline(classical_limit, color="red", linestyle="--", alpha=0.7,
               label=f"Classical limit (0.14×{d_classical}={classical_limit:.0f})")

    ax.set_xscale("log")
    ax.set_xlabel("Number of stored patterns", fontsize=12)
    ax.set_ylabel("Retrieval accuracy", fontsize=12)
    ax.set_title("Storage Capacity: Modern vs Classical Hopfield", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    out = RESULTS_DIR / "capacity_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.seed)
