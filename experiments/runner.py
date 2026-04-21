"""Shared result-saving utility for all experiments."""

from pathlib import Path
import numpy as np


RESULTS_DIR = Path("results")


def save_results(name: str, data_dict: dict, config_dict: dict, seed: int) -> Path:
    """Save experiment results to results/{name}_seed{seed}.npz.

    Args:
        name:        Experiment name (used in filename).
        data_dict:   Dict of array-like values to store in the .npz.
        config_dict: Hyper-parameter dict (stored as a separate key 'config').
        seed:        Random seed (used in filename).

    Returns:
        Path to the saved file.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}_seed{seed}.npz"

    save_dict = {k: np.asarray(v) for k, v in data_dict.items()}
    # Flatten config to string entries so numpy can store it
    for k, v in config_dict.items():
        save_dict[f"cfg_{k}"] = np.asarray(str(v))

    np.savez(path, **save_dict)
    print(f"Saved {name} — keys: {list(data_dict.keys())}  → {path}")
    return path
