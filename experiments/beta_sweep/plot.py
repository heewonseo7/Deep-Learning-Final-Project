"""Plot β-sweep and capacity curves (reproduces Figures 3 & 4 from the paper)."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_retrieval_vs_beta(results: dict, save_path: Path | None = None) -> None:
    """Line plot of retrieval error vs β for multiple pattern counts N.

    Args:
        results:   Output of run_beta_sweep — dict keyed by N with sub-dicts.
        save_path: If given, save figure instead of displaying.
    """
    # TODO: iterate results, plot error curves, add legend, axis labels
    pass


def plot_capacity_curve(results: dict, save_path: Path | None = None) -> None:
    """Plot storage capacity (max N before errors) vs pattern dimension d.

    Args:
        results:   Output of sweep_capacity — dict with 'n_patterns' and 'error_rate'.
        save_path: If given, save figure instead of displaying.
    """
    # TODO: plot capacity vs dimension, compare to classical Hopfield O(d) baseline
    pass


def main():
    # TODO: load or compute sweep results, call plot functions
    pass


if __name__ == "__main__":
    main()
