"""Sweep β (inverse temperature) and record retrieval error / storage capacity."""

import torch
import numpy as np

from hopfield.attention import hopfield_retrieve
from hopfield.energy import hopfield_energy


def run_beta_sweep(
    num_patterns: int,
    pattern_dim: int,
    beta_values: list[float],
    num_trials: int = 100,
    seed: int = 0,
) -> dict[str, list]:
    """Store `num_patterns` random ±1 patterns and measure retrieval accuracy across β.

    Args:
        num_patterns: Number of stored patterns N.
        pattern_dim:  Pattern dimensionality d.
        beta_values:  List of β values to sweep.
        num_trials:   Retrieval trials per β.
        seed:         RNG seed for reproducibility.

    Returns:
        Dict with keys 'beta', 'retrieval_error', 'energy' mapping to lists.
    """
    # TODO: generate random patterns, add noise to queries, measure convergence error
    pass


def sweep_capacity(
    pattern_dim: int,
    beta: float,
    max_patterns: int,
    step: int = 10,
) -> dict[str, list]:
    """Measure max storage capacity as a function of N for fixed β and d.

    Returns:
        Dict with keys 'n_patterns' and 'error_rate'.
    """
    # TODO: increase N until error rate exceeds threshold, record capacity curve
    pass
