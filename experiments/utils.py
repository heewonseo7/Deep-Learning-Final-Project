"""Shared utility functions for experiments."""

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def cosine_similarity(a: Tensor, b: Tensor) -> float:
    """Cosine similarity between two 1-D tensors."""
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def softmax_entropy(logits: Tensor) -> Tensor:
    """Shannon entropy of softmax distribution, per sample.

    Args:
        logits: (B, C) raw logits

    Returns:
        (B,) entropy values in nats
    """
    log_p = F.log_softmax(logits, dim=-1)
    p = torch.exp(log_p)
    return -(p * log_p).sum(dim=-1)


def reliability_diagram(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 10,
) -> plt.Figure:
    """Plot a calibration reliability diagram.

    Args:
        confidences: (N,) predicted confidence scores in [0, 1].
        accuracies:  (N,) binary correct/incorrect indicators.
        n_bins:      Number of equal-width confidence bins.

    Returns:
        matplotlib Figure with the reliability diagram.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs, bin_confs, bin_counts = [], [], []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() > 0:
            bin_accs.append(accuracies[mask].mean())
            bin_confs.append(confidences[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_accs.append(0.0)
            bin_confs.append((lo + hi) / 2)
            bin_counts.append(0)

    bin_accs = np.array(bin_accs)
    bin_confs = np.array(bin_confs)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.bar(
        bin_edges[:-1], bin_accs, width=1.0 / n_bins,
        align="edge", alpha=0.7, label="Model"
    )
    ax.plot(bin_confs, bin_accs, "ro-", markersize=4)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig
