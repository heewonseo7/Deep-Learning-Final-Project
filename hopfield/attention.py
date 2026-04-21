"""Core Hopfield update rule: xi_new = X @ softmax(beta * X^T @ xi)."""

import torch
import torch.nn.functional as F
from torch import Tensor


def hopfield_update(
    xi: Tensor,
    X: Tensor,
    beta: float = 1.0,
    num_iter: int = 1,
    normalize: bool = False,
) -> Tensor:
    """Run synchronous Hopfield update steps.

    Args:
        xi:        Query state (d,) or (B, d)
        X:         Stored patterns (d, N) — patterns as columns
        beta:      Inverse temperature
        num_iter:  Number of update steps
        normalize: If True, divide X columns by sqrt(d) before each step

    Returns:
        Updated state, same shape as xi.
    """
    squeeze = xi.dim() == 1
    if squeeze:
        xi = xi.unsqueeze(0)  # (1, d)

    if normalize:
        X = X / (X.shape[0] ** 0.5)

    for _ in range(num_iter):
        logits = beta * (xi @ X)               # (B, N)
        weights = F.softmax(logits, dim=-1)    # (B, N)
        xi = weights @ X.T                    # (B, d)

    return xi.squeeze(0) if squeeze else xi
