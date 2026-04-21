"""Log-sum-exp energy function for the modern continuous Hopfield network."""

import torch
from torch import Tensor


def hopfield_energy(xi: Tensor, X: Tensor, beta: float = 1.0) -> Tensor:
    """Compute E(xi) = -log(sum_i exp(beta * x_i^T xi)) + 0.5 * ||xi||^2.

    Args:
        xi:   Query state  (d,) or (B, d)
        X:    Stored patterns (d, N) — patterns as columns
        beta: Inverse temperature

    Returns:
        Scalar energy (or (B,) for batched input).
    """
    squeeze = xi.dim() == 1
    if squeeze:
        xi = xi.unsqueeze(0)  # (1, d)

    # (B, N) dot products:  xi @ X  ==  X^T @ xi^T  row-wise
    dots = xi @ X                                    # (B, N)
    lse = torch.logsumexp(beta * dots, dim=-1)       # (B,)
    energy = -lse + 0.5 * (xi * xi).sum(dim=-1)     # (B,)

    return energy.squeeze(0) if squeeze else energy
