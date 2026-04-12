"""Log-sum-exp energy function E(ξ) for the modern continuous Hopfield network."""

import torch
from torch import Tensor


def hopfield_energy(xi: Tensor, X: Tensor, beta: float = 1.0) -> Tensor:
    """Compute the Hopfield energy E(ξ) = -lse(β, Xᵀξ) + ½ξᵀξ + (1/β)·log(N).

    Args:
        xi:   Query state  (d,) or (B, d)
        X:    Stored patterns (N, d)
        beta: Inverse temperature controlling retrieval sharpness

    Returns:
        Scalar energy value (or batch of scalars).
    """
    # TODO: implement log-sum-exp energy
    pass
