"""Core Hopfield update rule: ξ_new = X · softmax(β · Xᵀ · ξ)."""

import torch
import torch.nn.functional as F
from torch import Tensor


def hopfield_update(xi: Tensor, X: Tensor, beta: float = 1.0) -> Tensor:
    """Run one synchronous update step toward the nearest stored pattern.

    Args:
        xi:   Current query state  (B, d) or (d,)
        X:    Stored patterns      (N, d)
        beta: Inverse temperature

    Returns:
        Updated state of the same shape as xi.
    """
    # TODO: compute attention weights = softmax(beta * X @ xi.T) then weighted sum
    pass


def hopfield_retrieve(xi: Tensor, X: Tensor, beta: float = 1.0, steps: int = 1) -> Tensor:
    """Iterate update rule for `steps` synchronous steps.

    Args:
        xi:    Initial query state (B, d)
        X:     Stored patterns     (N, d)
        beta:  Inverse temperature
        steps: Number of update iterations

    Returns:
        Retrieved pattern after convergence.
    """
    # TODO: loop hopfield_update `steps` times and return final state
    pass
