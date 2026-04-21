"""HopfieldPooling: single learned static query for set aggregation."""

import torch
import torch.nn as nn
from torch import Tensor

from hopfield.attention import hopfield_update


class HopfieldPooling(nn.Module):
    """Hopfield layer with one learned static query.

    Useful for multiple-instance learning: collapses a variable-size set
    of stored patterns to a fixed-size representation by retrieving the
    attractor closest to the learned query.

    Args:
        d:        Pattern / query dimensionality.
        beta:     Initial inverse temperature (learnable).
        num_iter: Number of Hopfield update steps.
    """

    def __init__(self, d: int, beta: float = 1.0, num_iter: int = 1) -> None:
        super().__init__()
        self.num_iter = num_iter
        self.query = nn.Parameter(torch.randn(d))
        self.beta = nn.Parameter(torch.tensor(float(beta)))

    def forward(self, X: Tensor) -> Tensor:
        """Retrieve from stored patterns using the learned static query.

        Args:
            X: Stored patterns (d, N)

        Returns:
            Retrieved pattern (d,)
        """
        return hopfield_update(
            self.query, X, beta=self.beta.item(), num_iter=self.num_iter
        )
