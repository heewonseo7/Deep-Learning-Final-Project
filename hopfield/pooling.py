"""HopfieldPooling: Hopfield layer with a single learned static query for set aggregation."""

import torch
import torch.nn as nn
from torch import Tensor


class HopfieldPooling(nn.Module):
    """Replace an attention query with a single trainable prototype vector.

    Useful for multiple-instance learning where the bag must be collapsed to
    a fixed-size representation regardless of the number of instances.

    Args:
        input_size:  Dimension of input (key/value) features.
        hidden_size: Dimension of the Hopfield association space.
        output_size: Dimension of the pooled output.
        beta:        Inverse temperature (scaling factor before softmax).
        num_heads:   Number of parallel association heads.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        beta: float = 1.0,
        num_heads: int = 1,
    ) -> None:
        super().__init__()
        # TODO: define learned static query, key/value projections, beta parameter
        pass

    def forward(self, X: Tensor, mask: Tensor | None = None) -> Tensor:
        """Pool a set X ∈ ℝ^{B×N×d} to a fixed representation ℝ^{B×output_size}.

        Args:
            X:    Input bag of instances (batch, instances, input_size).
            mask: Optional boolean mask (batch, instances) — True = keep.

        Returns:
            Pooled representation (batch, output_size).
        """
        # TODO: project X to keys/values, expand static query to batch dim,
        #       run Hopfield update, project output
        pass
