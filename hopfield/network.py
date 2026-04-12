"""Full Hopfield layer with learnable Q/K/V projections — drop-in transformer attention replacement."""

import torch
import torch.nn as nn
from torch import Tensor


class HopfieldLayer(nn.Module):
    """Hopfield association module with separate query, key, and value projections.

    Mimics the interface of nn.MultiheadAttention but uses the Hopfield update
    rule (scaled dot-product with softmax) and supports arbitrary beta scaling.

    Args:
        input_size:   Dimension of the raw input features.
        hidden_size:  Dimension of the association (Hopfield) space.
        output_size:  Dimension of the output projection.
        beta:         Inverse temperature; if None, uses 1/sqrt(hidden_size).
        num_heads:    Number of parallel Hopfield heads.
        dropout:      Dropout probability on association weights.
        bias:         Whether to add bias to projections.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        beta: float | None = None,
        num_heads: int = 1,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        # TODO: define W_Q, W_K, W_V, W_O linear layers and beta (learnable or fixed)
        pass

    def forward(
        self,
        query: Tensor,
        stored: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Compute Hopfield-attention output.

        Args:
            query:  Query tensor  (B, T_q, input_size)
            stored: Stored patterns / keys+values  (B, T_k, input_size)
            mask:   Key padding mask (B, T_k), True = ignore

        Returns:
            Output tensor (B, T_q, output_size).
        """
        # TODO: project Q/K/V, reshape for heads, compute beta-scaled softmax,
        #       aggregate values, project output
        pass
