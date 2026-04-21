"""HopfieldLayer: drop-in transformer attention replacement with Q/K/V projections."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class HopfieldLayer(nn.Module):
    """Hopfield association module with separate Q, K, V linear projections.

    Replaces nn.MultiheadAttention with beta-scaled dot-product softmax.

    Args:
        d_in:     Input feature dimension (for query and context).
        d_out:    Output / association space dimension.
        beta:     Inverse temperature; defaults to 1/sqrt(d_out).
        num_iter: Number of Hopfield update steps per forward pass.
        dropout:  Dropout on attention weights.
        bias:     Whether to add bias to projections.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        beta: float | None = None,
        num_iter: int = 1,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_iter = num_iter
        beta_val = beta if beta is not None else 1.0 / math.sqrt(d_out)
        self.beta = nn.Parameter(torch.tensor(beta_val))
        self.W_Q = nn.Linear(d_in, d_out, bias=bias)
        self.W_K = nn.Linear(d_in, d_out, bias=bias)
        self.W_V = nn.Linear(d_in, d_out, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        """Compute Hopfield-attention output.

        Args:
            query:   (B, T_q, d_in)
            context: (B, T_k, d_in)  — stored patterns (keys + values)

        Returns:
            (B, T_q, d_out)
        """
        Q = self.W_Q(query)    # (B, T_q, d_out)
        K = self.W_K(context)  # (B, T_k, d_out)
        V = self.W_V(context)  # (B, T_k, d_out)

        scores = self.beta * (Q @ K.transpose(-2, -1))  # (B, T_q, T_k)
        for _ in range(self.num_iter - 1):
            # Additional iterations: use retrieved state as next query
            Q = F.softmax(scores, dim=-1) @ V           # (B, T_q, d_out)
            scores = self.beta * (Q @ K.transpose(-2, -1))

        weights = self.dropout(F.softmax(scores, dim=-1))
        return weights @ V                              # (B, T_q, d_out)
