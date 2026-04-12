"""Standard soft-attention MIL baseline (Ilse et al., 2018) — no gating."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AttentionMIL(nn.Module):
    """MIL classifier with a single-layer attention pooling mechanism.

    Score each instance with a learned attention weight, then compute the
    weighted sum of instance embeddings as the bag representation.

    Args:
        input_dim:  Instance feature dimension.
        hidden_dim: Attention network hidden dimension (L in the paper).
        dropout:    Dropout on attention weights.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        # TODO: attention network: linear(input_dim → hidden_dim) → tanh → linear(hidden_dim → 1)
        # TODO: classifier head: linear(input_dim → 1)
        pass

    def forward(self, bags: Tensor, mask: Tensor | None = None) -> Tensor:
        """Classify bags.

        Args:
            bags: (B, N, input_dim)
            mask: (B, N) bool — True = valid instance

        Returns:
            Logits (B, 1).
        """
        # TODO: compute attention scores, mask, softmax, weighted sum, classify
        pass
