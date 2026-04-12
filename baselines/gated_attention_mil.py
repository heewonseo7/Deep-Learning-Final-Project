"""Gated attention MIL baseline (Ilse et al., 2018) — element-wise gate on attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GatedAttentionMIL(nn.Module):
    """MIL classifier with gated attention: attention weights use a tanh ⊙ sigmoid gate.

    The gate adds a second learned pathway that can suppress uninformative
    instances, often improving performance on harder benchmarks.

    Args:
        input_dim:  Instance feature dimension.
        hidden_dim: Attention + gate hidden dimension (L in the paper).
        dropout:    Dropout on gated attention weights.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        # TODO: attention path: linear → tanh; gate path: linear → sigmoid;
        #       combine element-wise, project to scalar score; classifier head
        pass

    def forward(self, bags: Tensor, mask: Tensor | None = None) -> Tensor:
        """Classify bags.

        Args:
            bags: (B, N, input_dim)
            mask: (B, N) bool — True = valid instance

        Returns:
            Logits (B, 1).
        """
        # TODO: compute tanh and sigmoid gate, multiply, score, softmax, pool, classify
        pass
