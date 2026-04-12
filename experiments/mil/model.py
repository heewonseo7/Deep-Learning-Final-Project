"""MIL classifier built on HopfieldPooling for MUSK / Fox / Tiger / Elephant benchmarks."""

import torch
import torch.nn as nn
from torch import Tensor

from hopfield.pooling import HopfieldPooling


class HopfieldMIL(nn.Module):
    """Multiple-instance learning classifier using Hopfield pooling.

    Architecture:
        feature_encoder → HopfieldPooling → MLP classifier

    Args:
        input_dim:   Raw instance feature dimension.
        hidden_dim:  Hopfield association space dimension.
        beta:        Inverse temperature for the pooling layer.
        num_heads:   Number of Hopfield heads.
        dropout:     Dropout rate in the classifier head.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        beta: float = 1.0,
        num_heads: int = 1,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        # TODO: build feature encoder (linear + ReLU), HopfieldPooling, binary classifier head
        pass

    def forward(self, bags: Tensor, mask: Tensor | None = None) -> Tensor:
        """Classify bags.

        Args:
            bags: (B, N, input_dim)
            mask: (B, N) bool — True = valid instance

        Returns:
            Logits (B, 1).
        """
        # TODO: encode instances, pool with Hopfield, classify
        pass
