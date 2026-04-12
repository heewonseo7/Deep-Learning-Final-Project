"""CNN + HopfieldPooling classifier for the MNIST-Bags MIL benchmark."""

import torch
import torch.nn as nn
from torch import Tensor

from hopfield.pooling import HopfieldPooling


class MNISTBagClassifier(nn.Module):
    """Encode each MNIST instance with a small CNN, then pool with HopfieldPooling.

    Args:
        hidden_dim:  Hopfield association space dimension.
        beta:        Inverse temperature for pooling.
        num_heads:   Number of Hopfield pooling heads.
        dropout:     Dropout in classifier head.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        beta: float = 1.0,
        num_heads: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # TODO: small CNN feature extractor (conv → pool → flatten), HopfieldPooling, classifier
        pass

    def forward(self, bags: Tensor) -> Tensor:
        """Classify bags of MNIST images.

        Args:
            bags: (B, N, 1, 28, 28)

        Returns:
            Logits (B, 1).
        """
        # TODO: reshape to (B*N, 1, 28, 28), encode, reshape back, pool, classify
        pass
