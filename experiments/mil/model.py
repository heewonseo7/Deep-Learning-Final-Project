"""MIL classifier using Hopfield-style attention pooling."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class HopfieldMIL(nn.Module):
    """Multiple-instance learning classifier with Hopfield attention pooling.

    Uses a learned static query + beta-scaled softmax to aggregate instance
    features, then classifies the pooled representation.
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
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.query = nn.Parameter(torch.randn(hidden_dim))
        self.beta = nn.Parameter(torch.tensor(beta))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, bags: Tensor, mask: Tensor | None = None) -> Tensor:
        B, N, _ = bags.shape
        instances = self.encoder(bags.view(B * N, -1)).view(B, N, -1)  # (B, N, h)

        # Beta-scaled softmax attention with learned static query
        q = self.query                                 # (h,)
        scores = self.beta * (instances @ q)           # (B, N)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)            # (B, N)
        pooled = (weights.unsqueeze(-1) * instances).sum(dim=1)  # (B, h)

        return self.classifier(pooled)                 # (B, 1)
