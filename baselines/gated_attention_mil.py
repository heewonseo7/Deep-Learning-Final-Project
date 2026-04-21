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
        self.attention_v = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.attention_u = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, bags: Tensor, mask: Tensor | None = None) -> Tensor:
        """Classify bags.

        Args:
            bags: (B, N, input_dim)
            mask: (B, N) bool — True = valid instance

        Returns:
            Logits (B, 1).
        """
        v = self.attention_v(bags)                          # (B, N, hidden_dim)
        u = self.attention_u(bags)                          # (B, N, hidden_dim)
        scores = self.attention_w(v * u).squeeze(-1)        # (B, N)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)                 # (B, N)
        weights = self.dropout(weights)
        pooled = (weights.unsqueeze(-1) * bags).sum(dim=1)  # (B, input_dim)
        return self.classifier(pooled)                      # (B, 1)
