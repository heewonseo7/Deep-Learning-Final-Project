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
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
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
        scores = self.attention(bags).squeeze(-1)  # (B, N)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)        # (B, N)
        weights = self.dropout(weights)
        pooled = (weights.unsqueeze(-1) * bags).sum(dim=1)  # (B, input_dim)
        return self.classifier(pooled)             # (B, 1)
