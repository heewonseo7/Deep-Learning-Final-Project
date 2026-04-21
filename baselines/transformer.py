"""SmallTransformer: standard attention vs Hopfield attention swap experiment."""

import torch
import torch.nn as nn
from torch import Tensor

from hopfield.network import HopfieldLayer


class _TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, use_hopfield: bool, dropout: float) -> None:
        super().__init__()
        if use_hopfield:
            self.attn = HopfieldLayer(d_model, d_model, dropout=dropout)
        else:
            self.attn = nn.MultiheadAttention(
                d_model, num_heads, dropout=dropout, batch_first=True
            )
        self.use_hopfield = use_hopfield
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.use_hopfield:
            attn_out = self.attn(x, x)
        else:
            attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class SmallTransformer(nn.Module):
    """Two-layer transformer for sequence classification.

    Optionally replaces nn.MultiheadAttention with HopfieldLayer in every block.

    Args:
        d_model:      Model / embedding dimension.
        num_classes:  Number of output classes.
        num_heads:    Attention heads (used only when use_hopfield=False).
        num_layers:   Number of transformer blocks.
        use_hopfield: If True, replace attention with HopfieldLayer.
        dropout:      Dropout probability throughout.
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        num_heads: int = 4,
        num_layers: int = 2,
        use_hopfield: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            _TransformerBlock(d_model, num_heads, use_hopfield, dropout)
            for _ in range(num_layers)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)   # mean-pool over sequence
        self.head = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Args:
            x: (B, seq_len, d_model)
        Returns:
            logits (B, num_classes)
        """
        for block in self.blocks:
            x = block(x)
        # mean-pool over sequence dimension
        pooled = x.mean(dim=1)           # (B, d_model)
        return self.head(self.dropout(pooled))
