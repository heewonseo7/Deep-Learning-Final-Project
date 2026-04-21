"""CNN + Hopfield attention pooling for MNIST-Bags MIL."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MNISTBagClassifier(nn.Module):
    def __init__(self, hidden_dim=128, beta=1.0, num_heads=1, dropout=0.0):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 20, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(20, 50, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(800, hidden_dim), nn.ReLU(),
        )
        self.query = nn.Parameter(torch.randn(hidden_dim))
        self.beta = nn.Parameter(torch.tensor(float(beta)))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, bags: Tensor) -> Tensor:
        B, N, C, H, W = bags.shape
        features = self.cnn(bags.view(B * N, C, H, W)).view(B, N, -1)
        scores = self.beta * (features @ self.query)   # (B, N)
        weights = F.softmax(scores, dim=-1)
        pooled = (weights.unsqueeze(-1) * features).sum(dim=1)
        return self.classifier(pooled)
