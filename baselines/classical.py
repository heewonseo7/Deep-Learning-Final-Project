"""Classical binary Hopfield network (Hopfield, 1982)."""

import numpy as np
import torch
from torch import Tensor


class ClassicalHopfield:
    """Binary {-1, +1} Hopfield network with Hebbian weight matrix.

    Capacity: ~0.138 * d patterns before catastrophic interference.

    Args:
        d: Pattern dimensionality.
    """

    def __init__(self, d: int) -> None:
        self.d = d
        self.W = np.zeros((d, d), dtype=np.float32)

    def store(self, patterns: Tensor) -> None:
        """Hebbian learning: W = X^T X / N, zero diagonal.

        Args:
            patterns: (N, d) binary {-1, +1} matrix.
        """
        X = patterns.numpy().astype(np.float32)  # (N, d)
        N = X.shape[0]
        self.W = (X.T @ X) / N
        np.fill_diagonal(self.W, 0.0)

    def retrieve(self, query: Tensor, num_iter: int = 20) -> Tensor:
        """Synchronous update: xi = sign(W @ xi).

        Args:
            query:    (d,) binary {-1, +1} initial state.
            num_iter: Number of synchronous update steps.

        Returns:
            (d,) retrieved binary pattern.
        """
        xi = query.numpy().astype(np.float32)
        for _ in range(num_iter):
            xi = np.sign(self.W @ xi)
            xi[xi == 0] = 1.0  # break ties to +1
        return torch.from_numpy(xi)

    def energy(self, xi: Tensor) -> float:
        """E(xi) = -0.5 * xi^T W xi.

        Args:
            xi: (d,) binary state.

        Returns:
            Scalar energy.
        """
        x = xi.numpy().astype(np.float32)
        return float(-0.5 * x @ self.W @ x)
