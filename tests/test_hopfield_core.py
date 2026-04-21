"""Original core tests — updated to use (d, N) X convention."""

import torch
import torch.nn.functional as F

from hopfield.attention import hopfield_update
from hopfield.energy import hopfield_energy


def cosine_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def test_retrieval_closer_to_queried_pattern():
    torch.manual_seed(0)
    N, d = 5, 64
    beta = 10.0
    X = torch.randn(d, N)                          # (d, N)
    xi_noisy = X[:, 3] + 0.1 * torch.randn(d)
    xi_new = hopfield_update(xi_noisy, X, beta=beta)
    sims = [cosine_sim(xi_new, X[:, i]) for i in range(N)]
    assert sims[3] == max(sims)


def test_energy_strictly_decreases():
    torch.manual_seed(42)
    N, d = 5, 64
    beta = 2.0
    X = torch.randn(d, N)                          # (d, N)
    xi = X[:, 1] + 0.5 * torch.randn(d)
    energies = []
    for _ in range(11):
        energies.append(hopfield_energy(xi, X, beta=beta).item())
        xi = hopfield_update(xi, X, beta=beta)
    for i in range(len(energies) - 1):
        assert energies[i + 1] <= energies[i] + 1e-6
    assert energies[-1] < energies[0]
