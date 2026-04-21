"""Five unit tests for the hopfield/ module — all must pass before Phase 6."""

import torch
import torch.nn.functional as F
import pytest

from hopfield.attention import hopfield_update
from hopfield.energy import hopfield_energy
from hopfield.network import HopfieldLayer
from baselines.classical import ClassicalHopfield


# ──────────────────────────────────────────────────────────────────────────────
# Test 1 — Basic retrieval
# ──────────────────────────────────────────────────────────────────────────────
def test_basic_retrieval():
    """Retrieved pattern must be closest to the queried pattern."""
    torch.manual_seed(0)
    d, N = 64, 5
    beta = 10.0

    X = torch.randn(d, N)                        # (d, N) — patterns as columns
    xi_noisy = X[:, 3] + 0.1 * torch.randn(d)   # noisy version of pattern #3

    retrieved = hopfield_update(xi_noisy, X, beta=beta, num_iter=1)

    sims = [F.cosine_similarity(retrieved.unsqueeze(0), X[:, i].unsqueeze(0)).item()
            for i in range(N)]
    print(f"\nCosine sims: {[f'{s:.4f}' for s in sims]}")
    assert sims[3] == max(sims), f"Expected pattern 3 closest, got {sims.index(max(sims))}"


# ──────────────────────────────────────────────────────────────────────────────
# Test 2 — Energy decrease
# ──────────────────────────────────────────────────────────────────────────────
def test_energy_decrease():
    """Energy must be non-increasing at every step, strictly lower at the end."""
    torch.manual_seed(1)
    d, N = 64, 5
    beta = 2.0

    X = torch.randn(d, N)
    xi = X[:, 0] + 0.5 * torch.randn(d)

    energies = []
    for _ in range(11):
        energies.append(hopfield_energy(xi, X, beta=beta).item())
        xi = hopfield_update(xi, X, beta=beta, num_iter=1)

    print(f"\nEnergies: {[f'{e:.5f}' for e in energies]}")

    for t in range(len(energies) - 1):
        assert energies[t + 1] <= energies[t] + 1e-5, (
            f"Energy increased at step {t+1}: {energies[t]:.6f} → {energies[t+1]:.6f}"
        )
    assert energies[-1] < energies[0], "Energy did not decrease overall"


# ──────────────────────────────────────────────────────────────────────────────
# Test 3 — Classical capacity wall
# ──────────────────────────────────────────────────────────────────────────────
def test_classical_capacity():
    """Classical Hopfield: high accuracy at N=5, degraded at N=20 (capacity ~ 0.14*d=14).

    Uses pattern-level accuracy (cosine_sim > 0.9 = correct retrieval) with 15% bit-flip
    noise — the same metric used in the capacity experiment.
    """
    torch.manual_seed(2)
    d = 100
    net = ClassicalHopfield(d)

    def measure_accuracy(N: int) -> float:
        patterns = torch.sign(torch.randn(N, d))
        net.store(patterns)
        correct = 0
        for i in range(N):
            noisy = patterns[i].clone()
            flip = torch.randperm(d)[:int(0.15 * d)]   # 15 bits flipped
            noisy[flip] *= -1
            retrieved = net.retrieve(noisy, num_iter=20)
            # Pattern-level: cosine_sim > 0.9 → correct
            sim = F.cosine_similarity(
                retrieved.float().unsqueeze(0), patterns[i].float().unsqueeze(0)
            ).item()
            correct += float(sim > 0.9)
        return correct / N

    acc5 = measure_accuracy(5)
    acc20 = measure_accuracy(20)
    print(f"\nClassical accuracy — N=5: {acc5:.3f}, N=20: {acc20:.3f}")
    assert acc5 > 0.95, f"Expected acc>0.95 at N=5, got {acc5:.3f}"
    assert acc20 < 0.80, f"Expected acc<0.80 at N=20, got {acc20:.3f}"


# ──────────────────────────────────────────────────────────────────────────────
# Test 4 — Batch support
# ──────────────────────────────────────────────────────────────────────────────
def test_batch_support():
    """hopfield_update must accept batched queries and return (B, d)."""
    torch.manual_seed(3)
    d, N, B = 64, 10, 8
    X = torch.randn(d, N)
    xi_batch = torch.randn(B, d)

    out = hopfield_update(xi_batch, X, beta=5.0, num_iter=1)
    assert out.shape == (B, d), f"Expected ({B}, {d}), got {out.shape}"


# ──────────────────────────────────────────────────────────────────────────────
# Test 5 — HopfieldLayer forward pass
# ──────────────────────────────────────────────────────────────────────────────
def test_hopfield_layer_shape():
    """HopfieldLayer output shape must match (B, T_q, d_out)."""
    torch.manual_seed(4)
    layer = HopfieldLayer(d_in=64, d_out=64)
    query = torch.randn(1, 10, 64)
    context = torch.randn(1, 20, 64)
    out = layer(query, context)
    assert out.shape == (1, 10, 64), f"Expected (1, 10, 64), got {out.shape}"
