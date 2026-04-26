"""Attention swap: SmallTransformer with standard vs Hopfield attention on MNIST few-shot."""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from baselines.transformer import SmallTransformer
from data.loaders import get_mnist, get_fashion_mnist
from experiments.runner import save_results
from experiments.utils import softmax_entropy

D_MODEL = 128
NUM_CLASSES = 10
PATCH_SIZE = 7          # 28/4 = 7 → 16 patches of shape (7,7) = 49 pixels → projected to D_MODEL
NUM_PATCHES = 16
PATCH_DIM = 49
EPOCHS = 50
SEEDS = [0, 1, 2]
FEW_SHOT_K = [5, 10, 20, 50, 100, 500]


class MNISTPatcher(nn.Module):
    """Flatten MNIST into 16 non-overlapping 7×7 patches and project to d_model."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(PATCH_DIM, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28)
        B = x.shape[0]
        # Split into 4×4 grid of 7×7 patches → (B, 16, 49)
        x = x.view(B, 1, 4, 7, 4, 7).permute(0, 2, 4, 1, 3, 5)   # (B,4,4,1,7,7)
        x = x.contiguous().view(B, 16, PATCH_DIM)
        return self.proj(x)   # (B, 16, d_model)


def train_and_eval(k: int, use_hopfield: bool, seed: int, device: torch.device) -> float:
    torch.manual_seed(seed)
    patcher = MNISTPatcher(D_MODEL).to(device)
    model = SmallTransformer(D_MODEL, NUM_CLASSES, use_hopfield=use_hopfield).to(device)
    params = list(patcher.parameters()) + list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)

    batch_size = min(32, k * 10)
    train_loader = get_mnist(train=True, few_shot_k=k, seed=seed, batch_size=batch_size)
    test_loader = get_mnist(train=False, batch_size=256)

    for epoch in range(EPOCHS):
        patcher.train(); model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            patches = patcher(x)
            logits = model(patches)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Test accuracy
    patcher.eval(); model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(patcher(x))
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
    return correct / total


def ood_entropy(use_hopfield: bool, k: int, seed: int, device: torch.device) -> float:
    """Train on MNIST k=100, evaluate OOD entropy on Fashion-MNIST."""
    torch.manual_seed(seed)
    patcher = MNISTPatcher(D_MODEL).to(device)
    model = SmallTransformer(D_MODEL, NUM_CLASSES, use_hopfield=use_hopfield).to(device)
    params = list(patcher.parameters()) + list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)

    train_loader = get_mnist(train=True, few_shot_k=100, seed=seed, batch_size=32)
    for epoch in range(EPOCHS):
        patcher.train(); model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(patcher(x))
            F.cross_entropy(logits, y).backward()
            optimizer.step()

    fmnist_loader = get_fashion_mnist(batch_size=256)
    patcher.eval(); model.eval()
    all_ent = []
    with torch.no_grad():
        for x, _ in fmnist_loader:
            logits = model(patcher(x.to(device)))
            all_ent.append(softmax_entropy(logits).cpu())
    return torch.cat(all_ent).mean().item()


def main(seed: int = 0) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    acc_std_all = {k: [] for k in FEW_SHOT_K}
    acc_hop_all = {k: [] for k in FEW_SHOT_K}

    for s in SEEDS:
        for k in tqdm(FEW_SHOT_K, desc=f"seed={s}"):
            acc_std_all[k].append(train_and_eval(k, use_hopfield=False, seed=s, device=device))
            acc_hop_all[k].append(train_and_eval(k, use_hopfield=True, seed=s, device=device))

    acc_std = [np.mean(acc_std_all[k]) for k in FEW_SHOT_K]
    acc_hop = [np.mean(acc_hop_all[k]) for k in FEW_SHOT_K]
    std_std = [np.std(acc_std_all[k]) for k in FEW_SHOT_K]
    std_hop = [np.std(acc_hop_all[k]) for k in FEW_SHOT_K]

    print(f"\nShots: {FEW_SHOT_K}")
    print(f"Standard acc: {[f'{a:.3f}' for a in acc_std]}")
    print(f"Hopfield acc: {[f'{a:.3f}' for a in acc_hop]}")

    ent_std = ood_entropy(use_hopfield=False, k=100, seed=seed, device=device)
    ent_hop = ood_entropy(use_hopfield=True, k=100, seed=seed, device=device)
    print(f"\nOOD entropy — standard: {ent_std:.4f}, hopfield: {ent_hop:.4f}")

    save_results(
        name="attn_swap",
        data_dict={
            "shots": FEW_SHOT_K,
            "acc_standard": acc_std,
            "acc_hopfield": acc_hop,
            "std_standard": std_std,
            "std_hopfield": std_hop,
            "ood_entropy_standard": [ent_std],
            "ood_entropy_hopfield": [ent_hop],
        },
        config_dict={"d_model": D_MODEL, "epochs": EPOCHS, "seeds": str(SEEDS)},
        seed=seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.seed)
