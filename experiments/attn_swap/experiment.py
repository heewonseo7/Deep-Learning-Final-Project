"""Attention swap: SmallTransformer with standard vs Hopfield attention on MNIST few-shot."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from baselines.transformer import SmallTransformer
from data.loaders import get_fashion_mnist, get_mnist
from experiments.config_utils import load_yaml_config
from experiments.runner import save_results
from experiments.utils import softmax_entropy

PATCH_SIZE = 7
NUM_PATCHES = 16
PATCH_DIM = 49


class MNISTPatcher(nn.Module):
    """Flatten MNIST into 16 non-overlapping 7×7 patches and project to d_model."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(PATCH_DIM, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = x.view(B, 1, 4, 7, 4, 7).permute(0, 2, 4, 1, 3, 5)
        x = x.contiguous().view(B, 16, PATCH_DIM)
        return self.proj(x)


def train_and_eval(
    k: int,
    use_hopfield: bool,
    seed: int,
    device: torch.device,
    *,
    d_model: int,
    num_classes: int,
    num_heads: int,
    num_layers: int,
    dropout: float,
    epochs: int,
    lr: float,
) -> float:
    torch.manual_seed(seed)
    patcher = MNISTPatcher(d_model).to(device)
    model = SmallTransformer(
        d_model,
        num_classes,
        num_heads=num_heads,
        num_layers=num_layers,
        use_hopfield=use_hopfield,
        dropout=dropout,
    ).to(device)
    params = list(patcher.parameters()) + list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    batch_size = min(32, k * 10)
    train_loader = get_mnist(train=True, few_shot_k=k, seed=seed, batch_size=batch_size)
    test_loader = get_mnist(train=False, batch_size=256)

    for _ in range(epochs):
        patcher.train()
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            patches = patcher(x)
            logits = model(patches)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    patcher.eval()
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(patcher(x))
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
    return correct / total


def ood_entropy(
    use_hopfield: bool,
    k: int,
    seed: int,
    device: torch.device,
    *,
    d_model: int,
    num_classes: int,
    num_heads: int,
    num_layers: int,
    dropout: float,
    epochs: int,
    lr: float,
    ood_dataset: str,
) -> float:
    """Train on MNIST with k shots per class; evaluate mean softmax entropy on OOD data."""
    if ood_dataset != "fashion_mnist":
        raise ValueError(f"Unsupported ood_dataset: {ood_dataset!r} (only 'fashion_mnist')")

    torch.manual_seed(seed)
    patcher = MNISTPatcher(d_model).to(device)
    model = SmallTransformer(
        d_model,
        num_classes,
        num_heads=num_heads,
        num_layers=num_layers,
        use_hopfield=use_hopfield,
        dropout=dropout,
    ).to(device)
    params = list(patcher.parameters()) + list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    train_loader = get_mnist(train=True, few_shot_k=k, seed=seed, batch_size=32)
    for _ in range(epochs):
        patcher.train()
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(patcher(x))
            F.cross_entropy(logits, y).backward()
            optimizer.step()

    fmnist_loader = get_fashion_mnist(batch_size=256)
    patcher.eval()
    model.eval()
    all_ent = []
    with torch.no_grad():
        for x, _ in fmnist_loader:
            logits = model(patcher(x.to(device)))
            all_ent.append(softmax_entropy(logits).cpu())
    return torch.cat(all_ent).mean().item()


def main(seed: int, config_path: Path) -> None:
    cfg = load_yaml_config(config_path)

    d_model = int(cfg["d_model"])
    num_classes = int(cfg["num_classes"])
    num_heads = int(cfg["num_heads"])
    num_layers = int(cfg["num_layers"])
    dropout = float(cfg["dropout"])
    epochs = int(cfg["epochs"])
    lr = float(cfg["lr"])
    few_shot_k = list(cfg["few_shot_k"])
    seeds = list(cfg["seeds"])
    ood_dataset = str(cfg.get("ood_dataset", "fashion_mnist"))
    ood_few_shot_k = int(cfg.get("ood_few_shot_k", 100))

    common = dict(
        d_model=d_model,
        num_classes=num_classes,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        epochs=epochs,
        lr=lr,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {config_path}")

    acc_std_all = {k: [] for k in few_shot_k}
    acc_hop_all = {k: [] for k in few_shot_k}

    for s in seeds:
        for k in tqdm(few_shot_k, desc=f"seed={s}"):
            acc_std_all[k].append(
                train_and_eval(k, use_hopfield=False, seed=s, device=device, **common)
            )
            acc_hop_all[k].append(
                train_and_eval(k, use_hopfield=True, seed=s, device=device, **common)
            )

    acc_std = [np.mean(acc_std_all[k]) for k in few_shot_k]
    acc_hop = [np.mean(acc_hop_all[k]) for k in few_shot_k]
    std_std = [np.std(acc_std_all[k]) for k in few_shot_k]
    std_hop = [np.std(acc_hop_all[k]) for k in few_shot_k]

    print(f"\nShots: {few_shot_k}")
    print(f"Standard acc: {[f'{a:.3f}' for a in acc_std]}")
    print(f"Hopfield acc: {[f'{a:.3f}' for a in acc_hop]}")

    ent_std = ood_entropy(
        use_hopfield=False,
        k=ood_few_shot_k,
        seed=seed,
        device=device,
        ood_dataset=ood_dataset,
        **common,
    )
    ent_hop = ood_entropy(
        use_hopfield=True,
        k=ood_few_shot_k,
        seed=seed,
        device=device,
        ood_dataset=ood_dataset,
        **common,
    )
    print(f"\nOOD entropy — standard: {ent_std:.4f}, hopfield: {ent_hop:.4f}")

    save_results(
        name="attn_swap",
        data_dict={
            "shots": few_shot_k,
            "acc_standard": acc_std,
            "acc_hopfield": acc_hop,
            "std_standard": std_std,
            "std_hopfield": std_hop,
            "ood_entropy_standard": [ent_std],
            "ood_entropy_hopfield": [ent_hop],
        },
        config_dict={
            "d_model": d_model,
            "epochs": epochs,
            "seeds": str(seeds),
            "lr": lr,
            "ood_few_shot_k": ood_few_shot_k,
            "ood_dataset": ood_dataset,
            "config_path": str(config_path),
        },
        seed=seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Attention swap: standard vs Hopfield attention on MNIST."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/attn_swap.yaml"),
        help="YAML hyperparameters (default: configs/attn_swap.yaml)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for OOD entropy pair of runs (main few-shot uses seeds from config).",
    )
    args = parser.parse_args()
    main(args.seed, args.config)
