"""Dataset loaders for MNIST, Fashion-MNIST, and CIFAR-10 embeddings."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

CACHE_DIR = Path("cache")
DATA_DIR = Path("data")


def get_mnist(
    train: bool,
    few_shot_k: int | None = None,
    seed: int = 0,
    batch_size: int = 32,
) -> DataLoader:
    """Return a DataLoader for MNIST.

    Args:
        train:       If True, return training split; else test split.
        few_shot_k:  If set, subsample k examples per class (training only).
        seed:        RNG seed for few-shot subsampling.
        batch_size:  DataLoader batch size.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    ds = datasets.MNIST(
        root=DATA_DIR / "mnist", train=train, download=True, transform=transform
    )

    if few_shot_k is not None and train:
        rng = torch.Generator()
        rng.manual_seed(seed)
        targets = torch.as_tensor(ds.targets)
        indices = []
        for c in range(10):
            class_idx = (targets == c).nonzero(as_tuple=True)[0]
            perm = torch.randperm(len(class_idx), generator=rng)
            indices.extend(class_idx[perm[:few_shot_k]].tolist())
        ds = Subset(ds, indices)

    return DataLoader(ds, batch_size=batch_size, shuffle=train)


def get_fashion_mnist(batch_size: int = 256) -> DataLoader:
    """Return a DataLoader for the Fashion-MNIST test split (OOD evaluation).

    Args:
        batch_size: DataLoader batch size.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    ds = datasets.FashionMNIST(
        root=DATA_DIR / "fashion_mnist", train=False, download=True, transform=transform
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def get_cifar10_embeddings() -> tuple[torch.Tensor, torch.Tensor]:
    """Return ResNet-18 embeddings for all 50 000 CIFAR-10 training images.

    Extracts features on first call and caches to cache/cifar10_embeddings.pt.

    Returns:
        emb:    (50000, 512) float32 embedding tensor
        labels: (50000,)     int64 label tensor
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "cifar10_embeddings.pt"

    if cache_path.exists():
        saved = torch.load(cache_path, weights_only=True)
        return saved["emb"], saved["labels"]

    print("Extracting ResNet-18 embeddings for CIFAR-10 (first run, cached afterwards)...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    ds = datasets.CIFAR10(
        root=DATA_DIR / "cifar10", train=True, download=True, transform=transform
    )
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2)

    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    backbone.fc = torch.nn.Identity()  # remove classification head → 512-dim output
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = backbone.to(device).eval()

    all_emb, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            all_emb.append(backbone(x.to(device)).cpu())
            all_labels.append(y)

    emb = torch.cat(all_emb)        # (50000, 512)
    labels = torch.cat(all_labels)  # (50000,)
    torch.save({"emb": emb, "labels": labels}, cache_path)
    print(f"Cached embeddings → {cache_path}")
    return emb, labels
