"""MNIST-Bags MIL dataset: bags of MNIST digits with positive label iff target digit present."""

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from pathlib import Path


class MNISTBags(Dataset):
    """Each bag is a collection of MNIST images; label=1 if target digit appears ≥ once.

    Reproduces the synthetic MIL benchmark from Ilse et al. (2018) used as
    an additional evaluation in the Hopfield paper.

    Args:
        root:          Directory for MNIST download/cache.
        train:         Use training split of MNIST.
        num_bags:      Total number of bags to generate.
        bag_size:      Number of instances per bag (fixed).
        target_digit:  The digit whose presence makes a bag positive.
        pos_fraction:  Fraction of bags that are positive.
        seed:          RNG seed.
    """

    def __init__(
        self,
        root: Path = Path("data/mnist"),
        train: bool = True,
        num_bags: int = 200,
        bag_size: int = 10,
        target_digit: int = 9,
        pos_fraction: float = 0.5,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.bag_size = bag_size
        self.target_digit = target_digit

        # Download/load raw MNIST
        mnist = datasets.MNIST(
            root=str(root),
            train=train,
            download=True,
            transform=transforms.ToTensor(),
        )

        # Split raw indices by digit
        labels = mnist.targets  # (60000,) or (10000,)
        target_idx = (labels == target_digit).nonzero(as_tuple=True)[0]
        other_idx = (labels != target_digit).nonzero(as_tuple=True)[0]

        rng = torch.Generator()
        rng.manual_seed(seed)

        num_pos = int(round(num_bags * pos_fraction))
        num_neg = num_bags - num_pos

        self.bags: list[Tensor] = []
        self.labels: list[int] = []

        # Build positive bags (contain ≥1 target digit)
        for _ in range(num_pos):
            # At least one instance must be the target digit
            n_target = torch.randint(1, bag_size, (1,), generator=rng).item()
            t_pick = target_idx[torch.randint(len(target_idx), (n_target,), generator=rng)]
            o_pick = other_idx[torch.randint(len(other_idx), (bag_size - n_target,), generator=rng)]
            pick = torch.cat([t_pick, o_pick])
            # Shuffle order within bag
            pick = pick[torch.randperm(bag_size, generator=rng)]
            bag = torch.stack([mnist[i][0] for i in pick.tolist()])  # (bag_size, 1, 28, 28)
            self.bags.append(bag)
            self.labels.append(1)

        # Build negative bags (no target digit)
        for _ in range(num_neg):
            pick = other_idx[torch.randint(len(other_idx), (bag_size,), generator=rng)]
            bag = torch.stack([mnist[i][0] for i in pick.tolist()])
            self.bags.append(bag)
            self.labels.append(0)

        # Shuffle bag order
        perm = torch.randperm(num_bags, generator=rng).tolist()
        self.bags = [self.bags[i] for i in perm]
        self.labels = [self.labels[i] for i in perm]

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        """Return (bag, label) where bag is (bag_size, 1, 28, 28)."""
        return self.bags[idx], self.labels[idx]
