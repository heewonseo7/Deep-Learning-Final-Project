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
        # TODO: download MNIST, sample bags according to pos_fraction
        pass

    def __len__(self) -> int:
        # TODO: return num_bags
        pass

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        """Return (bag, label) where bag is (bag_size, 1, 28, 28)."""
        # TODO: return sampled bag tensor and label
        pass
