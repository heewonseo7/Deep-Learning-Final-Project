"""Loaders for MIL benchmark datasets: MUSK1, MUSK2, Fox, Tiger, Elephant."""

import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


DATASET_NAMES = ["musk1", "musk2", "fox", "tiger", "elephant"]


class MILBagDataset(Dataset):
    """Generic multiple-instance learning bag dataset.

    Each sample is a bag (variable-length set of feature vectors) with a
    binary bag-level label (positive if ≥1 instance is positive).

    Args:
        name:      One of DATASET_NAMES.
        data_dir:  Root directory containing raw .mat / .csv files.
        train:     If True, return training split; else test split.
        seed:      Random seed for train/test split.
    """

    def __init__(self, name: str, data_dir: Path, train: bool = True, seed: int = 42) -> None:
        # TODO: load raw file, parse bags and instance features, stratified split
        pass

    def __len__(self) -> int:
        # TODO: return number of bags
        pass

    def __getitem__(self, idx: int):
        """Return (bag_tensor, label) where bag_tensor is (N_i, d)."""
        # TODO: return padded or ragged bag tensor and scalar label
        pass


def mil_collate_fn(batch):
    """Collate bags of variable length into padded tensors + mask.

    Returns:
        bags:   (B, max_N, d) float tensor
        labels: (B,) long tensor
        mask:   (B, max_N) bool tensor — True = valid instance
    """
    # TODO: pad bags to max bag size in batch, build mask
    pass
