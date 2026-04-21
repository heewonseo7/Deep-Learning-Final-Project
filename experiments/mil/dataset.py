"""Loaders for MIL benchmark datasets: MUSK1, MUSK2, Fox, Tiger, Elephant."""

import numpy as np
import torch
from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold


DATASET_NAMES = ["musk1", "musk2", "fox", "tiger", "elephant"]

# Expected feature dimensions for each dataset
FEATURE_DIMS = {
    "musk1": 166,
    "musk2": 166,
    "fox": 230,
    "tiger": 230,
    "elephant": 230,
}


def _load_mat(path: Path) -> tuple[list[np.ndarray], np.ndarray]:
    """Load a .mat file and return (bags, bag_labels).

    Supports the format used by Ilse et al. (2018) / Ramsauer et al. (2020):
        mat['X']  — object array of shape (N_bags,); each element is (n_i, d) instances
        mat['Y']  — float array of shape (N_bags, 1) with bag-level binary labels

    Also supports the flat format:
        mat['data']   — (total_instances, d + 1) where column 0 is bag_id
        mat['labels'] — (N_bags,) bag-level labels
    """
    import scipy.io

    mat = scipy.io.loadmat(str(path))

    # Format 1: cell-array bags
    if "X" in mat and "Y" in mat:
        X_cell = mat["X"].squeeze()  # object array of (n_i, d) arrays
        Y = mat["Y"].squeeze().astype(int)
        bags = [np.array(X_cell[i], dtype=np.float32) for i in range(len(X_cell))]
        return bags, Y

    # Format 2: flat table with bag_id column
    if "data" in mat and "labels" in mat:
        data = mat["data"].astype(np.float32)
        labels = mat["labels"].squeeze().astype(int)
        bag_ids = data[:, 0].astype(int)
        features = data[:, 1:]
        unique_ids = np.unique(bag_ids)
        bags = [features[bag_ids == bid] for bid in unique_ids]
        return bags, labels

    raise ValueError(
        f"Unrecognised .mat format in {path}. "
        "Expected keys ('X','Y') or ('data','labels')."
    )


class MILBagDataset(Dataset):
    """Generic multiple-instance learning bag dataset.

    Each sample is a bag (variable-length set of feature vectors) with a
    binary bag-level label (positive if ≥1 instance is positive).

    Args:
        name:      One of DATASET_NAMES.
        data_dir:  Root directory containing raw .mat files.
        indices:   Explicit list of bag indices to include (for CV splits).
                   If None, all bags are used.
    """

    def __init__(
        self,
        name: str,
        data_dir: Path,
        indices: list[int] | None = None,
    ) -> None:
        assert name in DATASET_NAMES, f"Unknown dataset '{name}'. Choose from {DATASET_NAMES}."
        data_dir = Path(data_dir)
        mat_path = data_dir / f"{name}.mat"

        if not mat_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {mat_path}\n"
                f"Run `uv run python data/download_mil.py` to fetch all MIL datasets."
            )

        all_bags, all_labels = _load_mat(mat_path)

        if indices is None:
            indices = list(range(len(all_bags)))

        self.bags: list[np.ndarray] = [all_bags[i] for i in indices]
        self.labels: list[int] = [int(all_labels[i]) for i in indices]
        self.name = name

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        """Return (bag_tensor, label) where bag_tensor is (N_i, d)."""
        bag = torch.from_numpy(self.bags[idx])
        return bag, self.labels[idx]

    # ------------------------------------------------------------------
    # Cross-validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def make_cv_splits(
        name: str,
        data_dir: Path,
        n_folds: int = 10,
        seed: int = 42,
    ) -> list[tuple["MILBagDataset", "MILBagDataset"]]:
        """Return list of (train_dataset, val_dataset) pairs for k-fold CV.

        Stratified so each fold preserves the positive/negative ratio.
        """
        data_dir = Path(data_dir)
        mat_path = data_dir / f"{name}.mat"
        all_bags, all_labels = _load_mat(mat_path)
        all_indices = np.arange(len(all_bags))

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = []
        for train_idx, val_idx in skf.split(all_indices, all_labels):
            train_ds = MILBagDataset.__new__(MILBagDataset)
            train_ds.bags = [all_bags[i] for i in train_idx]
            train_ds.labels = [int(all_labels[i]) for i in train_idx]
            train_ds.name = name

            val_ds = MILBagDataset.__new__(MILBagDataset)
            val_ds.bags = [all_bags[i] for i in val_idx]
            val_ds.labels = [int(all_labels[i]) for i in val_idx]
            val_ds.name = name

            splits.append((train_ds, val_ds))
        return splits


def mil_collate_fn(batch: list[tuple[Tensor, int]]) -> tuple[Tensor, Tensor, Tensor]:
    """Collate bags of variable length into padded tensors + mask.

    Returns:
        bags:   (B, max_N, d) float tensor
        labels: (B,) long tensor
        mask:   (B, max_N) bool tensor — True = valid instance
    """
    bags, labels = zip(*batch)
    max_n = max(b.shape[0] for b in bags)
    d = bags[0].shape[1]

    padded = torch.zeros(len(bags), max_n, d)
    mask = torch.zeros(len(bags), max_n, dtype=torch.bool)

    for i, bag in enumerate(bags):
        n = bag.shape[0]
        padded[i, :n] = bag
        mask[i, :n] = True

    return padded, torch.tensor(labels, dtype=torch.long), mask
