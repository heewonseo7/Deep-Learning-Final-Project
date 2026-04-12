"""Training and cross-validation loop for the MIL experiments (Table 1 in the paper)."""

import argparse
from pathlib import Path

import torch
import yaml

from experiments.mil.dataset import MILBagDataset, mil_collate_fn
from experiments.mil.model import HopfieldMIL


def train_one_fold(model, loader, optimizer, criterion, device):
    """Run one training epoch over a single CV fold.

    Returns:
        Mean loss over the epoch.
    """
    # TODO: standard train loop — zero_grad, forward, loss, backward, step
    pass


def evaluate(model, loader, device):
    """Compute accuracy and AUC on a data loader.

    Returns:
        dict with keys 'accuracy' and 'auc'.
    """
    # TODO: collect logits, compute sklearn metrics
    pass


def cross_validate(config: dict, data_dir: Path, device: torch.device):
    """Run k-fold cross-validation and report mean ± std accuracy.

    Args:
        config:   Experiment hyper-parameters loaded from YAML.
        data_dir: Path to raw dataset files.
        device:   Torch device.
    """
    # TODO: loop over folds, train, evaluate, aggregate results
    pass


def main():
    parser = argparse.ArgumentParser(description="Train MIL model with Hopfield pooling.")
    parser.add_argument("--config", type=Path, required=True, help="YAML config file.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/mil"))
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TODO: call cross_validate(config, args.data_dir, device)


if __name__ == "__main__":
    main()
