"""Training script for the MNIST-Bags MIL experiment."""

import argparse
from pathlib import Path

import torch
import yaml

from experiments.mnist_bags.dataset import MNISTBags
from experiments.mnist_bags.model import MNISTBagClassifier


def train(model, loader, optimizer, criterion, device):
    """One training epoch.

    Returns:
        Mean binary cross-entropy loss.
    """
    # TODO: standard train loop
    pass


def evaluate(model, loader, device):
    """Compute accuracy and AUC on the given loader.

    Returns:
        dict with 'accuracy' and 'auc'.
    """
    # TODO: collect predictions, compute metrics
    pass


def main():
    parser = argparse.ArgumentParser(description="Train MNIST-Bags MIL with Hopfield pooling.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TODO: build datasets, loaders, model, optimizer; call train / evaluate per epoch
    pass


if __name__ == "__main__":
    main()
