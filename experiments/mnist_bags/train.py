"""Training script for the MNIST-Bags MIL experiment."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import yaml

from experiments.mnist_bags.dataset import MNISTBags
from experiments.mnist_bags.model import MNISTBagClassifier


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for bags, labels in loader:
        bags, labels = bags.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(bags).squeeze(-1)          # (B,)
        loss = criterion(logits, labels.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for bags, labels in loader:
            logits = model(bags.to(device)).squeeze(-1).cpu()
            all_logits.append(logits)
            all_labels.append(labels)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    preds = (logits > 0).long()
    acc = (preds == labels).float().mean().item()
    try:
        auc = roc_auc_score(labels.numpy(), logits.numpy())
    except ValueError:
        auc = float("nan")
    return {"accuracy": acc, "auc": auc}


def main():
    parser = argparse.ArgumentParser(description="Train MNIST-Bags MIL with Hopfield pooling.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    bags_cfg = config["bags"]
    t_cfg = config["training"]
    m_cfg = config["model"]
    data_dir = Path(config.get("data_dir", "data/mnist"))

    train_ds = MNISTBags(
        root=data_dir,
        train=True,
        num_bags=bags_cfg.get("num_bags_train", 200),
        bag_size=bags_cfg.get("bag_size", 10),
        target_digit=bags_cfg.get("target_digit", 9),
        pos_fraction=bags_cfg.get("pos_fraction", 0.5),
        seed=42,
    )
    test_ds = MNISTBags(
        root=data_dir,
        train=False,
        num_bags=bags_cfg.get("num_bags_test", 50),
        bag_size=bags_cfg.get("bag_size", 10),
        target_digit=bags_cfg.get("target_digit", 9),
        pos_fraction=bags_cfg.get("pos_fraction", 0.5),
        seed=0,
    )

    train_loader = DataLoader(train_ds, batch_size=t_cfg.get("batch_size", 32), shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=t_cfg.get("batch_size", 32), shuffle=False)

    model = MNISTBagClassifier(
        hidden_dim=m_cfg.get("hidden_dim", 128),
        beta=m_cfg.get("beta", 1.0),
        num_heads=m_cfg.get("num_heads", 1),
        dropout=m_cfg.get("dropout", 0.0),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=t_cfg.get("lr", 1e-4),
        weight_decay=t_cfg.get("weight_decay", 1e-4),
    )
    criterion = nn.BCEWithLogitsLoss()

    epochs = t_cfg.get("epochs", 50)
    log_every = config.get("logging", {}).get("log_every", 5)

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        if epoch % log_every == 0:
            metrics = evaluate(model, test_loader, device)
            print(
                f"epoch {epoch:3d}/{epochs}  loss {loss:.4f}"
                f"  acc {metrics['accuracy']:.3f}  auc {metrics['auc']:.3f}"
            )

    final = evaluate(model, test_loader, device)
    print(f"\nFinal — acc {final['accuracy']:.3f}  auc {final['auc']:.3f}")

    save_dir = Path(config.get("logging", {}).get("save_dir", "results/mnist_bags"))
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "results.txt", "w") as f:
        f.write(f"accuracy: {final['accuracy']:.4f}\n")
        f.write(f"auc: {final['auc']:.4f}\n")
    print(f"Results saved to {save_dir}/results.txt")


if __name__ == "__main__":
    main()
