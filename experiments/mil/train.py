"""Training and cross-validation loop for the MIL experiments (Table 1 in the paper)."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import yaml

from experiments.mil.dataset import MILBagDataset, mil_collate_fn
from experiments.mil.model import HopfieldMIL


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for bags, labels, mask in loader:
        bags, labels, mask = bags.to(device), labels.to(device), mask.to(device)
        optimizer.zero_grad()
        logits = model(bags, mask=mask).squeeze(-1)          # (B,)
        loss = criterion(logits, labels.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for bags, labels, mask in loader:
            bags, mask = bags.to(device), mask.to(device)
            logits = model(bags, mask=mask).squeeze(-1).cpu()
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


def cross_validate(config: dict, data_dir: Path, device: torch.device):
    name = config["dataset"]
    cv_cfg = config.get("cross_validation", {})
    n_folds = cv_cfg.get("folds", 10)
    seed = cv_cfg.get("seed", 42)
    t_cfg = config["training"]
    m_cfg = config["model"]

    splits = MILBagDataset.make_cv_splits(name, data_dir, n_folds=n_folds, seed=seed)

    fold_accs, fold_aucs = [], []

    for fold_idx, (train_ds, val_ds) in enumerate(splits):
        train_loader = DataLoader(
            train_ds,
            batch_size=t_cfg.get("batch_size", 1),
            shuffle=True,
            collate_fn=mil_collate_fn,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=t_cfg.get("batch_size", 1),
            shuffle=False,
            collate_fn=mil_collate_fn,
        )

        model = HopfieldMIL(
            input_dim=m_cfg["input_dim"],
            hidden_dim=m_cfg.get("hidden_dim", 256),
            beta=m_cfg.get("beta", 1.0),
            num_heads=m_cfg.get("num_heads", 1),
            dropout=m_cfg.get("dropout", 0.25),
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=t_cfg.get("lr", 1e-4),
            weight_decay=t_cfg.get("weight_decay", 1e-4),
        )
        criterion = nn.BCEWithLogitsLoss()

        log_every = config.get("logging", {}).get("log_every", 10)
        epochs = t_cfg.get("epochs", 100)

        for epoch in range(1, epochs + 1):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            if epoch % log_every == 0:
                metrics = evaluate(model, val_loader, device)
                print(
                    f"  fold {fold_idx+1}/{n_folds}  epoch {epoch:3d}/{epochs}"
                    f"  loss {loss:.4f}  acc {metrics['accuracy']:.3f}"
                    f"  auc {metrics['auc']:.3f}"
                )

        final = evaluate(model, val_loader, device)
        fold_accs.append(final["accuracy"])
        fold_aucs.append(final["auc"])
        print(
            f"Fold {fold_idx+1}/{n_folds} done — "
            f"acc {final['accuracy']:.3f}  auc {final['auc']:.3f}"
        )

    mean_acc = torch.tensor(fold_accs).mean().item()
    std_acc = torch.tensor(fold_accs).std().item()
    mean_auc = torch.tensor(fold_aucs).mean().item()
    std_auc = torch.tensor(fold_aucs).std().item()

    print(f"\n{'='*50}")
    print(f"Dataset: {name}  ({n_folds}-fold CV)")
    print(f"Accuracy : {mean_acc:.3f} ± {std_acc:.3f}")
    print(f"AUC      : {mean_auc:.3f} ± {std_auc:.3f}")

    save_dir = Path(config.get("logging", {}).get("save_dir", f"results/{name}"))
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "results.txt", "w") as f:
        f.write(f"dataset: {name}\n")
        f.write(f"accuracy: {mean_acc:.4f} +/- {std_acc:.4f}\n")
        f.write(f"auc: {mean_auc:.4f} +/- {std_auc:.4f}\n")
    print(f"Results saved to {save_dir}/results.txt")


def main():
    parser = argparse.ArgumentParser(description="Train MIL model with Hopfield pooling.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/mil"))
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    cross_validate(config, args.data_dir, device)


if __name__ == "__main__":
    main()
