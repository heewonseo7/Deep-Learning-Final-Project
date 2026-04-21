"""Download and convert MIL benchmark datasets to the .mat format expected by MILBagDataset.

Datasets:
  MUSK1, MUSK2  — UCI ML Repository (Dietterich et al., 1997); downloaded automatically.
  Fox, Tiger, Elephant — Corel image MIL benchmarks (Andrews et al., 2003).
                          These are not freely redistributable; this script generates
                          statistically-equivalent synthetic surrogates for development.
                          Replace with real files (name them fox.mat, tiger.mat,
                          elephant.mat in data/mil/) when available.

Usage:
    uv run python data/download_mil.py [--out-dir data/mil]
"""

import argparse
import csv
import io
import urllib.request
import shutil
import tempfile
from pathlib import Path

import numpy as np
import scipy.io


# ---------------------------------------------------------------------------
# MUSK1 / MUSK2 from UCI
# ---------------------------------------------------------------------------
_UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/musk"
_UCI_SOURCES = {
    "musk1": f"{_UCI_BASE}/clean1.data.Z",
    "musk2": f"{_UCI_BASE}/clean2.data.Z",
}

_MUSK_DIMS = 166

# Approximate bag/instance stats after parsing
_EXPECTED: dict[str, dict] = {
    "musk1":    {"n_bags": 92,  "n_instances": 476,  "n_features": 166},
    "musk2":    {"n_bags": 102, "n_instances": 6598, "n_features": 166},
    "elephant": {"n_bags": 200, "n_instances": 1391, "n_features": 230},
    "fox":      {"n_bags": 200, "n_instances": 1302, "n_features": 230},
    "tiger":    {"n_bags": 200, "n_instances": 1220, "n_features": 230},
}

# Corel datasets: approximate stats used for synthetic generation
_COREL_STATS = {
    "elephant": {"n_bags": 200, "avg_instances": 7,  "n_features": 230, "pos_ratio": 0.5},
    "fox":      {"n_bags": 200, "avg_instances": 6,  "n_features": 230, "pos_ratio": 0.5},
    "tiger":    {"n_bags": 200, "avg_instances": 6,  "n_features": 230, "pos_ratio": 0.5},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_mat(bags: list[np.ndarray], labels: np.ndarray, path: Path) -> None:
    """Save bags + bag-level labels in the (X, Y) cell format."""
    X_cell = np.empty(len(bags), dtype=object)
    for i, b in enumerate(bags):
        X_cell[i] = b.astype(np.float32)
    Y = labels.reshape(-1, 1).astype(float)
    scipy.io.savemat(str(path), {"X": X_cell, "Y": Y})


def _verify(path: Path, name: str) -> None:
    exp = _EXPECTED[name]
    mat = scipy.io.loadmat(str(path))
    X = mat["X"].squeeze()
    Y = mat["Y"].squeeze()
    n_bags = len(X)
    n_instances = sum(X[i].shape[0] for i in range(n_bags))
    n_features = X[0].shape[1]
    ok = n_bags == exp["n_bags"] and n_features == exp["n_features"]
    tag = "OK" if ok else "MISMATCH"
    print(
        f"  [{tag}] {name}: {n_bags} bags, {n_instances} instances, "
        f"{n_features} features, {int(Y.sum())} positive"
    )


# ---------------------------------------------------------------------------
# MUSK downloader
# ---------------------------------------------------------------------------

def _fetch_musk(name: str, out_dir: Path) -> None:
    """Download and convert one MUSK dataset from UCI.

    UCI files are LZW-compressed (.data.Z).  We use /usr/bin/uncompress
    (present on macOS and most Linux installs).
    """
    import subprocess

    UNCOMPRESS = "/usr/bin/uncompress"
    out_path = out_dir / f"{name}.mat"
    url_z = _UCI_SOURCES[name]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        z_file = tmp / f"{name}.data.Z"

        # Download the compressed file
        try:
            print(f"  Downloading {name} from UCI ...", end=" ", flush=True)
            with urllib.request.urlopen(url_z, timeout=30) as resp:
                z_file.write_bytes(resp.read())
            print(f"done ({z_file.stat().st_size // 1024} KB)")
        except Exception as exc:
            print(f"failed ({exc})\n  Generating synthetic substitute.")
            _generate_synthetic_musk(name, out_path)
            return

        # Decompress: uncompress removes the .Z file and leaves <name>.data
        data_file = tmp / f"{name}.data"
        try:
            subprocess.run(
                [UNCOMPRESS, "-f", str(z_file)],
                check=True, capture_output=True,
            )
        except Exception as exc:
            print(f"  uncompress failed ({exc})\n  Generating synthetic substitute.")
            _generate_synthetic_musk(name, out_path)
            return

        # Parse .data CSV:  molecule,conformer,f1,...,f166,label.
        # Label has a trailing period in the raw file; strip it.
        bags: dict[str, list] = {}
        bag_labels: dict[str, int] = {}

        with open(data_file, newline="") as f:
            for row in csv.reader(f):
                row = [c.strip().rstrip(".") for c in row]
                if not row or not row[0]:
                    continue
                mol = row[0]
                features = np.array(row[2 : 2 + _MUSK_DIMS], dtype=np.float32)
                instance_label = int(float(row[-1]))
                bags.setdefault(mol, []).append(features)
                bag_labels[mol] = max(bag_labels.get(mol, 0), instance_label)

        bag_list = [np.stack(bags[m]) for m in bags]
        label_arr = np.array([bag_labels[m] for m in bags], dtype=int)
        _save_mat(bag_list, label_arr, out_path)
        _verify(out_path, name)


def _generate_synthetic_musk(name: str, out_path: Path) -> None:
    """Generate a synthetic MUSK-like dataset for development/testing."""
    exp = _EXPECTED[name]
    rng = np.random.default_rng(42)
    n_bags, n_feat = exp["n_bags"], exp["n_features"]

    bags, labels = [], []
    for i in range(n_bags):
        n_i = rng.integers(2, max(3, exp["n_instances"] // n_bags * 2))
        bag = rng.standard_normal((n_i, n_feat)).astype(np.float32)
        lbl = int(i < n_bags // 2)
        if lbl:
            bag[0] += 1.5  # positive prototype shift
        bags.append(bag)
        labels.append(lbl)

    _save_mat(bags, np.array(labels, dtype=int), out_path)
    print(f"  [SYNTHETIC] {name}: {n_bags} bags, {n_feat} features (real data unavailable)")


# ---------------------------------------------------------------------------
# Corel image sets (Fox / Tiger / Elephant)
# ---------------------------------------------------------------------------

def _generate_corel(name: str, out_path: Path) -> None:
    """Generate a synthetic stand-in for the Corel MIL image datasets.

    The real Fox/Tiger/Elephant datasets are not freely redistributable.
    This synthetic version preserves:
      - Bag/instance count statistics matching the published datasets
      - Feature dimensionality (230)
      - 50% positive bag fraction
    Replace with real .mat files when available.
    """
    stats = _COREL_STATS[name]
    rng = np.random.default_rng({"elephant": 1, "fox": 2, "tiger": 3}[name])

    n_bags = stats["n_bags"]
    avg_n = stats["avg_instances"]
    n_feat = stats["n_features"]
    n_pos = int(n_bags * stats["pos_ratio"])

    bags, labels = [], []
    for i in range(n_bags):
        n_i = max(1, int(rng.integers(avg_n - 2, avg_n + 3)))
        bag = rng.standard_normal((n_i, n_feat)).astype(np.float32)
        lbl = int(i < n_pos)
        if lbl:
            bag[0] += 1.5
        bags.append(bag)
        labels.append(lbl)

    _save_mat(bags, np.array(labels, dtype=int), out_path)
    print(
        f"  [SYNTHETIC] {name}: {n_bags} bags, {n_feat} features "
        "(replace with real .mat file for paper results)"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def download_all(out_dir: Path, force: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in ("musk1", "musk2"):
        path = out_dir / f"{name}.mat"
        if path.exists() and not force:
            print(f"  Skipping {name} (already present — use --force to re-download)")
            _verify(path, name)
        else:
            _fetch_musk(name, out_dir)

    for name in ("fox", "tiger", "elephant"):
        path = out_dir / f"{name}.mat"
        if path.exists() and not force:
            print(f"  Skipping {name} (already present — use --force to regenerate)")
            _verify(path, name)
        else:
            _generate_corel(name, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download / prepare MIL benchmark datasets.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/mil"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    print(f"Output directory: {args.out_dir.resolve()}\n")
    download_all(args.out_dir, force=args.force)

    print("\nAll datasets ready. Next:")
    print("  uv run python -m experiments.mil.train --config configs/mil_musk1.yaml")


if __name__ == "__main__":
    main()
