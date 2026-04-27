#!/usr/bin/env python3
"""Print all artifacts under results/ for reporting (txt, npz, png)."""

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "results"
    if not root.is_dir():
        print(f"No results dir: {root}")
        return

    exts = {".txt", ".npz", ".png", ".csv"}
    paths = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts
    )
    if not paths:
        print(f"No result files ({', '.join(sorted(exts))}) under {root}")
        return

    print(f"Results under {root}:\n")
    for p in paths:
        rel = p.relative_to(root)
        try:
            sz = p.stat().st_size
        except OSError:
            sz = -1
        print(f"  {rel}  ({sz} bytes)")


if __name__ == "__main__":
    main()
