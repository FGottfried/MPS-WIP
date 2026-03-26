import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_rows(csv_path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "n_qubits": int(row["n_qubits"]),
                    "bond_dim": int(row["bond_dim"]),
                    "knn_accuracy": float(row["knn_accuracy"]),
                }
            )
    if len(rows) == 0:
        raise ValueError(f"No rows found in {csv_path}")
    return rows


def make_heatmap(rows: List[Dict[str, float]], out_path: Path) -> None:
    n_qubits = sorted({int(r["n_qubits"]) for r in rows})
    bond_dims = sorted({int(r["bond_dim"]) for r in rows})

    q_idx = {q: i for i, q in enumerate(n_qubits)}
    b_idx = {b: i for i, b in enumerate(bond_dims)}

    mat = np.full((len(n_qubits), len(bond_dims)), np.nan, dtype=float)
    for r in rows:
        mat[q_idx[int(r["n_qubits"])], b_idx[int(r["bond_dim"])]] = float(r["knn_accuracy"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(bond_dims)))
    ax.set_yticks(range(len(n_qubits)))
    ax.set_xticklabels(bond_dims)
    ax.set_yticklabels(n_qubits)
    ax.set_xlabel("Bond dimension")
    ax.set_ylabel("Number of qubits (embedding dim)")
    ax.set_title("kNN accuracy sweep")

    for i in range(len(n_qubits)):
        for j in range(len(bond_dims)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{100.0 * mat[i, j]:.1f}%", ha="center", va="center", fontsize=8, color="white")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Accuracy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_line_plot(rows: List[Dict[str, float]], out_path: Path) -> None:
    grouped: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for r in rows:
        grouped[int(r["n_qubits"])].append((int(r["bond_dim"]), float(r["knn_accuracy"])))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for q in sorted(grouped.keys()):
        pairs = sorted(grouped[q], key=lambda t: t[0])
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        ax.plot(xs, ys, marker="o", label=f"n_qubits={q}")

    ax.set_xlabel("Bond dimension")
    ax.set_ylabel("kNN accuracy")
    ax.set_title("kNN accuracy vs bond dimension")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize MNIST MPS kNN sweep results")
    parser.add_argument("--csv", type=Path, required=True, help="Path to knn_sweep_results.csv")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for figures")
    args = parser.parse_args()

    rows = load_rows(args.csv)
    best = max(rows, key=lambda r: r["knn_accuracy"])

    out_dir = args.out_dir if args.out_dir is not None else args.csv.parent
    heatmap_path = out_dir / "knn_accuracy_heatmap.png"
    lineplot_path = out_dir / "knn_accuracy_vs_bond.png"

    make_heatmap(rows, heatmap_path)
    make_line_plot(rows, lineplot_path)

    print(f"Saved: {heatmap_path}")
    print(f"Saved: {lineplot_path}")
    print(
        "Best run: "
        f"n_qubits={best['n_qubits']}, bond_dim={best['bond_dim']}, "
        f"kNN={100.0 * best['knn_accuracy']:.2f}%"
    )


if __name__ == "__main__":
    main()
