import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate per-run metric JSONs into one CSV")
    parser.add_argument("--mode-dir", type=Path, required=True, help="Path like full_sweep or sanity_check")
    args = parser.parse_args()

    metrics_dir = args.mode_dir / "results" / "metrics"
    out_csv = metrics_dir / "knn_sweep_results.csv"

    json_files = sorted(p for p in metrics_dir.glob("nq*_bond*.json") if p.is_file())
    if len(json_files) == 0:
        print(f"No per-run metrics found in {metrics_dir}")
        return

    rows = []
    for path in json_files:
        with path.open("r", encoding="utf-8") as f:
            rows.append(json.load(f))

    rows.sort(key=lambda r: (int(r["n_qubits"]), int(r["bond_dim"])))
    fieldnames = [
        "n_qubits",
        "bond_dim",
        "train_samples",
        "test_samples",
        "num_epochs",
        "batch_size",
        "knn_k",
        "knn_accuracy",
        "final_train_loss",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
