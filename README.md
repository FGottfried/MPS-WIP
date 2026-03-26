# MNIST MPS kNN Sweep

Minimal experiment repository for MNIST classification using the **MPS-based encoder** and **kNN evaluation**.

## What this repo runs

For each combination of:
- `bond_dim in [1,2,4,6,8,10,12,14,16,18,20]`
- `n_qubits (embedding_dim) in [4,8,16,32]`

it will:
1. Train an MPS encoder with a quantum-fidelity pairwise contrastive objective.
2. Extract train/test embeddings.
3. Compute MNIST classification accuracy with kNN.
4. Save per-run metrics and model snapshots.

## Files

- `run_mps_knn_sweep.py`: core sweep module.
- `run_mps_knn_from_config.py`: JSON-driven runner with `sanity`/`full` modes.
- `visualize_knn_sweep.py`: visualization script for CSV sweep results.
- `mps_embedding_config.json`: experiment config (including sanity-check subsets).
- `requirements.txt`: Python dependencies.
- `scripts/job.slurm`: SLURM template for remote execution.
- `results/`: output root.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run sanity check first (2x2)

```bash
python run_mps_knn_from_config.py --config mps_embedding_config.json --mode sanity
```

This writes to:
- `sanity_check/results/metrics/knn_sweep_results.csv`
- `sanity_check/results/logs/`
- `sanity_check/results/models/`

## Run full sweep

```bash
python run_mps_knn_from_config.py --config mps_embedding_config.json --mode full
```

This writes to:
- `full_sweep/results/metrics/knn_sweep_results.csv`
- `full_sweep/results/logs/`
- `full_sweep/results/models/`

## Visualize results

Sanity results:
```bash
python visualize_knn_sweep.py \
  --csv sanity_check/results/metrics/knn_sweep_results.csv \
  --out-dir sanity_check/results/metrics
```

Full sweep results:
```bash
python visualize_knn_sweep.py \
  --csv full_sweep/results/metrics/knn_sweep_results.csv \
  --out-dir full_sweep/results/metrics
```

Generated figures:
- `knn_accuracy_heatmap.png`
- `knn_accuracy_vs_bond.png`

## Notes

- Use `max_train=0` and `max_test=0` in the JSON for full MNIST.
- Default setup is CPU-oriented.
- Runtime can be substantial for the full sweep; sanity mode is intended as a quick pre-flight test.
