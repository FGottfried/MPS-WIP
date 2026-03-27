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
- `run_mps_knn_one_from_config.py`: runs exactly one `(n_qubits, bond_dim)` job.
- `visualize_knn_sweep.py`: visualization script for CSV sweep results.
- `mps_embedding_config.json`: experiment config (including sanity-check subsets).
- `requirements.txt`: Python dependencies.
- `scripts/job.slurm`: SLURM array worker (one combo per task).
- `scripts/prepare_pending_jobs.py`: creates pending job list by skipping completed runs.
- `scripts/submit_remaining.sh`: submits only remaining jobs (resumable workflow).
- `scripts/aggregate_metrics.py`: merges per-run metric JSONs into one CSV.
- `results/`: output root.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Local single-process mode

```bash
python run_mps_knn_from_config.py --config mps_embedding_config.json --mode sanity
```

```bash
python run_mps_knn_from_config.py --config mps_embedding_config.json --mode full
```

## Recommended SLURM mode (many small resumable jobs)

Sanity set (2x2):
```bash
./scripts/submit_remaining.sh sanity mps_embedding_config.json
```

Full set:
```bash
./scripts/submit_remaining.sh full mps_embedding_config.json
```

If jobs are interrupted, run the same command again.  
Only unfinished combinations are submitted.

After jobs finish, aggregate per-run metrics:
```bash
python scripts/aggregate_metrics.py --mode-dir sanity_check
python scripts/aggregate_metrics.py --mode-dir full_sweep
```

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
