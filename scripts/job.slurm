#!/bin/bash
#SBATCH --job-name=mps-knn-sweep
#SBATCH --output=results/logs/slurm-%j.out
#SBATCH --error=results/logs/slurm-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

# Optional: adapt to your cluster module setup.
# module load python/3.11

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# First run a small sanity check, then the full sweep.
python run_mps_knn_from_config.py --config mps_embedding_config.json --mode sanity
python run_mps_knn_from_config.py --config mps_embedding_config.json --mode full
