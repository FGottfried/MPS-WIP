#!/bin/bash
#SBATCH --job-name=mps-knn
#SBATCH --output=results/logs/slurm-%A_%a.out
#SBATCH --error=results/logs/slurm-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-0

set -euo pipefail

MODE=${MODE:-full}
CONFIG_PATH=${CONFIG_PATH:-mps_embedding_config.json}
PENDING_FILE=${PENDING_FILE:-}

if [ -z "${PENDING_FILE}" ]; then
  echo "PENDING_FILE is not set. Submit via scripts/submit_remaining.sh"
  exit 1
fi
if [ ! -f "${PENDING_FILE}" ]; then
  echo "Pending file not found: ${PENDING_FILE}"
  exit 1
fi

line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${PENDING_FILE}")
if [ -z "${line}" ]; then
  echo "No pending line for task ${SLURM_ARRAY_TASK_ID}; exiting."
  exit 0
fi

IFS=',' read -r N_QUBITS BOND_DIM <<< "${line}"
echo "Task ${SLURM_ARRAY_TASK_ID}: mode=${MODE}, n_qubits=${N_QUBITS}, bond_dim=${BOND_DIM}"

# Optional: adapt to your cluster module setup.
# module load python/3.11

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
else
  echo "No Python interpreter found on compute node. Load a Python module first."
  exit 1
fi

if [ ! -d .venv ]; then
  ${PYTHON_BIN} -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
else
  source .venv/bin/activate
fi

${PYTHON_BIN} run_mps_knn_one_from_config.py \
  --config "${CONFIG_PATH}" \
  --mode "${MODE}" \
  --n-qubits "${N_QUBITS}" \
  --bond-dim "${BOND_DIM}" \
  --skip-if-done
