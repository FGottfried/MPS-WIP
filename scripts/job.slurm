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
PYTHON_MODULE=${PYTHON_MODULE:-}
VENV_DIR=${VENV_DIR:-.venv}

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

# Some clusters require initializing the module function in non-login shells.
if [ -f /etc/profile.d/modules.sh ]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

if command -v module >/dev/null 2>&1 && [ -n "${PYTHON_MODULE}" ]; then
  echo "Trying to load module: ${PYTHON_MODULE}"
  module load "${PYTHON_MODULE}" || true
fi

pick_python() {
  for candidate in python3 python; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      candidate_path=$(command -v "${candidate}")
      if "${candidate_path}" -c "import sys; print(sys.version)" >/dev/null 2>&1; then
        echo "${candidate_path}"
        return 0
      fi
    fi
  done
  return 1
}

if PYTHON_BIN=$(pick_python); then
  echo "Using interpreter: ${PYTHON_BIN}"
else
  echo "No working Python interpreter found."
  echo "Set a module, e.g.: PYTHON_MODULE=python/3.11"
  if command -v module >/dev/null 2>&1; then
    echo "Loaded modules:"
    module list || true
  fi
  exit 1
fi

# Build/repair a shared venv once, guarded by an atomic lock directory.
need_bootstrap=0
if [ ! -x "${VENV_DIR}/bin/python" ]; then
  need_bootstrap=1
else
  if ! "${VENV_DIR}/bin/python" -c "import sys; print(sys.version)" >/dev/null 2>&1; then
    need_bootstrap=1
  fi
fi

if [ "${need_bootstrap}" -eq 1 ]; then
  lock_dir="${VENV_DIR}.lock"
  while ! mkdir "${lock_dir}" 2>/dev/null; do
    echo "Waiting for venv lock: ${lock_dir}"
    sleep 2
  done
  trap 'rmdir "${lock_dir}" >/dev/null 2>&1 || true' EXIT

  # Re-check after acquiring lock in case another task completed bootstrap.
  if [ ! -x "${VENV_DIR}/bin/python" ] || ! "${VENV_DIR}/bin/python" -c "import sys; print(sys.version)" >/dev/null 2>&1; then
    echo "Bootstrapping virtual environment at ${VENV_DIR}"
    rm -rf "${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
    "${VENV_DIR}/bin/python" -m pip install --upgrade pip
    "${VENV_DIR}/bin/python" -m pip install -r requirements.txt
  fi
fi

"${VENV_DIR}/bin/python" run_mps_knn_one_from_config.py \
  --config "${CONFIG_PATH}" \
  --mode "${MODE}" \
  --n-qubits "${N_QUBITS}" \
  --bond-dim "${BOND_DIM}" \
  --skip-if-done
