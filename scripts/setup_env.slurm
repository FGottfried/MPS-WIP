#!/bin/bash
#SBATCH --job-name=mps-env
#SBATCH --output=results/logs/setup-env-%j.out
#SBATCH --error=results/logs/setup-env-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

PYTHON_MODULE=${PYTHON_MODULE:-python/3.11.8}
VENV_DIR=${VENV_DIR:-.venv}

mkdir -p results/logs

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
  echo "Set a module, e.g.: python/3.11"
  if command -v module >/dev/null 2>&1; then
    module list || true
  fi
  exit 1
fi

echo "Preparing virtual environment at ${VENV_DIR}"
rm -rf "${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r requirements.txt

"${VENV_DIR}/bin/python" -c "import torch, torchvision, pennylane, matplotlib, numpy; print('Environment OK')"
echo "setup_env completed successfully"
