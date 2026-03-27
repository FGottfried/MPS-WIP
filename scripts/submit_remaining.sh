#!/bin/bash
set -euo pipefail

MODE=${1:-full}
CONFIG_PATH=${2:-mps_embedding_config.json}

if [[ "${MODE}" != "full" && "${MODE}" != "sanity" ]]; then
  echo "Usage: $0 [full|sanity] [config_path]"
  exit 1
fi

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
else
  echo "No Python interpreter found in PATH."
  exit 1
fi

${PYTHON_BIN} scripts/prepare_pending_jobs.py --config "${CONFIG_PATH}" --mode "${MODE}"

if [ "${MODE}" = "full" ]; then
  MODE_DIR=full_sweep
else
  MODE_DIR=sanity_check
fi

PENDING_FILE="${MODE_DIR}/pending_jobs.txt"
if [ ! -f "${PENDING_FILE}" ]; then
  echo "Pending file not found: ${PENDING_FILE}"
  exit 1
fi

N=$(wc -l < "${PENDING_FILE}")
N=$(echo "${N}" | tr -d ' ')
if [ "${N}" -eq 0 ]; then
  echo "No pending jobs for mode=${MODE}."
  exit 0
fi

LAST=$((N - 1))
echo "Submitting ${N} jobs as array 0-${LAST} (mode=${MODE})"
sbatch --array=0-${LAST}%4 --export=ALL,MODE=${MODE},CONFIG_PATH=${CONFIG_PATH},PENDING_FILE=${PENDING_FILE} scripts/job.slurm
