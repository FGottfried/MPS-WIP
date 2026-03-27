#!/bin/bash
set -euo pipefail

MODE=${1:-full}
CONFIG_PATH=${2:-mps_embedding_config.json}
PYTHON_MODULE=${3:-python/3.11.8}
RUN_SETUP_ENV=${RUN_SETUP_ENV:-1}

if [[ "${MODE}" != "full" && "${MODE}" != "sanity" ]]; then
  echo "Usage: $0 [full|sanity] [config_path] [python_module]"
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

DEPENDENCY_ARGS=()
if [ "${RUN_SETUP_ENV}" = "1" ]; then
  echo "Submitting setup_env.slurm first (RUN_SETUP_ENV=1)"
  SETUP_JOB_ID=$(sbatch --parsable --export=ALL,PYTHON_MODULE=${PYTHON_MODULE},VENV_DIR=.venv scripts/setup_env.slurm)
  echo "setup_env job id: ${SETUP_JOB_ID}"
  DEPENDENCY_ARGS+=(--dependency=afterok:${SETUP_JOB_ID})
fi

sbatch "${DEPENDENCY_ARGS[@]}" \
  --array=0-${LAST}%4 \
  --export=ALL,MODE=${MODE},CONFIG_PATH=${CONFIG_PATH},PENDING_FILE=${PENDING_FILE},PYTHON_MODULE=${PYTHON_MODULE},VENV_DIR=.venv \
  scripts/job.slurm
