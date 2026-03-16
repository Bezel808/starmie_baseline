#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  ./run_starmie_0316.sh

Environment variables (all optional):
  PYTHON_BIN             Python interpreter used to run runner and subcommands.
  DATASET_ROOT           Root path containing *_1218 datasets.
  OUTPUT_ROOT            Output root for logs/summary.
  GPU                    CUDA_VISIBLE_DEVICES value (default: 1).
  DATASETS               Space-separated dataset list.
                         Default: "wikidbs_1218 santos_benchmark_1218 magellan_1218"
  N_EPOCHS               Pretrain epochs (default: 3)
  BATCH_SIZE             Batch size (default: 32)
  PRETRAIN_SIZE          Pretrain size cap (default: 10000)
  LR                     Learning rate (default: 5e-5)
  MAX_LEN                Max token length (default: 128)
  AUGMENT_OP             Augmentation op (default: drop_col)
  SAMPLE_METH            Sample method (default: tfidf_entity)
  TABLE_ORDER            Table order (default: column)
  FP16                   1 to enable fp16 (default), 0 to disable

Examples:
  DATASET_ROOT=/home/mengshi/table_quality/datasets_joint_discovery_integration ./run_starmie_0316.sh
  GPU=0 BATCH_SIZE=16 DATASETS="santos_benchmark_1218" ./run_starmie_0316.sh
EOF
  exit 0
fi

TS="$(date +%Y%m%d_%H%M%S)"
PYTHON_BIN="${PYTHON_BIN:-${STARMIE_PYTHON:-python}}"
DATASET_ROOT="${DATASET_ROOT:-${STARMIE_DATASET_ROOT:-}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/runs/starmie_0316_shell_${TS}}"
GPU="${GPU:-1}"
DATASETS="${DATASETS:-wikidbs_1218 santos_benchmark_1218 magellan_1218}"
N_EPOCHS="${N_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PRETRAIN_SIZE="${PRETRAIN_SIZE:-10000}"
LR="${LR:-5e-5}"
MAX_LEN="${MAX_LEN:-128}"
AUGMENT_OP="${AUGMENT_OP:-drop_col}"
SAMPLE_METH="${SAMPLE_METH:-tfidf_entity}"
TABLE_ORDER="${TABLE_ORDER:-column}"
FP16="${FP16:-1}"

if [[ -z "${DATASET_ROOT}" ]]; then
  echo "ERROR: DATASET_ROOT is required (or set STARMIE_DATASET_ROOT)." >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: PYTHON_BIN not found: ${PYTHON_BIN}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
LOG_FILE="${OUTPUT_ROOT}/run_starmie_0316_${TS}.log"
TIME_FILE="${OUTPUT_ROOT}/time_${TS}.txt"

read -r -a DATASET_ARR <<< "${DATASETS}"

CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/run_starmie_0316.py"
  --datasets "${DATASET_ARR[@]}"
  --dataset-root "${DATASET_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --python-bin "${PYTHON_BIN}"
  --gpu "${GPU}"
  --n-epochs "${N_EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --pretrain-size "${PRETRAIN_SIZE}"
  --lr "${LR}"
  --max-len "${MAX_LEN}"
  --augment-op "${AUGMENT_OP}"
  --sample-meth "${SAMPLE_METH}"
  --table-order "${TABLE_ORDER}"
)

if [[ "${FP16}" == "1" ]]; then
  CMD+=(--fp16)
else
  CMD+=(--no-fp16)
fi

echo "========================================="
echo " Start Running: $(date)"
echo " Repo Root: ${REPO_ROOT}"
echo " Python Bin: ${PYTHON_BIN}"
echo " Dataset Root: ${DATASET_ROOT}"
echo " Output Root: ${OUTPUT_ROOT}"
echo " GPU: ${GPU}"
echo " Datasets: ${DATASETS}"
echo " Epochs: ${N_EPOCHS} | Batch: ${BATCH_SIZE} | Size: ${PRETRAIN_SIZE}"
echo " LR: ${LR} | Max Len: ${MAX_LEN}"
echo " Augment: ${AUGMENT_OP} | Sample: ${SAMPLE_METH} | Order: ${TABLE_ORDER} | FP16: ${FP16}"
echo " Log: ${LOG_FILE}"
echo "========================================="

CUDA_VISIBLE_DEVICES="${GPU}" /usr/bin/time -v -o "${TIME_FILE}" "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"

echo "========================================="
echo " Finished: $(date)"
echo " Summary: ${OUTPUT_ROOT}/summary.json"
echo " Time Stats: ${TIME_FILE}"
echo "========================================="
