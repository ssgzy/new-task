#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="Lion"
MODEL_LABEL=""
DOWNLOAD_MODEL=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_assigned_model_v1.sh --label "Orca-2-7B"
  bash scripts/run_assigned_model_v1.sh --label "Orca-2-7B" --download-model

Options:
  --label           Exact model label from 模型注册表
  --env-name        Conda environment name, default Lion
  --download-model  Download the assigned model before running
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --label)
      MODEL_LABEL="${2:-}"
      shift 2
      ;;
    --env-name)
      ENV_NAME="${2:-}"
      shift 2
      ;;
    --download-model)
      DOWNLOAD_MODEL=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${MODEL_LABEL}" ]]; then
  echo "Missing required --label" >&2
  usage >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

if [[ "${DOWNLOAD_MODEL}" -eq 1 ]]; then
  conda run -n "${ENV_NAME}" python scripts/ensure_candidate_models.py --labels "${MODEL_LABEL}" --download-missing --provisional
fi

conda run -n "${ENV_NAME}" python scripts/ensure_candidate_models.py --labels "${MODEL_LABEL}" --provisional
conda run -n "${ENV_NAME}" python scripts/run_length_calibration_v1.py --labels "${MODEL_LABEL}" --resume --provisional
conda run -n "${ENV_NAME}" python scripts/run_qualification_v1.py --labels "${MODEL_LABEL}" --resume --provisional
