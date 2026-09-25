#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULT_DIR=""
OUTPUT_CSV=""

usage() {
  cat <<'EOF'
Usage:
  bash scripts/score_close_loop.sh --result-dir <path> [options]

Options:
  --result-dir <path>   Existing close-loop result root
  --output-csv <path>   Output CSV path, default: <result-dir>/result_summary.csv
  --python-bin <path>   Python executable, default: $PYTHON_BIN or python
  -h, --help            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --result-dir)
      RESULT_DIR="$2"
      shift 2
      ;;
    --output-csv)
      OUTPUT_CSV="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$RESULT_DIR" ]]; then
  echo "--result-dir is required" >&2
  usage
  exit 2
fi

if [[ ! -d "$RESULT_DIR" ]]; then
  echo "--result-dir does not exist: $RESULT_DIR" >&2
  exit 2
fi

SCORE_CMD=(
  "$PYTHON_BIN" "${ROOT_DIR}/scripts/score_close_loop.py"
  --result-dir "$RESULT_DIR"
)
if [[ -n "$OUTPUT_CSV" ]]; then
  SCORE_CMD+=(--output-csv "$OUTPUT_CSV")
fi

echo "[score] ${SCORE_CMD[*]}"
"${SCORE_CMD[@]}"

