#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULT_DIR=""
SKIP_L1_4="0"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/score_open_loop.sh --result-dir <path> [options]

Options:
  --result-dir <path>   Existing model result directory to score
  --python-bin <path>   Python executable, default: $PYTHON_BIN or python
  --skip-l1-4           Skip R1_4 Lingo-Judge scoring
  -h, --help            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --result-dir)
      RESULT_DIR="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --skip-l1-4)
      SKIP_L1_4="1"
      shift
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
  "$PYTHON_BIN" "${ROOT_DIR}/scripts/score_vlm_total_entry.py"
  --result-dir "$RESULT_DIR"
)
if [[ "$SKIP_L1_4" == "1" ]]; then
  SCORE_CMD+=(--skip-l1-4)
fi

echo "[score] ${SCORE_CMD[*]}"
"${SCORE_CMD[@]}"
