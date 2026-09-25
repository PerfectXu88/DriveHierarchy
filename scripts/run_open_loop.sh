#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="vllm"
CONFIG=""
DATASETS="all"
DATASET_CONFIG=""
OUTPUT_DIR="result/open_loop"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_SCORE="0"
SKIP_L1_4="0"
DRY_RUN="0"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_open_loop.sh --backend <vllm|transformers> --config <preset_or_json_path> [options]

Options:
  --backend <name>         Inference backend: vllm or transformers
  --config <value>         Model config preset name or JSON path
  --datasets <value>       Dataset selection, default: all
  --dataset-config <value> Dataset catalog preset name or JSON path
  --output-dir <path>      Output root, default: result/open_loop
  --python-bin <path>      Python executable, default: $PYTHON_BIN or python
  --score                  Run final scoring after inference
  --skip-l1-4              Skip L1_4 Lingo-Judge scoring
  --dry-run                Validate config and dataset resolution only
  -h, --help               Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      BACKEND="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --datasets)
      DATASETS="$2"
      shift 2
      ;;
    --dataset-config)
      DATASET_CONFIG="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --score)
      RUN_SCORE="1"
      shift
      ;;
    --skip-l1-4)
      SKIP_L1_4="1"
      shift
      ;;
    --dry-run)
      DRY_RUN="1"
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

if [[ -z "$CONFIG" ]]; then
  echo "--config is required" >&2
  usage
  exit 2
fi

if [[ "$BACKEND" != "vllm" && "$BACKEND" != "transformers" ]]; then
  echo "--backend must be one of: vllm, transformers" >&2
  exit 2
fi

RUNNER="${ROOT_DIR}/scripts/run_open_loop_${BACKEND}.py"
RUN_CMD=(
  "$PYTHON_BIN" "$RUNNER"
  --config "$CONFIG"
  --datasets "$DATASETS"
  --output-dir "$OUTPUT_DIR"
)

if [[ -n "$DATASET_CONFIG" ]]; then
  RUN_CMD+=(--dataset-config "$DATASET_CONFIG")
fi
if [[ "$DRY_RUN" == "1" ]]; then
  RUN_CMD+=(--dry-run)
fi

echo "[run] ${RUN_CMD[*]}"
"${RUN_CMD[@]}"

if [[ "$RUN_SCORE" != "1" || "$DRY_RUN" == "1" ]]; then
  exit 0
fi

RESULT_DIR="$("$PYTHON_BIN" -c 'import json, sys; from pathlib import Path
config_arg = sys.argv[1]
output_dir = sys.argv[2]
config_path = Path(config_arg)
if config_path.exists():
    path = config_path
else:
    path = Path("scripts/configs/models") / f"{config_arg}.json"
with path.open(encoding="utf-8") as f:
    cfg = json.load(f)
print(str(Path(output_dir) / cfg["name"]))' "$CONFIG" "$OUTPUT_DIR")"

SCORE_CMD=(
  "$PYTHON_BIN" "${ROOT_DIR}/scripts/score_vlm_total_entry.py"
  --result-dir "$RESULT_DIR"
)
if [[ "$SKIP_L1_4" == "1" ]]; then
  SCORE_CMD+=(--skip-l1-4)
fi

echo "[score] ${SCORE_CMD[*]}"
"${SCORE_CMD[@]}"
