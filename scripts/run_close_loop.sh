#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG=""
SCENARIO_FILE=""
SCENARIO_ROOT=""
RESULT_ROOT="${RESULT_ROOT:-$ROOT_DIR/result}"
RESUME_MARKER_FILE="${RESUME_MARKER_FILE:-scene_score.json}"
DRY_RUN="0"
LIST_CONFIGS="0"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_close_loop.sh --config <preset_or_json_path> [options]

Options:
  --config <value>         Close-loop model preset name or preset JSON path
  --scenario-file <path>   Run a single scenario file
  --scenario-root <path>   Scenario root for batch mode
  --result-root <path>     Result root, default: result
  --resume-marker <name>   Resume marker file, default: scene_score.json
  --python-bin <path>      Python executable, default: $PYTHON_BIN or python
  --dry-run                Validate config and pending scenario resolution only
  --list-model-configs     List available close-loop presets
  -h, --help               Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --scenario-file)
      SCENARIO_FILE="$2"
      shift 2
      ;;
    --scenario-root)
      SCENARIO_ROOT="$2"
      shift 2
      ;;
    --result-root)
      RESULT_ROOT="$2"
      shift 2
      ;;
    --resume-marker)
      RESUME_MARKER_FILE="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    --list-model-configs)
      LIST_CONFIGS="1"
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

CLI="${ROOT_DIR}/scripts/close_loop_eval_cli.py"
if [[ "$LIST_CONFIGS" == "1" ]]; then
  "$PYTHON_BIN" "$CLI" list-model-configs
  exit 0
fi

if [[ -z "$CONFIG" ]]; then
  echo "--config is required" >&2
  usage
  exit 2
fi

resolve_field() {
  local field="$1"
  "$PYTHON_BIN" "$CLI" resolve --config "$CONFIG" --field "$field"
}

bool_true() {
  local value="${1:-}"
  [[ "$value" == "1" || "$value" == "true" || "$value" == "True" ]]
}

CLOSE_LOOP_ROOT="${CLOSE_LOOP_ROOT:-$ROOT_DIR/Close_Loop_Evaluation}"
SCENARIO_ROOT="${SCENARIO_ROOT:-$CLOSE_LOOP_ROOT/Scenario_Onsite}"
CARLA_HOST="${CARLA_HOST:-127.0.0.1}"
DEFAULT_CARLA_PORT="2000"
DEFAULT_VLM_PORT="8000"
CARLA_PORT="${CARLA_PORT:-$(resolve_field runtime.carla_port)}"
CARLA_GRAPHICS_ADAPTER="${CARLA_GRAPHICS_ADAPTER:-0}"
SUMO_HOME_DEFAULT="/share/apps/sumo-1.20"
SUMO_HOME="${SUMO_HOME:-$SUMO_HOME_DEFAULT}"
START_CARLA="${START_CARLA:-1}"
START_VLM_SERVICE="${START_VLM_SERVICE:-1}"
DEBUG_VLM_IO="${DEBUG_VLM_IO:-1}"
DEBUG_VLM_IO_STDOUT="${DEBUG_VLM_IO_STDOUT:-0}"
VLM_ENABLE_WAYPOINT_INFERENCE="${VLM_ENABLE_WAYPOINT_INFERENCE:-0}"
VLM_ACTION_CONSIDERATIONS="${VLM_ACTION_CONSIDERATIONS:-}"
VLM_DESTINATION_DIRECTION_ONLY="${VLM_DESTINATION_DIRECTION_ONLY:-0}"
CARLA_LOG_FILE="${CARLA_LOG_FILE:-$ROOT_DIR/.close_loop_carla.log}"
CARLA_FILTER_NOISY_WARNINGS="${CARLA_FILTER_NOISY_WARNINGS:-1}"
CARLA_START_WAIT_SECONDS="${CARLA_START_WAIT_SECONDS:-$(resolve_field runtime.carla_start_wait_seconds)}"
VLM_START_WAIT_SECONDS="${VLM_START_WAIT_SECONDS:-$(resolve_field runtime.vlm_start_wait_seconds)}"
VLM_CUDA_VISIBLE_DEVICES="${VLM_CUDA_VISIBLE_DEVICES:-$(resolve_field runtime.vlm_visible_devices)}"
RUNTIME_VLM_GPU_ID="${RUNTIME_VLM_GPU_ID:-$(resolve_field runtime.runtime_gpu_id)}"
PIN_SINGLE_VISIBLE_DEVICE_FROM_CURRENT_CUDA_VISIBLE_DEVICES="${PIN_SINGLE_VISIBLE_DEVICE_FROM_CURRENT_CUDA_VISIBLE_DEVICES:-$(resolve_field runtime.pin_single_visible_device_from_current_cuda_visible_devices)}"
VLM_SERVICE_CONDA_ENV="${VLM_SERVICE_CONDA_ENV:-$(resolve_field runtime.service_conda_env)}"
VLM_SERVICE_LABEL="${VLM_SERVICE_LABEL:-$(resolve_field runtime.service_label)}"
CONFIG_NAME="$(resolve_field name)"
MODEL_DIR_NAME="${MODEL_DIR_NAME:-$(resolve_field resolved_model_dir_name)}"
VLM_PORT="${VLM_PORT:-$(resolve_field inference_config.PORT)}"
VLM_HOST="${VLM_HOST:-127.0.0.1}"
VLM_ENDPOINT="${VLM_ENDPOINT:-}"
VLM_USE_BEV="${VLM_USE_BEV:-$(resolve_field inference_config.INFERENCE_BASICS.USE_BEV)}"
CONDA_SH="${CONDA_SH:-/share/apps/miniconda3/etc/profile.d/conda.sh}"
CARLA_PYTHON_ENV="${CARLA_PYTHON_ENV:-carla_base}"
VLM_SERVICE_PYTHON_BIN="${VLM_SERVICE_PYTHON_BIN:-$PYTHON_BIN}"
ENABLE_MODULE_LOAD="${ENABLE_MODULE_LOAD:-1}"
EXTRA_LD_LIBRARY_PATH="${EXTRA_LD_LIBRARY_PATH:-}"
STOP_VLM_BETWEEN_SCENARIOS="${STOP_VLM_BETWEEN_SCENARIOS:-0}"

CARLA_PID=""
VLM_PID=""
OWN_CARLA=0
OWN_VLM=0
RUNTIME_CONFIG=""
VLM_LOG_FILE="${VLM_LOG_FILE:-$ROOT_DIR/.close_loop_${CONFIG_NAME}_vlm_service.log}"

if [[ -z "$CARLA_PORT" ]]; then
  CARLA_PORT="$DEFAULT_CARLA_PORT"
fi

if [[ -z "$VLM_PORT" ]]; then
  VLM_PORT="$DEFAULT_VLM_PORT"
fi

if [[ -z "$VLM_ENDPOINT" ]]; then
  VLM_ENDPOINT="http://${VLM_HOST}:${VLM_PORT}"
fi

if [[ ! -d "$CLOSE_LOOP_ROOT" ]]; then
  echo "Close-loop root not found: $CLOSE_LOOP_ROOT" >&2
  exit 1
fi

if [[ ! -d "$SCENARIO_ROOT" ]]; then
  echo "Scenario root not found: $SCENARIO_ROOT" >&2
  exit 1
fi

if [[ -n "$SCENARIO_FILE" && ! -f "$SCENARIO_FILE" ]]; then
  echo "Scenario file not found: $SCENARIO_FILE" >&2
  exit 1
fi

if [[ ! -d "$SUMO_HOME" ]]; then
  echo "SUMO_HOME not found: $SUMO_HOME" >&2
  exit 1
fi

CARLA_EGG="$(find "$CLOSE_LOOP_ROOT/Carla_Simulation/PythonAPI/carla/dist" -maxdepth 1 -type f -name 'carla-*.egg' | head -n 1)"
if [[ -z "$CARLA_EGG" ]]; then
  echo "CARLA Python egg not found under $CLOSE_LOOP_ROOT/Carla_Simulation/PythonAPI/carla/dist" >&2
  exit 1
fi

if [[ -f "$CONDA_SH" ]]; then
  # shellcheck source=/dev/null
  source "$CONDA_SH"
fi

if bool_true "$ENABLE_MODULE_LOAD" && command -v module >/dev/null 2>&1; then
  module load cuda/12.8 || true
  module load cmake || true
  module load ninja || true
  module load sumo || true
fi

cd "$CLOSE_LOOP_ROOT"

export SUMO_HOME
export PATH="$SUMO_HOME/bin:$PATH"
export PYTHONPATH="$SUMO_HOME/tools:$CARLA_EGG:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$SUMO_HOME/lib:${LD_LIBRARY_PATH:-}"
if [[ -n "$EXTRA_LD_LIBRARY_PATH" ]]; then
  export LD_LIBRARY_PATH="$EXTRA_LD_LIBRARY_PATH:$LD_LIBRARY_PATH"
fi

scenario_output_dir() {
  local scenario_file="$1"
  "$PYTHON_BIN" "$CLI" scenario-output-dir \
    --scenario-file "$scenario_file" \
    --scenario-root "$SCENARIO_ROOT" \
    --result-root "$RESULT_ROOT" \
    --model-dir-name "$MODEL_DIR_NAME"
}

tcp_port_open() {
  local host="$1"
  local port="$2"
  "$PYTHON_BIN" - "$host" "$port" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(1.0)
try:
    sock.connect((host, port))
except OSError:
    sys.exit(1)
finally:
    sock.close()
PY
}

check_vlm_service() {
  local host="$1"
  local port="$2"
  "$PYTHON_BIN" - "$host" "$port" <<'PY'
import json
import sys
import urllib.request

host = sys.argv[1]
port = int(sys.argv[2])
health_url = f"http://{host}:{port}/healthz"
openapi_url = f"http://{host}:{port}/openapi.json"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

try:
    with opener.open(health_url, timeout=5.0) as resp:
        data = json.load(resp)
    if data.get("status") == "ok":
        sys.exit(0)
except Exception:
    pass

try:
    with opener.open(openapi_url, timeout=5.0) as resp:
        data = json.load(resp)
except Exception:
    sys.exit(1)

if "/interact" in data.get("paths", {}):
    sys.exit(0)
sys.exit(1)
PY
}

wait_for_vlm_service() {
  local host="$1"
  local port="$2"
  local max_seconds="$3"
  local start_ts
  start_ts="$(date +%s)"
  while true; do
    if check_vlm_service "$host" "$port"; then
      return 0
    fi
    if (( "$(date +%s)" - start_ts >= max_seconds )); then
      return 1
    fi
    sleep 1
  done
}

check_carla_service() {
  local host="$1"
  local port="$2"
  conda run --no-capture-output -n "$CARLA_PYTHON_ENV" python - "$host" "$port" <<'PY'
import sys
import carla

host = sys.argv[1]
port = int(sys.argv[2])
try:
    client = carla.Client(host, port)
    client.set_timeout(5.0)
    world = client.get_world()
    world.get_map()
except Exception as exc:
    print(f"CARLA probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    sys.exit(1)
PY
}

wait_for_carla_service() {
  local pid="$1"
  local host="$2"
  local port="$3"
  local max_seconds="$4"
  local start_ts
  start_ts="$(date +%s)"
  while true; do
    if check_carla_service "$host" "$port" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      return 2
    fi
    if (( "$(date +%s)" - start_ts >= max_seconds )); then
      return 1
    fi
    sleep 1
  done
}

filter_carla_log_stream() {
  if [[ "$CARLA_FILTER_NOISY_WARNINGS" != "1" ]]; then
    cat
    return 0
  fi

  awk '
    /WARNING: Traffic sign [0-9]+ overlaps a driving lane\. Moving out of the road\.\.\./ { next }
    /WARNING: Failed to find suitable place for signal\./ { next }
    { print; fflush() }
  '
}

detect_vlm_no_inference_failure() {
  local scene_dir="$1"
  "$PYTHON_BIN" - "$scene_dir" <<'PY'
import json
import os
import sys

scene_dir = sys.argv[1]
log_path = os.path.join(scene_dir, "vlm_bridge.log")
if not os.path.isfile(log_path):
    raise SystemExit(1)

ctrl_files = sorted(name for name in os.listdir(scene_dir) if name.endswith("_ctrl.json"))
if not ctrl_files:
    raise SystemExit(1)

total_frames = 0
null_decision_frames = 0
valid_decision_frames = 0
for ctrl_name in ctrl_files:
    ctrl_path = os.path.join(scene_dir, ctrl_name)
    try:
        with open(ctrl_path, "r", encoding="utf-8") as handle:
            ctrl_data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        continue

    total_frames += 1
    steer_cmd = ctrl_data.get("steer_cmd")
    target_speed_mps = ctrl_data.get("target_speed_mps")
    if steer_cmd is None and target_speed_mps is None:
        null_decision_frames += 1
    if isinstance(steer_cmd, (int, float)) and isinstance(target_speed_mps, (int, float)):
        valid_decision_frames += 1

if total_frames == 0:
    raise SystemExit(1)

with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
    log_lines = handle.read().splitlines()

valid_action_lines = sum(1 for line in log_lines if " A=" in line and "A=Error:" not in line)
if null_decision_frames == total_frames and valid_decision_frames == 0 and valid_action_lines == 0:
    print(
        f"Detected VLM no-inference failure in {scene_dir}: "
        f"all {total_frames} control frames are null and the VLM log has no valid action output.",
        file=sys.stderr,
    )
    raise SystemExit(0)

raise SystemExit(1)
PY
}

kill_pid_tree() {
  local pid="$1"
  local child_pid
  while read -r child_pid; do
    [[ -n "$child_pid" ]] || continue
    kill_pid_tree "$child_pid"
  done < <(pgrep -P "$pid" 2>/dev/null || true)
  kill "$pid" >/dev/null 2>&1 || true
}

force_kill_pid_tree() {
  local pid="$1"
  local child_pid
  while read -r child_pid; do
    [[ -n "$child_pid" ]] || continue
    force_kill_pid_tree "$child_pid"
  done < <(pgrep -P "$pid" 2>/dev/null || true)
  kill -9 "$pid" >/dev/null 2>&1 || true
}

stop_process_tree() {
  local pid="$1"
  local label="$2"
  local waited=0

  [[ -n "$pid" ]] || return 0
  if ! kill -0 "$pid" >/dev/null 2>&1; then
    return 0
  fi

  echo "Stopping $label (pid=$pid)..."
  kill_pid_tree "$pid"
  while kill -0 "$pid" >/dev/null 2>&1 && (( waited < 5 )); do
    sleep 1
    waited=$((waited + 1))
  done
  if kill -0 "$pid" >/dev/null 2>&1; then
    force_kill_pid_tree "$pid"
  fi
}

stop_matching_processes() {
  local pattern="$1"
  local label="$2"
  local pid
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    stop_process_tree "$pid" "$label"
  done < <(pgrep -f "$pattern" 2>/dev/null || true)
}

CARLA_CMD_PATTERN="$CLOSE_LOOP_ROOT/Carla_Simulation/CarlaUE4.sh|$CLOSE_LOOP_ROOT/Carla_Simulation/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"
VLM_CMD_PATTERN="$CLOSE_LOOP_ROOT/VLM_Service/web_interact_app.py"

stop_owned_vlm() {
  if [[ "$OWN_VLM" != "1" ]]; then
    return 0
  fi
  stop_process_tree "$VLM_PID" "VLM service"
  stop_matching_processes "$VLM_CMD_PATTERN" "VLM service"
  if [[ -n "$RUNTIME_CONFIG" && -f "$RUNTIME_CONFIG" ]]; then
    rm -f "$RUNTIME_CONFIG"
  fi
  RUNTIME_CONFIG=""
  VLM_PID=""
  OWN_VLM=0
}

stop_owned_carla() {
  if [[ "$OWN_CARLA" != "1" ]]; then
    return 0
  fi
  stop_process_tree "$CARLA_PID" "CARLA server"
  stop_matching_processes "$CARLA_CMD_PATTERN" "CARLA server"
  CARLA_PID=""
  OWN_CARLA=0
}

start_carla_server() {
  : >"$CARLA_LOG_FILE"
  echo "Starting CARLA server for current scenario..."
  (
    cd "$CLOSE_LOOP_ROOT"
    ./Carla_Simulation/CarlaUE4.sh \
      -RenderOffScreen \
      -nosound \
      -carla-rpc-port="$CARLA_PORT" \
      -graphicsadapter="$CARLA_GRAPHICS_ADAPTER" \
      2>&1 | filter_carla_log_stream >"$CARLA_LOG_FILE"
  ) &
  CARLA_PID="$!"
  OWN_CARLA=1

  if wait_for_carla_service "$CARLA_PID" "$CARLA_HOST" "$CARLA_PORT" "$CARLA_START_WAIT_SECONDS"; then
    echo "CARLA ready in background, pid=$CARLA_PID"
  else
    wait_status="$?"
    if [[ "$wait_status" == "2" ]]; then
      echo "CARLA exited before becoming ready on ${CARLA_HOST}:${CARLA_PORT}" >&2
    else
      echo "CARLA failed to become ready on ${CARLA_HOST}:${CARLA_PORT}" >&2
    fi
    tail -n 40 "$CARLA_LOG_FILE" >&2 || true
    check_carla_service "$CARLA_HOST" "$CARLA_PORT" >&2 || true
    exit 1
  fi
}

compute_pinned_visible_device() {
  local runtime_gpu_id="$1"
  if [[ -z "$runtime_gpu_id" ]]; then
    echo "runtime_gpu_id must be set when pinning a single visible CUDA device." >&2
    exit 1
  fi
  "$PYTHON_BIN" - "$runtime_gpu_id" "${CUDA_VISIBLE_DEVICES:-}" <<'PY'
import sys

gpu_id = int(sys.argv[1])
visible_devices = sys.argv[2].strip()
if gpu_id < 0:
    raise SystemExit(f"runtime_gpu_id must be >= 0, got {gpu_id}")
if not visible_devices:
    print(gpu_id)
    raise SystemExit(0)

tokens = [token.strip() for token in visible_devices.split(",") if token.strip()]
if gpu_id >= len(tokens):
    raise SystemExit(
        f"runtime_gpu_id={gpu_id} is out of range for CUDA_VISIBLE_DEVICES={visible_devices}"
    )
print(tokens[gpu_id])
PY
}

start_vlm_service() {
  local launch_visible_devices=""
  local runtime_gpu_id_for_config="$RUNTIME_VLM_GPU_ID"
  local run_cmd=()

  if [[ "$START_VLM_SERVICE" != "1" ]]; then
    echo "Skipping VLM service start because START_VLM_SERVICE=$START_VLM_SERVICE"
    return 0
  fi

  if check_vlm_service "$VLM_HOST" "$VLM_PORT"; then
    echo "Reusing existing VLM service on $VLM_ENDPOINT"
    return 0
  fi

  if tcp_port_open "$VLM_HOST" "$VLM_PORT"; then
    echo "Port $VLM_PORT is already in use, but it is not a healthy VLM service." >&2
    exit 1
  fi

  if bool_true "$PIN_SINGLE_VISIBLE_DEVICE_FROM_CURRENT_CUDA_VISIBLE_DEVICES"; then
    launch_visible_devices="$(compute_pinned_visible_device "$RUNTIME_VLM_GPU_ID")"
    runtime_gpu_id_for_config="0"
  elif [[ -n "$VLM_CUDA_VISIBLE_DEVICES" ]]; then
    launch_visible_devices="$VLM_CUDA_VISIBLE_DEVICES"
  fi

  if [[ -n "$RUNTIME_CONFIG" && -f "$RUNTIME_CONFIG" ]]; then
    rm -f "$RUNTIME_CONFIG"
  fi
  RUNTIME_CONFIG="$(mktemp "$ROOT_DIR/.close_loop_${CONFIG_NAME}.XXXXXX.json")"
  run_cmd=(
    "$PYTHON_BIN" "$CLI" write-runtime-config
    --config "$CONFIG"
    --output "$RUNTIME_CONFIG"
  )
  if [[ -n "$runtime_gpu_id_for_config" ]]; then
    run_cmd+=(--runtime-gpu-id "$runtime_gpu_id_for_config")
  fi
  "${run_cmd[@]}" >/dev/null

  echo "Starting $VLM_SERVICE_LABEL ..."
  (
    cd "$CLOSE_LOOP_ROOT"
    export PYTHONUNBUFFERED=1
    if [[ -n "$launch_visible_devices" ]]; then
      export CUDA_VISIBLE_DEVICES="$launch_visible_devices"
    fi
    if [[ -n "$VLM_SERVICE_CONDA_ENV" ]]; then
      exec conda run --no-capture-output -n "$VLM_SERVICE_CONDA_ENV" python \
        "$CLOSE_LOOP_ROOT/VLM_Service/web_interact_app.py" \
        --config "$RUNTIME_CONFIG"
    fi
    exec "$VLM_SERVICE_PYTHON_BIN" \
      "$CLOSE_LOOP_ROOT/VLM_Service/web_interact_app.py" \
      --config "$RUNTIME_CONFIG"
  ) >"$VLM_LOG_FILE" 2>&1 &
  VLM_PID="$!"
  OWN_VLM=1

  if ! wait_for_vlm_service "$VLM_HOST" "$VLM_PORT" "$VLM_START_WAIT_SECONDS"; then
    echo "VLM service failed to become healthy on $VLM_ENDPOINT" >&2
    tail -n 40 "$VLM_LOG_FILE" >&2 || true
    exit 1
  fi

  echo "VLM service started in background, pid=$VLM_PID"
}

cleanup() {
  set +e
  stop_owned_vlm
  stop_owned_carla
  if [[ -n "$RUNTIME_CONFIG" && -f "$RUNTIME_CONFIG" ]]; then
    rm -f "$RUNTIME_CONFIG"
  fi
}

trap cleanup EXIT INT TERM

SCENARIO_FILES=()
if [[ -n "$SCENARIO_FILE" ]]; then
  SCENARIO_FILES=("$SCENARIO_FILE")
else
  mapfile -t SCENARIO_FILES < <(find "$SCENARIO_ROOT" -type f -name 'entity_sumo.json' | sort)
fi

if [[ "${#SCENARIO_FILES[@]}" -eq 0 ]]; then
  echo "No scenario files found under $SCENARIO_ROOT" >&2
  exit 1
fi

DISCOVERED_SCENARIO_COUNT="${#SCENARIO_FILES[@]}"
SKIPPED_SCENARIO_COUNT=0
PENDING_SCENARIO_FILES=()
for scenario_file in "${SCENARIO_FILES[@]}"; do
  output_dir="$(scenario_output_dir "$scenario_file")"
  marker_file="$output_dir/$RESUME_MARKER_FILE"
  if [[ -f "$marker_file" ]]; then
    echo "Skipping completed scenario: $scenario_file"
    echo "  Found resume marker: $marker_file"
    SKIPPED_SCENARIO_COUNT=$((SKIPPED_SCENARIO_COUNT + 1))
    continue
  fi
  PENDING_SCENARIO_FILES+=("$scenario_file")
done
SCENARIO_FILES=("${PENDING_SCENARIO_FILES[@]}")

echo "Close-loop root: $CLOSE_LOOP_ROOT"
echo "Config:          $CONFIG"
echo "Model dir:       $MODEL_DIR_NAME"
echo "Result root:     $RESULT_ROOT"
echo "Resume marker:   $RESUME_MARKER_FILE"
echo "Scenario root:   $SCENARIO_ROOT"
if [[ -n "$SCENARIO_FILE" ]]; then
  echo "Scenario mode:   single"
  echo "Scenario file:   $SCENARIO_FILE"
else
  echo "Scenario mode:   batch"
  echo "Scenario count:  $DISCOVERED_SCENARIO_COUNT"
  echo "Pending count:   ${#SCENARIO_FILES[@]}"
  echo "Skipped count:   $SKIPPED_SCENARIO_COUNT"
fi
echo "SUMO_HOME:       $SUMO_HOME"
echo "VLM endpoint:    $VLM_ENDPOINT"
echo "VLM service env: $VLM_SERVICE_CONDA_ENV"
echo "VLM visible:     ${VLM_CUDA_VISIBLE_DEVICES:-<inherit>}"
echo "Runtime GPU ID:  $RUNTIME_VLM_GPU_ID"
echo "CARLA host:port: ${CARLA_HOST}:${CARLA_PORT}"
echo "CARLA log:       $CARLA_LOG_FILE"
echo "VLM log:         $VLM_LOG_FILE"
echo "Dry run:         $DRY_RUN"

if [[ "${#SCENARIO_FILES[@]}" -eq 0 ]]; then
  echo "All scenarios already completed. Nothing to run."
  exit 0
fi

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

nvidia-smi -L || true

if [[ "$START_CARLA" != "1" ]]; then
  echo "Per-scenario clean world requires START_CARLA=1" >&2
  exit 1
fi

run_single_scenario() {
  local scenario_file="$1"
  local output_dir="$2"

  echo "Starting Tongji VLM control..."
  echo "Scenario file: $scenario_file"
  echo "Output dir:    $output_dir"

  CONTROL_ARGS=()
  if [[ "$DEBUG_VLM_IO" == "1" ]]; then
    CONTROL_ARGS+=(--debug-vlm-io)
  fi
  if [[ "$DEBUG_VLM_IO_STDOUT" == "1" ]]; then
    CONTROL_ARGS+=(--debug-vlm-io-stdout)
  fi
  if bool_true "$VLM_USE_BEV"; then
    CONTROL_ARGS+=(--use-bev)
  fi
  if [[ "$VLM_ENABLE_WAYPOINT_INFERENCE" == "1" ]]; then
    CONTROL_ARGS+=(--enable-waypoint-inference)
  fi
  if [[ -n "$VLM_ACTION_CONSIDERATIONS" ]]; then
    CONTROL_ARGS+=(--action-considerations "$VLM_ACTION_CONSIDERATIONS")
  fi
  if [[ "$VLM_DESTINATION_DIRECTION_ONLY" == "1" ]]; then
    CONTROL_ARGS+=(--destination-direction-only)
  fi

  conda run -n "$CARLA_PYTHON_ENV" \
    env SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
    python \
    "$CLOSE_LOOP_ROOT/Carla_Simulation/Co-Simulation/Sumo/vlm_control_sumo_tongji_entity.py" \
    "$scenario_file" \
    --output-dir "$output_dir" \
    --host "$CARLA_HOST" \
    --port "$CARLA_PORT" \
    --vlm-endpoint "$VLM_ENDPOINT" \
    --camera-view chase \
    --follow-spectator \
    --spectator-view bev \
    --spectator-height 60 \
    --show-ego-destination \
    --show-ego-via-points \
    --scripted-vehicle-end-mode stop \
    --end-on-hero-collision \
    --end-on-hero-offroad \
    --hero-offroad-grace-time 1.5 \
    --end-on-ego-destination \
    --scenario-duration-scale 2 \
    "${CONTROL_ARGS[@]}"
}

scenario_index=0
for scenario_file in "${SCENARIO_FILES[@]}"; do
  scenario_index=$((scenario_index + 1))
  output_dir="$(scenario_output_dir "$scenario_file")"
  mkdir -p "$output_dir"

  echo
  echo "[$scenario_index/${#SCENARIO_FILES[@]}] Running scenario"
  start_carla_server
  start_vlm_service
  run_single_scenario "$scenario_file" "$output_dir"
  if detect_vlm_no_inference_failure "$output_dir"; then
    echo "Aborting batch because the scenario produced results without valid VLM inference." >&2
    echo "Scenario file: $scenario_file" >&2
    echo "Output dir:    $output_dir" >&2
    exit 1
  fi
  stop_owned_carla
  if [[ "$STOP_VLM_BETWEEN_SCENARIOS" == "1" ]]; then
    stop_owned_vlm
  fi
done
