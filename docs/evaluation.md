# Evaluation Guide

[← Back to DriveHierarchy](../README.md)

Complete setup, inference, scoring, and configuration reference. Run all shell commands from the **repository root**, including commands on this page.

[Data](#benchmark-data) · [External assets](#external-assets) · [Environment](#environment-preparation) · [Open-loop evaluation](#open-loop-evaluation) · [Closed-loop evaluation](#closed-loop-evaluation) · [Configuration](#configuration)

## Repository Layout

```text
DriveHierarchy/
├── assest/                        # Paper PDF and teaser figure
├── docs/evaluation.md             # Detailed evaluation guide
├── Open_Loop_Evaluation/          # Open-loop benchmark JSONL files
├── Close_Loop_Evaluation/
│   ├── Carla_Simulation/          # CARLA + SUMO simulation code; `CarlaUE4/` and `HDMaps/` are distributed separately
│   ├── Scenario_Onsite/           # Closed-loop scenario JSON files
│   ├── VLM_Service/               # Standalone VLM inference service used in closed-loop mode
│   └── requirements.txt           # Closed-loop Python dependencies
├── scripts/
│   ├── run_open_loop.sh
│   ├── run_close_loop.sh
│   ├── score_open_loop.sh
│   ├── score_close_loop.sh
│   ├── score_vlm_total_entry.py
│   ├── close_loop_eval_cli.py
│   └── configs/                   # Model presets, dataset catalogs, inference configs
├── result/                        # Default output directory
└── requirements.txt
```

## Benchmark Data

### Open-loop tasks

The active evaluation catalog is [scripts/configs/datasets/open_loop_all.json](../scripts/configs/datasets/open_loop_all.json). It reproduces the 14-task paper protocol:

- `R1_1_A_Existence`
- `R1_1_B_Counting`
- `R1_1_C_State_Attribute`
- `R1_2_A_Nearest_Object`
- `R1_2_B_Certain_Object`
- `R1_2_C_Distance_Bucket`
- `R1_3_Location_Questions`
- `R1_4_Situation_Description`
- `R2_1_Multi_view_Memory`
- `R2_2_A_Temporal_Counting`
- `R2_2_B_Temporal_Status_Recognition`
- `R2_3_Spatial_Relations`
- `R3_1_Outcome_Prediction`
- `R3_2_Sequential_Planning`

Each task is stored under [Open_Loop_Evaluation](../Open_Loop_Evaluation) as one `.jsonl` file. The original release documentation describes a separate Hugging Face export, `open_loop_records.jsonl`, with 15,000 records and a `task_name` field. For paper reproduction, use the checked-in 14-task catalog above (14,000 records).

### Closed-loop scenarios

The default scenario root is [Close_Loop_Evaluation/Scenario_Onsite](../Close_Loop_Evaluation/Scenario_Onsite). It contains 100 curated simulator scenarios. The current layout is:

- scenario category directory, for example `07_sudden_brake`
- scenario instance directory, for example `SC20260323153401DBRL`
- JSON assets required for execution, including `entity.json` and `entity_sumo.json`

## External Assets

To keep the public Git repository lightweight, the following heavyweight directories are not tracked in Git:

- `Close_Loop_Evaluation/Carla_Simulation/CarlaUE4`
- `Close_Loop_Evaluation/Carla_Simulation/HDMaps`

These assets are required for closed-loop evaluation and must be restored locally at the exact same paths. You can download them from [here](https://huggingface.co/datasets/anonymous-2FD5/DriveHierarchy).

```text
DriveHierarchy/
└── Close_Loop_Evaluation/
    └── Carla_Simulation/
        ├── CarlaUE4/
        └── HDMaps/
```

After extraction, the two restored directories should be:

- `Close_Loop_Evaluation/Carla_Simulation/CarlaUE4`
- `Close_Loop_Evaluation/Carla_Simulation/HDMaps`

The open-loop benchmark and the closed-loop orchestration code remain fully available in this repository. Only the heavyweight runtime assets are excluded from version control.

## Environment Preparation

### General

- Python 3.10+ is recommended.
- Run all commands from the repository root.
- A practical setup is to use separate Conda environments for open-loop and closed-loop evaluation.

### Open-loop environment

Before inference, obtain the source imagery required by the task JSONL files under the applicable dataset terms. The checked-in records include image paths such as `/data/sets/nuscenes/...`; mount the data at those paths or update the image references to your local files. Depending on the task, references appear in `image_path`, `image_paths`, or nested image metadata. A config dry-run does not check that every image is accessible.

Open-loop inference supports two backends:

- `vllm`
- `transformers`

Model presets are stored in [scripts/configs/models](../scripts/configs/models). A recommended environment setup is:

```bash
conda create -n drivehierarchy_openloop python=3.10 -y
conda activate drivehierarchy_openloop
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:

- `vllm` is only needed when you use the `vllm` backend.
- `peft` is only needed for adapter / LoRA-style Transformers presets.
- If a specific model family requires extra vendor packages, install them in the same Conda environment.

### Closed-loop environment

Closed-loop evaluation additionally requires:

- CARLA runtime assets under [Close_Loop_Evaluation/Carla_Simulation](../Close_Loop_Evaluation/Carla_Simulation), including separately distributed `CarlaUE4/` and `HDMaps/`
- SUMO with a valid `SUMO_HOME`
- Python dependencies from [Close_Loop_Evaluation/requirements.txt](../Close_Loop_Evaluation/requirements.txt)
- A working environment for [Close_Loop_Evaluation/VLM_Service](../Close_Loop_Evaluation/VLM_Service)

Typical closed-loop setup:

```bash
conda create -n drivehierarchy_closeloop python=3.10 -y
conda activate drivehierarchy_closeloop
pip install --upgrade pip
pip install -r Close_Loop_Evaluation/requirements.txt
export SUMO_HOME=/path/to/sumo
```

The open-source closed-loop presets use neutral runtime defaults:

- `CARLA_PORT=2000`
- `VLM_PORT=8000`
- no fixed `runtime_gpu_id`
- no fixed `service_conda_env`

If needed, override these at runtime with environment variables such as `CARLA_PORT`, `VLM_PORT`, `RUNTIME_VLM_GPU_ID`, `VLM_SERVICE_CONDA_ENV`, and `VLM_SERVICE_PYTHON_BIN`.

## Open-Loop Evaluation

### 1. List available model presets

For `vllm`:

```bash
python scripts/run_open_loop_vllm.py --list-model-configs
```

For `transformers`:

```bash
python scripts/run_open_loop_transformers.py --list-model-configs
```

### 2. Run inference

Run all open-loop tasks with a `vllm` preset:

```bash
bash scripts/run_open_loop.sh \
  --backend vllm \
  --config qwen3_vl_8b_instruct_vllm
```

Run with a `transformers` preset:

```bash
bash scripts/run_open_loop.sh \
  --backend transformers \
  --config qwen25_vl_drivelm_adapter_transformers
```

Run selected tasks only. `--datasets` accepts `all`, dataset names, file names, or comma-separated combinations:

```bash
bash scripts/run_open_loop.sh \
  --backend vllm \
  --config qwen3_vl_8b_instruct_vllm \
  --datasets R1_1_A_Existence,R2_3_Spatial_Relations,R3_2_Sequential_Planning
```

Validate config and dataset resolution without loading a model:

```bash
bash scripts/run_open_loop.sh \
  --backend vllm \
  --config qwen3_vl_8b_instruct_vllm \
  --dry-run
```

### 3. Score results

Run inference and scoring in one command:

```bash
bash scripts/run_open_loop.sh \
  --backend vllm \
  --config qwen3_vl_8b_instruct_vllm \
  --score
```

Or score an existing result directory:

```bash
bash scripts/score_open_loop.sh \
  --result-dir result/open_loop/qwen3_vl_8b_instruct_vllm
```

If you want to skip `R1_4` Lingo-Judge scoring:

```bash
bash scripts/score_open_loop.sh \
  --result-dir result/open_loop/qwen3_vl_8b_instruct_vllm \
  --skip-l1-4
```

### 4. Open-loop outputs

By default, outputs are written under `result/open_loop`.

For one model, the main artifacts are:

- prediction files such as `R1_1_A_Existence_pred.jsonl`
- evaluation summary at `<result_dir>/evaluation/final_score.json`

## Closed-Loop Evaluation

The standardized closed-loop runner is based on the `resume_stop_on_vlm_failure` workflow and uses:

- [Close_Loop_Evaluation/Carla_Simulation](../Close_Loop_Evaluation/Carla_Simulation)
- [Close_Loop_Evaluation/VLM_Service](../Close_Loop_Evaluation/VLM_Service)
- [Close_Loop_Evaluation/Scenario_Onsite](../Close_Loop_Evaluation/Scenario_Onsite)

### 1. List available closed-loop presets

```bash
python scripts/close_loop_eval_cli.py list-model-configs
```

### 2. Run locally

Dry-run first to validate config resolution and pending scenarios:

```bash
SUMO_HOME=/path/to/sumo \
bash scripts/run_close_loop.sh \
  --config gemma_3_12b_it \
  --dry-run
```

Run the full closed-loop benchmark:

```bash
SUMO_HOME=/path/to/sumo \
bash scripts/run_close_loop.sh \
  --config gemma_3_12b_it
```

Run only one scenario file:

```bash
SUMO_HOME=/path/to/sumo \
bash scripts/run_close_loop.sh \
  --config gemma_3_12b_it \
  --scenario-file Close_Loop_Evaluation/Scenario_Onsite/07_sudden_brake/SC20260323153401DBRL/entity.json
```

Override the output root:

```bash
SUMO_HOME=/path/to/sumo \
bash scripts/run_close_loop.sh \
  --config gemma_3_12b_it \
  --result-root result_close_loop
```

### 3. Score closed-loop results

Summarize all discovered `scene_score.json` files into a single CSV:

```bash
bash scripts/score_close_loop.sh --result-dir result
```

Or specify the output CSV path explicitly:

```bash
bash scripts/score_close_loop.sh \
  --result-dir result \
  --output-csv result/result_summary.csv
```

### 4. Closed-loop outputs

For each executed scenario, the pipeline writes a `scene_score.json` file under the model-specific result tree. The final summary script writes:

- `<result_dir>/result_summary.csv`

The CSV contains averaged closed-loop metrics per scene and per model:

- `avg_score_route`
- `avg_score_safety`
- `avg_score_efficiency`
- `avg_score_penalty`
- `avg_score_composed`

## Configuration

### Open-loop

- Model presets: [scripts/configs/models](../scripts/configs/models)
- Dataset catalogs: [scripts/configs/datasets](../scripts/configs/datasets)

Each open-loop model preset uses the same schema, including:

- `name`
- `backend`
- `family`
- `model`
- `generation`
- `runtime`
- `dataset_config`
- `datasets`

### Closed-loop

- Model catalog: [scripts/configs/close_loop_models/catalog.json](../scripts/configs/close_loop_models/catalog.json)
- Runtime inference JSON files: [scripts/configs/close_loop_inference](../scripts/configs/close_loop_inference)

Each closed-loop preset defines:

- a preset name
- the inference JSON file to materialize at runtime
- runtime service settings
- CARLA port
- GPU visibility

If you change dataset file names or scenario roots, update the corresponding JSON files under [scripts/configs](../scripts/configs).
