from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
CATALOG_PATH = SCRIPTS_DIR / "configs" / "close_loop_models" / "catalog.json"
INFERENCE_CONFIG_DIR = SCRIPTS_DIR / "configs" / "close_loop_inference"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def load_catalog() -> dict[str, Any]:
    return load_json(CATALOG_PATH)


def list_model_configs() -> list[str]:
    catalog = load_catalog()
    models = catalog.get("models", {})
    if not isinstance(models, dict):
        raise ValueError(f"Invalid catalog schema in {CATALOG_PATH}")
    return sorted(models)


def _load_catalog_preset(name: str) -> dict[str, Any]:
    catalog = load_catalog()
    models = catalog.get("models", {})
    if name not in models:
        available = ", ".join(sorted(models))
        raise KeyError(f"Unknown close-loop config '{name}'. Available: {available}")
    preset = copy.deepcopy(models[name])
    if not isinstance(preset, dict):
        raise ValueError(f"Invalid preset entry for {name}")
    preset.setdefault("name", name)
    return preset


def _load_path_preset(path: Path) -> dict[str, Any]:
    preset = load_json(path)
    preset.setdefault("name", path.stem)
    return preset


def resolve_preset(config_arg: str) -> dict[str, Any]:
    path = Path(config_arg)
    preset = _load_path_preset(path.resolve()) if path.exists() else _load_catalog_preset(config_arg)

    inference_name = preset.get("inference_config_file")
    if not inference_name or not isinstance(inference_name, str):
        raise ValueError(f"Preset {preset.get('name', config_arg)!r} is missing inference_config_file")

    inference_path = Path(inference_name)
    if not inference_path.is_absolute():
        inference_path = (INFERENCE_CONFIG_DIR / inference_name).resolve()
    if not inference_path.is_file():
        raise FileNotFoundError(f"Inference config not found: {inference_path}")

    inference_config = load_json(inference_path)
    preset["inference_config_path"] = str(inference_path)
    preset["inference_config"] = inference_config
    preset["resolved_model_dir_name"] = derive_model_dir_name(preset)
    return preset


def derive_model_dir_name(preset: dict[str, Any]) -> str:
    explicit = str(preset.get("model_dir_name", "")).strip()
    if explicit:
        return explicit

    inference_config = preset.get("inference_config", {})
    if not isinstance(inference_config, dict):
        raise ValueError("Preset is missing resolved inference_config")

    model_path = str(inference_config.get("MODEL_PATH", "")).strip()
    model_name = str(inference_config.get("MODEL_NAME", "")).strip()
    derived = os.path.basename(model_path.rstrip("/")) if model_path else model_name
    if not derived:
        raise ValueError("Unable to derive model directory name from MODEL_PATH or MODEL_NAME")
    return derived


def lookup_field(mapping: Any, dotted_path: str) -> Any:
    value = mapping
    for part in dotted_path.split("."):
        if not part:
            continue
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"Field not found: {dotted_path}")
        value = value[part]
    return value


def write_runtime_inference_config(
    preset: dict[str, Any],
    output_path: Path,
    runtime_gpu_id: int | None,
) -> Path:
    inference_config = copy.deepcopy(preset.get("inference_config", {}))
    if not isinstance(inference_config, dict):
        raise ValueError("Preset is missing resolved inference_config")
    if runtime_gpu_id is not None:
        inference_config["GPU_ID"] = int(runtime_gpu_id)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(inference_config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return output_path


def scenario_output_dir(
    scenario_file: str | Path,
    scenario_root: str | Path,
    result_root: str | Path,
    model_dir_name: str,
) -> Path:
    scenario_file_path = Path(scenario_file).resolve()
    scenario_root_path = Path(scenario_root).resolve()
    result_root_path = Path(result_root).resolve()
    scenario_dir = scenario_file_path.parent

    if os.path.commonpath([str(scenario_dir), str(scenario_root_path)]) != str(scenario_root_path):
        raise ValueError(
            f"Scenario directory {scenario_dir} is not under scenario root {scenario_root_path}"
        )

    relative_dir = scenario_dir.relative_to(scenario_root_path)
    return result_root_path / model_dir_name / relative_dir
