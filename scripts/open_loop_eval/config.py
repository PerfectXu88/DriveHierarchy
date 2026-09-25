"""Configuration loading for model presets and dataset catalogs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .constants import DEFAULT_SYSTEM_PROMPT
from .types import DatasetCatalog, DatasetSpec, GenerationConfig, ModelConfig, RuntimeConfig


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PACKAGE_ROOT / "configs"
MODEL_CONFIG_ROOT = CONFIG_ROOT / "models"
DATASET_CONFIG_ROOT = CONFIG_ROOT / "datasets"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def resolve_model_config_path(spec: str) -> Path:
    candidate = Path(spec)
    if candidate.exists():
        return candidate.resolve()

    preset = MODEL_CONFIG_ROOT / f"{spec}.json"
    if preset.exists():
        return preset.resolve()

    raise FileNotFoundError(f"Model config not found: {spec}")


def resolve_dataset_config_path(spec: str) -> Path:
    candidate = Path(spec)
    if candidate.exists():
        return candidate.resolve()

    preset = DATASET_CONFIG_ROOT / f"{spec}.json"
    if preset.exists():
        return preset.resolve()

    raise FileNotFoundError(f"Dataset config not found: {spec}")


def _parse_generation(data: dict[str, Any] | None) -> GenerationConfig:
    payload = data or {}
    return GenerationConfig(
        max_new_tokens=int(payload.get("max_new_tokens", 128)),
        temperature=float(payload.get("temperature", 0.0)),
        top_p=float(payload.get("top_p", 1.0)),
        top_k=int(payload.get("top_k", -1)),
        repetition_penalty=float(payload.get("repetition_penalty", 1.0)),
        presence_penalty=float(payload.get("presence_penalty", 0.0)),
    )


def _parse_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    raise ValueError(f"Unsupported boolean value: {value!r}")


def _parse_runtime(data: dict[str, Any] | None, trust_remote_code: bool) -> RuntimeConfig:
    payload = data or {}
    return RuntimeConfig(
        batch_size=int(payload.get("batch_size", 1)),
        seed=int(payload.get("seed", 0)),
        device=str(payload.get("device", "auto")),
        device_map=str(payload.get("device_map", "auto")),
        dtype=payload.get("dtype"),
        revision=str(payload.get("revision", "")),
        attn_implementation=str(payload.get("attn_implementation", "")),
        tensor_parallel_size=payload.get("tensor_parallel_size"),
        gpu_memory_utilization=float(payload.get("gpu_memory_utilization", 0.90)),
        max_model_len=payload.get("max_model_len"),
        max_images_per_prompt=payload.get("max_images_per_prompt"),
        max_dynamic_patch=payload.get("max_dynamic_patch"),
        image_patch_size=int(payload.get("image_patch_size", 16)),
        mm_processor_cache_gb=float(payload.get("mm_processor_cache_gb", 0.0)),
        mm_processor_cache_type=str(payload.get("mm_processor_cache_type", "shm")),
        mm_shm_cache_max_object_size_mb=int(payload.get("mm_shm_cache_max_object_size_mb", 512)),
        enable_mm_preprocessor_cache=bool(payload.get("enable_mm_preprocessor_cache", False)),
        prefer_bf16=bool(payload.get("prefer_bf16", False)),
        use_fast_processor=_parse_optional_bool(payload.get("use_fast_processor")),
        use_fast_tokenizer=bool(payload.get("use_fast_tokenizer", False)),
        trust_remote_code=bool(payload.get("trust_remote_code", trust_remote_code)),
        num_beams=int(payload.get("num_beams", 1)),
        do_sample=_parse_optional_bool(payload.get("do_sample")),
    )


def load_model_config(spec: str) -> ModelConfig:
    path = resolve_model_config_path(spec)
    raw = _load_json(path)

    trust_remote_code = bool(raw.get("trust_remote_code", True))
    config = ModelConfig(
        name=str(raw["name"]),
        backend=str(raw["backend"]),
        family=str(raw["family"]),
        model=str(raw["model"]),
        base_model=raw.get("base_model"),
        adapter_model=raw.get("adapter_model"),
        merge_adapter=bool(raw.get("merge_adapter", False)),
        trust_remote_code=trust_remote_code,
        system_prompt=str(raw.get("system_prompt", DEFAULT_SYSTEM_PROMPT)),
        output_dir=str(raw.get("output_dir", "result/open_loop")),
        dataset_config=str(raw.get("dataset_config", "open_loop_all")),
        datasets=str(raw.get("datasets", "all")),
        generation=_parse_generation(raw.get("generation")),
        runtime=_parse_runtime(raw.get("runtime"), trust_remote_code=trust_remote_code),
        family_options=dict(raw.get("family_options", {})),
        notes=str(raw.get("notes", "")),
        config_path=path,
    )
    return config


def load_dataset_catalog(spec: str) -> DatasetCatalog:
    path = resolve_dataset_config_path(spec)
    raw = _load_json(path)
    root = (path.parent / raw["root"]).resolve()
    datasets = [
        DatasetSpec(
            name=str(item["name"]),
            path=str(item["path"]),
            task=str(item["task"]),
        )
        for item in raw["datasets"]
    ]
    return DatasetCatalog(name=str(raw["name"]), root=root, datasets=datasets)


def list_model_configs() -> list[Path]:
    return sorted(MODEL_CONFIG_ROOT.glob("*.json"))

