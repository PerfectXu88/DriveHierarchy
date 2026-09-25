"""Dataclasses used across the evaluation package."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DatasetSpec:
    name: str
    path: str
    task: str


@dataclass(slots=True)
class DatasetCatalog:
    name: str
    root: Path
    datasets: list[DatasetSpec]


@dataclass(slots=True)
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0


@dataclass(slots=True)
class RuntimeConfig:
    batch_size: int = 1
    seed: int = 0
    device: str = "auto"
    device_map: str = "auto"
    dtype: str | None = None
    revision: str = ""
    attn_implementation: str = ""
    tensor_parallel_size: int | None = None
    gpu_memory_utilization: float = 0.90
    max_model_len: int | None = None
    max_images_per_prompt: int | None = None
    max_dynamic_patch: int | None = None
    image_patch_size: int = 16
    mm_processor_cache_gb: float = 0.0
    mm_processor_cache_type: str = "shm"
    mm_shm_cache_max_object_size_mb: int = 512
    enable_mm_preprocessor_cache: bool = False
    prefer_bf16: bool = False
    use_fast_processor: bool | None = None
    use_fast_tokenizer: bool = False
    trust_remote_code: bool = True
    num_beams: int = 1
    do_sample: bool | None = None


@dataclass(slots=True)
class ModelConfig:
    name: str
    backend: str
    family: str
    model: str
    base_model: str | None = None
    adapter_model: str | None = None
    merge_adapter: bool = False
    trust_remote_code: bool = True
    system_prompt: str = ""
    output_dir: str = "result/open_loop"
    dataset_config: str = "open_loop_all"
    datasets: str = "all"
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    family_options: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    config_path: Path | None = None


@dataclass(slots=True)
class PreparedSample:
    qid: str | None
    id_key: str
    question_original: str
    question_for_model: str
    ground_truth: Any
    task: str
    image_entries: list[dict[str, Any]]
    image_paths: list[str]
    labels: list[str]
    labeled_images: bool
    extra: dict[str, Any] = field(default_factory=dict)

