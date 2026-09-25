"""Dataset selection and resolution helpers."""

from __future__ import annotations

from pathlib import Path

from .constants import TASK_TEXT
from .types import DatasetCatalog, DatasetSpec


def select_datasets(catalog: DatasetCatalog, selection: str) -> list[DatasetSpec]:
    if not selection or selection.strip().lower() == "all":
        return list(catalog.datasets)

    tokens = [item.strip() for item in selection.split(",") if item.strip()]
    selected: list[DatasetSpec] = []
    for token in tokens:
        matched = False
        for dataset in catalog.datasets:
            if token in {dataset.name, dataset.path, Path(dataset.path).name}:
                selected.append(dataset)
                matched = True
                break
        if not matched:
            selected.append(
                DatasetSpec(
                    name=Path(token).stem,
                    path=token,
                    task=TASK_TEXT,
                )
            )
    return selected


def resolve_dataset_path(catalog: DatasetCatalog, dataset: DatasetSpec) -> Path:
    candidate = Path(dataset.path)
    if candidate.exists():
        return candidate.resolve()
    return (catalog.root / dataset.path).resolve()

