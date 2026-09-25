"""Shared evaluation runners for Transformers and vLLM backends."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .config import load_dataset_catalog
from .datasets import resolve_dataset_path, select_datasets
from .io_utils import (
    close_images,
    close_images_from_meta,
    count_lines,
    load_images,
    read_processed_ids,
    sanitize_name,
    setup_logger,
    write_json,
)
from .prompting import build_output_record, compose_system_prompt, prepare_sample
from .types import ModelConfig


def _snapshot_config(config: ModelConfig) -> dict[str, Any]:
    payload = asdict(config)
    if config.config_path is not None:
        payload["config_path"] = str(config.config_path)
    return payload


def _prepare_output_base(config: ModelConfig, output_dir_override: str | None) -> Path:
    base = Path(output_dir_override or config.output_dir)
    return base / sanitize_name(config.name)


def _prepare_run(config: ModelConfig, output_dir_override: str | None, backend_label: str) -> tuple[Path, Any]:
    output_base = _prepare_output_base(config, output_dir_override)
    log_path = output_base / "logs" / f"{backend_label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_path)
    write_json(output_base / "run_config.json", _snapshot_config(config))
    try:
        import torch

        if config.runtime.seed is not None:
            torch.manual_seed(config.runtime.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.runtime.seed)
    except ModuleNotFoundError:
        logger.warning("torch is not installed in the current Python environment; dry-run/config validation still works.")
    return output_base, logger


def run_transformers_evaluation(
    config: ModelConfig,
    datasets_override: str | None = None,
    dataset_config_override: str | None = None,
    output_dir_override: str | None = None,
    dry_run: bool = False,
) -> None:
    dataset_catalog = load_dataset_catalog(dataset_config_override or config.dataset_config)
    datasets = select_datasets(dataset_catalog, datasets_override or config.datasets)

    output_base, logger = _prepare_run(config, output_dir_override, backend_label="transformers")
    if dry_run:
        logger.info("Dry run only; no model will be loaded.")
        for dataset in datasets:
            logger.info("Dataset | name=%s | path=%s", dataset.name, resolve_dataset_path(dataset_catalog, dataset))
        return

    from .adapters import build_transformers_adapter

    adapter = build_transformers_adapter(config, logger)
    adapter.load()

    for dataset in datasets:
        dataset_path = resolve_dataset_path(dataset_catalog, dataset)
        if not dataset_path.exists():
            logger.error("Dataset not found: %s", dataset_path)
            continue

        output_path = output_base / f"{dataset.name}_pred.jsonl"
        processed_ids = read_processed_ids(output_path)
        if processed_ids:
            logger.info("Resuming dataset | name=%s | processed=%d", dataset.name, len(processed_ids))

        total = count_lines(dataset_path)
        logger.info("Running dataset | name=%s | path=%s", dataset.name, dataset_path)

        with dataset_path.open("r", encoding="utf-8") as fin, output_path.open("a", encoding="utf-8") as fout:
            for line_number, line in enumerate(tqdm(fin, total=total, desc=dataset.name), start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample_obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("JSON decode error | dataset=%s | line=%d | error=%s", dataset.name, line_number, exc)
                    continue

                prepared = prepare_sample(sample_obj, dataset.task)
                if prepared.qid is None:
                    logger.warning("Missing id field | dataset=%s | line=%d", dataset.name, line_number)
                if prepared.qid in processed_ids:
                    continue
                if not prepared.image_entries:
                    logger.error("No image entries | dataset=%s | line=%d | qid=%s", dataset.name, line_number, prepared.qid)
                    continue

                images, labels, ok = load_images(prepared.image_entries, logger, prepared.qid)
                if not ok:
                    continue

                system_prompt = compose_system_prompt(config.system_prompt, prepared.task)
                try:
                    prediction = adapter.predict(
                        system_prompt=system_prompt,
                        question=prepared.question_for_model,
                        images=images,
                        labels=labels,
                        labeled_images=prepared.labeled_images,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Inference failed | dataset=%s | line=%d | qid=%s | error=%s", dataset.name, line_number, prepared.qid, exc)
                    close_images(images)
                    continue

                record = build_output_record(prepared, prediction)
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                if prepared.qid is not None:
                    processed_ids.add(prepared.qid)
                close_images(images)


def run_vllm_evaluation(
    config: ModelConfig,
    datasets_override: str | None = None,
    dataset_config_override: str | None = None,
    output_dir_override: str | None = None,
    dry_run: bool = False,
) -> None:
    dataset_catalog = load_dataset_catalog(dataset_config_override or config.dataset_config)
    datasets = select_datasets(dataset_catalog, datasets_override or config.datasets)

    output_base, logger = _prepare_run(config, output_dir_override, backend_label="vllm")
    if dry_run:
        logger.info("Dry run only; no model will be loaded.")
        for dataset in datasets:
            logger.info("Dataset | name=%s | path=%s", dataset.name, resolve_dataset_path(dataset_catalog, dataset))
        return

    from .adapters import build_vllm_adapter

    adapter = build_vllm_adapter(config, logger)
    adapter.load()

    for dataset in datasets:
        dataset_path = resolve_dataset_path(dataset_catalog, dataset)
        if not dataset_path.exists():
            logger.error("Dataset not found: %s", dataset_path)
            continue

        output_path = output_base / f"{dataset.name}_pred.jsonl"
        processed_ids = read_processed_ids(output_path)
        if processed_ids:
            logger.info("Resuming dataset | name=%s | processed=%d", dataset.name, len(processed_ids))

        total = count_lines(dataset_path)
        logger.info("Running dataset | name=%s | path=%s", dataset.name, dataset_path)

        batch_requests: list[Any] = []
        batch_meta: list[dict[str, Any]] = []

        def write_output(meta: dict[str, Any], prediction: str, fout: Any) -> None:
            record = build_output_record(meta["prepared"], prediction)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            if meta["prepared"].qid is not None:
                processed_ids.add(meta["prepared"].qid)

        def flush_batch(fout: Any) -> None:
            if not batch_requests:
                return
            requests = list(batch_requests)
            metas = list(batch_meta)
            try:
                predictions = adapter.generate_batch(requests)
                for prediction, meta in zip(predictions, metas):
                    write_output(meta, prediction, fout)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Batch inference failed; falling back to single | dataset=%s | error=%s", dataset.name, exc)
                for request, meta in zip(requests, metas):
                    try:
                        prediction = adapter.generate_one(request)
                    except Exception as inner_exc:  # noqa: BLE001
                        logger.exception(
                            "Single inference failed | dataset=%s | qid=%s | error=%s",
                            dataset.name,
                            meta["prepared"].qid,
                            inner_exc,
                        )
                        continue
                    write_output(meta, prediction, fout)
            finally:
                close_images_from_meta(metas)
                batch_requests.clear()
                batch_meta.clear()

        with dataset_path.open("r", encoding="utf-8") as fin, output_path.open("a", encoding="utf-8") as fout:
            for line_number, line in enumerate(tqdm(fin, total=total, desc=dataset.name), start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample_obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("JSON decode error | dataset=%s | line=%d | error=%s", dataset.name, line_number, exc)
                    continue

                prepared = prepare_sample(sample_obj, dataset.task)
                if prepared.qid is None:
                    logger.warning("Missing id field | dataset=%s | line=%d", dataset.name, line_number)
                if prepared.qid in processed_ids:
                    continue
                if not prepared.image_entries:
                    logger.error("No image entries | dataset=%s | line=%d | qid=%s", dataset.name, line_number, prepared.qid)
                    continue

                images, labels, ok = load_images(prepared.image_entries, logger, prepared.qid)
                if not ok:
                    continue

                system_prompt = compose_system_prompt(config.system_prompt, prepared.task)
                try:
                    request = adapter.build_request(
                        system_prompt=system_prompt,
                        question=prepared.question_for_model,
                        images=images,
                        labels=labels,
                        labeled_images=prepared.labeled_images,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Failed to build request | dataset=%s | qid=%s | error=%s", dataset.name, prepared.qid, exc)
                    close_images(images)
                    continue

                batch_requests.append(request)
                batch_meta.append({"prepared": prepared, "_images_to_close": images})
                if len(batch_requests) >= config.runtime.batch_size:
                    flush_batch(fout)

            if batch_requests:
                flush_batch(fout)
