"""Shared file, image, and logging utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

from PIL import Image


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"open_loop_eval::{log_path.stem}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def sanitize_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def count_lines(path: Path) -> int:
    total = 0
    with path.open("r", encoding="utf-8") as handle:
        for _ in handle:
            total += 1
    return total


def read_processed_ids(output_path: Path) -> set[str]:
    processed: set[str] = set()
    if not output_path.exists():
        return processed
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = record.get("Question_id") or record.get("task_id") or record.get("id")
            if qid is not None:
                processed.add(str(qid))
    return processed


def load_images(
    image_entries: Sequence[dict[str, Any]],
    logger: logging.Logger,
    qid: str | None,
) -> tuple[list[Image.Image], list[str], bool]:
    images: list[Image.Image] = []
    labels: list[str] = []
    for index, entry in enumerate(image_entries):
        path = entry.get("path")
        if not path or not Path(path).exists():
            logger.error("Missing image | qid=%s | path=%s", qid, path)
            return [], [], False
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load image | qid=%s | path=%s | error=%s", qid, path, exc)
            return [], [], False
        images.append(image)
        label = entry.get("label")
        labels.append(str(label if label is not None else index + 1))
    return images, labels, True


def close_images(images: Sequence[Image.Image]) -> None:
    for image in images:
        try:
            image.close()
        except Exception:  # noqa: BLE001
            pass


def close_images_from_meta(items: Sequence[dict[str, Any]]) -> None:
    for item in items:
        images = item.pop("_images_to_close", None)
        if images:
            close_images(images)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

