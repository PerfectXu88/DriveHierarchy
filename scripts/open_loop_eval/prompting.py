"""Shared sample parsing and prompting helpers."""

from __future__ import annotations

import json
from typing import Any, Sequence

from .constants import (
    TASK_BBOX,
    TASK_MULTI_CHOICE,
    TASK_NUMERIC,
    TASK_SEQUENCE_ORDER,
    TASK_YES_NO,
)
from .types import PreparedSample


def normalize_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def compose_system_prompt(base: str, task_kind: str) -> str:
    base_clean = (base or "").strip()
    if task_kind == TASK_YES_NO:
        extra = "Answer with only 'yes' or 'no' in lowercase. Do not output any other text."
    elif task_kind == TASK_NUMERIC:
        extra = "Answer with only the numeric result (integer or decimal). Do not include units or extra text."
    elif task_kind == TASK_MULTI_CHOICE:
        extra = "Choose the correct option and output only the option letter (e.g., A/B/C) without any extra text."
    elif task_kind == TASK_BBOX:
        extra = "Output only the bounding box as [x1, y1, x2, y2] in pixels with no extra text."
    elif task_kind == TASK_SEQUENCE_ORDER:
        extra = "Output only the ordered list of option labels in the format [\"A\", \"B\", ...] with no extra text."
    else:
        extra = "Output only the final answer without extra explanation."

    if base_clean:
        return f"{base_clean} {extra}"
    return extra


def format_question(sample: dict[str, Any], task_kind: str) -> str:
    question = normalize_to_text(sample.get("question") or sample.get("prompt") or "")

    if task_kind == TASK_MULTI_CHOICE:
        options = sample.get("options") or []
        if options:
            options_text = "\n".join(str(option) for option in options)
            question = f"{question}\nOptions:\n{options_text}" if question else f"Options:\n{options_text}"
    elif task_kind == TASK_SEQUENCE_ORDER:
        if not question:
            question = "Please determine the correct order of the images."
        labels = [str(item["label"]) for item in sample.get("images") or [] if item.get("label")]
        if labels:
            question = f"{question}\nOption labels: {', '.join(labels)}."
        question = f'{question}\nOutput format: ["A", "B", ...].'

    return question


def extract_id(sample: dict[str, Any]) -> tuple[str, str | None]:
    if "Question_id" in sample:
        return "Question_id", sample.get("Question_id")
    if "task_id" in sample:
        return "task_id", sample.get("task_id")
    return "id", sample.get("id")


def extract_ground_truth(sample: dict[str, Any], task_kind: str) -> Any:
    if task_kind == TASK_SEQUENCE_ORDER:
        return sample.get("ground_truth_order")
    if task_kind == TASK_BBOX:
        return sample.get("ground_truth")
    return sample.get("answer")


def extract_image_entries(sample: dict[str, Any], task_kind: str) -> tuple[list[dict[str, Any]], bool]:
    if task_kind == TASK_SEQUENCE_ORDER:
        return [
            {"path": item.get("path"), "label": item.get("label")}
            for item in (sample.get("images") or [])
        ], True

    image_paths = sample.get("image_paths")
    if not image_paths:
        image_paths = (sample.get("meta_data") or {}).get("image_paths")
    if image_paths:
        return [{"path": path} for path in image_paths], False

    image_path = sample.get("image_path")
    if image_path:
        return [{"path": image_path}], False

    return [], False


def prepare_sample(sample: dict[str, Any], task_kind: str) -> PreparedSample:
    id_key, qid = extract_id(sample)
    question_for_model = format_question(sample, task_kind)
    question_original = normalize_to_text(sample.get("question") or sample.get("prompt") or question_for_model)
    ground_truth = extract_ground_truth(sample, task_kind)
    image_entries, labeled_images = extract_image_entries(sample, task_kind)
    image_paths = [entry["path"] for entry in image_entries if entry.get("path")]
    labels = [str(entry.get("label")) for entry in image_entries if entry.get("label") is not None]
    extra: dict[str, Any] = {}
    if task_kind == TASK_SEQUENCE_ORDER:
        extra["ground_truth_order"] = sample.get("ground_truth_order")

    return PreparedSample(
        qid=str(qid) if qid is not None else None,
        id_key=id_key,
        question_original=question_original,
        question_for_model=question_for_model,
        ground_truth=ground_truth,
        task=task_kind,
        image_entries=image_entries,
        image_paths=image_paths,
        labels=labels,
        labeled_images=labeled_images,
        extra=extra,
    )


def build_output_record(prepared: PreparedSample, prediction: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "Question_id": prepared.qid,
        "question": prepared.question_original,
        "ground_truth": prepared.ground_truth,
        "model_prediction": prediction,
    }
    if prepared.id_key and prepared.id_key != "Question_id":
        record[prepared.id_key] = prepared.qid
    if prepared.extra.get("ground_truth_order") is not None:
        record["ground_truth_order"] = prepared.extra["ground_truth_order"]
    if prepared.labeled_images:
        record["images"] = list(prepared.image_entries)
    if len(prepared.image_paths) == 1:
        record["image_path"] = prepared.image_paths[0]
    elif prepared.image_paths:
        record["image_paths"] = prepared.image_paths
    return record


def merge_system_and_question(system_prompt: str, question: str) -> str:
    system_clean = (system_prompt or "").strip()
    question_clean = (question or "").strip()
    if system_clean and question_clean:
        return f"{system_clean}\n\n{question_clean}"
    if system_clean:
        return system_clean
    return question_clean


def content_with_images(
    question: str,
    images: Sequence[Any],
    labels: Sequence[str] | None,
    labeled_images: bool,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if labeled_images and labels:
        for label, image in zip(labels, images):
            content.append({"type": "text", "text": f"Image {label}"})
            content.append({"type": "image", "image": image})
    else:
        for image in images:
            content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": question})
    return content

