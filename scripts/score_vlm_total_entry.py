#!/usr/bin/env python3
"""Single-entry scoring for Open Loop VLM evaluation results.

This script scores one model result directory and writes exactly one final JSON
summary by default:

`<result_dir>/evaluation/final_score.json`

It supports:
- classification / numeric / bbox / sequence tasks
- optional L1_4 scoring with wayveai/Lingo-Judge
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?")
LEADING_CHOICE_RE = re.compile(r'^\s*["\'\(\[\{]*\s*([A-Z])(?:\s*[\)\]\}"\'\.,:;-]|$)')
CHOICE_HINT_RE = re.compile(r"\b(?:OPTION|ANSWER|CHOICE)\s*[:\-]?\s*([A-Z])\b")

TASK_TEXT_LINGO = "text_lingo_judge"
METRIC_LINGO = "lingo_judge_probability"

FILE_TASKS: Dict[str, str] = {
    "R1_1_A_Existence_pred.jsonl": "yes_no",
    "R1_1_B_Counting_pred.jsonl": "numeric_count",
    "R1_1_C_State_Attribute_pred.jsonl": "multi_choice",
    "R1_2_A_Nearest_Object_pred.jsonl": "numeric_distance",
    "R1_2_B_Certain_Object_pred.jsonl": "numeric_distance",
    "R1_2_C_Distance_Bucket_pred.jsonl": "numeric_count",
    "R1_3_Location_Questions_pred.jsonl": "bbox",
    "R1_4_Situation_Description_pred.jsonl": TASK_TEXT_LINGO,
    "R2_1_Multi_view_Memory_pred.jsonl": "numeric_count",
    "R2_2_A_Temporal_Counting_pred.jsonl": "numeric_count",
    "R2_2_B_Temporal_Status_Recognition_pred.jsonl": "multi_choice",
    "R2_3_Spatial_Relations_pred.jsonl": "multi_choice",
    "R3_1_Outcome_Prediction_pred.jsonl": "multi_choice",
    "R3_2_Sequential_Planning_pred.jsonl": "sequence_order",
    "R1_1_A_filtered_Existence_nuscenes_trainval_w_path_pred.jsonl": "yes_no",
    "R1_1_B_filtered_Counting_nuscenes_trainval_w_path_pred.jsonl": "numeric_count",
    "R1_1_C_filtered_State_nuscenes_trainval_w_path_pred.jsonl": "multi_choice",
    "R1_2_A_filtered_Distance_global_nuscenes_trainval_w_path_pred.jsonl": "numeric_distance",
    "R1_2_B_filtered_Distance_pixel_nuscenes_trainval_w_path_pred.jsonl": "numeric_distance",
    "R1_2_C_filtered_DistanceBucket_nuscenes_trainval_w_path_pred.jsonl": "numeric_count",
    "R1_3_location_drama_w_path_pred.jsonl": "bbox",
    "R1_4_LingoQA_situation_w_path_pred.jsonl": TASK_TEXT_LINGO,
    "R2_1_A_filtered_Multiview_memory_nuscenes_trainval_w_paths_pred.jsonl": "numeric_count",
    "R2_1_B_Temporal_Sequence_nuscenes_trainval_w_paths_pred.jsonl": "numeric_count",
    "R2_1_C_nuscenes_trainval_w_paths_pred.jsonl": "multi_choice",
    "R2_3_filtered_Spatial_rel_nuscenes_trainval_w_path_pred.jsonl": "multi_choice",
    "R3_1_Outcome_prediction_drivelm_trainnus_w_path_pred.jsonl": "multi_choice",
    "R3_2_Sequence_planning_navsim_trainval_w_path_pred.jsonl": "sequence_order"
}


@dataclass(frozen=True)
class RuntimeConfig:
    count_c: float
    count_tau: float
    count_sigma: float
    distance_epsilon: float
    distance_tau: float
    distance_sigma: float


@dataclass
class FileScoreResult:
    source_file: str
    task_type: str
    total: int
    parse_fail: int
    mean: float
    p50: float
    p90: float
    pass_rate: Optional[float]
    metric: str
    scores: List[float]


def clamp_0_1(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def clamp_0_100(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    if p <= 0:
        return float(min(values))
    if p >= 100:
        return float(max(values))
    vals = sorted(values)
    rank = (len(vals) - 1) * (p / 100.0)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return float(vals[low])
    frac = rank - low
    return float(vals[low] * (1.0 - frac) + vals[high] * frac)


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


def resolve_sigma(name: str, tau: float, sigma: Optional[float], e50: Optional[float]) -> float:
    if tau < 0:
        raise ValueError(f"{name}: tau must be >= 0")
    if sigma is not None:
        if sigma <= 0:
            raise ValueError(f"{name}: sigma must be > 0")
        return sigma
    if e50 is None:
        raise ValueError(f"{name}: sigma is not provided, so e50 must be provided")
    if e50 <= tau:
        raise ValueError(f"{name}: e50 must be > tau when deriving sigma")
    return (e50 - tau) / math.log(2.0)


def unified_exp_score(error: float, tau: float, sigma: float) -> float:
    if error <= tau:
        return 100.0
    return clamp_0_100(100.0 * math.exp(-(error - tau) / sigma))


def round_half_up(value: float) -> int:
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def normalize_text(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    return text.strip()


def strip_choice_prefix(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"^\s*[A-Za-z]\s*[\.\):\-]\s*", "", text)
    text = re.sub(r"^\s*(?:option|answer|choice)\s*[:\-]?\s*[A-Za-z]\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def extract_yes_no(value: Any) -> Optional[str]:
    text = normalize_text(value)
    if not text:
        return None
    if text in {"yes", "y", "true"} or text.startswith("yes"):
        return "yes"
    if text in {"no", "n", "false"} or text.startswith("no"):
        return "no"
    return None


def extract_choice_letter(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    if len(text) == 1 and text.isalpha():
        return text
    match = LEADING_CHOICE_RE.search(text)
    if match:
        return match.group(1)
    match = CHOICE_HINT_RE.search(text)
    if match:
        return match.group(1)
    return None


def extract_first_number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = NUMBER_RE.search(value.replace(",", " "))
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return None
    return None


def _canonical_bbox(values: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = values[:4]
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    return [left, top, right, bottom]


def extract_bbox(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, dict):
        if "bbox" in value:
            return extract_bbox(value["bbox"])
        return None
    if isinstance(value, list):
        if len(value) >= 4:
            numbers: List[float] = []
            for item in value[:4]:
                number = extract_first_number(item)
                if number is None:
                    numbers = []
                    break
                numbers.append(number)
            if len(numbers) == 4:
                return _canonical_bbox(numbers)
        if value and isinstance(value[0], list):
            return extract_bbox(value[0])
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
            parsed_bbox = extract_bbox(parsed)
            if parsed_bbox is not None:
                return parsed_bbox
        except json.JSONDecodeError:
            pass
        numbers = [float(item) for item in NUMBER_RE.findall(text)]
        if len(numbers) >= 4:
            return _canonical_bbox(numbers[:4])
    return None


def bbox_iou(gt_bbox: Sequence[float], pred_bbox: Sequence[float]) -> float:
    gx1, gy1, gx2, gy2 = gt_bbox
    px1, py1, px2, py2 = pred_bbox
    inter_w = max(0.0, min(gx2, px2) - max(gx1, px1))
    inter_h = max(0.0, min(gy2, py2) - max(gy1, py1))
    inter = inter_w * inter_h
    g_area = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
    p_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    union = g_area + p_area - inter
    if union <= 0:
        return 0.0
    return inter / union


def extract_label_sequence(value: Any) -> List[str]:
    raw_items: List[Any] = []
    if isinstance(value, list):
        raw_items = list(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            raw_items = parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            raw_items = re.findall(r"\b([A-Z])\b", text.upper())
    else:
        return []

    labels: List[str] = []
    for item in raw_items:
        label = extract_choice_letter(item)
        if label is None:
            token = str(item).strip().upper()
            if len(token) == 1 and token.isalpha():
                label = token
        if label and label not in labels:
            labels.append(label)
    return labels


def score_yes_no(gt: Any, pred: Any) -> tuple[float, bool]:
    gt_norm = extract_yes_no(gt)
    pred_norm = extract_yes_no(pred)
    if gt_norm is None or pred_norm is None:
        return 0.0, False
    return (100.0 if gt_norm == pred_norm else 0.0), True


def score_multi_choice(gt: Any, pred: Any) -> tuple[float, bool]:
    gt_letter = extract_choice_letter(gt)
    pred_letter = extract_choice_letter(pred)
    if gt_letter is not None and pred_letter is not None:
        return (100.0 if gt_letter == pred_letter else 0.0), True
    gt_text = normalize_text(strip_choice_prefix(gt))
    pred_text = normalize_text(strip_choice_prefix(pred))
    if not gt_text or not pred_text:
        return 0.0, False
    return (100.0 if gt_text == pred_text else 0.0), True


def score_numeric_count(gt: Any, pred: Any, cfg: RuntimeConfig) -> tuple[float, bool]:
    gt_num = extract_first_number(gt)
    pred_num = extract_first_number(pred)
    if gt_num is None or pred_num is None:
        return 0.0, False
    gt_count = max(0, round_half_up(gt_num))
    pred_count = max(0, round_half_up(pred_num))
    error = abs(pred_count - gt_count) / max(float(gt_count), cfg.count_c)
    return unified_exp_score(error, cfg.count_tau, cfg.count_sigma), True


def score_numeric_distance(gt: Any, pred: Any, cfg: RuntimeConfig) -> tuple[float, bool]:
    gt_num = extract_first_number(gt)
    pred_num = extract_first_number(pred)
    if gt_num is None or pred_num is None:
        return 0.0, False
    gt_dist = max(0.0, float(gt_num))
    pred_dist = max(0.0, float(pred_num))
    error = abs(math.log((pred_dist + cfg.distance_epsilon) / (gt_dist + cfg.distance_epsilon)))
    return unified_exp_score(error, cfg.distance_tau, cfg.distance_sigma), True


def score_bbox(gt: Any, pred: Any) -> tuple[float, bool]:
    gt_bbox = extract_bbox(gt)
    pred_bbox = extract_bbox(pred)
    if gt_bbox is None or pred_bbox is None:
        return 0.0, False
    return clamp_0_100(100.0 * bbox_iou(gt_bbox, pred_bbox)), True


def score_sequence_order(gt: Any, pred: Any) -> tuple[float, bool]:
    gt_seq = extract_label_sequence(gt)
    pred_seq = extract_label_sequence(pred)
    if not gt_seq or not pred_seq:
        return 0.0, False
    gt_set = set(gt_seq)
    pred_filtered: List[str] = []
    for label in pred_seq:
        if label in gt_set and label not in pred_filtered:
            pred_filtered.append(label)
    for label in gt_seq:
        if label not in pred_filtered:
            pred_filtered.append(label)
    n = len(gt_seq)
    if n == 1:
        return (100.0 if pred_filtered[0] == gt_seq[0] else 0.0), True
    order = {label: index for index, label in enumerate(pred_filtered)}
    pair_total = n * (n - 1) // 2
    inversions = 0
    for i in range(n):
        for j in range(i + 1, n):
            if order[gt_seq[i]] > order[gt_seq[j]]:
                inversions += 1
    norm_inv = inversions / pair_total
    return clamp_0_100(100.0 * (1.0 - norm_inv)), True


def infer_task_type(filename: str) -> Optional[str]:
    task = FILE_TASKS.get(filename)
    if task is not None:
        return task
    lower = filename.lower()
    if "existence" in lower:
        return "yes_no"
    if "counting" in lower or "distance_bucket" in lower or "multi_view_memory" in lower or "temporal_counting" in lower:
        return "numeric_count"
    if "nearest_object" in lower or "certain_object" in lower or "distance_" in lower:
        return "numeric_distance"
    if "state_attribute" in lower or "status_recognition" in lower or "spatial_relations" in lower or "outcome_prediction" in lower:
        return "multi_choice"
    if "location_questions" in lower:
        return "bbox"
    if "situation_description" in lower:
        return TASK_TEXT_LINGO
    if "sequential_planning" in lower:
        return "sequence_order"
    return None


def score_structured_file(path: Path, task_type: str, cfg: RuntimeConfig) -> FileScoreResult:
    scores: List[float] = []
    parse_fail = 0
    metric = task_type
    for record in read_jsonl(path):
        gt = record.get("ground_truth")
        pred = record.get("model_prediction")
        if task_type == "yes_no":
            score, valid = score_yes_no(gt, pred)
            metric = "em_yes_no"
        elif task_type == "multi_choice":
            score, valid = score_multi_choice(gt, pred)
            metric = "em_multi_choice"
        elif task_type == "numeric_count":
            score, valid = score_numeric_count(gt, pred, cfg)
            metric = "count_exp_decay"
        elif task_type == "numeric_distance":
            score, valid = score_numeric_distance(gt, pred, cfg)
            metric = "distance_log_ratio_exp_decay"
        elif task_type == "bbox":
            score, valid = score_bbox(gt, pred)
            metric = "bbox_iou"
        elif task_type == "sequence_order":
            score, valid = score_sequence_order(gt, pred)
            metric = "sequence_kendall"
        else:
            score, valid = 0.0, False
        if not valid:
            parse_fail += 1
        scores.append(score)

    total = len(scores)
    mean = (sum(scores) / total) if total else 0.0
    return FileScoreResult(
        source_file=path.name,
        task_type=task_type,
        total=total,
        parse_fail=parse_fail,
        mean=mean,
        p50=percentile(scores, 50),
        p90=percentile(scores, 90),
        pass_rate=None,
        metric=metric,
        scores=scores
    )


def resolve_lingo_device(device_arg: str) -> int:
    if device_arg != "auto":
        return int(device_arg)
    try:
        import torch
        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1


def score_lingo_file(path: Path, model_name: str, device_arg: str, threshold: float) -> FileScoreResult:
    from transformers import pipeline

    device = resolve_lingo_device(device_arg)
    judge = pipeline("text-classification", model=model_name, device=device)

    scores: List[float] = []
    parse_fail = 0
    pass_count = 0

    for record in read_jsonl(path):
        question = str(record.get("question", "")).strip()
        answer = str(record.get("ground_truth", "")).strip()
        prediction = str(record.get("model_prediction", "")).strip()
        if not question or not answer or not prediction:
            parse_fail += 1
            scores.append(0.0)
            continue
        prompt = f"[CLS]\nQuestion: {question}\nAnswer: {answer}\nStudent: {prediction}"
        raw_output = judge(prompt)
        judge_score: Optional[float] = None
        if isinstance(raw_output, list) and raw_output and isinstance(raw_output[0], dict):
            raw_score = raw_output[0].get("score")
            if raw_score is not None:
                judge_score = clamp_0_1(float(raw_score))
        if judge_score is None:
            parse_fail += 1
            scores.append(0.0)
            continue
        final_score = clamp_0_100(judge_score * 100.0)
        if judge_score >= threshold:
            pass_count += 1
        scores.append(final_score)

    total = len(scores)
    mean = (sum(scores) / total) if total else 0.0
    pass_rate = (pass_count / total) if total else 0.0
    return FileScoreResult(
        source_file=path.name,
        task_type=TASK_TEXT_LINGO,
        total=total,
        parse_fail=parse_fail,
        mean=mean,
        p50=percentile(scores, 50),
        p90=percentile(scores, 90),
        pass_rate=pass_rate,
        metric=METRIC_LINGO,
        scores=scores
    )


def summarize_file_results(results: Sequence[FileScoreResult]) -> dict[str, Any]:
    overall_scores: List[float] = []
    overall_parse_fail = 0
    per_task_scores: Dict[str, List[float]] = {}
    per_task_parse_fail: Dict[str, int] = {}
    per_task_totals: Dict[str, int] = {}

    for result in results:
        overall_scores.extend(result.scores)
        overall_parse_fail += result.parse_fail
        per_task_scores.setdefault(result.task_type, []).extend(result.scores)
        per_task_parse_fail[result.task_type] = per_task_parse_fail.get(result.task_type, 0) + result.parse_fail
        per_task_totals[result.task_type] = per_task_totals.get(result.task_type, 0) + result.total

    per_file = []
    for result in results:
        item = {
            "source_file": result.source_file,
            "task_type": result.task_type,
            "metric": result.metric,
            "n": result.total,
            "parse_fail": result.parse_fail,
            "mean": result.mean,
            "p50": result.p50,
            "p90": result.p90
        }
        if result.pass_rate is not None:
            item["pass_rate"] = result.pass_rate
        per_file.append(item)

    per_task = []
    for task_type in sorted(per_task_scores):
        task_scores = per_task_scores[task_type]
        per_task.append(
            {
                "task_type": task_type,
                "n": per_task_totals[task_type],
                "parse_fail": per_task_parse_fail[task_type],
                "mean": (sum(task_scores) / len(task_scores)) if task_scores else 0.0,
                "p50": percentile(task_scores, 50),
                "p90": percentile(task_scores, 90)
            }
        )

    overall = {
        "n": len(overall_scores),
        "parse_fail": overall_parse_fail,
        "mean": (sum(overall_scores) / len(overall_scores)) if overall_scores else 0.0,
        "p50": percentile(overall_scores, 50),
        "p90": percentile(overall_scores, 90)
    }
    return {
        "overall": overall,
        "per_task": per_task,
        "per_file": per_file
    }


def resolve_model_dir(result_root: Path, model_name: Optional[str], result_dir: Optional[Path]) -> Path:
    if result_dir is not None:
        if not result_dir.exists():
            raise FileNotFoundError(f"result_dir not found: {result_dir}")
        return result_dir
    if model_name:
        candidate = result_root / model_name
        if not candidate.exists():
            raise FileNotFoundError(f"model dir not found: {candidate}")
        return candidate
    candidates = sorted(path for path in result_root.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"no model directory found under {result_root}")
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise ValueError(f"multiple model directories found under {result_root}: {names}; use --model-name")
    return candidates[0]


def check_module_available(python_bin: str, module_name: str) -> bool:
    import subprocess
    result = subprocess.run(
        [python_bin, "-c", f"import {module_name}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False
    )
    return result.returncode == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-file summary scorer for one model result directory.")
    parser.add_argument("--result-root", type=Path, default=Path("result/open_loop"), help="Root folder containing model result dirs.")
    parser.add_argument("--model-name", type=str, default="", help="Model directory name under result root.")
    parser.add_argument("--result-dir", type=Path, default=None, help="Explicit model result directory.")
    parser.add_argument("--output-subdir", type=str, default="evaluation", help="Subdirectory under result dir for score output.")
    parser.add_argument("--output-file", type=str, default="final_score.json", help="Final score filename.")
    parser.add_argument("--python-bin", type=str, default=sys.executable, help="Python executable used for dependency checks.")
    parser.add_argument("--count-c", type=float, default=1.0)
    parser.add_argument("--count-tau", type=float, default=0.05)
    parser.add_argument("--count-sigma", type=float, default=None)
    parser.add_argument("--count-e50", type=float, default=0.20)
    parser.add_argument("--distance-epsilon", type=float, default=0.1)
    parser.add_argument("--distance-tau", type=float, default=0.0)
    parser.add_argument("--distance-sigma", type=float, default=None)
    parser.add_argument("--distance-e50", type=float, default=math.log(2.0))
    parser.add_argument("--skip-l1-4", action="store_true", help="Skip R1_4 Lingo-Judge scoring.")
    parser.add_argument("--l1-4-model-name", type=str, default="wayveai/Lingo-Judge")
    parser.add_argument("--l1-4-device", type=str, default="auto")
    parser.add_argument("--l1-4-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RuntimeConfig(
        count_c=args.count_c,
        count_tau=args.count_tau,
        count_sigma=resolve_sigma("count", args.count_tau, args.count_sigma, args.count_e50),
        distance_epsilon=args.distance_epsilon,
        distance_tau=args.distance_tau,
        distance_sigma=resolve_sigma("distance", args.distance_tau, args.distance_sigma, args.distance_e50)
    )

    model_dir = resolve_model_dir(args.result_root, args.model_name or None, args.result_dir)
    output_dir = model_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_file

    pred_files = sorted(model_dir.glob("*_pred.jsonl"))
    results: List[FileScoreResult] = []
    skipped_files: List[Dict[str, str]] = []
    l1_4_skip_reason: Optional[str] = None
    l1_4_seen = False

    for pred_file in pred_files:
        task_type = infer_task_type(pred_file.name)
        if task_type is None:
            skipped_files.append({"source_file": pred_file.name, "reason": "unknown_task_type"})
            continue

        if task_type == TASK_TEXT_LINGO:
            l1_4_seen = True
            if args.skip_l1_4:
                l1_4_skip_reason = "skip_l1_4_flag"
                skipped_files.append({"source_file": pred_file.name, "reason": l1_4_skip_reason})
                continue
            if not check_module_available(args.python_bin, "transformers"):
                l1_4_skip_reason = "transformers_unavailable"
                skipped_files.append({"source_file": pred_file.name, "reason": l1_4_skip_reason})
                continue
            result = score_lingo_file(pred_file, args.l1_4_model_name, args.l1_4_device, args.l1_4_threshold)
        else:
            result = score_structured_file(pred_file, task_type, cfg)
        results.append(result)

    if not l1_4_seen and not args.skip_l1_4:
        l1_4_skip_reason = "l1_4_result_file_missing"

    summary = summarize_file_results(results)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result_dir": str(model_dir),
        "output_file": str(output_path),
        "config": {
            "count_c": args.count_c,
            "count_tau": args.count_tau,
            "count_sigma": cfg.count_sigma,
            "distance_epsilon": args.distance_epsilon,
            "distance_tau": args.distance_tau,
            "distance_sigma": cfg.distance_sigma,
            "l1_4": {
                "enabled": not bool(l1_4_skip_reason) and not args.skip_l1_4,
                "skip_reason": l1_4_skip_reason,
                "model_name": args.l1_4_model_name,
                "device": args.l1_4_device,
                "threshold": args.l1_4_threshold
            }
        },
        "scored_files": len(results),
        "skipped_files": skipped_files,
        "overall": summary["overall"],
        "per_task": summary["per_task"],
        "per_file": summary["per_file"]
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Final scoring completed.")
    print(f"result_dir: {model_dir}")
    print(f"final_score: {output_path}")


if __name__ == "__main__":
    main()
