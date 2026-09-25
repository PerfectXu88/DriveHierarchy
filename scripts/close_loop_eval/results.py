from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


SCORE_FIELDS = (
    "score_route",
    "score_safety",
    "score_efficiency",
    "score_penalty",
    "score_composed",
)


def iter_score_files(result_dir: Path) -> Iterable[Path]:
    yield from sorted(result_dir.rglob("scene_score.json"))


def read_score_file(score_file: Path, result_dir: Path) -> dict[str, object] | None:
    relative_path = score_file.resolve().relative_to(result_dir.resolve())
    if len(relative_path.parts) < 4:
        return None

    model_name = relative_path.parts[0]
    scene_name = relative_path.parts[1]

    with score_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    return {
        "model": model_name,
        "scene": scene_name,
        **{field: float(data[field]) for field in SCORE_FIELDS},
    }


def summarize_scores(result_dir: Path) -> list[dict[str, object]]:
    grouped_scores = defaultdict(lambda: {field: 0.0 for field in SCORE_FIELDS} | {"count": 0})

    for score_file in iter_score_files(result_dir):
        record = read_score_file(score_file, result_dir)
        if record is None:
            continue
        group = grouped_scores[(record["scene"], record["model"])]
        group["count"] += 1
        for field in SCORE_FIELDS:
            group[field] += float(record[field])

    summary_rows: list[dict[str, object]] = []
    for (scene_name, model_name), group in sorted(grouped_scores.items()):
        count = int(group["count"])
        if count == 0:
            continue
        summary_rows.append(
            {
                "scene": scene_name,
                "model": model_name,
                "sample_count": count,
                **{f"avg_{field}": round(float(group[field]) / count, 6) for field in SCORE_FIELDS},
            }
        )

    return summary_rows


def write_summary_csv(rows: list[dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scene",
        "model",
        "sample_count",
        *[f"avg_{field}" for field in SCORE_FIELDS],
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

