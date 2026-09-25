#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from close_loop_eval.results import summarize_scores, write_summary_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize close-loop scene_score.json outputs into one CSV.")
    parser.add_argument(
        "--result-dir",
        default="result",
        help="Close-loop result root. Default: result",
    )
    parser.add_argument(
        "--output-csv",
        help="Output CSV path. Default: <result-dir>/result_summary.csv",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result_dir = Path(args.result_dir).resolve()
    if not result_dir.is_dir():
        raise FileNotFoundError(f"result_dir not found: {result_dir}")

    output_csv = Path(args.output_csv).resolve() if args.output_csv else result_dir / "result_summary.csv"
    rows = summarize_scores(result_dir)
    write_summary_csv(rows, output_csv)
    print(f"wrote {len(rows)} rows to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

