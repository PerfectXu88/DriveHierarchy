#!/usr/bin/env python3
"""Run Open Loop evaluation with vLLM backends."""

from __future__ import annotations

import argparse
import sys

from open_loop_eval.config import list_model_configs, load_model_config
from open_loop_eval.constants import BACKEND_VLLM
from open_loop_eval.runners import run_vllm_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open Loop evaluation entrypoint for vLLM models.")
    parser.add_argument("--config", required=False, help="Model config preset name or path.")
    parser.add_argument("--datasets", default="", help="Optional dataset override.")
    parser.add_argument("--dataset-config", default="", help="Optional dataset config preset name or path.")
    parser.add_argument("--output-dir", default="", help="Optional output dir override.")
    parser.add_argument("--list-model-configs", action="store_true", help="List available vLLM model presets and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and dataset selection without loading model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_model_configs:
        for path in list_model_configs():
            config = load_model_config(str(path))
            if config.backend == BACKEND_VLLM:
                print(path.stem)
        return

    if not args.config:
        print("--config is required unless --list-model-configs is used.", file=sys.stderr)
        sys.exit(2)

    config = load_model_config(args.config)
    if config.backend != BACKEND_VLLM:
        print(f"Config backend mismatch: expected {BACKEND_VLLM}, got {config.backend}", file=sys.stderr)
        sys.exit(2)

    run_vllm_evaluation(
        config=config,
        datasets_override=args.datasets or None,
        dataset_config_override=args.dataset_config or None,
        output_dir_override=args.output_dir or None,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

