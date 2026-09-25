#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from close_loop_eval.config import (
    derive_model_dir_name,
    list_model_configs,
    lookup_field,
    resolve_preset,
    scenario_output_dir,
    write_runtime_inference_config,
)


def print_scalar(value: Any) -> None:
    if isinstance(value, bool):
        print("true" if value else "false")
    elif value is None:
        print("")
    elif isinstance(value, (int, float)):
        print(value)
    elif isinstance(value, str):
        print(value)
    else:
        print(json.dumps(value, ensure_ascii=False))


def parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return int(text)


def cmd_list_model_configs(_: argparse.Namespace) -> int:
    for name in list_model_configs():
        print(name)
    return 0


def cmd_resolve(args: argparse.Namespace) -> int:
    preset = resolve_preset(args.config)
    if args.field:
        print_scalar(lookup_field(preset, args.field))
        return 0
    print(json.dumps(preset, ensure_ascii=False, indent=2))
    return 0


def cmd_write_runtime_config(args: argparse.Namespace) -> int:
    preset = resolve_preset(args.config)
    output_path = write_runtime_inference_config(
        preset,
        Path(args.output).resolve(),
        runtime_gpu_id=parse_optional_int(args.runtime_gpu_id),
    )
    print(output_path)
    return 0


def cmd_derive_model_dir(args: argparse.Namespace) -> int:
    preset = resolve_preset(args.config)
    print(derive_model_dir_name(preset))
    return 0


def cmd_scenario_output_dir(args: argparse.Namespace) -> int:
    path = scenario_output_dir(
        scenario_file=args.scenario_file,
        scenario_root=args.scenario_root,
        result_root=args.result_root,
        model_dir_name=args.model_dir_name,
    )
    print(path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Utilities for close-loop evaluation presets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-model-configs", help="List available close-loop config presets.")
    list_parser.set_defaults(func=cmd_list_model_configs)

    resolve_parser = subparsers.add_parser("resolve", help="Resolve one close-loop preset.")
    resolve_parser.add_argument("--config", required=True, help="Preset name or preset JSON path.")
    resolve_parser.add_argument("--field", help="Optional dotted field path.")
    resolve_parser.set_defaults(func=cmd_resolve)

    runtime_parser = subparsers.add_parser(
        "write-runtime-config",
        help="Write a runtime inference config with an overridden GPU_ID.",
    )
    runtime_parser.add_argument("--config", required=True, help="Preset name or preset JSON path.")
    runtime_parser.add_argument("--output", required=True, help="Output JSON path.")
    runtime_parser.add_argument(
        "--runtime-gpu-id",
        help="Optional GPU_ID override to write. Empty keeps the config default.",
    )
    runtime_parser.set_defaults(func=cmd_write_runtime_config)

    model_dir_parser = subparsers.add_parser("derive-model-dir", help="Resolve the output model directory name.")
    model_dir_parser.add_argument("--config", required=True, help="Preset name or preset JSON path.")
    model_dir_parser.set_defaults(func=cmd_derive_model_dir)

    output_dir_parser = subparsers.add_parser(
        "scenario-output-dir",
        help="Compute the output directory for one scenario file.",
    )
    output_dir_parser.add_argument("--scenario-file", required=True)
    output_dir_parser.add_argument("--scenario-root", required=True)
    output_dir_parser.add_argument("--result-root", required=True)
    output_dir_parser.add_argument("--model-dir-name", required=True)
    output_dir_parser.set_defaults(func=cmd_scenario_output_dir)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
