#!/usr/bin/env python

"""Convert Scenario_Platform entity.json into a SUMO/CARLA benchmark scene file."""

import argparse
import os

from entity_scenario_lib import (
    DEFAULT_BACKGROUND_BLUEPRINTS,
    DEFAULT_BACKGROUND_BLUEPRINT_WEIGHTS,
    build_scene_from_entity,
    save_scene,
    split_blueprint_spec,
    split_weight_spec,
)


def parse_args():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entity_json", help="Path to Scenario_Platform entity.json")
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for converted scene JSON",
    )
    parser.add_argument(
        "--net-file",
        default=os.path.join(script_dir, "tongji", "tongji.net.xml"),
        help="SUMO net.xml used for map matching",
    )
    parser.add_argument(
        "--vtypes-json",
        default=os.path.join(script_dir, "data", "vtypes.json"),
        help="CARLA blueprint to SUMO vClass mapping JSON",
    )
    parser.add_argument(
        "--hero-blueprint",
        default="vehicle.lincoln.mkz_2020",
        help="CARLA blueprint used for the main actor",
    )
    parser.add_argument(
        "--background-blueprints",
        default=",".join(DEFAULT_BACKGROUND_BLUEPRINTS),
        help="Comma-separated CARLA blueprints used for slave actors",
    )
    parser.add_argument(
        "--background-blueprint-weights",
        default=",".join(str(item) for item in DEFAULT_BACKGROUND_BLUEPRINT_WEIGHTS),
        help="Comma-separated weights aligned with --background-blueprints",
    )
    parser.add_argument(
        "--random-seed",
        default=7,
        type=int,
        help="Deterministic seed used to assign background blueprints",
    )
    parser.add_argument(
        "--search-radius",
        default=6.0,
        type=float,
        help="Initial lane match radius in meters",
    )
    parser.add_argument(
        "--radius-step",
        default=4.0,
        type=float,
        help="Increment applied while expanding the lane match radius",
    )
    parser.add_argument(
        "--max-radius",
        default=30.0,
        type=float,
        help="Maximum lane match radius in meters",
    )
    parser.add_argument(
        "--fallback-speed-kmh",
        default=20.0,
        type=float,
        help="Fallback speed used when trajectory points provide zero speed",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    background_blueprints = split_blueprint_spec(args.background_blueprints)
    if not background_blueprints:
        raise RuntimeError("At least one background blueprint is required")
    background_weights = split_weight_spec(args.background_blueprint_weights, len(background_blueprints))

    scene = build_scene_from_entity(
        entity_json_path=args.entity_json,
        net_file=args.net_file,
        vtypes_json_path=args.vtypes_json,
        hero_blueprint=args.hero_blueprint,
        background_blueprints=background_blueprints,
        background_weights=background_weights,
        random_seed=args.random_seed,
        search_radius=args.search_radius,
        radius_step=args.radius_step,
        max_radius=args.max_radius,
        fallback_speed_kmh=args.fallback_speed_kmh,
    )
    save_scene(scene, args.output)
    print("saved scene to %s" % os.path.abspath(args.output))
    print("hero route edges: %d" % len(scene["hero"]["route_edges"]))
    print("background vehicles: %d" % len(scene["background_vehicles"]))
    print("conversion warnings: %d" % len(scene["conversion_warnings"]))


if __name__ == "__main__":
    main()
