#!/usr/bin/env python

"""Utilities for converting Scenario_Platform entity.json into SUMO-ready scenes."""

import json
import math
import os
import random
import site
import sys
import glob
from typing import Dict, List, Optional, Sequence, Tuple


def ensure_sumo_tools() -> None:
    """Expose SUMO python tools without requiring a hardcoded local setup."""
    if any(path.endswith(os.path.join("sumo", "tools")) for path in sys.path):
        return

    candidate_paths = []
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        candidate_paths.append(os.path.join(sumo_home, "tools"))

    for root in site.getsitepackages() + [site.getusersitepackages()]:
        candidate_paths.append(os.path.join(root, "sumo", "tools"))

    home_dir = os.path.expanduser("~")
    candidate_paths.extend(
        glob.glob(os.path.join(home_dir, "anaconda3", "envs", "*", "lib", "python*", "site-packages", "sumo", "tools"))
    )
    candidate_paths.extend(
        glob.glob(os.path.join(home_dir, "miniconda3", "envs", "*", "lib", "python*", "site-packages", "sumo", "tools"))
    )

    for path in candidate_paths:
        if path and os.path.isdir(path):
            site_packages = os.path.dirname(os.path.dirname(path))
            os.environ.setdefault("SUMO_HOME", os.path.dirname(path))
            if site_packages not in sys.path:
                sys.path.append(site_packages)
            if path not in sys.path:
                sys.path.append(path)
            return

    raise RuntimeError("SUMO python tools were not found. Set SUMO_HOME or install sumo tools.")


ensure_sumo_tools()


def ensure_proj_data() -> None:
    """Expose PROJ data for sumolib lon/lat conversion in conda/pip installs."""
    candidate_dirs = []

    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        candidate_dirs.append(os.path.join(sumo_home, "data", "proj"))

    for root in site.getsitepackages() + [site.getusersitepackages()]:
        candidate_dirs.append(os.path.join(root, "pyproj", "proj_dir", "share", "proj"))
        candidate_dirs.append(os.path.join(root, "sumo", "data", "proj"))

    home_dir = os.path.expanduser("~")
    candidate_dirs.extend(
        glob.glob(
            os.path.join(
                home_dir,
                "anaconda3",
                "envs",
                "*",
                "lib",
                "python*",
                "site-packages",
                "pyproj",
                "proj_dir",
                "share",
                "proj",
            )
        )
    )
    candidate_dirs.extend(
        glob.glob(
            os.path.join(
                home_dir,
                "anaconda3",
                "envs",
                "*",
                "lib",
                "python*",
                "site-packages",
                "sumo",
                "data",
                "proj",
            )
        )
    )
    candidate_dirs.extend(
        glob.glob(
            os.path.join(
                home_dir,
                "miniconda3",
                "envs",
                "*",
                "lib",
                "python*",
                "site-packages",
                "pyproj",
                "proj_dir",
                "share",
                "proj",
            )
        )
    )
    candidate_dirs.extend(
        glob.glob(
            os.path.join(
                home_dir,
                "miniconda3",
                "envs",
                "*",
                "lib",
                "python*",
                "site-packages",
                "sumo",
                "data",
                "proj",
            )
        )
    )

    for path in candidate_dirs:
        if path and os.path.isfile(os.path.join(path, "proj.db")):
            os.environ.setdefault("PROJ_LIB", path)
            os.environ.setdefault("PROJ_DATA", path)
            return


ensure_proj_data()

import sumolib  # pylint: disable=wrong-import-position,import-error


SCHEMA_VERSION = 1
DEFAULT_BACKGROUND_BLUEPRINTS = [
    "vehicle.dodge.charger_2020",
    "vehicle.volkswagen.t2_2021",
    "vehicle.ford.ambulance",
]
DEFAULT_BACKGROUND_BLUEPRINT_WEIGHTS = [0.6, 0.2, 0.2]
DEFAULT_CLEAR_WEATHER = {
    "cloudiness": 0.0,
    "precipitation": 0.0,
    "precipitation_deposits": 0.0,
    "wind_intensity": 0.0,
    "sun_azimuth_angle": 270.0,
    "sun_altitude_angle": 13.0,
    "fog_density": 0.0,
    "fog_distance": 0.0,
    "wetness": 0.0,
    "fog_falloff": 0.0,
    "scattering_intensity": 0.0,
    "mie_scattering_scale": 0.0,
    "rayleigh_scattering_scale": 0.0331,
}


def load_entity_participants(entity_json_path: str) -> Sequence[dict]:
    with open(entity_json_path, encoding="utf-8") as file_obj:
        entity_obj = json.load(file_obj)
    trajectory_info = json.loads(entity_obj["trajectoryInfo"])
    return trajectory_info["participantTrajectories"]


def load_vtypes(vtypes_json_path: str) -> Dict[str, dict]:
    with open(vtypes_json_path, encoding="utf-8") as file_obj:
        return json.load(file_obj)["carla_blueprints"]


def load_scene(scene_path: str) -> dict:
    with open(scene_path, encoding="utf-8") as file_obj:
        scene = json.load(file_obj)
    if scene.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            "Unsupported scene schema_version %s in %s" % (scene.get("schema_version"), scene_path)
        )
    return scene


def save_scene(scene: dict, output_path: str) -> None:
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(scene, file_obj, ensure_ascii=False, indent=2)


def split_blueprint_spec(spec: str) -> List[str]:
    return [item.strip() for item in spec.split(",") if item.strip()]


def split_weight_spec(spec: Optional[str], count: int) -> List[float]:
    if not spec:
        return []
    weights = [float(item.strip()) for item in spec.split(",") if item.strip()]
    if len(weights) != count:
        raise RuntimeError("Weight count does not match blueprint count")
    return weights


def choose_actor_blueprint(
    participant: dict,
    hero_blueprint: str,
    background_blueprints: Sequence[str],
    background_weights: Sequence[float],
    rng_seed: int,
) -> str:
    if participant["type"] == "main":
        return hero_blueprint

    rng = random.Random(f"{rng_seed}:{participant['id']}")
    if background_weights and len(background_weights) == len(background_blueprints):
        return rng.choices(list(background_blueprints), weights=list(background_weights), k=1)[0]
    return rng.choice(list(background_blueprints))


def estimate_duration_seconds(points: Sequence[dict], fallback_speed_kmh: float) -> float:
    fallback_speed_mps = max(fallback_speed_kmh / 3.6, 0.1)
    duration = 0.0
    for start_point, end_point in zip(points, points[1:]):
        dx = end_point["sumo_x"] - start_point["sumo_x"]
        dy = end_point["sumo_y"] - start_point["sumo_y"]
        distance = math.hypot(dx, dy)
        start_speed = max(start_point["speed_kmh"] / 3.6, 0.0)
        end_speed = max(end_point["speed_kmh"] / 3.6, 0.0)
        avg_speed = (start_speed + end_speed) / 2.0
        if avg_speed < 0.1:
            avg_speed = fallback_speed_mps
        duration += distance / avg_speed
    return duration


def points_from_trajectory(net, net_offset: Tuple[float, float], trajectory: Sequence[dict]) -> List[dict]:
    points = []
    offset_x, offset_y = net_offset
    for raw_point in trajectory:
        longitude = float(raw_point["longitude"])
        latitude = float(raw_point["latitude"])
        sumo_x, sumo_y = net.convertLonLat2XY(longitude, latitude)
        carla_x = sumo_x - offset_x
        carla_y = -(sumo_y - offset_y)
        points.append(
            {
                "frame_id": int(raw_point.get("frameId", len(points))),
                "point_type": raw_point.get("type", "pathway"),
                "pass": bool(raw_point.get("pass", True)),
                "speed_kmh": float(raw_point.get("speed", 0.0)),
                "longitude": longitude,
                "latitude": latitude,
                "sumo_x": sumo_x,
                "sumo_y": sumo_y,
                "carla_x": carla_x,
                "carla_y": carla_y,
            }
        )
    return points


def _find_neighbor_lanes(net, x: float, y: float, vclass: str, search_radius: float, radius_step: float, max_radius: float):
    radius = search_radius
    candidates = []
    seen = set()
    while radius <= max_radius and not candidates:
        for lane, distance in net.getNeighboringLanes(x, y, radius):
            edge = lane.getEdge()
            if edge.isSpecial() or not edge.allows(vclass):
                continue
            lane_id = lane.getID()
            if lane_id in seen:
                continue
            seen.add(lane_id)
            candidates.append((lane, distance))
        radius += radius_step
    candidates.sort(key=lambda item: item[1])
    return candidates


def _path_between_edges(net, from_edge, to_edge, vclass: str, path_cache: Dict[Tuple[str, str, str], Optional[List[str]]]):
    cache_key = (from_edge.getID(), to_edge.getID(), vclass)
    if cache_key in path_cache:
        return path_cache[cache_key]

    if from_edge.getID() == to_edge.getID():
        path_cache[cache_key] = [from_edge.getID()]
        return path_cache[cache_key]

    if to_edge in from_edge.getOutgoing():
        path_cache[cache_key] = [from_edge.getID(), to_edge.getID()]
        return path_cache[cache_key]

    path = net.getShortestPath(from_edge, to_edge, vClass=vclass)
    if path is None or path[0] is None:
        path_cache[cache_key] = None
        return None

    path_cache[cache_key] = [edge.getID() for edge in path[0]]
    return path_cache[cache_key]


def match_points_to_route(
    net,
    points: Sequence[dict],
    vclass: str,
    search_radius: float,
    radius_step: float,
    max_radius: float,
) -> Tuple[List[str], dict, List[dict], List[str]]:
    if not points:
        raise RuntimeError("Cannot match route without points")

    route_edges: List[str] = []
    matched_points: List[dict] = []
    warnings: List[str] = []
    path_cache: Dict[Tuple[str, str, str], Optional[List[str]]] = {}
    previous_edge = None
    first_lane = None
    first_lane_pos = None

    for point_index, point in enumerate(points):
        candidates = _find_neighbor_lanes(
            net,
            point["sumo_x"],
            point["sumo_y"],
            vclass,
            search_radius=search_radius,
            radius_step=radius_step,
            max_radius=max_radius,
        )
        if not candidates:
            warnings.append("point %d could not be matched to any %s lane" % (point_index, vclass))
            continue

        best_lane = None
        best_path = None
        best_score = None
        for lane, distance in candidates[:8]:
            edge = lane.getEdge()
            if previous_edge is None:
                candidate_path = [edge.getID()]
                score = distance
            else:
                candidate_path = _path_between_edges(net, previous_edge, edge, vclass, path_cache)
                if candidate_path is None:
                    continue
                score = len(candidate_path) * 5.0 + distance

            if best_score is None or score < best_score:
                best_lane = lane
                best_path = candidate_path
                best_score = score

        if best_lane is None or best_path is None:
            warnings.append(
                "point %d matched nearby lanes but no connected route was found from previous edge" % point_index
            )
            continue

        lane_pos, lane_distance = best_lane.getClosestLanePosAndDist((point["sumo_x"], point["sumo_y"]))
        matched_points.append(
            {
                "point_index": point_index,
                "edge_id": best_lane.getEdge().getID(),
                "lane_id": best_lane.getID(),
                "lane_index": int(best_lane.getIndex()),
                "lane_pos": float(lane_pos),
                "distance_to_lane": float(lane_distance),
            }
        )

        if first_lane is None:
            first_lane = best_lane
            first_lane_pos = lane_pos

        if not route_edges:
            route_edges.extend(best_path)
        else:
            for edge_id in best_path[1:]:
                if route_edges[-1] != edge_id:
                    route_edges.append(edge_id)

        previous_edge = best_lane.getEdge()

    if not route_edges or first_lane is None or first_lane_pos is None:
        raise RuntimeError("No valid SUMO route could be matched from entity trajectory")

    depart = {
        "edge_id": route_edges[0],
        "lane_id": first_lane.getID(),
        "lane_index": int(first_lane.getIndex()),
        "lane_pos": max(0.0, float(first_lane_pos)),
    }
    return route_edges, depart, matched_points, warnings


def compute_initial_yaw(points: Sequence[dict]) -> float:
    if len(points) < 2:
        return 0.0
    dx = points[1]["carla_x"] - points[0]["carla_x"]
    dy = points[1]["carla_y"] - points[0]["carla_y"]
    if abs(dx) < 1e-3 and abs(dy) < 1e-3:
        return 0.0
    return math.degrees(math.atan2(dy, dx))


def build_scene_from_entity(
    entity_json_path: str,
    net_file: str,
    vtypes_json_path: str,
    hero_blueprint: str,
    background_blueprints: Sequence[str],
    background_weights: Sequence[float],
    random_seed: int,
    search_radius: float,
    radius_step: float,
    max_radius: float,
    fallback_speed_kmh: float,
) -> dict:
    participants = load_entity_participants(entity_json_path)
    vtypes = load_vtypes(vtypes_json_path)
    net = sumolib.net.readNet(net_file)
    net_offset = tuple(float(item) for item in net.getLocationOffset())

    actors = []
    warnings: List[str] = []
    for participant in participants:
        blueprint = choose_actor_blueprint(
            participant,
            hero_blueprint=hero_blueprint,
            background_blueprints=background_blueprints,
            background_weights=background_weights,
            rng_seed=random_seed,
        )
        if blueprint not in vtypes:
            raise RuntimeError("Blueprint %s is missing from %s" % (blueprint, vtypes_json_path))

        points = points_from_trajectory(net, net_offset, participant["trajectory"])
        route_edges, depart, matched_points, actor_warnings = match_points_to_route(
            net,
            points,
            vclass=vtypes[blueprint]["vClass"],
            search_radius=search_radius,
            radius_step=radius_step,
            max_radius=max_radius,
        )
        warnings.extend("actor %s: %s" % (participant["id"], warning) for warning in actor_warnings)

        actor_record = {
            "actor_id": str(participant["id"]),
            "name": participant.get("name", str(participant["id"])),
            "role": participant["type"],
            "sumo_type_id": blueprint,
            "vclass": vtypes[blueprint]["vClass"],
            "route_id": "entity_route_%s" % participant["id"],
            "route_edges": route_edges,
            "depart": {
                "time_seconds": 0.0,
                "edge_id": depart["edge_id"],
                "lane_index": depart["lane_index"],
                "lane_id": depart["lane_id"],
                "lane_pos": depart["lane_pos"],
                "speed_kmh": max(points[0]["speed_kmh"], 0.0),
            },
            "arrival": {
                "edge_id": route_edges[-1],
            },
            "carla_spawn": {
                "x": points[0]["carla_x"],
                "y": points[0]["carla_y"],
                "z": 14.0,
                "yaw": compute_initial_yaw(points),
            },
            "reference_points": points,
            "matched_points": matched_points,
            "estimated_duration_seconds": estimate_duration_seconds(points, fallback_speed_kmh=fallback_speed_kmh),
        }
        actors.append(actor_record)

    hero_actor = next((actor for actor in actors if actor["role"] == "main"), None)
    if hero_actor is None:
        raise RuntimeError("No main participant found in entity.json")

    background_actors = [actor for actor in actors if actor["role"] != "main"]
    scenario_duration = max(actor["estimated_duration_seconds"] for actor in actors) if actors else 0.0

    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "entity_json": os.path.abspath(entity_json_path),
            "net_file": os.path.abspath(net_file),
            "vtypes_json": os.path.abspath(vtypes_json_path),
        },
        "map": {
            "name": "tongji",
            "net_offset": [net_offset[0], net_offset[1]],
        },
        "hero": hero_actor,
        "background_vehicles": background_actors,
        "scenario_duration_seconds": scenario_duration,
        "weather": DEFAULT_CLEAR_WEATHER,
        "random_seed": random_seed,
        "conversion_warnings": warnings,
    }
