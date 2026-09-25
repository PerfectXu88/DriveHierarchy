#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Single-scene scoring for Tongji CARLA+SUMO runs."""

import json
import math
import os
import time


COLLISION_PENALTIES = {
    'collision_pedestrian': 0.0,
    'collision_vehicle': 0.2,
    'collision_static': 0.3,
}
SCENARIO_TIMEOUT_PENALTY = 0.70
BLOCKED_TIMEOUT_PENALTY = 0.70
OFFROAD_TIMEOUT_PENALTY = 0.0
MANUAL_INTERRUPT_PENALTY = 0.0
SYSTEM_ERROR_PENALTY = 0.0

FAILURE_REASONS = {
    'collision_pedestrian',
    'collision_vehicle',
    'collision_static',
    'offroad_timeout',
    'scenario_timeout',
    'blocked_timeout',
    'manual_interrupt',
    'system_error',
    'unknown',
}

ROUTE_PROGRESS_MAX_DEVIATION_M = 6.0
ROUTE_PROGRESS_MAX_LOOKAHEAD_M = 20.0
DISTANCE_EFFICIENCY_GRACE_FACTOR = 1.10
TIME_EFFICIENCY_GRACE_FACTOR = 1.50
DISTANCE_EFFICIENCY_WEIGHT = 0.7
TIME_EFFICIENCY_WEIGHT = 0.3


def _clamp(value, lower, upper):
    return max(lower, min(upper, value))


def _to_xy(point):
    if point is None:
        return None

    if 'carla_x' in point and 'carla_y' in point:
        return float(point['carla_x']), float(point['carla_y'])
    if 'x' in point and 'y' in point:
        return float(point['x']), float(point['y'])
    return None


def _distance_xy(point_a, point_b):
    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    return math.hypot(dx, dy)


def _interp_point(point_a, point_b, ratio):
    return (
        point_a[0] + (point_b[0] - point_a[0]) * ratio,
        point_a[1] + (point_b[1] - point_a[1]) * ratio,
    )


class SceneScoreTracker(object):
    """Track progress / infractions and emit a single-scene score JSON."""

    def __init__(self, scene, scene_id, output_dir,
                 destination_point=None, via_points=None,
                 destination_radius=8.0):
        self.scene = scene
        self.scene_id = scene_id
        self.output_dir = output_dir
        self.destination_point = destination_point
        self.via_points = via_points or []
        self.destination_radius = float(destination_radius)
        self.output_path = os.path.join(output_dir, 'scene_score.json')

        self._system_start_time = time.time()
        self._sim_duration = 0.0
        self._route_source = 'none'
        self._route_points = self._build_route_points()
        self._route_length = self._compute_route_length(self._route_points)
        self._reference_duration = self._build_reference_duration()
        self._max_completed_distance = 0.0
        self._events = []
        self._event_penalty_product = 1.0
        self._travel_distance = 0.0
        self._offroad_time = 0.0
        self._last_location = None
        self._last_xy = None
        self._last_update_time = None
        self._last_is_offroad = False
        self._last_route_distance = None
        self._destination_reached = False
        self._finalized = False
        self.result = None

        os.makedirs(self.output_dir, exist_ok=True)

    def get_route_points(self):
        return list(self._route_points)

    def _build_reference_duration(self):
        hero_config = self.scene.get('hero', {})
        reference_duration = float(hero_config.get('estimated_duration_seconds') or 0.0)
        if reference_duration > 1e-6:
            return reference_duration
        if self._route_length > 1e-6:
            return self._route_length / 5.56
        return 0.0

    def _build_route_points(self):
        hero_config = self.scene.get('hero', {})

        route_points = self._build_route_points_from_route_edges(hero_config)
        if route_points:
            self._route_source = 'route_edges'
            return route_points

        reference_points = hero_config.get('reference_points') or []
        route_points = []
        for point in reference_points:
            xy = _to_xy(point)
            if xy is not None:
                route_points.append(xy)

        if route_points:
            self._route_source = 'reference_points'
            return self._dedupe_points(route_points)

        fallback_points = []
        for point in self.via_points:
            xy = _to_xy(point)
            if xy is not None:
                fallback_points.append(xy)
        destination_xy = _to_xy(self.destination_point)
        if destination_xy is not None:
            fallback_points.append(destination_xy)
        if fallback_points:
            self._route_source = 'via_points_destination'
        return self._dedupe_points(fallback_points)

    def _build_route_points_from_route_edges(self, hero_config):
        source = self.scene.get('source', {})
        net_file = source.get('net_file')
        route_edges = hero_config.get('route_edges') or []
        net_offset = self.scene.get('map', {}).get('net_offset') or [0.0, 0.0]

        if not net_file or not os.path.isfile(net_file) or not route_edges:
            return None

        try:
            import sumolib
        except ImportError:
            return None

        try:
            net = sumolib.net.readNet(net_file)
        except Exception:
            return None

        polyline = []
        for edge_id in route_edges:
            try:
                edge = net.getEdge(edge_id)
            except Exception:
                return None

            try:
                shape = edge.getShape(includeJunctions=True)
            except TypeError:
                shape = edge.getShape()
            except Exception:
                return None

            for shape_point in shape:
                polyline.append(self._sumo_to_carla_xy(shape_point, net_offset))

        polyline = self._dedupe_points(polyline)
        if len(polyline) < 2:
            return None

        reference_points = hero_config.get('reference_points') or []
        start_xy = _to_xy(reference_points[0]) if reference_points else None
        end_xy = _to_xy(reference_points[-1]) if reference_points else None
        if start_xy is None or end_xy is None:
            return polyline

        start_progress, _, _ = self._project_onto_polyline(start_xy, polyline)
        end_progress, _, _ = self._project_onto_polyline(end_xy, polyline)
        if end_progress <= start_progress + 1e-6:
            return polyline

        clipped = self._slice_polyline(polyline, start_progress, end_progress)
        clipped = self._dedupe_points(clipped)
        return clipped if len(clipped) >= 2 else polyline

    @staticmethod
    def _sumo_to_carla_xy(shape_point, net_offset):
        return (
            float(shape_point[0]) - float(net_offset[0]),
            float(net_offset[1]) - float(shape_point[1]),
        )

    @staticmethod
    def _dedupe_points(points):
        deduped = []
        for point in points:
            if deduped and _distance_xy(deduped[-1], point) <= 1e-6:
                continue
            deduped.append(point)
        return deduped

    @staticmethod
    def _compute_route_length(route_points):
        if len(route_points) < 2:
            return 0.0

        total = 0.0
        for prev_point, curr_point in zip(route_points[:-1], route_points[1:]):
            total += _distance_xy(prev_point, curr_point)
        return total

    @staticmethod
    def _project_onto_segment(point, seg_start, seg_end):
        seg_dx = seg_end[0] - seg_start[0]
        seg_dy = seg_end[1] - seg_start[1]
        seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy
        if seg_len_sq <= 1e-9:
            return seg_start, 0.0

        rel_x = point[0] - seg_start[0]
        rel_y = point[1] - seg_start[1]
        ratio = (rel_x * seg_dx + rel_y * seg_dy) / seg_len_sq
        ratio = _clamp(ratio, 0.0, 1.0)
        proj_point = (
            seg_start[0] + ratio * seg_dx,
            seg_start[1] + ratio * seg_dy,
        )
        return proj_point, ratio

    @classmethod
    def _project_onto_polyline(cls, point, route_points):
        if len(route_points) < 2:
            return 0.0, None, point

        best_distance = None
        best_progress = 0.0
        best_point = route_points[0]
        accum_length = 0.0

        for seg_start, seg_end in zip(route_points[:-1], route_points[1:]):
            seg_length = _distance_xy(seg_start, seg_end)
            if seg_length <= 1e-9:
                continue
            proj_point, ratio = cls._project_onto_segment(point, seg_start, seg_end)
            distance = _distance_xy(point, proj_point)
            progress = accum_length + ratio * seg_length
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_progress = progress
                best_point = proj_point
            accum_length += seg_length

        return best_progress, best_distance, best_point

    @classmethod
    def _slice_polyline(cls, route_points, start_progress, end_progress):
        if len(route_points) < 2:
            return list(route_points)

        total_length = cls._compute_route_length(route_points)
        start_progress = _clamp(start_progress, 0.0, total_length)
        end_progress = _clamp(end_progress, start_progress, total_length)

        sliced = []
        accum_length = 0.0
        for seg_start, seg_end in zip(route_points[:-1], route_points[1:]):
            seg_length = _distance_xy(seg_start, seg_end)
            if seg_length <= 1e-9:
                continue

            seg_progress_start = accum_length
            seg_progress_end = accum_length + seg_length
            overlap_start = max(start_progress, seg_progress_start)
            overlap_end = min(end_progress, seg_progress_end)
            if overlap_end < overlap_start - 1e-9:
                accum_length += seg_length
                continue

            start_ratio = (overlap_start - seg_progress_start) / seg_length
            end_ratio = (overlap_end - seg_progress_start) / seg_length
            clipped_start = _interp_point(seg_start, seg_end, start_ratio)
            clipped_end = _interp_point(seg_start, seg_end, end_ratio)

            if not sliced or _distance_xy(sliced[-1], clipped_start) > 1e-6:
                sliced.append(clipped_start)
            if _distance_xy(sliced[-1], clipped_end) > 1e-6:
                sliced.append(clipped_end)

            accum_length += seg_length

        return sliced

    @staticmethod
    def _location_to_dict(location):
        return {
            'x': round(float(location.x), 3),
            'y': round(float(location.y), 3),
            'z': round(float(location.z), 3),
        }

    def update(self, hero_actor, sim_time, is_offroad=False):
        if hero_actor is None or not hero_actor.is_alive:
            return

        sim_time = float(sim_time)
        self._sim_duration = max(self._sim_duration, sim_time)

        hero_location = hero_actor.get_location()
        hero_xy = (float(hero_location.x), float(hero_location.y))

        if self._last_update_time is not None:
            delta_time = max(0.0, sim_time - self._last_update_time)
            if self._last_is_offroad:
                self._offroad_time += delta_time
            if self._last_xy is not None:
                self._travel_distance += _distance_xy(self._last_xy, hero_xy)

        self._last_update_time = sim_time
        self._last_is_offroad = bool(is_offroad)
        self._last_xy = hero_xy
        self._last_location = self._location_to_dict(hero_location)

        if not is_offroad and self._route_length > 1e-6:
            progress, route_distance, _ = self._project_onto_polyline(
                hero_xy, self._route_points)
            self._last_route_distance = route_distance
            if (route_distance is not None
                    and route_distance <= ROUTE_PROGRESS_MAX_DEVIATION_M):
                progress = min(
                    progress,
                    self._max_completed_distance + ROUTE_PROGRESS_MAX_LOOKAHEAD_M,
                )
                self._max_completed_distance = max(self._max_completed_distance, progress)

        destination_xy = _to_xy(self.destination_point)
        if destination_xy is not None:
            if _distance_xy(hero_xy, destination_xy) <= self.destination_radius:
                self._destination_reached = True
                self._max_completed_distance = max(
                    self._max_completed_distance,
                    self._route_length,
                )

    def _append_event(self, event_type, sim_time, details=None, penalty_multiplier=None):
        event = {
            'type': event_type,
            'time': round(float(sim_time), 3),
        }
        if details:
            event.update(details)
        if penalty_multiplier is not None:
            event['penalty_multiplier'] = penalty_multiplier
            self._event_penalty_product *= penalty_multiplier
        self._events.append(event)

    @staticmethod
    def classify_collision(other_actor_type):
        actor_type = other_actor_type or 'unknown'
        if actor_type.startswith('walker.'):
            return 'collision_pedestrian'
        if actor_type.startswith('vehicle.'):
            return 'collision_vehicle'
        return 'collision_static'

    def record_collision(self, sim_time, other_actor_type, intensity):
        event_type = self.classify_collision(other_actor_type)
        self._append_event(
            event_type=event_type,
            sim_time=sim_time,
            details={
                'other_actor': other_actor_type or 'unknown',
                'intensity': round(float(intensity), 3),
            },
            penalty_multiplier=COLLISION_PENALTIES[event_type],
        )
        return event_type

    def record_timeout(self, sim_time, max_sim_time):
        self._append_event(
            event_type='scenario_timeout',
            sim_time=sim_time,
            details={
                'max_sim_time': None if max_sim_time is None else round(float(max_sim_time), 3),
            },
            penalty_multiplier=SCENARIO_TIMEOUT_PENALTY,
        )
        return 'scenario_timeout'

    def record_blocked(self, sim_time, blocked_steps, speed_kmh, speed_threshold_kmh):
        self._append_event(
            event_type='blocked_timeout',
            sim_time=sim_time,
            details={
                'blocked_steps': int(blocked_steps),
                'speed_kmh': round(float(speed_kmh), 3),
                'speed_threshold_kmh': round(float(speed_threshold_kmh), 3),
            },
            penalty_multiplier=BLOCKED_TIMEOUT_PENALTY,
        )
        return 'blocked_timeout'

    def record_offroad(self, sim_time, offroad_duration):
        self._offroad_time = max(self._offroad_time, float(offroad_duration))
        self._append_event(
            event_type='route_deviation',
            sim_time=sim_time,
            details={
                'offroad_duration': round(float(offroad_duration), 3),
            },
            penalty_multiplier=OFFROAD_TIMEOUT_PENALTY,
        )
        return 'offroad_timeout'

    def record_destination(self, sim_time):
        self._destination_reached = True
        self._max_completed_distance = max(self._max_completed_distance, self._route_length)
        self._append_event(
            event_type='destination_reached',
            sim_time=sim_time,
            details={
                'destination_radius': round(self.destination_radius, 3),
            },
        )
        return 'destination_reached'

    def record_manual_interrupt(self, sim_time):
        self._append_event(
            'manual_interrupt',
            sim_time,
            penalty_multiplier=MANUAL_INTERRUPT_PENALTY,
        )
        return 'manual_interrupt'

    def record_system_error(self, sim_time, error_message):
        self._append_event(
            event_type='system_error',
            sim_time=sim_time,
            details={'error': str(error_message)},
            penalty_multiplier=SYSTEM_ERROR_PENALTY,
        )
        return 'system_error'

    def _compute_route_score(self):
        if self._destination_reached:
            return 100.0
        if self._route_length <= 1e-6:
            return 0.0
        score = 100.0 * self._max_completed_distance / self._route_length
        return _clamp(score, 0.0, 100.0)

    def _compute_safety_score(self):
        if self._sim_duration > 1e-6:
            offroad_ratio = _clamp(self._offroad_time / self._sim_duration, 0.0, 1.0)
            lane_keep_penalty = 1.0 - offroad_ratio
        else:
            offroad_ratio = 0.0
            lane_keep_penalty = 1.0

        score = self._event_penalty_product * lane_keep_penalty
        return _clamp(score, 0.0, 1.0), offroad_ratio, lane_keep_penalty

    def _compute_efficiency_score(self):
        if self._route_length <= 1e-6:
            return 1.0, 1.0, 1.0

        distance_reference = self._route_length * DISTANCE_EFFICIENCY_GRACE_FACTOR
        if self._travel_distance <= distance_reference + 1e-6:
            distance_efficiency = 1.0
        else:
            distance_efficiency = distance_reference / self._travel_distance

        if self._reference_duration <= 1e-6 or self._sim_duration <= 1e-6:
            time_efficiency = 1.0
        else:
            duration_reference = self._reference_duration * TIME_EFFICIENCY_GRACE_FACTOR
            if self._sim_duration <= duration_reference + 1e-6:
                time_efficiency = 1.0
            else:
                time_efficiency = duration_reference / self._sim_duration

        distance_efficiency = _clamp(distance_efficiency, 0.0, 1.0)
        time_efficiency = _clamp(time_efficiency, 0.0, 1.0)
        score = (
            DISTANCE_EFFICIENCY_WEIGHT * distance_efficiency
            + TIME_EFFICIENCY_WEIGHT * time_efficiency
        )
        return _clamp(score, 0.0, 1.0), distance_efficiency, time_efficiency

    def finalize(self, termination_reason):
        if self._finalized:
            return self.output_path

        self._finalized = True
        score_route = round(self._compute_route_score(), 6)
        score_safety, offroad_ratio, lane_keep_penalty = self._compute_safety_score()
        score_efficiency, distance_efficiency, time_efficiency = self._compute_efficiency_score()
        score_safety = round(score_safety, 6)
        score_efficiency = round(score_efficiency, 6)
        score_penalty = round(max(score_safety * score_efficiency, 0.0), 6)
        score_composed = round(max(score_route * score_penalty, 0.0), 6)
        system_duration = round(time.time() - self._system_start_time, 3)

        if score_route >= 100.0 and termination_reason not in FAILURE_REASONS:
            if score_safety >= 0.999999 and score_efficiency >= 0.999999:
                status = 'Perfect'
            else:
                status = 'Completed'
        else:
            status = 'Failed'

        result = {
            'scene_id': self.scene_id,
            'status': status,
            'termination_reason': termination_reason,
            'score_route': score_route,
            'score_safety': score_safety,
            'score_efficiency': score_efficiency,
            'score_penalty': score_penalty,
            'score_composed': score_composed,
            'events': self._events,
            'meta': {
                'route_source': self._route_source,
                'route_length_m': round(self._route_length, 3),
                'completed_distance_m': round(self._max_completed_distance, 3),
                'travel_distance_m': round(self._travel_distance, 3),
                'offroad_time_s': round(self._offroad_time, 3),
                'offroad_ratio': round(offroad_ratio, 6),
                'lane_keep_penalty': round(lane_keep_penalty, 6),
                'duration_game': round(self._sim_duration, 3),
                'duration_system': system_duration,
                'reference_duration_s': round(self._reference_duration, 3),
                'distance_efficiency': round(distance_efficiency, 6),
                'time_efficiency': round(time_efficiency, 6),
                'destination_radius': round(self.destination_radius, 3),
                'last_route_distance_m': None if self._last_route_distance is None else round(self._last_route_distance, 3),
                'last_location': self._last_location,
            },
        }

        with open(self.output_path, 'w', encoding='utf-8') as output_file:
            json.dump(result, output_file, ensure_ascii=False, indent=2)

        self.result = result
        return self.output_path
