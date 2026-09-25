# -*- coding: utf-8 -*-
"""Route guidance and waypoint tracking helpers for online VLM control."""

import math
import os

import carla

from agents.navigation.controller import VehiclePIDController

try:
    import sumolib
except ImportError:  # pragma: no cover - optional at import time
    sumolib = None


COMMAND_TEXT = {
    1: 'turn left at the intersection',
    2: 'turn right at the intersection',
    3: 'drive straight at the intersection',
    4: 'follow the road',
    5: 'do a lane change to the left',
    6: 'do a lane change to the right',
}

ROUTE_PROGRESS_MAX_DEVIATION_M = 8.0
ROUTE_SAMPLE_SPACING_M = 4.0
ROUTE_SAMPLE_LOOKAHEAD_M = 64.0
COMMAND_NEAR_LOOKAHEAD_M = 10.0
COMMAND_FAR_LOOKAHEAD_M = 28.0
COMMAND_FARTHER_LOOKAHEAD_M = 42.0


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


def _world_to_ego_local(hero_transform, world_xy):
    delta_x = world_xy[0] - hero_transform.location.x
    delta_y = world_xy[1] - hero_transform.location.y
    yaw = math.radians(hero_transform.rotation.yaw)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    forward = delta_x * cos_yaw + delta_y * sin_yaw
    right = -delta_x * sin_yaw + delta_y * cos_yaw
    return forward, right


def _ego_local_to_world(hero_transform, forward, right, z=None):
    yaw = math.radians(hero_transform.rotation.yaw)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    world_x = (
        hero_transform.location.x
        + forward * cos_yaw
        - right * sin_yaw
    )
    world_y = (
        hero_transform.location.y
        + forward * sin_yaw
        + right * cos_yaw
    )
    world_z = hero_transform.location.z if z is None else z
    return world_x, world_y, world_z


class RouteGuidancePlanner(object):
    """Build a stable route polyline and expose online guidance snapshots."""

    def __init__(self, scene, destination_point=None, via_points=None):
        self.scene = scene
        self.destination_point = destination_point
        self.via_points = via_points or []
        self._route_points = self._build_route_points()
        self._route_length = self._compute_route_length(self._route_points)
        self._max_progress = 0.0

    def snapshot(self, hero_actor):
        hero_transform = hero_actor.get_transform()
        hero_xy = (hero_transform.location.x, hero_transform.location.y)

        progress, route_distance, _ = self._project_onto_polyline(
            hero_xy, self._route_points)
        if route_distance is None or route_distance > ROUTE_PROGRESS_MAX_DEVIATION_M:
            progress = self._max_progress
        else:
            progress = max(progress, self._max_progress)
            self._max_progress = progress

        route_world_xy = self._sample_polyline(
            start_progress=progress,
            step=ROUTE_SAMPLE_SPACING_M,
            max_length=ROUTE_SAMPLE_LOOKAHEAD_M,
        )
        route_local = [
            _world_to_ego_local(hero_transform, point_xy)
            for point_xy in route_world_xy
        ]

        near_progress = min(
            self._route_length, progress + COMMAND_NEAR_LOOKAHEAD_M)
        far_progress = min(
            self._route_length, progress + COMMAND_FAR_LOOKAHEAD_M)
        farther_progress = min(
            self._route_length, progress + COMMAND_FARTHER_LOOKAHEAD_M)

        near_xy = self._point_at_progress(near_progress)
        far_xy = self._point_at_progress(far_progress)
        farther_xy = self._point_at_progress(farther_progress)

        near_local = _world_to_ego_local(hero_transform, near_xy)
        far_local = _world_to_ego_local(hero_transform, far_xy)
        farther_local = _world_to_ego_local(hero_transform, farther_xy)

        world_map = hero_actor.get_world().get_map()
        current_wp = self._map_waypoint(world_map, hero_transform.location)
        near_wp = self._map_waypoint_xy(world_map, near_xy)
        far_wp = self._map_waypoint_xy(world_map, far_xy)
        farther_wp = self._map_waypoint_xy(world_map, farther_xy)

        near_command = self._infer_command(
            primary_local=near_local,
            reference_local=far_local,
            current_wp=current_wp,
            primary_wp=near_wp,
            reference_wp=far_wp,
        )
        far_command = self._infer_command(
            primary_local=far_local,
            reference_local=farther_local,
            current_wp=current_wp,
            primary_wp=far_wp,
            reference_wp=farther_wp,
        )

        next_command = near_command if near_command != 4 else far_command
        if next_command == near_command and near_command != 4:
            next_command_distance = near_progress - progress
        elif far_command != 4:
            next_command_distance = far_progress - progress
        else:
            next_command_distance = 0.0

        route_locations = [
            carla.Location(x=point_xy[0], y=point_xy[1], z=hero_transform.location.z + 0.2)
            for point_xy in route_world_xy
        ]

        return {
            'route_progress_m': progress,
            'route_distance_m': route_distance,
            'route_length_m': self._route_length,
            'route_points_world': route_world_xy,
            'route_points_local': route_local,
            'route_locations': route_locations,
            'command_near': near_command,
            'command_far': far_command,
            'x_command_near': near_xy[0],
            'y_command_near': near_xy[1],
            'x_command_far': far_xy[0],
            'y_command_far': far_xy[1],
            'next_command': next_command,
            'next_command_distance_m': max(0.0, next_command_distance),
            'route_text': self._format_route_text(route_local, next_command, next_command_distance),
        }

    def _build_route_points(self):
        hero_config = self.scene.get('hero', {})

        route_points = self._build_route_points_from_route_edges(hero_config)
        if route_points:
            return route_points

        reference_points = hero_config.get('reference_points') or []
        route_points = []
        for point in reference_points:
            xy = _to_xy(point)
            if xy is not None:
                route_points.append(xy)
        if route_points:
            return self._dedupe_points(route_points)

        fallback_points = []
        for point in self.via_points:
            xy = _to_xy(point)
            if xy is not None:
                fallback_points.append(xy)
        destination_xy = _to_xy(self.destination_point)
        if destination_xy is not None:
            fallback_points.append(destination_xy)
        return self._dedupe_points(fallback_points)

    def _build_route_points_from_route_edges(self, hero_config):
        source = self.scene.get('source', {})
        net_file = source.get('net_file')
        route_edges = hero_config.get('route_edges') or []
        net_offset = self.scene.get('map', {}).get('net_offset') or [0.0, 0.0]

        if sumolib is None or not net_file or not os.path.isfile(net_file) or not route_edges:
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
                polyline.append((
                    float(shape_point[0]) - float(net_offset[0]),
                    float(net_offset[1]) - float(shape_point[1]),
                ))

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
        for start_point, end_point in zip(route_points[:-1], route_points[1:]):
            total += _distance_xy(start_point, end_point)
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

    def _point_at_progress(self, progress):
        if len(self._route_points) < 2:
            if self._route_points:
                return self._route_points[0]
            return (0.0, 0.0)

        progress = _clamp(progress, 0.0, self._route_length)
        accum_length = 0.0
        for seg_start, seg_end in zip(self._route_points[:-1], self._route_points[1:]):
            seg_length = _distance_xy(seg_start, seg_end)
            if seg_length <= 1e-9:
                continue
            if accum_length + seg_length >= progress:
                ratio = (progress - accum_length) / seg_length
                return _interp_point(seg_start, seg_end, ratio)
            accum_length += seg_length
        return self._route_points[-1]

    def _sample_polyline(self, start_progress, step, max_length):
        if not self._route_points:
            return []

        sampled = []
        distance = 0.0
        while distance <= max_length + 1e-6:
            progress = min(self._route_length, start_progress + distance)
            point_xy = self._point_at_progress(progress)
            if not sampled or _distance_xy(sampled[-1], point_xy) > 0.5:
                sampled.append(point_xy)
            if progress >= self._route_length - 1e-6:
                break
            distance += step
        return sampled

    @staticmethod
    def _map_waypoint(world_map, location):
        return world_map.get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

    @classmethod
    def _map_waypoint_xy(cls, world_map, point_xy):
        return cls._map_waypoint(
            world_map,
            carla.Location(x=point_xy[0], y=point_xy[1], z=0.5),
        )

    @staticmethod
    def _infer_command(primary_local, reference_local,
                       current_wp, primary_wp, reference_wp):
        primary_forward, primary_right = primary_local
        ref_forward, ref_right = reference_local

        primary_forward = max(primary_forward, 0.1)
        ref_forward = max(ref_forward, 0.1)

        primary_angle = math.degrees(math.atan2(primary_right, primary_forward))
        ref_angle = math.degrees(math.atan2(ref_right, ref_forward))

        in_junction = False
        for waypoint in (current_wp, primary_wp, reference_wp):
            if waypoint is not None and waypoint.is_junction:
                in_junction = True
                break

        if not in_junction and abs(ref_right) >= 2.2 and abs(ref_angle) <= 18.0:
            return 6 if ref_right > 0.0 else 5

        if in_junction or abs(ref_angle) >= 18.0 or abs(primary_angle) >= 18.0:
            turn_angle = ref_angle if abs(ref_angle) >= abs(primary_angle) else primary_angle
            if turn_angle >= 18.0:
                return 2
            if turn_angle <= -18.0:
                return 1
            return 3

        return 4

    @staticmethod
    def _format_route_text(route_local, next_command, next_command_distance):
        anchors = []
        for forward, right in route_local[:6]:
            if forward < -1.0:
                continue
            anchors.append('({:.1f},{:.1f})'.format(forward, right))
        anchor_text = ', '.join(anchors) if anchors else 'none'

        if next_command in COMMAND_TEXT and next_command != 4:
            maneuver_text = '{} in {:.1f} m'.format(
                COMMAND_TEXT[next_command], max(0.0, next_command_distance))
        else:
            maneuver_text = 'follow the road'

        return {
            'maneuver': maneuver_text,
            'anchors': anchor_text,
        }


class _TargetWaypoint(object):
    """Simple stand-in for CARLA map waypoint objects."""

    def __init__(self, location, yaw_deg):
        self.transform = carla.Transform(
            location,
            carla.Rotation(yaw=yaw_deg),
        )


class WaypointTrajectoryTracker(object):
    """Track VLM-predicted short-horizon waypoint deltas with deterministic PID."""

    def __init__(self, vehicle, control_period_seconds):
        pid_dt = max(0.05, float(control_period_seconds))
        self._pid = VehiclePIDController(
            vehicle,
            args_lateral={
                'K_P': 1.4,
                'K_I': 0.05,
                'K_D': 0.2,
                'dt': pid_dt,
            },
            args_longitudinal={
                'K_P': 1.0,
                'K_I': 0.05,
                'K_D': 0.0,
                'dt': pid_dt,
            },
            max_throttle=0.75,
            max_brake=0.5,
            max_steering=0.8,
        )

    def run_step(self, hero_actor, rel_waypoint_deltas):
        local_waypoints = self._accumulate_local_waypoints(rel_waypoint_deltas)
        world_waypoints = self._local_waypoints_to_world(hero_actor, local_waypoints)
        if not world_waypoints:
            return None, {
                'target_speed_kmh': None,
                'selected_index': None,
                'world_waypoints': [],
            }

        ego_speed_kmh = hero_actor.get_velocity().length() * 3.6
        lookahead_distance = _clamp(4.0 + ego_speed_kmh * 0.12, 4.0, 12.0)
        selected_index = self._select_target_index(
            hero_actor.get_location(), world_waypoints, lookahead_distance)

        target_speed_kmh = self._estimate_target_speed_kmh(local_waypoints)
        target_waypoint = self._make_target_waypoint(world_waypoints, selected_index)
        control = self._pid.run_step(target_speed_kmh, target_waypoint)

        if target_speed_kmh <= 0.5 and ego_speed_kmh <= 2.0:
            control.throttle = 0.0
            control.brake = max(control.brake, 0.45)

        debug = {
            'target_speed_kmh': round(float(target_speed_kmh), 3),
            'selected_index': int(selected_index),
            'selected_world_point': self._location_to_dict(world_waypoints[selected_index]),
            'world_waypoints': [
                self._location_to_dict(location)
                for location in world_waypoints
            ],
        }
        return control, debug

    @staticmethod
    def _accumulate_local_waypoints(rel_waypoint_deltas):
        local_waypoints = []
        forward_total = 0.0
        right_total = 0.0
        for delta in rel_waypoint_deltas:
            if not isinstance(delta, (list, tuple)) or len(delta) != 2:
                continue
            try:
                forward_delta = float(delta[0])
                right_delta = float(delta[1])
            except (TypeError, ValueError):
                continue
            forward_total += forward_delta
            right_total += right_delta
            if forward_total < -1.0:
                continue
            local_waypoints.append((forward_total, right_total))
        return local_waypoints

    @staticmethod
    def _local_waypoints_to_world(hero_actor, local_waypoints):
        hero_transform = hero_actor.get_transform()
        world_waypoints = []
        for forward, right in local_waypoints:
            world_x, world_y, world_z = _ego_local_to_world(
                hero_transform, forward, right, hero_transform.location.z)
            world_waypoints.append(
                carla.Location(x=world_x, y=world_y, z=world_z))
        return world_waypoints

    @staticmethod
    def _select_target_index(hero_location, world_waypoints, lookahead_distance):
        for index, waypoint in enumerate(world_waypoints):
            if hero_location.distance(waypoint) >= lookahead_distance:
                return index
        return len(world_waypoints) - 1

    @staticmethod
    def _estimate_target_speed_kmh(local_waypoints):
        if not local_waypoints:
            return 0.0

        speeds = []
        previous = (0.0, 0.0)
        for forward, right in local_waypoints[:4]:
            segment_distance = _distance_xy(previous, (forward, right))
            speeds.append(segment_distance / 0.5)
            previous = (forward, right)

        if not speeds:
            return 0.0

        speed_mps = sum(speeds) / len(speeds)
        return _clamp(speed_mps * 3.6, 0.0, 36.0)

    @staticmethod
    def _make_target_waypoint(world_waypoints, selected_index):
        target = world_waypoints[selected_index]
        if selected_index + 1 < len(world_waypoints):
            target_next = world_waypoints[selected_index + 1]
        elif selected_index > 0:
            target_next = world_waypoints[selected_index]
            target = world_waypoints[selected_index - 1]
        else:
            target_next = world_waypoints[selected_index]

        delta_x = target_next.x - target.x
        delta_y = target_next.y - target.y
        if abs(delta_x) <= 1e-6 and abs(delta_y) <= 1e-6:
            yaw_deg = 0.0
        else:
            yaw_deg = math.degrees(math.atan2(delta_y, delta_x))
        return _TargetWaypoint(world_waypoints[selected_index], yaw_deg)

    @staticmethod
    def _location_to_dict(location):
        return {
            'x': round(float(location.x), 3),
            'y': round(float(location.y), 3),
            'z': round(float(location.z), 3),
        }
