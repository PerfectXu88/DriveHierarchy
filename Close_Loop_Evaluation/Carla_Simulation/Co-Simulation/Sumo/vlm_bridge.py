# -*- coding: utf-8 -*-
"""
VLM communication bridge for the Tongji CARLA+SUMO pipeline.

This module runs inside **carla_base** (Python 3.8) and communicates with
the VLM service (running in bench4lmad / Python 3.10) via the project HTTP
``/interact`` REST protocol.

Responsibilities
----------------
1. Capture front-camera RGB images from CARLA and save to disk.
2. Build ``Bubble`` payloads that are wire-compatible with the VLM service.
3. POST to ``http://{host}:{port}/interact`` and parse the response.
4. Decode VLM output (waypoint tokens or behaviour commands) into
   ``carla.VehicleControl``.
"""

import json
import logging
import math
import os
import re
import time
import tempfile
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

# Fall back to stdlib if ``requests`` is not installed in carla_base
import urllib.request
import urllib.error

from b2dvl_compat import (
    Bubble, Context, create_query, create_response,
    decode_xy_token, decode_polar_token,
)
from b2dvl_compat.commands import (
    extract_control_values,
)
from vqa_adapter import (
    build_anno_from_carla,
    generate_condition_from_anno,
    QID42_QUESTION, build_action_decision_question,
    infer_nav_command_from_future_path,
)
from scene_scoring import SceneScoreTracker

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants aligned with B2DVL defaults
# ---------------------------------------------------------------------------
DEFAULT_CHAIN_ORDER = [50]  # default: direct action decision only
DEFAULT_CHAIN_PREV = {42: [], 50: []}
DEFAULT_CHAIN_INHERIT = {}

# Control tuning
MAX_TARGET_SPEED_MPS = 30.0
MAX_THROTTLE = 0.85
MAX_BRAKE = 0.85
TARGET_SPEED_THROTTLE_KP = 0.22
TARGET_SPEED_BRAKE_KP = 0.30
STEER_RATE_LIMIT = 0.12
MIN_MOVING_SPEED_MPS = 0.8
ROUTE_LOOKAHEAD_DISTANCE_M = 60.0
FUTURE_PATH_SAMPLE_DISTANCES_M = (5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0)


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib-only fallback)
# ---------------------------------------------------------------------------

def _post_json(url, payload, timeout=30):
    """POST JSON and return parsed response dict."""
    if _HAS_REQUESTS:
        session = _requests.Session()
        session.trust_env = False
        resp = session.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    else:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            url, data=data,
            headers={'Content-Type': 'application/json'},
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode('utf-8'))


# ---------------------------------------------------------------------------
# Image capture helper
# ---------------------------------------------------------------------------

class CarlaCameraCapture:
    """Attach an RGB camera to the hero vehicle and save frames to disk.

    The saved path is used as ``image_dict['CAM_FRONT']`` in the Bubble,
    which the VLM service reads when ``USE_BASE64=false``.
    """

    def __init__(self, vehicle, width=800, height=450, fov=100,
                 save_dir=None):
        self._vehicle = vehicle
        self._world = vehicle.get_world()
        self._width = width
        self._height = height
        self._latest_image = None
        self._frame_path = None

        if save_dir is None:
            save_dir = os.path.join(tempfile.gettempdir(), 'vlm_bridge_imgs')
        self._save_dir = save_dir
        os.makedirs(self._save_dir, exist_ok=True)

        bp_lib = self._world.get_blueprint_library()
        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(width))
        cam_bp.set_attribute('image_size_y', str(height))
        cam_bp.set_attribute('fov', str(fov))

        import carla
        spawn_point = carla.Transform(
            carla.Location(x=1.5, z=2.4),
            carla.Rotation(pitch=-5.0),
        )
        self._sensor = self._world.spawn_actor(
            cam_bp, spawn_point, attach_to=vehicle,
        )
        self._sensor.listen(self._on_image)

    def _on_image(self, image):
        self._latest_image = image

    def save_frame(self, frame_number):
        """Save the latest captured frame and return its file path."""
        if self._latest_image is None:
            return None
        path = os.path.join(self._save_dir, '{:06d}.jpg'.format(frame_number))
        self._latest_image.save_to_disk(path)
        self._frame_path = path
        return path

    @property
    def latest_path(self):
        return self._frame_path

    @property
    def sensor(self):
        return self._sensor

    @property
    def image_size(self):
        return self._width, self._height

    @property
    def fov(self):
        return 100.0

    def destroy(self):
        if self._sensor is not None and self._sensor.is_alive:
            self._sensor.stop()
            self._sensor.destroy()
            self._sensor = None


class CarlaBEVCapture:
    """Attach a top-down semantic/RGB camera as a simplified BEV source."""

    def __init__(self, vehicle, size=800, height=50.0, save_dir=None):
        self._vehicle = vehicle
        self._world = vehicle.get_world()
        self._latest_image = None
        self._frame_path = None
        self._size = size
        self._fov = 90.0

        if save_dir is None:
            save_dir = os.path.join(tempfile.gettempdir(), 'vlm_bridge_bev')
        self._save_dir = save_dir
        os.makedirs(self._save_dir, exist_ok=True)

        bp_lib = self._world.get_blueprint_library()
        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(size))
        cam_bp.set_attribute('image_size_y', str(size))
        cam_bp.set_attribute('fov', '90')

        import carla
        spawn_point = carla.Transform(
            carla.Location(x=0.0, z=height),
            carla.Rotation(pitch=-90.0),
        )
        self._sensor = self._world.spawn_actor(
            cam_bp, spawn_point, attach_to=vehicle,
        )
        self._sensor.listen(self._on_image)

    def _on_image(self, image):
        self._latest_image = image

    def save_frame(self, frame_number):
        if self._latest_image is None:
            return None
        path = os.path.join(self._save_dir, 'bev_{:06d}.jpg'.format(frame_number))
        self._latest_image.save_to_disk(path)
        self._frame_path = path
        return path

    @property
    def latest_path(self):
        return self._frame_path

    @property
    def sensor(self):
        return self._sensor

    @property
    def image_size(self):
        return self._size, self._size

    @property
    def fov(self):
        return self._fov

    def destroy(self):
        if self._sensor is not None and self._sensor.is_alive:
            self._sensor.stop()
            self._sensor.destroy()
            self._sensor = None


# ---------------------------------------------------------------------------
# Core bridge
# ---------------------------------------------------------------------------

class VLMBridge:
    """Orchestrates VLM inference for one scenario episode.

    Typical usage inside the simulation loop::

        bridge = VLMBridge(hero_actor, vlm_endpoint, scenario_name,
                           use_bev=False)
        ...
        while running:
            if tick_counter % control_interval == 0:
                ctrl = bridge.step(hero_actor, frame_number)
                hero_actor.apply_control(ctrl)
            ...
        bridge.destroy()
    """

    def __init__(self, hero_actor, vlm_endpoint, scenario_name,
                 use_bev=False, output_dir=None,
                 conversation_window=1, no_history=True,
                 img_width=800, img_height=450,
                 debug_vlm_io=False, debug_vlm_io_stdout=False,
                 enable_waypoint_inference=False,
                 action_considerations=None,
                 destination_point=None, via_points=None,
                 destination_direction_only=False,
                 route_points=None,
                 max_target_speed_mps=MAX_TARGET_SPEED_MPS):
        self.vlm_endpoint = vlm_endpoint.rstrip('/')
        self.scenario_name = scenario_name
        self.use_bev = use_bev
        self.debug_vlm_io = debug_vlm_io
        self.debug_vlm_io_stdout = debug_vlm_io_stdout
        self.enable_waypoint_inference = enable_waypoint_inference
        self.action_question = build_action_decision_question(
            considerations_text=action_considerations,
            max_target_speed_mps=max_target_speed_mps,
        )
        self.destination_point = destination_point
        self.via_points = via_points or []
        self.destination_direction_only = destination_direction_only
        self.route_points = [tuple(point) for point in (route_points or [])]
        self.max_target_speed_mps = float(max_target_speed_mps)

        if output_dir is None:
            output_dir = os.path.join(
                tempfile.gettempdir(), 'vlm_bridge_output', scenario_name)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._debug_dir = os.path.join(self.output_dir, 'vlm_debug')
        if self.debug_vlm_io:
            os.makedirs(self._debug_dir, exist_ok=True)
        self._anno_front_dir = os.path.join(self.output_dir, 'anno_front')
        self._anno_bev_dir = os.path.join(self.output_dir, 'anno_bev')
        os.makedirs(self._anno_front_dir, exist_ok=True)
        os.makedirs(self._anno_bev_dir, exist_ok=True)

        # Cameras
        self.front_cam = CarlaCameraCapture(
            hero_actor, width=img_width, height=img_height,
            save_dir=os.path.join(self.output_dir, 'rgb_front'),
        )
        self.bev_cam = None
        if use_bev:
            self.bev_cam = CarlaBEVCapture(
                hero_actor,
                save_dir=os.path.join(self.output_dir, 'bev'),
            )

        # B2DVL context (conversation history)
        self.context = Context(
            conversation_window=conversation_window,
            no_history=no_history,
        )

        # Chain config (minimal: only qid 42 and 50)
        self.chain_order = [42, 50] if self.enable_waypoint_inference else list(DEFAULT_CHAIN_ORDER)
        self.chain_prev = DEFAULT_CHAIN_PREV
        self.chain_inherit = DEFAULT_CHAIN_INHERIT

        # Last control for hold-between-inferences
        import carla
        self._last_control = carla.VehicleControl()
        self._last_steer_cmd = 0.0
        self._last_target_speed_mps = 0.0

        # Logging
        self._log_path = os.path.join(self.output_dir, 'vlm_bridge.log')

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def step(self, hero_actor, frame_number, nav_command=None):
        """Run one VLM inference cycle and return ``carla.VehicleControl``.

        Parameters
        ----------
        hero_actor : carla.Vehicle
        frame_number : int
        nav_command : int
            Navigation command (1-6).

        Returns
        -------
        carla.VehicleControl
        """
        # 1. Save images
        front_path = self.front_cam.save_frame(frame_number)
        bev_path = None
        if self.bev_cam is not None:
            bev_path = self.bev_cam.save_frame(frame_number)

        if front_path is None:
            log.warning('No front camera image available at frame %d', frame_number)
            return self._last_control

        route_guidance = self._build_route_guidance(hero_actor)
        anno_front_path, anno_bev_path = self._build_annotated_images(
            hero_actor=hero_actor,
            frame_number=frame_number,
            front_path=front_path,
            bev_path=bev_path,
            route_locations=route_guidance['route_locations'],
        )

        # 2. Build image dict (B2DVL format)
        image_dict = {
            'frame_number': frame_number,
            'CAM_FRONT': front_path,
        }
        if bev_path is not None:
            image_dict['ANNO_BEV'] = anno_bev_path or bev_path

        # 3. Build anno
        if nav_command is None:
            nav_command = route_guidance['nav_command']
        anno = build_anno_from_carla(
            hero_actor,
            nav_command=nav_command,
            future_path_points=route_guidance['future_path_points'],
            distance_to_destination=route_guidance['distance_to_destination'],
        )

        # 4. Ask each question in chain order
        self.context.fifo()
        steer_cmd = None
        target_speed_mps = None
        action_response_text = None

        for qid in self.chain_order:
            question_text = QID42_QUESTION if qid == 42 else self.action_question
            extra_condition = generate_condition_from_anno(anno, qid)

            question_bubble = create_query(
                words=question_text,
                images=[image_dict],
                frame_number=frame_number,
                scenario=self.scenario_name,
                qid=qid,
                gt='',
                extra_words=extra_condition,
            )

            # Get conversation context for this question
            curr_context = self.context.get_context_for_question(
                qid=qid,
                prev=self.chain_prev,
                inherit=self.chain_inherit,
                frame_number=frame_number,
            )

            # Call VLM service
            response_text = self._ask_vlm(question_bubble, curr_context)
            self._log('frame={} qid={} Q={}'.format(
                frame_number, qid, question_bubble.get_full_words()[:120]))
            self._log('frame={} qid={} A={}'.format(
                frame_number, qid, response_text[:200]))
            self._record_debug_io(
                frame_number=frame_number,
                qid=qid,
                image_dict=image_dict,
                question_bubble=question_bubble,
                curr_context=curr_context,
                response_text=response_text,
            )

            answer_bubble = create_response(
                words=response_text,
                frame_number=frame_number,
                scenario=self.scenario_name,
                qid=qid,
                gt='',
            )

            self.context.update(question_bubble)
            self.context.update(answer_bubble)

            # Parse answer
            if qid == 50:
                action_response_text = response_text
                steer_cmd, target_speed_mps = extract_control_values(response_text)
            elif qid == 42:
                # Waypoint tokens – decode but currently we rely on qid=50
                try:
                    wp = decode_xy_token(response_text)
                    if not wp:
                        wp = decode_polar_token(response_text)
                    self._log('frame={} decoded_wp={}'.format(frame_number, wp))
                except Exception as e:
                    self._log('frame={} wp_decode_error={}'.format(frame_number, e))

        # 5. Convert behaviour commands → VehicleControl
        ego_speed = self._get_speed(hero_actor)
        if steer_cmd is None or target_speed_mps is None:
            self._log(
                'frame={} invalid_action_response={!r}'.format(
                    frame_number, action_response_text
                )
            )
            self._save_frame_result(
                frame_number=frame_number,
                steer_cmd=None,
                target_speed_mps=None,
                ego_speed=ego_speed,
                future_path_points=route_guidance['future_path_points'],
                nav_command=nav_command,
                raw_response=action_response_text,
            )
            return self._last_control

        control, steer_cmd, target_speed_mps = self._control_values_to_control(
            steer_cmd=steer_cmd,
            target_speed_mps=target_speed_mps,
            ego_speed=ego_speed,
        )

        # Save result
        self._save_frame_result(
            frame_number=frame_number,
            steer_cmd=steer_cmd,
            target_speed_mps=target_speed_mps,
            ego_speed=ego_speed,
            future_path_points=route_guidance['future_path_points'],
            nav_command=nav_command,
            raw_response=action_response_text,
        )

        self._last_control = control
        self._last_steer_cmd = steer_cmd
        self._last_target_speed_mps = target_speed_mps
        return control

    def reset(self):
        """Reset context between scenarios."""
        self.context.reset()

    def destroy(self):
        """Clean up sensors."""
        self.front_cam.destroy()
        if self.bev_cam is not None:
            self.bev_cam.destroy()

    # ------------------------------------------------------------------
    # VLM communication (aligned with B2DVL ask_model)
    # ------------------------------------------------------------------

    def _ask_vlm(self, question_bubble, curr_context):
        """POST to /interact – same protocol as InferenceWorker.ask_model()."""
        payload = {
            "bubble": question_bubble.to_dict(),
            "conversation": [b.to_dict() for b in curr_context],
        }
        url = self.vlm_endpoint + '/interact'
        try:
            result = _post_json(url, payload, timeout=60)
            return result.get('response', '')
        except Exception as e:
            log.error('VLM request failed: %s', e)
            return 'Error: {}'.format(e)

    @staticmethod
    def _carla_location_from_nav_point(point):
        import carla
        return carla.Location(
            x=float(point['x']),
            y=float(point['y']),
            z=float(point.get('z', 0.5)),
        )

    @staticmethod
    def _carla_location_from_xy(point_xy, z=0.5):
        import carla
        return carla.Location(
            x=float(point_xy[0]),
            y=float(point_xy[1]),
            z=float(z),
        )

    @staticmethod
    def _distance_xy(point_a, point_b):
        return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])

    @classmethod
    def _sample_route_points(cls, route_points, sample_distances):
        if not route_points:
            return []
        if len(route_points) == 1:
            return [route_points[0]]

        sampled_points = []
        current_distance = 0.0
        segment_index = 0
        seg_start = route_points[0]
        seg_end = route_points[1]
        seg_length = cls._distance_xy(seg_start, seg_end)

        for target_distance in sample_distances:
            while segment_index < len(route_points) - 2 and current_distance + seg_length < target_distance:
                current_distance += seg_length
                segment_index += 1
                seg_start = route_points[segment_index]
                seg_end = route_points[segment_index + 1]
                seg_length = cls._distance_xy(seg_start, seg_end)

            if seg_length <= 1e-6:
                sampled_points.append(seg_end)
                continue

            remaining = max(0.0, target_distance - current_distance)
            ratio = min(max(remaining / seg_length, 0.0), 1.0)
            sampled_points.append((
                seg_start[0] + (seg_end[0] - seg_start[0]) * ratio,
                seg_start[1] + (seg_end[1] - seg_start[1]) * ratio,
            ))

        return sampled_points

    def _build_route_guidance(self, hero_actor):
        hero_location = hero_actor.get_location()
        hero_xy = (float(hero_location.x), float(hero_location.y))

        if self.route_points and len(self.route_points) >= 2 and not self.destination_direction_only:
            route_length = SceneScoreTracker._compute_route_length(self.route_points)
            progress, _, _ = SceneScoreTracker._project_onto_polyline(hero_xy, self.route_points)
            end_progress = min(route_length, progress + ROUTE_LOOKAHEAD_DISTANCE_M)
            local_route_points = SceneScoreTracker._slice_polyline(
                self.route_points,
                progress,
                end_progress,
            )
            local_route_points = local_route_points or [self.route_points[-1]]
            sampled_route_points = self._sample_route_points(
                local_route_points,
                FUTURE_PATH_SAMPLE_DISTANCES_M,
            )
            future_path_points = [
                self._world_to_ego_local(hero_actor.get_transform(), self._carla_location_from_xy(point_xy))
                for point_xy in sampled_route_points
            ]
            route_locations = [
                self._carla_location_from_xy(point_xy)
                for point_xy in local_route_points
            ]
            nav_command = infer_nav_command_from_future_path(future_path_points)
            distance_to_destination = max(0.0, route_length - progress)
            return {
                'route_locations': route_locations,
                'future_path_points': future_path_points,
                'nav_command': nav_command,
                'distance_to_destination': distance_to_destination,
            }

        route_locations = self._get_remaining_route_locations(hero_actor)
        future_path_points = [
            self._world_to_ego_local(hero_actor.get_transform(), location)
            for location in route_locations[:len(FUTURE_PATH_SAMPLE_DISTANCES_M)]
        ]
        nav_command = infer_nav_command_from_future_path(future_path_points)
        distance_to_destination = None
        if self.destination_point is not None:
            destination_loc = self._carla_location_from_nav_point(self.destination_point)
            distance_to_destination = hero_location.distance(destination_loc)
        return {
            'route_locations': route_locations,
            'future_path_points': future_path_points,
            'nav_command': nav_command,
            'distance_to_destination': distance_to_destination,
        }

    def _get_remaining_route_locations(self, hero_actor):
        if self.destination_direction_only:
            if self.destination_point is None:
                return []
            return [self._carla_location_from_nav_point(self.destination_point)]

        route_points = []
        hero_location = hero_actor.get_location()
        for point in self.via_points:
            point_loc = self._carla_location_from_nav_point(point)
            if hero_location.distance(point_loc) > 3.0:
                route_points.append(point_loc)
        if self.destination_point is not None:
            route_points.append(self._carla_location_from_nav_point(self.destination_point))
        return route_points[:32]

    @staticmethod
    def _build_camera_intrinsic(width, height, fov_deg):
        focal = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
        return np.array([
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ])

    @staticmethod
    def _project_world_to_image(sensor, world_location, width, height, fov_deg):
        sensor_transform = sensor.get_transform()
        world_to_camera = np.array(sensor_transform.get_inverse_matrix())
        point = np.array([world_location.x, world_location.y, world_location.z, 1.0])
        point_camera = world_to_camera.dot(point)
        point_camera = np.array([point_camera[1], -point_camera[2], point_camera[0]])
        if point_camera[2] <= 0.1:
            return None

        intrinsic = VLMBridge._build_camera_intrinsic(width, height, fov_deg)
        point_img = intrinsic.dot(point_camera)
        u = float(point_img[0] / point_img[2])
        v = float(point_img[1] / point_img[2])
        if u < 0 or u >= width or v < 0 or v >= height:
            return None
        return u, v

    @staticmethod
    def _world_to_ego_local(hero_transform, world_location):
        delta = world_location - hero_transform.location
        yaw = math.radians(hero_transform.rotation.yaw)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        forward = delta.x * cos_yaw + delta.y * sin_yaw
        right = -delta.x * sin_yaw + delta.y * cos_yaw
        return forward, right

    @staticmethod
    def _clip_line_to_rect(start_point, end_point, width, height):
        x0, y0 = start_point
        x1, y1 = end_point
        dx = x1 - x0
        dy = y1 - y0
        t0 = 0.0
        t1 = 1.0

        def update(p, q, cur_t0, cur_t1):
            if abs(p) <= 1e-9:
                if q < 0.0:
                    return None
                return cur_t0, cur_t1
            ratio = q / p
            if p < 0.0:
                if ratio > cur_t1:
                    return None
                if ratio > cur_t0:
                    cur_t0 = ratio
            else:
                if ratio < cur_t0:
                    return None
                if ratio < cur_t1:
                    cur_t1 = ratio
            return cur_t0, cur_t1

        for p, q in (
                (-dx, x0),
                (dx, (width - 1.0) - x0),
                (-dy, y0),
                (dy, (height - 1.0) - y0)):
            updated = update(p, q, t0, t1)
            if updated is None:
                return None
            t0, t1 = updated

        clipped_start = (x0 + t0 * dx, y0 + t0 * dy)
        clipped_end = (x0 + t1 * dx, y0 + t1 * dy)
        return clipped_start, clipped_end

    def _draw_route_on_bev_image(self, src_path, dst_path, hero_actor, route_locations):
        if self.bev_cam is None or not route_locations:
            return None

        width, height = self.bev_cam.image_size
        fov_rad = math.radians(self.bev_cam.fov)
        bev_height = float(self.bev_cam.sensor.get_transform().location.z)
        visible_span_m = 2.0 * bev_height * math.tan(fov_rad / 2.0)
        if visible_span_m <= 1e-6:
            return None

        pixels_per_meter = width / visible_span_m
        hero_transform = hero_actor.get_transform()

        ego_point = (width / 2.0, height / 2.0)
        projected_points = [ego_point]
        for location in route_locations:
            forward, right = self._world_to_ego_local(hero_transform, location)
            projected_points.append((
                width / 2.0 + right * pixels_per_meter,
                height / 2.0 - forward * pixels_per_meter,
            ))

        clipped_segments = []
        visible_markers = []
        for prev_point, curr_point in zip(projected_points[:-1], projected_points[1:]):
            clipped = self._clip_line_to_rect(prev_point, curr_point, width, height)
            if clipped is None:
                continue
            clipped_segments.append(clipped)
            if 0.0 <= curr_point[0] < width and 0.0 <= curr_point[1] < height:
                visible_markers.append(curr_point)

        if not clipped_segments:
            return None

        image = Image.open(src_path).convert('RGB')
        draw = ImageDraw.Draw(image)

        for segment_start, segment_end in clipped_segments:
            draw.line((segment_start, segment_end), fill=(255, 90, 90), width=3)

        for index, point in enumerate(visible_markers, start=1):
            radius = 4 if index < len(visible_markers) else 5
            draw.ellipse(
                (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
                fill=(255, 255, 255),
                outline=(255, 90, 90),
                width=2,
            )
        draw.ellipse(
            (ego_point[0] - 6, ego_point[1] - 6, ego_point[0] + 6, ego_point[1] + 6),
            fill=(255, 230, 80),
            outline=(20, 20, 20),
            width=2,
        )
        image.save(dst_path)
        return dst_path

    def _build_annotated_images(self, hero_actor, frame_number, front_path, bev_path,
                                route_locations):
        if not route_locations:
            return None, None

        del front_path
        anno_bev_path = None

        if bev_path is not None and self.bev_cam is not None:
            anno_bev_path = os.path.join(
                self._anno_bev_dir, 'bev_{:06d}.jpg'.format(frame_number))
            anno_bev_path = self._draw_route_on_bev_image(
                src_path=bev_path,
                dst_path=anno_bev_path,
                hero_actor=hero_actor,
                route_locations=route_locations,
            )

        return None, anno_bev_path

    # ------------------------------------------------------------------
    # Control conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _get_speed(hero_actor):
        v = hero_actor.get_velocity()
        return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)

    def _control_values_to_control(self, steer_cmd, target_speed_mps, ego_speed):
        """Convert ``steer`` + ``target_speed_mps`` to ``carla.VehicleControl``."""
        import carla
        ctrl = carla.VehicleControl()
        ctrl.manual_gear_shift = False

        steer_cmd = max(-1.0, min(1.0, float(steer_cmd)))
        target_speed_mps = max(0.0, min(self.max_target_speed_mps, float(target_speed_mps)))
        steer_delta_min = self._last_steer_cmd - STEER_RATE_LIMIT
        steer_delta_max = self._last_steer_cmd + STEER_RATE_LIMIT
        steer_cmd = max(steer_delta_min, min(steer_delta_max, steer_cmd))

        speed_error = target_speed_mps - ego_speed
        if target_speed_mps <= 0.2:
            ctrl.throttle = 0.0
            ctrl.brake = min(MAX_BRAKE, 0.35 + 0.25 * max(ego_speed, 0.0))
        elif speed_error >= 0.0:
            throttle = 0.25 + TARGET_SPEED_THROTTLE_KP * speed_error
            if target_speed_mps >= MIN_MOVING_SPEED_MPS and ego_speed < MIN_MOVING_SPEED_MPS:
                throttle = max(throttle, 0.45)
            ctrl.throttle = min(MAX_THROTTLE, max(0.0, throttle))
            ctrl.brake = 0.0
        else:
            ctrl.throttle = 0.0
            ctrl.brake = min(MAX_BRAKE, max(0.0, -speed_error * TARGET_SPEED_BRAKE_KP))

        ctrl.steer = steer_cmd
        return ctrl, steer_cmd, target_speed_mps

    # ------------------------------------------------------------------
    # Logging / persistence
    # ------------------------------------------------------------------

    def _log(self, msg):
        with open(self._log_path, 'a') as f:
            f.write(msg + '\n')

    @staticmethod
    def _bubble_debug_dict(bubble):
        return {
            'qid': bubble.qid,
            'actor': bubble.actor,
            'frame_number': bubble.frame_number,
            'scenario': bubble.scenario,
            'words': bubble.words,
            'extra_words': bubble.extra_words,
            'full_words': bubble.get_full_words(),
            'images': bubble.images,
            'extra_images': bubble.extra_images,
            'full_images': bubble.get_full_images(),
        }

    def _record_debug_io(self, frame_number, qid, image_dict, question_bubble,
                         curr_context, response_text):
        if not self.debug_vlm_io and not self.debug_vlm_io_stdout:
            return

        payload = {
            "bubble": question_bubble.to_dict(),
            "conversation": [b.to_dict() for b in curr_context],
        }
        debug_record = {
            'timestamp': datetime.now().isoformat(),
            'frame': frame_number,
            'qid': qid,
            'vlm_endpoint': self.vlm_endpoint,
            'question': self._bubble_debug_dict(question_bubble),
            'conversation': [self._bubble_debug_dict(b) for b in curr_context],
            'image_paths': image_dict,
            'response_text': response_text,
            'raw_payload': payload,
        }

        if self.debug_vlm_io:
            path = os.path.join(
                self._debug_dir,
                '{:06d}_qid{}_io.json'.format(frame_number, qid),
            )
            with open(path, 'w') as f:
                json.dump(debug_record, f, indent=2, ensure_ascii=False)

        if self.debug_vlm_io_stdout:
            print('[vlm-debug] frame={} qid={}'.format(frame_number, qid))
            print('[vlm-debug] images={}'.format(json.dumps(image_dict, ensure_ascii=False)))
            print('[vlm-debug] question={}'.format(question_bubble.get_full_words()))
            if curr_context:
                print('[vlm-debug] context={}'.format(json.dumps(
                    [self._bubble_debug_dict(b) for b in curr_context],
                    ensure_ascii=False)))
            print('[vlm-debug] response={}'.format(response_text))

    def _save_frame_result(self, frame_number, steer_cmd, target_speed_mps,
                           ego_speed, future_path_points, nav_command,
                           raw_response):
        result = {
            'frame': frame_number,
            'steer_cmd': steer_cmd,
            'target_speed_mps': (
                None if target_speed_mps is None else round(float(target_speed_mps), 3)
            ),
            'ego_speed': round(ego_speed, 2),
            'nav_command': nav_command,
            'future_path_points': [
                [round(float(point[0]), 3), round(float(point[1]), 3)]
                for point in (future_path_points or [])
            ],
            'raw_response': raw_response,
        }
        path = os.path.join(self.output_dir, '{:06d}_ctrl.json'.format(frame_number))
        with open(path, 'w') as f:
            json.dump(result, f, indent=2)
