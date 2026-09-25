#!/usr/bin/env python

"""Manual hero driving on Tongji with entity.json-derived SUMO background traffic."""

import argparse
from configparser import ConfigParser
import glob
import logging
import math
import numpy as np
import os
import sys
import time
import weakref

import lxml.etree as ET

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    sys.path.append(
        glob.glob(
            os.path.join(
                SCRIPT_DIR,
                '../../PythonAPI/carla/dist/carla-*%d.%d-%s.egg'
                % (
                    sys.version_info.major,
                    sys.version_info.minor,
                    'win-amd64' if os.name == 'nt' else 'linux-x86_64',
                ),
            )
        )[0]
    )
except IndexError:
    pass

pythonapi_dir = os.path.join(SCRIPT_DIR, '../../PythonAPI/carla')
if pythonapi_dir not in sys.path:
    sys.path.append(pythonapi_dir)

try:
    sys.path.append(
        glob.glob(
            os.path.join(
                SCRIPT_DIR,
                '../../PythonAPI/carla/dist/carla-*%d.%d-%s.whl'
                % (
                    sys.version_info.major,
                    sys.version_info.minor,
                    'win-amd64' if os.name == 'nt' else 'linux-x86_64',
                ),
            )
        )[0]
    )
except IndexError:
    pass

import carla
import pygame

from entity_scenario_lib import load_scene
import sumolib
from run_synchronization import SimulationSynchronization
from sumo_integration.carla_simulation import CarlaSimulation
from sumo_integration.sumo_simulation import SumoSimulation

import traci  # pylint: disable=wrong-import-position,import-error


class ManualHeroController(object):
    """Minimal keyboard controller for the hero vehicle."""

    def __init__(self):
        self._control = carla.VehicleControl()
        self._steer_cache = 0.0

    def parse_events(self, camera):
        for event in pygame.event.get():
            if self._handle_common_event(event, camera):
                return True
        return False

    def _handle_common_event(self, event, camera):
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
            if event.key == pygame.K_q:
                self._control.reverse = not self._control.reverse
                self._control.gear = -1 if self._control.reverse else 1
            if event.key == pygame.K_TAB:
                camera.toggle()
        return False

    def apply(self, vehicle, milliseconds):
        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self._control.throttle = min(self._control.throttle + 0.02, 1.0)
        else:
            self._control.throttle = 0.0

        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self._control.brake = min(self._control.brake + 0.2, 1.0)
        else:
            self._control.brake = 0.0

        steer_increment = 5e-4 * milliseconds
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0.0
            self._steer_cache -= steer_increment
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0.0
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0

        self._steer_cache = min(0.8, max(-0.8, self._steer_cache))
        self._control.steer = round(self._steer_cache, 2)
        self._control.hand_brake = keys[pygame.K_SPACE]
        vehicle.apply_control(self._control)


class G29HeroController(ManualHeroController):
    """Logitech G29 controller backed by pygame joystick."""

    def __init__(self, config_path):
        super(G29HeroController, self).__init__()
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count < 1:
            raise RuntimeError('No joystick detected for G29 mode')
        if joystick_count > 1:
            raise RuntimeError('Multiple joysticks detected; please keep only the G29 connected')

        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()

        parser = ConfigParser()
        if not parser.read(config_path):
            raise RuntimeError('Could not read wheel config: %s' % config_path)

        section = 'G29 Racing Wheel'
        self._steer_idx = parser.getint(section, 'steering_wheel')
        self._throttle_idx = parser.getint(section, 'throttle')
        self._brake_idx = parser.getint(section, 'brake')
        self._reverse_idx = parser.getint(section, 'reverse')
        self._handbrake_idx = parser.getint(section, 'handbrake')

    def parse_events(self, camera):
        for event in pygame.event.get():
            if self._handle_common_event(event, camera):
                return True
            if event.type == pygame.JOYBUTTONDOWN and event.button == self._reverse_idx:
                self._control.reverse = not self._control.reverse
                self._control.gear = -1 if self._control.reverse else 1
        return False

    def apply(self, vehicle, milliseconds):
        del milliseconds
        num_axes = self._joystick.get_numaxes()
        js_inputs = [float(self._joystick.get_axis(i)) for i in range(num_axes)]
        js_buttons = [float(self._joystick.get_button(i)) for i in range(self._joystick.get_numbuttons())]

        steer_cmd = math.tan(1.1 * js_inputs[self._steer_idx])
        throttle_cmd = 1.6 + (2.05 * math.log10(-0.7 * js_inputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
        brake_cmd = 1.6 + (2.05 * math.log10(-0.7 * js_inputs[self._brake_idx] + 1.4) - 1.2) / 0.92

        self._control.steer = max(-1.0, min(1.0, steer_cmd))
        self._control.throttle = max(0.0, min(1.0, throttle_cmd))
        self._control.brake = max(0.0, min(1.0, brake_cmd))
        self._control.hand_brake = bool(js_buttons[self._handbrake_idx])
        vehicle.apply_control(self._control)


class HeroCamera(object):
    """RGB camera manager with a few useful viewpoints."""

    def __init__(self, vehicle, width, height, initial_view='chase'):
        self._vehicle = vehicle
        self._surface = None
        self._sensor = None
        bound_x = 0.5 + vehicle.bounding_box.extent.x
        bound_y = 0.5 + vehicle.bounding_box.extent.y
        bound_z = 0.5 + vehicle.bounding_box.extent.z
        attachment = carla.AttachmentType
        self._views = [
            (
                'chase',
                carla.Transform(
                    carla.Location(x=-2.5 * bound_x, y=0.0 * bound_y, z=2.0 * bound_z),
                    carla.Rotation(pitch=8.0),
                ),
                attachment.SpringArm,
            ),
            (
                'hood',
                carla.Transform(carla.Location(x=0.8 * bound_x, y=0.0, z=1.3 * bound_z)),
                attachment.Rigid,
            ),
            (
                'high_follow',
                carla.Transform(
                    carla.Location(x=-3.0 * bound_x, y=0.0, z=5.0 * bound_z),
                    carla.Rotation(pitch=-12.0),
                ),
                attachment.SpringArm,
            ),
            (
                'bev',
                carla.Transform(
                    carla.Location(x=0.0, y=0.0, z=max(18.0, 12.0 * bound_z)),
                    carla.Rotation(pitch=-90.0),
                ),
                attachment.Rigid,
            ),
        ]
        available_views = [name for name, _, _ in self._views]
        if initial_view not in available_views:
            raise RuntimeError('Unsupported camera view %s, choose from %s' % (initial_view, available_views))
        self._transform_index = available_views.index(initial_view)

        world = vehicle.get_world()
        blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
        blueprint.set_attribute('image_size_x', str(width))
        blueprint.set_attribute('image_size_y', str(height))
        blueprint.set_attribute('fov', '100')
        self._blueprint = blueprint
        self._spawn_sensor()

    def _spawn_sensor(self):
        if self._sensor is not None:
            self._sensor.destroy()
            self._sensor = None
        _, transform, attachment = self._views[self._transform_index]
        self._sensor = self._vehicle.get_world().spawn_actor(
            self._blueprint,
            transform,
            attach_to=self._vehicle,
            attachment_type=attachment,
        )
        weak_self = weakref.ref(self)
        self._sensor.listen(lambda image: HeroCamera._on_image(weak_self, image))

    @staticmethod
    def _on_image(weak_self, image):
        self = weak_self()
        if self is None:
            return
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self._surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    def toggle(self):
        self._transform_index = (self._transform_index + 1) % len(self._views)
        self._spawn_sensor()

    @property
    def view_name(self):
        return self._views[self._transform_index][0]

    def render(self, display):
        if self._surface is not None:
            display.blit(self._surface, (0, 0))

    def destroy(self):
        if self._sensor is not None:
            if self._sensor.is_alive:
                self._sensor.stop()
                self._sensor.destroy()
            self._sensor = None


class HeroCollisionMonitor(object):
    """Small collision monitor used for optional hero termination rules."""

    def __init__(self, hero_actor):
        self._sensor = None
        self.collided = False
        self.last_other_actor = 'unknown'
        self.last_intensity = 0.0

        world = hero_actor.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self._sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=hero_actor)
        weak_self = weakref.ref(self)
        self._sensor.listen(lambda event: HeroCollisionMonitor._on_collision(weak_self, event))

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if self is None:
            return

        other_actor = event.other_actor
        if other_actor is not None:
            self.last_other_actor = other_actor.type_id
        impulse = event.normal_impulse
        self.last_intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.collided = True

    def destroy(self):
        if self._sensor is not None:
            if self._sensor.is_alive:
                self._sensor.stop()
                self._sensor.destroy()
            self._sensor = None


class Overlay(object):
    """Small on-screen status panel."""

    def __init__(self, width, height, scenario_name):
        pygame.font.init()
        self.dim = (width, height)
        self.font = pygame.font.Font(pygame.font.get_default_font(), 20)
        self.scenario_name = scenario_name

    def render(
        self,
        display,
        hero,
        actor_count,
        active_scene_actors,
        sim_time,
        control_mode,
        max_sim_time,
        camera_view,
        navigation_summary=None,
    ):
        panel = pygame.Surface((460, 176))
        panel.set_alpha(160)
        panel.fill((0, 0, 0))
        display.blit(panel, (12, 12))

        velocity = hero.get_velocity()
        speed_kmh = 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        control_hint = 'G29 pedals+wheel, Tab camera, Q reverse, Esc quit'
        if control_mode == 'keyboard':
            control_hint = 'WASD/arrows drive, Space brake, Tab camera, Q reverse, Esc quit'
        lines = [
            'Tongji entity-scene manual drive',
            'Scene: %s' % self.scenario_name,
            'Control: %s' % control_mode,
            'Camera: %s' % camera_view,
            'Speed: %.1f km/h' % speed_kmh,
            'Actors: %d total / %d scripted active' % (actor_count, active_scene_actors),
            'Sim time: %.1f / %s s' % (sim_time, 'inf' if max_sim_time is None else '%.1f' % max_sim_time),
            control_hint,
        ]
        if navigation_summary:
            lines.extend(navigation_summary)

        y = 24
        for line in lines:
            text = self.font.render(line, True, (255, 255, 255))
            display.blit(text, (24, y))
            y += 22


def write_sumocfg_xml(cfg_file, net_file, route_file, viewsettings_file, additional_traci_clients=0):
    root = ET.Element('configuration')

    input_tag = ET.SubElement(root, 'input')
    ET.SubElement(input_tag, 'net-file', {'value': net_file})
    ET.SubElement(input_tag, 'route-files', {'value': route_file})

    gui_tag = ET.SubElement(root, 'gui_only')
    ET.SubElement(gui_tag, 'gui-settings-file', {'value': viewsettings_file})

    ET.SubElement(root, 'num-clients', {'value': str(additional_traci_clients + 1)})

    tree = ET.ElementTree(root)
    tree.write(cfg_file, pretty_print=True, encoding='UTF-8', xml_declaration=True)


def configure_world(carla_simulation, weather_config):
    env_objs1 = carla_simulation.world.get_environment_objects(carla.CityObjectLabel.TrafficLight)
    env_objs2 = carla_simulation.world.get_environment_objects(carla.CityObjectLabel.Poles)
    env_ids = [obj.id for obj in env_objs1] + [obj.id for obj in env_objs2]
    if env_ids:
        carla_simulation.world.enable_environment_objects(env_ids, False)

    weather = carla.WeatherParameters(**weather_config)
    carla_simulation.world.set_weather(weather)


def maybe_reload_world(carla_simulation):
    try:
        new_world = carla_simulation.client.reload_world(False)
    except TypeError:
        new_world = carla_simulation.client.reload_world()
    carla_simulation.world = new_world
    carla_simulation.blueprint_library = new_world.get_blueprint_library()
    carla_simulation._active_actors = set()  # pylint: disable=protected-access
    carla_simulation.spawned_actors = set()
    carla_simulation.destroyed_actors = set()
    carla_simulation._tls = {}  # pylint: disable=protected-access
    tmp_map = new_world.get_map()
    for landmark in tmp_map.get_all_landmarks_of_type('1000001'):
        if landmark.id != '':
            traffic_light = new_world.get_traffic_light(landmark)
            if traffic_light is not None:
                carla_simulation._tls[landmark.id] = traffic_light  # pylint: disable=protected-access


def cleanup_existing_actors(carla_simulation):
    actors = carla_simulation.world.get_actors()
    actor_ids = []
    for actor in actors:
        if actor.type_id.startswith('vehicle.'):
            actor_ids.append(actor.id)
        elif actor.type_id.startswith('sensor.'):
            actor_ids.append(actor.id)
        elif actor.type_id.startswith('walker.'):
            actor_ids.append(actor.id)
        elif actor.type_id == 'controller.ai.walker':
            actor_ids.append(actor.id)

    if not actor_ids:
        return

    logging.info('Cleaning up %d pre-existing CARLA actors', len(actor_ids))
    commands = [carla.command.DestroyActor(actor_id) for actor_id in actor_ids]
    carla_simulation.client.apply_batch_sync(commands, True)
    try:
        carla_simulation.world.tick()
    except RuntimeError:
        carla_simulation.world.wait_for_tick()


def set_spectator(carla_simulation, hero_actor, view_mode='chase', height=45.0):
    spectator = carla_simulation.world.get_spectator()
    hero_transform = hero_actor.get_transform()
    if view_mode == 'bev':
        follow_location = hero_transform.location + carla.Location(z=height)
        follow_rotation = carla.Rotation(pitch=-90.0, yaw=hero_transform.rotation.yaw)
    else:
        follow_location = hero_transform.location + carla.Location(x=-10.0, z=max(6.0, height * 0.15))
        follow_rotation = carla.Rotation(pitch=-18.0, yaw=hero_transform.rotation.yaw)
    spectator.set_transform(carla.Transform(follow_location, follow_rotation))


def make_nav_point(point, default_label):
    if point is None:
        return None
    return {
        'x': float(point['x']) if 'x' in point else float(point['carla_x']),
        'y': float(point['y']) if 'y' in point else float(point['carla_y']),
        'z': float(point.get('z', 0.8)),
        'label': point.get('label', default_label),
    }


def get_ego_navigation_display(hero_config, via_point_stride):
    navigation_display = hero_config.get('navigation_display')
    if navigation_display:
        destination = make_nav_point(navigation_display.get('destination'), 'goal')
        via_points = [
            make_nav_point(point, 'via_%d' % index)
            for index, point in enumerate(navigation_display.get('via_points', []), start=1)
        ]
        via_points = [point for point in via_points if point is not None]
        return destination, via_points

    reference_points = hero_config.get('reference_points', [])
    if not reference_points:
        return None, []

    destination = make_nav_point(reference_points[-1], 'goal')
    stride = max(1, via_point_stride)
    via_candidates = []
    for index, point in enumerate(reference_points[1:-1], start=1):
        if point.get('point_type') == 'start':
            continue
        via_candidates.append(
            make_nav_point(
                {
                    'carla_x': point['carla_x'],
                    'carla_y': point['carla_y'],
                    'label': 'via_%d' % index,
                },
                'via_%d' % index,
            )
        )
    return destination, via_candidates[::stride]


def draw_ego_navigation_markers(world, destination_point, via_points, show_destination, show_via_points, life_time):
    debug = world.debug
    if show_via_points:
        for via_point in via_points:
            location = carla.Location(x=via_point['x'], y=via_point['y'], z=via_point['z'])
            debug.draw_point(location, size=0.12, color=carla.Color(0, 200, 255), life_time=life_time)
            debug.draw_string(
                location + carla.Location(z=0.35),
                via_point['label'],
                draw_shadow=False,
                color=carla.Color(0, 200, 255),
                life_time=life_time,
                persistent_lines=False,
            )

    if show_destination and destination_point is not None:
        location = carla.Location(
            x=destination_point['x'],
            y=destination_point['y'],
            z=destination_point['z'],
        )
        debug.draw_point(location, size=0.18, color=carla.Color(255, 80, 80), life_time=life_time)
        debug.draw_string(
            location + carla.Location(z=0.45),
            destination_point['label'],
            draw_shadow=False,
            color=carla.Color(255, 80, 80),
            life_time=life_time,
            persistent_lines=False,
        )


def build_navigation_summary(destination_point, via_points):
    summary = []
    if destination_point is not None:
        summary.append('Goal: (%.1f, %.1f)' % (destination_point['x'], destination_point['y']))
    if via_points:
        summary.append('Via points: %d' % len(via_points))
    return summary


def hero_reached_destination(hero_actor, destination_point, distance_threshold):
    if hero_actor is None or destination_point is None:
        return False
    hero_location = hero_actor.get_transform().location
    dx = hero_location.x - destination_point['x']
    dy = hero_location.y - destination_point['y']
    return math.hypot(dx, dy) <= distance_threshold


def hero_is_offroad(world_map, hero_actor):
    if hero_actor is None or not hero_actor.is_alive:
        return False
    hero_location = hero_actor.get_location()
    waypoint = world_map.get_waypoint(
        hero_location,
        project_to_road=False,
        lane_type=carla.LaneType.Driving,
    )
    return waypoint is None


def default_vehicle_color(type_id):
    if type_id == 'vehicle.volkswagen.t2_2021':
        return (0, 255, 0)
    if type_id == 'vehicle.ford.ambulance':
        return (0, 0, 255)
    return None


def split_scene_background_actors(background_actors):
    scene_vehicles = []
    scene_pedestrians = []
    for actor in background_actors:
        if actor.get("role") == "pedestrian":
            scene_pedestrians.append(actor)
        else:
            scene_vehicles.append(actor)
    return scene_vehicles, scene_pedestrians


def snap_transform(world_map, x, y, yaw_deg, z_offset):
    query_location = carla.Location(x=x, y=y, z=0.0)
    waypoint = world_map.get_waypoint(
        query_location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if waypoint is not None:
        snapped_transform = waypoint.transform
        location = snapped_transform.location
        location.z += z_offset
        rotation = snapped_transform.rotation
        return carla.Transform(location, rotation)
    return carla.Transform(
        carla.Location(x=x, y=y, z=z_offset),
        carla.Rotation(pitch=0.0, yaw=yaw_deg, roll=0.0),
    )


def spawn_scene_background_vehicles(background_vehicles):
    scripted_vehicle_ids = {}
    for vehicle in background_vehicles:
        route_id = vehicle["route_id"]
        actor_id = "entity_%s" % vehicle["actor_id"]
        route_edges = vehicle["route_edges"]
        depart = vehicle["depart"]
        traci.route.add(route_id, route_edges)
        depart_time = str(depart.get("time_seconds", 0.0))
        depart_lane = str(depart.get("lane_index", "best"))
        depart_pos = str(depart.get("lane_pos", "base"))
        depart_speed = str(max(depart.get("speed_kmh", 0.0) / 3.6, 0.0))
        try:
            traci.vehicle.add(
                actor_id,
                route_id,
                typeID=vehicle["sumo_type_id"],
                depart=depart_time,
                departLane=depart_lane,
                departPos=depart_pos,
                departSpeed=depart_speed,
            )
        except traci.exceptions.TraCIException as error:
            logging.info(
                "Fallback depart settings for %s on edge %s due to SUMO rejection: %s",
                actor_id,
                depart.get("edge_id"),
                error,
            )
            traci.vehicle.add(
                actor_id,
                route_id,
                typeID=vehicle["sumo_type_id"],
                depart=depart_time,
                departLane="best",
                departPos="base",
                departSpeed=depart_speed,
            )
        color = default_vehicle_color(vehicle["sumo_type_id"])
        if color is not None:
            traci.vehicle.setColor(actor_id, color)
        scripted_vehicle_ids[actor_id] = vehicle
    return scripted_vehicle_ids


def pedestrian_lane_mask():
    lane_type = carla.LaneType.Driving
    for lane_name in ('Sidewalk', 'Shoulder', 'Parking'):
        if hasattr(carla.LaneType, lane_name):
            lane_type = lane_type | getattr(carla.LaneType, lane_name)
    return lane_type


def snap_pedestrian_transform(world_map, x, y, yaw_deg, z_offset):
    waypoint = world_map.get_waypoint(
        carla.Location(x=x, y=y, z=0.0),
        project_to_road=True,
        lane_type=pedestrian_lane_mask(),
    )
    if waypoint is not None:
        location = waypoint.transform.location
        location.z += z_offset
        return carla.Transform(
            location,
            carla.Rotation(
                pitch=0.0,
                yaw=yaw_deg,
                roll=0.0,
            ),
        )
    return carla.Transform(
        carla.Location(x=x, y=y, z=z_offset),
        carla.Rotation(pitch=0.0, yaw=yaw_deg, roll=0.0),
    )


def choose_pedestrian_blueprint(world, actor_id):
    blueprints = list(world.get_blueprint_library().filter('walker.pedestrian.*'))
    if not blueprints:
        raise RuntimeError('No walker.pedestrian.* blueprints found in CARLA')
    blueprints.sort(key=lambda blueprint: blueprint.id)
    blueprint = blueprints[sum(str(actor_id).encode('utf-8')) % len(blueprints)]
    if blueprint.has_attribute('is_invincible'):
        blueprint.set_attribute('is_invincible', 'false')
    blueprint.set_attribute('role_name', 'scripted_pedestrian')
    return blueprint


def spawn_scripted_pedestrian(world, pedestrian_config):
    blueprint = choose_pedestrian_blueprint(world, pedestrian_config["actor_id"])
    spawn_x = pedestrian_config["carla_spawn"]["x"]
    spawn_y = pedestrian_config["carla_spawn"]["y"]
    spawn_yaw = pedestrian_config["carla_spawn"]["yaw"]
    world_map = world.get_map()

    candidate_transforms = []
    seen_transforms = set()

    def add_candidate(transform):
        key = (
            round(transform.location.x, 2),
            round(transform.location.y, 2),
            round(transform.location.z, 2),
            round(transform.rotation.yaw, 2),
        )
        if key not in seen_transforms:
            seen_transforms.add(key)
            candidate_transforms.append(transform)

    for z_offset in (0.4, 0.8, 1.2, 1.6):
        add_candidate(snap_pedestrian_transform(world_map, spawn_x, spawn_y, spawn_yaw, z_offset))

    for candidate_transform in candidate_transforms:
        actor = world.try_spawn_actor(blueprint, candidate_transform)
        if actor is not None:
            logging.info(
                'Spawned scripted pedestrian %s at x=%.2f y=%.2f z=%.2f',
                pedestrian_config["actor_id"],
                candidate_transform.location.x,
                candidate_transform.location.y,
                candidate_transform.location.z,
            )
            return actor

    raise RuntimeError(
        'Could not spawn pedestrian %s near scenario pose (x=%.2f, y=%.2f)'
        % (pedestrian_config["actor_id"], spawn_x, spawn_y)
    )


class ScriptedPedestrianController(object):
    """Client-side walker controller for scenes without SUMO pedestrian support."""

    def __init__(self, actor, pedestrian_config, reach_radius, default_speed_mps):
        self.actor = actor
        self.pedestrian_config = pedestrian_config
        self.reach_radius = max(reach_radius, 0.2)
        self.default_speed_mps = max(default_speed_mps, 0.1)
        self.reference_points = pedestrian_config.get("reference_points", [])
        self.target_index = 1 if len(self.reference_points) > 1 else 0
        self.finished = len(self.reference_points) <= 1

    def _stop(self):
        if self.actor is None or not self.actor.is_alive:
            return
        control = carla.WalkerControl()
        control.direction = carla.Vector3D(0.0, 0.0, 0.0)
        control.speed = 0.0
        control.jump = False
        self.actor.apply_control(control)

    def tick(self):
        if self.finished:
            self._stop()
            return

        if self.actor is None or not self.actor.is_alive:
            self.finished = True
            return

        actor_location = self.actor.get_location()
        while self.target_index < len(self.reference_points):
            target_point = self.reference_points[self.target_index]
            delta_x = float(target_point["carla_x"]) - actor_location.x
            delta_y = float(target_point["carla_y"]) - actor_location.y
            if math.hypot(delta_x, delta_y) > self.reach_radius:
                break
            self.target_index += 1

        if self.target_index >= len(self.reference_points):
            self.finished = True
            self._stop()
            return

        target_point = self.reference_points[self.target_index]
        delta_x = float(target_point["carla_x"]) - actor_location.x
        delta_y = float(target_point["carla_y"]) - actor_location.y
        distance = math.hypot(delta_x, delta_y)
        if distance < 1e-3:
            return

        control = carla.WalkerControl()
        control.direction = carla.Vector3D(delta_x / distance, delta_y / distance, 0.0)
        target_speed = max(float(target_point.get("speed_kmh", 0.0)) / 3.6, 0.0)
        control.speed = target_speed if target_speed > 0.1 else self.default_speed_mps
        control.jump = False
        self.actor.apply_control(control)

    def destroy(self):
        self._stop()
        if self.actor is not None and self.actor.is_alive:
            self.actor.destroy()


def spawn_scene_pedestrians(world, pedestrian_configs, reach_radius, default_speed_mps):
    controllers = []
    if not pedestrian_configs:
        return controllers

    logging.warning(
        'Scene contains %d pedestrians. SUMO pedestrian support is unavailable here, '
        'so they will be simulated directly in CARLA.',
        len(pedestrian_configs),
    )
    for pedestrian_config in pedestrian_configs:
        actor = spawn_scripted_pedestrian(world, pedestrian_config)
        controllers.append(
            ScriptedPedestrianController(
                actor,
                pedestrian_config,
                reach_radius=reach_radius,
                default_speed_mps=default_speed_mps,
            )
        )
    return controllers


def tick_scripted_pedestrians(controllers):
    for controller in controllers:
        controller.tick()


def active_scripted_pedestrian_count(controllers):
    count = 0
    for controller in controllers:
        if controller.finished:
            continue
        if controller.actor is None or not controller.actor.is_alive:
            continue
        count += 1
    return count


def all_scripted_pedestrians_finished(controllers):
    return active_scripted_pedestrian_count(controllers) == 0


def spawn_hero(world, hero_config, hero_blueprint, hero_z_override=None):
    blueprint = world.get_blueprint_library().find(hero_blueprint)
    blueprint.set_attribute('role_name', 'hero')
    if blueprint.has_attribute('color'):
        blueprint.set_attribute('color', '255,0,0')

    base_x = hero_config["carla_spawn"]["x"]
    base_y = hero_config["carla_spawn"]["y"]
    base_yaw = hero_config["carla_spawn"]["yaw"]
    base_z = hero_config["carla_spawn"]["z"] if hero_z_override is None else hero_z_override
    world_map = world.get_map()

    candidate_transforms = []
    seen_transforms = set()

    def add_candidate(transform):
        key = (
            round(transform.location.x, 2),
            round(transform.location.y, 2),
            round(transform.location.z, 2),
            round(transform.rotation.yaw, 2),
        )
        if key not in seen_transforms:
            seen_transforms.add(key)
            candidate_transforms.append(transform)

    waypoint = world_map.get_waypoint(
        carla.Location(x=base_x, y=base_y, z=0.0),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if waypoint is not None:
        # Prefer the local lane heading over the raw scene yaw so the ego faces along the lane.
        for z_offset in [0.8, 1.2, 2.0, 3.0, 5.0, base_z]:
            add_candidate(snap_transform(world_map, base_x, base_y, base_yaw, z_offset=z_offset))

        waypoint_candidates = [waypoint]
        for distance in [2.0, 4.0, 6.0, 8.0, 10.0]:
            waypoint_candidates.extend(waypoint.next(distance))
            waypoint_candidates.extend(waypoint.previous(distance))

        for candidate_waypoint in waypoint_candidates:
            for z_offset in [0.8, 1.2, 2.0, 3.0]:
                candidate_transform = candidate_waypoint.transform
                candidate_location = candidate_transform.location
                candidate_rotation = candidate_transform.rotation
                add_candidate(
                    carla.Transform(
                        carla.Location(
                            x=candidate_location.x,
                            y=candidate_location.y,
                            z=candidate_location.z + z_offset,
                        ),
                        carla.Rotation(
                            pitch=candidate_rotation.pitch,
                            yaw=candidate_rotation.yaw,
                            roll=candidate_rotation.roll,
                        ),
                    )
                )

    fallback_yaw = base_yaw
    if waypoint is not None:
        fallback_yaw = waypoint.transform.rotation.yaw
    add_candidate(
        carla.Transform(
            carla.Location(x=base_x, y=base_y, z=base_z),
            carla.Rotation(pitch=0.0, yaw=fallback_yaw, roll=0.0),
        )
    )

    for candidate_transform in candidate_transforms:
        hero = world.try_spawn_actor(blueprint, candidate_transform)
        if hero is not None:
            logging.info(
                'Spawned hero at x=%.2f y=%.2f z=%.2f yaw=%.2f',
                candidate_transform.location.x,
                candidate_transform.location.y,
                candidate_transform.location.z,
                candidate_transform.rotation.yaw,
            )
            return hero

    waypoint_location = 'none'
    if waypoint is not None:
        waypoint_location = '(%.2f, %.2f, %.2f)' % (
            waypoint.transform.location.x,
            waypoint.transform.location.y,
            waypoint.transform.location.z,
        )
    raise RuntimeError(
        'Could not spawn hero vehicle near scenario pose '
        '(x=%.2f, y=%.2f, z=%.2f, yaw=%.2f); nearest driving waypoint=%s'
        % (base_x, base_y, base_z, base_yaw, waypoint_location)
    )


def active_scripted_vehicle_count(scripted_vehicle_ids):
    current_ids = set(traci.vehicle.getIDList())
    return len(current_ids.intersection(scripted_vehicle_ids.keys()))


def all_scripted_vehicles_stopped_or_gone(scripted_vehicle_ids, stopped_vehicle_ids):
    current_ids = set(traci.vehicle.getIDList())
    for actor_id in scripted_vehicle_ids.keys():
        if actor_id in current_ids and actor_id not in stopped_vehicle_ids:
            return False
    return True


def choose_next_edge(current_edge, vclass, actor_id):
    outgoing_edges = list(current_edge.getAllowedOutgoing(vclass).keys())
    if not outgoing_edges:
        return None
    outgoing_edges.sort(key=lambda edge: edge.getID())
    if len(outgoing_edges) == 1:
        return outgoing_edges[0]
    stable_key = '%s:%s' % (actor_id, current_edge.getID())
    choice_index = sum(stable_key.encode('utf-8')) % len(outgoing_edges)
    return outgoing_edges[choice_index]


def set_vehicle_terminal_stop(actor_id, current_edge, lane_index):
    current_pos = traci.vehicle.getLanePosition(actor_id)
    edge_length = current_edge.getLength()
    stop_pos = min(edge_length - 0.5, max(current_pos + 1.0, edge_length - 3.0))
    stop_pos = max(stop_pos, min(current_pos + 0.5, edge_length - 0.1))
    traci.vehicle.setStop(
        actor_id,
        current_edge.getID(),
        pos=stop_pos,
        laneIndex=max(lane_index, 0),
        duration=10 ** 8,
    )
    traci.vehicle.setSpeed(actor_id, 0.0)


def schedule_terminal_stop(actor_id, vehicle_config, current_edge, stopped_vehicle_ids):
    current_pos = traci.vehicle.getLanePosition(actor_id)
    remaining_distance = current_edge.getLength() - current_pos
    lane_index = traci.vehicle.getLaneIndex(actor_id)

    if remaining_distance >= 12.0:
        try:
            set_vehicle_terminal_stop(actor_id, current_edge, lane_index)
            stopped_vehicle_ids.add(actor_id)
            return
        except traci.exceptions.TraCIException:
            pass

    next_edge = choose_next_edge(current_edge, vehicle_config["vclass"], actor_id)
    if next_edge is not None:
        traci.vehicle.setRoute(actor_id, [current_edge.getID(), next_edge.getID()])
        next_stop_pos = min(max(8.0, next_edge.getLength() * 0.25), max(next_edge.getLength() - 1.0, 1.0))
        try:
            traci.vehicle.setStop(
                actor_id,
                next_edge.getID(),
                pos=next_stop_pos,
                laneIndex=0,
                duration=10 ** 8,
            )
            stopped_vehicle_ids.add(actor_id)
            return
        except traci.exceptions.TraCIException as error:
            logging.warning('Failed to schedule terminal stop for %s on %s: %s', actor_id, next_edge.getID(), error)

    # Final fallback: do not crash the scenario if SUMO cannot accept a stop command here.
    try:
        traci.vehicle.slowDown(actor_id, 0.0, 2.0)
    except traci.exceptions.TraCIException:
        traci.vehicle.setSpeed(actor_id, 0.0)


def maintain_scripted_vehicle_behaviour(scripted_vehicle_ids, sumo_net, end_mode, stopped_vehicle_ids):
    current_vehicle_ids = set(traci.vehicle.getIDList())
    for actor_id, vehicle_config in scripted_vehicle_ids.items():
        if actor_id not in current_vehicle_ids:
            continue

        route = traci.vehicle.getRoute(actor_id)
        route_index = traci.vehicle.getRouteIndex(actor_id)
        if not route or route_index != len(route) - 1:
            continue

        current_edge = sumo_net.getEdge(route[route_index])
        if end_mode == 'continue':
            next_edge = choose_next_edge(current_edge, vehicle_config["vclass"], actor_id)
            if next_edge is not None:
                traci.vehicle.setRoute(actor_id, [current_edge.getID(), next_edge.getID()])
                if actor_id in stopped_vehicle_ids:
                    stopped_vehicle_ids.remove(actor_id)
                continue

        if actor_id in stopped_vehicle_ids:
            continue

        schedule_terminal_stop(
            actor_id,
            vehicle_config,
            current_edge,
            stopped_vehicle_ids,
        )


def run(args):
    scene = load_scene(args.scenario_file)
    scenario_name = os.path.splitext(os.path.basename(args.scenario_file))[0]
    scene_vehicles, scene_pedestrians = split_scene_background_actors(scene["background_vehicles"])
    max_sim_time = None
    if args.max_sim_time < 0:
        max_sim_time = scene.get("scenario_duration_seconds", 0.0) * args.scenario_duration_scale + args.scenario_grace_period
    elif args.max_sim_time > 0:
        max_sim_time = args.max_sim_time

    pygame.init()
    display = pygame.display.set_mode((args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption('Tongji SUMO Manual Control (Entity Scene)')
    clock = pygame.time.Clock()

    hero_actor = None
    hero_camera = None
    hero_collision_monitor = None
    synchronization = None
    scripted_pedestrian_controllers = []

    carla_simulation = CarlaSimulation(args.host, args.port, args.step_length)
    if args.reload_world:
        logging.info('Reloading CARLA world before scenario start')
        maybe_reload_world(carla_simulation)
    if args.cleanup_existing_actors:
        cleanup_existing_actors(carla_simulation)
    configure_world(carla_simulation, scene.get("weather", {}))

    basedir = os.path.dirname(os.path.realpath(__file__))
    net_file = os.path.join(basedir, 'tongji', 'tongji.net.xml')
    cfg_file = os.path.join(basedir, 'tongji', 'tongji.sumocfg')
    vtypes_file = os.path.join(basedir, 'tongji', 'carlavtypes13.rou.xml')
    viewsettings_file = os.path.join(basedir, 'tongji', 'viewsettings.xml')
    sumo_net = sumolib.net.readNet(net_file)
    write_sumocfg_xml(cfg_file, net_file, vtypes_file, viewsettings_file, args.additional_traci_clients)

    sumo_simulation = SumoSimulation(
        cfg_file,
        args.step_length,
        host=args.sumo_host,
        port=args.sumo_port,
        sumo_gui=args.sumo_gui,
        client_order=args.client_order,
        suppress_warnings=args.suppress_sumo_warnings,
    )
    synchronization = SimulationSynchronization(
        sumo_simulation,
        carla_simulation,
        args.tls_manager,
        args.sync_vehicle_color,
        args.sync_vehicle_lights,
    )

    try:
        scripted_vehicle_ids = spawn_scene_background_vehicles(scene_vehicles)
        stopped_vehicle_ids = set()
        hero_blueprint = args.hero_blueprint or scene["hero"]["sumo_type_id"]
        hero_actor = spawn_hero(carla_simulation.world, scene["hero"], hero_blueprint, hero_z_override=args.hero_z)
        hero_actor.set_autopilot(False)
        if args.end_on_hero_collision:
            hero_collision_monitor = HeroCollisionMonitor(hero_actor)
        scripted_pedestrian_controllers = spawn_scene_pedestrians(
            carla_simulation.world,
            scene_pedestrians,
            reach_radius=args.pedestrian_reach_radius,
            default_speed_mps=args.pedestrian_default_speed,
        )
        scenario_start_time = carla_simulation.world.get_snapshot().timestamp.elapsed_seconds
        destination_point, via_points = get_ego_navigation_display(scene["hero"], args.ego_via_point_stride)
        navigation_summary = build_navigation_summary(destination_point, via_points)
        next_marker_draw_time = 0.0
        offroad_since_time = None
        world_map = carla_simulation.world.get_map()

        if args.control_mode == 'g29':
            controller = G29HeroController(args.wheel_config)
        else:
            controller = ManualHeroController()
        hero_camera = HeroCamera(hero_actor, args.width, args.height, initial_view=args.camera_view)
        overlay = Overlay(args.width, args.height, scenario_name)

        while True:
            start = time.time()
            clock.tick()

            if controller.parse_events(hero_camera):
                break
            if not hero_actor.is_alive:
                raise RuntimeError('Hero vehicle was destroyed')

            world_time = carla_simulation.world.get_snapshot().timestamp.elapsed_seconds
            sim_time = world_time - scenario_start_time
            if max_sim_time is not None and sim_time >= max_sim_time:
                logging.info('Reached max simulation time %.2f s', max_sim_time)
                break

            if (
                (args.show_ego_destination or args.show_ego_via_points)
                and sim_time >= next_marker_draw_time
            ):
                draw_ego_navigation_markers(
                    carla_simulation.world,
                    destination_point,
                    via_points,
                    args.show_ego_destination,
                    args.show_ego_via_points,
                    args.ego_marker_life_time,
                )
                next_marker_draw_time = sim_time + args.ego_marker_refresh_interval

            controller.apply(hero_actor, max(clock.get_time(), int(args.step_length * 1000)))
            tick_scripted_pedestrians(scripted_pedestrian_controllers)
            maintain_scripted_vehicle_behaviour(
                scripted_vehicle_ids,
                sumo_net,
                args.scripted_vehicle_end_mode,
                stopped_vehicle_ids,
            )
            synchronization.tick()
            tick_scripted_pedestrians(scripted_pedestrian_controllers)
            maintain_scripted_vehicle_behaviour(
                scripted_vehicle_ids,
                sumo_net,
                args.scripted_vehicle_end_mode,
                stopped_vehicle_ids,
            )

            if (
                args.end_on_hero_collision
                and hero_collision_monitor is not None
                and hero_collision_monitor.collided
            ):
                logging.info(
                    'Hero collided with %s (impulse %.3f)',
                    hero_collision_monitor.last_other_actor,
                    hero_collision_monitor.last_intensity,
                )
                break

            if args.end_on_hero_offroad:
                if hero_is_offroad(world_map, hero_actor):
                    if offroad_since_time is None:
                        offroad_since_time = sim_time
                    elif sim_time - offroad_since_time >= args.hero_offroad_grace_time:
                        logging.info(
                            'Hero left the driving lane for %.2f s',
                            sim_time - offroad_since_time,
                        )
                        break
                else:
                    offroad_since_time = None

            if (
                args.end_on_ego_destination
                and hero_reached_destination(hero_actor, destination_point, args.ego_destination_radius)
            ):
                logging.info(
                    'Hero reached destination within %.2f m',
                    args.ego_destination_radius,
                )
                break

            if args.end_on_scenario_complete:
                pedestrians_finished = all_scripted_pedestrians_finished(scripted_pedestrian_controllers)
                if args.scripted_vehicle_end_mode == 'stop':
                    vehicles_finished = all_scripted_vehicles_stopped_or_gone(
                        scripted_vehicle_ids,
                        stopped_vehicle_ids,
                    )
                    if vehicles_finished and pedestrians_finished:
                        logging.info('All scripted scene actors reached their terminal states')
                        break
                elif active_scripted_vehicle_count(scripted_vehicle_ids) == 0 and pedestrians_finished:
                    logging.info('All scripted scene actors completed their routes')
                    break

            if args.follow_spectator:
                set_spectator(
                    carla_simulation,
                    hero_actor,
                    view_mode=args.spectator_view,
                    height=args.spectator_height,
                )

            display.fill((0, 0, 0))
            hero_camera.render(display)
            overlay.render(
                display,
                hero_actor,
                len(carla_simulation.world.get_actors().filter('vehicle.*'))
                + len(carla_simulation.world.get_actors().filter('walker.pedestrian.*')),
                active_scripted_vehicle_count(scripted_vehicle_ids) + active_scripted_pedestrian_count(scripted_pedestrian_controllers),
                sim_time,
                args.control_mode,
                max_sim_time,
                hero_camera.view_name,
                navigation_summary,
            )
            pygame.display.flip()

            elapsed = time.time() - start
            if elapsed < args.step_length:
                time.sleep(args.step_length - elapsed)

    except KeyboardInterrupt:
        logging.info('Cancelled by user.')
    finally:
        if hero_camera is not None:
            hero_camera.destroy()
        if hero_collision_monitor is not None:
            hero_collision_monitor.destroy()
        for controller in scripted_pedestrian_controllers:
            controller.destroy()
        if synchronization is None and hero_actor is not None and hero_actor.is_alive:
            hero_actor.destroy()
        if synchronization is not None:
            synchronization.close()
        pygame.quit()


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('scenario_file', help='Converted scene JSON produced from entity.json')
    argparser.add_argument('--host', default='127.0.0.1', help='CARLA host')
    argparser.add_argument('--port', default=2000, type=int, help='CARLA port')
    argparser.add_argument('--reload-world', action='store_true', help='Reload CARLA world before running the scenario')
    argparser.add_argument('--cleanup-existing-actors', action='store_true', default=True, help='Destroy pre-existing vehicles/sensors/walkers before running the scenario')
    argparser.add_argument('--no-cleanup-existing-actors', dest='cleanup_existing_actors', action='store_false', help='Keep pre-existing CARLA actors')
    argparser.add_argument('--sumo-host', default=None, help='External SUMO host')
    argparser.add_argument('--sumo-port', default=None, type=int, help='External SUMO port')
    argparser.add_argument('--sumo-gui', action='store_true', help='Run SUMO with GUI')
    argparser.add_argument('--suppress-sumo-warnings', action='store_true', default=True, help='Suppress SUMO warning output when launching a local SUMO server')
    argparser.add_argument('--show-sumo-warnings', dest='suppress_sumo_warnings', action='store_false', help='Show SUMO warning output')
    argparser.add_argument('--step-length', default=0.05, type=float, help='Fixed simulation step')
    argparser.add_argument('--additional-traci-clients', default=0, type=int, help='Additional TraCI clients')
    argparser.add_argument('--client-order', default=1, type=int, help='TraCI client order')
    argparser.add_argument('--sync-vehicle-lights', action='store_true', help='Sync vehicle lights')
    argparser.add_argument('--sync-vehicle-color', action='store_true', help='Sync vehicle color')
    argparser.add_argument('--sync-vehicle-all', action='store_true', help='Sync all vehicle properties')
    argparser.add_argument(
        '--tls-manager',
        choices=['none', 'sumo', 'carla'],
        default='none',
        help='Traffic light manager',
    )
    argparser.add_argument('--hero-blueprint', default=None, help='Optional override for hero blueprint')
    argparser.add_argument('--hero-z', default=None, type=float, help='Optional override for hero spawn z')
    argparser.add_argument('--control-mode', choices=['keyboard', 'g29'], default='keyboard', help='Hero input device')
    argparser.add_argument('--camera-view', choices=['chase', 'hood', 'high_follow', 'bev'], default='chase', help='Initial hero camera view')
    argparser.add_argument('--show-ego-destination', action='store_true', help='Draw the ego destination marker from scene navigation data')
    argparser.add_argument('--show-ego-via-points', action='store_true', help='Draw the ego via-point markers from scene navigation data')
    argparser.add_argument('--ego-via-point-stride', default=1, type=int, help='Display every Nth via point when deriving markers from reference_points')
    argparser.add_argument('--ego-marker-life-time', default=2.0, type=float, help='Lifetime in seconds for ego navigation markers')
    argparser.add_argument('--ego-marker-refresh-interval', default=1.0, type=float, help='Refresh interval in seconds for ego navigation markers')
    argparser.add_argument('--end-on-ego-destination', action='store_true', default=False, help='Stop the scenario when the ego reaches the destination vicinity')
    argparser.add_argument('--no-end-on-ego-destination', dest='end_on_ego_destination', action='store_false', help='Do not stop the scenario when the ego reaches the destination')
    argparser.add_argument('--ego-destination-radius', default=8.0, type=float, help='Meters around the destination that count as ego arrival, regardless of lane')
    argparser.add_argument('--wheel-config', default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'wheel_config.ini'), help='Wheel config ini for G29 mode')
    argparser.add_argument('--width', default=1280, type=int, help='Display width')
    argparser.add_argument('--height', default=720, type=int, help='Display height')
    argparser.add_argument('--follow-spectator', action='store_true', help='Follow hero with spectator')
    argparser.add_argument('--spectator-view', choices=['chase', 'bev'], default='bev', help='CARLA server spectator follow mode')
    argparser.add_argument('--spectator-height', default=55.0, type=float, help='Height in meters for BEV spectator follow')
    argparser.add_argument('--scripted-vehicle-end-mode', choices=['continue', 'stop'], default='continue', help='Behaviour when scripted SUMO vehicles reach the end of their planned route')
    argparser.add_argument('--pedestrian-default-speed', default=1.2, type=float, help='Fallback pedestrian speed in m/s when trajectory points provide no speed')
    argparser.add_argument('--pedestrian-reach-radius', default=0.9, type=float, help='Meters used to advance scripted pedestrian waypoint targets')
    argparser.add_argument('--end-on-hero-collision', action='store_true', default=True, help='Stop the scenario when the hero collides with another actor or object')
    argparser.add_argument('--no-end-on-hero-collision', dest='end_on_hero_collision', action='store_false', help='Do not stop the scenario on hero collision')
    argparser.add_argument('--end-on-hero-offroad', action='store_true', default=True, help='Stop the scenario when the hero leaves driving lanes')
    argparser.add_argument('--no-end-on-hero-offroad', dest='end_on_hero_offroad', action='store_false', help='Do not stop the scenario when the hero leaves driving lanes')
    argparser.add_argument('--hero-offroad-grace-time', default=0.5, type=float, help='Seconds the hero may stay outside driving lanes before the scenario stops')
    argparser.add_argument('--end-on-scenario-complete', action='store_true', help='Stop once scripted SUMO actors complete their routes')
    argparser.add_argument('--max-sim-time', default=-1.0, type=float, help='Seconds before auto-stop; 0 disables; negative uses scene duration * scale + grace')
    argparser.add_argument('--scenario-duration-scale', default=2.0, type=float, help='Multiplier applied to scene_duration_seconds when --max-sim-time is negative')
    argparser.add_argument('--scenario-grace-period', default=0.0, type=float, help='Extra seconds added when --max-sim-time is negative')
    argparser.add_argument('--debug', action='store_true', help='Enable debug logs')
    args = argparser.parse_args()

    if args.sync_vehicle_all:
        args.sync_vehicle_lights = True
        args.sync_vehicle_color = True

    logging.basicConfig(
        format='%(levelname)s: %(message)s',
        level=logging.DEBUG if args.debug else logging.INFO,
    )
    run(args)


if __name__ == '__main__':
    main()
