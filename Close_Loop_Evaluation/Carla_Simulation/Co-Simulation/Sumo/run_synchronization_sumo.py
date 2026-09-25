#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""
Script to integrate CARLA and SUMO simulations
"""

# ==================================================================================================
# -- imports ---------------------------------------------------------------------------------------
# ==================================================================================================
import random
import pandas as pd
import argparse
import logging
import time
import numpy as np
from carla import Transform, Location, Rotation
import carla
import cv2
from queue import Queue, Empty
import copy
from datetime import datetime
import json
from pyproj import Proj
import asyncio
import json
import time
import threading


# ==================================================================================================
# -- find carla module -----------------------------------------------------------------------------
# ==================================================================================================

import glob
import os
import sys

try:
    sys.path.append(
        glob.glob('../../PythonAPI/carla/dist/carla-*%d.%d-%s.egg' %
                  (sys.version_info.major, sys.version_info.minor,
                   'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==================================================================================================
# -- find traci module -----------------------------------------------------------------------------
# ==================================================================================================

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import traci

# ==================================================================================================
# -- sumo integration imports ----------------------------------------------------------------------
# ==================================================================================================

from sumo_integration.bridge_helper import BridgeHelper  # pylint: disable=wrong-import-position
from sumo_integration.carla_simulation import CarlaSimulation  # pylint: disable=wrong-import-position
from sumo_integration.constants import INVALID_ACTOR_ID  # pylint: disable=wrong-import-position
from sumo_integration.sumo_simulation import SumoSimulation  # pylint: disable=wrong-import-position

type_dic = {'PASSENGER': 4, 'TAXI': 4, 'MOTORCYCLE': 202, 'TRUCK': 603}
color_dic = {'PASSENGER': 6, 'TAXI': 5, 'BUS': 1, 'TRUCK': 3}

import math
# 计算两点之间的距离

def eucliDist(A, B):
    return math.sqrt(sum([(a - b)**2 for (a, b) in zip(A, B)]))

# ==================================================================================================
# -- synchronization_loop --------------------------------------------------------------------------
# ==================================================================================================


def points2pcd(PCD_FILE_PATH, points):
    # 存放路径
    # PCD_DIR_PATH = os.path.join(os.path.abspath('.'), 'pcd')
    # PCD_FILE_PATH = os.path.join(PCD_DIR_PATH, 'cache.pcd')
    if os.path.exists(PCD_FILE_PATH):
        os.remove(PCD_FILE_PATH)

    # 写文件句柄
    handle = open(PCD_FILE_PATH, 'a')

    # 得到点云点数
    point_num = points.shape[0]

    # pcd头部（重要）
    handle.write(
        '# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1')
    string = '\nWIDTH ' + str(point_num)
    handle.write(string)
    handle.write('\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0')
    string = '\nPOINTS ' + str(point_num)
    handle.write(string)
    handle.write('\nDATA ascii')

    # 依次写入点
    for i in range(point_num):
        string = '\n' + str(points[i, 0]) + ' ' + str(points[i, 1]) + ' ' + str(points[i, 2]) + ' ' + str(points[i, 3])
        handle.write(string)
    handle.close()


def get_time_stamp(ct):
    """

    :param ct: float时间
    :return: 带毫秒的格式化时间戳
    """
    # ct = time.time()
    # print(ct)
    local_time = time.localtime(ct)
    # data_head = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    data_head = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    data_secs = (ct - int(ct)) * 1000
    time_stamp = "%s.%03d" % (data_head, data_secs)
    # print(time_stamp, type(time_stamp))
    # stamp = ("".join(time_stamp.split()[0].split("-"))+"".join(time_stamp.split()[1].split(":"))).replace('.', '')
    # print(stamp)
    return time_stamp


def draw_waypoints(world, waypoints, road_id=None, node_num=0, life_time=150.0):
    """
    :param waypoints: 地图所有航点列表
    :param road_id: 目标路段id
    :param life_time: 高亮时间
    :return:
    """
    obj_waypoints = []

    for waypoint in waypoints:
        if waypoint.road_id == road_id:
            obj_waypoints.append(waypoint)

    # filtered_waypoints = draw_waypoints(world, waypoints, road_id=obj_roadid, life_time=200)  # 绘制指定路段的航点
    waypoints_num = len(obj_waypoints)
    if node_num:
        spawn_point = obj_waypoints[node_num].transform  # 获取第一个航点的位置
    else:
        spawn_point = obj_waypoints[int(waypoints_num / 2)].transform  # 获取第一个航点的位置
    spawn_point.location.z += 2  # 坐标高度增加2，避免车辆与地面碰撞报错
    return spawn_point


def sensor_callback(sensor_data, sensor_queue, sensor_name):
    # Do stuff with the sensor_data data like save it to disk
    # Then you just need to add to the queue
    sensor_queue.put((sensor_data.frame, sensor_name, sensor_data))


# modify from manual control
def _parse_image_cb(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array


# modify from leaderboard
def _parse_lidar_cb(lidar_data):
    points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
    points = copy.deepcopy(points)
    points = np.reshape(points, (int(points.shape[0] / 4), 4))
    return points


# modify from world on rail code
def lidar_to_bev(lidar, min_x=-24, max_x=24, min_y=-16, max_y=16, pixels_per_meter=4, hist_max_per_pixel=10):
    xbins = np.linspace(
        min_x, max_x + 1,
               (max_x - min_x) * pixels_per_meter + 1,
    )
    ybins = np.linspace(
        min_y, max_y + 1,
               (max_y - min_y) * pixels_per_meter + 1,
    )
    # Compute histogram of x and y coordinates of points.
    hist = np.histogramdd(lidar[..., :2], bins=(xbins, ybins))[0]
    # Clip histogram
    hist[hist > hist_max_per_pixel] = hist_max_per_pixel
    # Normalize histogram by the maximum number of points in a bin we care about.
    overhead_splat = hist / hist_max_per_pixel * 255.
    # Return splat in X x Y orientation, with X parallel to car axis, Y perp, both parallel to ground.
    # print(overhead_splat.shape)
    # print(lidar.shape)
    # print(lidar)
    # print(overhead_splat)
    # print(overhead_splat[::-1, :])
    return overhead_splat[::-1, :]
    # return overhead_splat


# modify from world on rail code
def visualize_data(rgb, lidars, text_args=(cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)):
    # print(type(rgb),rgb.shape)
    rgb_canvas = np.array(rgb[..., ::-1])
    # print(type(rgb_canvas), rgb_canvas.shape)
    # print(rgb_canvas.shape, rgb_canvas.size)
    canvas_list = []
    lidar_canvas = None

    if lidars is not None:
        for lidar in lidars:
            lidar_viz = lidar_to_bev(lidar).astype(np.uint8)
            lidar_viz = cv2.cvtColor(lidar_viz, cv2.COLOR_GRAY2RGB)
            canvas = cv2.resize(lidar_viz.astype(np.uint8), (rgb_canvas.shape[0], rgb_canvas.shape[0]))
            canvas_list.append(canvas)
        lidar_canvas = np.concatenate(canvas_list, axis=1)
    # cv2.putText(canvas, f'yaw angle: {imu_yaw:.3f}', (4, 10), *text_args)
    # cv2.putText(canvas, f'log: {gnss[0]:.3f} alt: {gnss[1]:.3f} brake: {gnss[2]:.3f}', (4, 20), *text_args)
    return lidar_canvas


def mkdir_folder(path):
    for s_type in sensor_type:
        if not os.path.isdir(os.path.join(path, s_type)):
            os.makedirs(os.path.join(path, s_type))
    return True


class SimulationSynchronization(object):
    """
    SimulationSynchronization class is responsible for the synchronization(同步) of sumo and carla
    simulations.
    """

    def __init__(self,
                 sumo_simulation,
                 carla_simulation,
                 tls_manager='none',
                 sync_vehicle_color=False,
                 sync_vehicle_lights=False,
                 real_ids=None):

        if real_ids is None:
            real_ids = {}
        self.sumo = sumo_simulation
        self.carla = carla_simulation

        self.tls_manager = tls_manager
        self.sync_vehicle_color = sync_vehicle_color
        self.sync_vehicle_lights = sync_vehicle_lights

        if tls_manager == 'carla':
            self.sumo.switch_off_traffic_lights()  # SUMO的红绿灯冻结然后全绿
        elif tls_manager == 'sumo':
            self.carla.switch_off_traffic_lights()  # CARLA的红绿灯冻结然后全绿

        # Mapped actor ids.
        self.sumo2carla_ids = {}  # Contains only actors controlled by sumo. 仅包含由SUMO控制的actors
        self.carla2sumo_ids = {}  # Contains only actors controlled by carla.  仅包含由CARLA控制的actors
        self.inser_ids = {}  # Contains only actors controlled by real_lidar.       仅包含由real_lidar控制的actors

        BridgeHelper.blueprint_library = self.carla.world.get_blueprint_library()
        # 返回carla里面可用的actor蓝图列表，以简化这些蓝图的生成

        BridgeHelper.offset = self.sumo.get_net_offset()
        # 从地理坐标转换为 UTM 后要添加的偏移量 是一个set(x,y)

        # Configuring carla simulation in sync mode.
        settings = self.carla.world.get_settings()
        settings.synchronous_mode = True  # 开启CARLA的同步模式
        settings.fixed_delta_seconds = self.carla.step_length  # CARLA仿真步长 仅限同步模式下
        self.carla.world.apply_settings(settings)

        # ===========================实例化传感器模型============================
        cam_bp = BridgeHelper.blueprint_library.find('sensor.camera.rgb')  # 相机
        lidar_bp_16 = BridgeHelper.blueprint_library.find('sensor.lidar.ray_cast')  # 雷达
        lidar_bp_8 = BridgeHelper.blueprint_library.find('sensor.lidar.ray_cast')  # 雷达
        lidar_bp_4_1 = BridgeHelper.blueprint_library.find('sensor.lidar.ray_cast')  # 雷达
        lidar_bp_4_2 = BridgeHelper.blueprint_library.find('sensor.lidar.ray_cast')  # 雷达
        lidar_bp_32 = BridgeHelper.blueprint_library.find('sensor.lidar.ray_cast')  # 雷达
        # gnss_bp = world.get_blueprint_library().find('sensor.other.gnss')
        # imu_bp = world.get_blueprint_library().find('sensor.other.imu')

        # =======================================设置actor模型==================================
        # ===================================传感器

        # 设置传感器的参数 set the attribute of camera
        cam_bp.set_attribute("image_size_x", "{}".format(IM_WIDTH))
        cam_bp.set_attribute("image_size_y", "{}".format(IM_HEIGHT))
        cam_bp.set_attribute("fov", "90")
        cam_bp.set_attribute('sensor_tick', '0.1')

        lidar_bp_32.set_attribute('channels', '32')
        lidar_bp_32.set_attribute('upper_fov', '0')
        lidar_bp_32.set_attribute('lower_fov', '-37')
        lidar_bp_32.set_attribute('dropoff_general_rate', '0.0')
        lidar_bp_32.set_attribute('points_per_second', '576000')
        lidar_bp_32.set_attribute('range', '100')
        lidar_bp_32.set_attribute('rotation_frequency', str(int(1 / settings.fixed_delta_seconds)))

        lidar_bp_16.set_attribute('channels', '16')
        lidar_bp_16.set_attribute('upper_fov', '0')
        lidar_bp_16.set_attribute('lower_fov', '-9')
        lidar_bp_16.set_attribute('dropoff_general_rate', '0.0')
        lidar_bp_16.set_attribute('points_per_second', '288000')
        lidar_bp_16.set_attribute('range', '100')
        lidar_bp_16.set_attribute('rotation_frequency', str(int(1 / settings.fixed_delta_seconds)))

        lidar_bp_8.set_attribute('channels', '8')
        lidar_bp_8.set_attribute('upper_fov', '-10')
        lidar_bp_8.set_attribute('lower_fov', '-17')
        lidar_bp_8.set_attribute('dropoff_general_rate', '0.0')
        lidar_bp_8.set_attribute('points_per_second', '144000')
        lidar_bp_8.set_attribute('range', '100')
        lidar_bp_8.set_attribute('rotation_frequency', str(int(1 / settings.fixed_delta_seconds)))

        lidar_bp_4_1.set_attribute('channels', '4')
        lidar_bp_4_1.set_attribute('upper_fov', '-19')
        lidar_bp_4_1.set_attribute('lower_fov', '-25')
        lidar_bp_4_1.set_attribute('dropoff_general_rate', '0.0')
        lidar_bp_4_1.set_attribute('points_per_second', '72000')
        lidar_bp_4_1.set_attribute('range', '100')
        lidar_bp_4_1.set_attribute('rotation_frequency', str(int(1 / settings.fixed_delta_seconds)))

        lidar_bp_4_2.set_attribute('channels', '4')
        lidar_bp_4_2.set_attribute('upper_fov', '-28')
        lidar_bp_4_2.set_attribute('lower_fov', '-37')
        lidar_bp_4_2.set_attribute('dropoff_general_rate', '0.0')
        lidar_bp_4_2.set_attribute('points_per_second', '72000')
        lidar_bp_4_2.set_attribute('range', '100')
        lidar_bp_4_2.set_attribute('rotation_frequency', str(int(1 / settings.fixed_delta_seconds)))

        world = self.carla.world

        cam01 = world.spawn_actor(cam_bp, camera01_trans, attach_to=None)
        cam02 = world.spawn_actor(cam_bp, camera02_trans, attach_to=None)

        cam01.listen(lambda data: sensor_callback(data, sensor_queue, "rgb_camera01"))
        cam02.listen(lambda data: sensor_callback(data, sensor_queue, "rgb_camera02"))
        sensor_list.append(cam01)
        sensor_list.append(cam02)

        lidar_16 = world.spawn_actor(lidar_bp_16, lidar_trans, attach_to=None)
        lidar_8 = world.spawn_actor(lidar_bp_8, lidar_trans, attach_to=None)
        lidar_4_1 = world.spawn_actor(lidar_bp_4_1, lidar_trans, attach_to=None)
        lidar_4_2 = world.spawn_actor(lidar_bp_4_2, lidar_trans, attach_to=None)
        lidar_32 = world.spawn_actor(lidar_bp_32, lidar_trans, attach_to=None)
        # lidar_32 = world.spawn_actor(lidar_bp_32, carla.Transform(carla.Location(z=4)), attach_to = ego_vehicle)

        lidar_16.listen(lambda data: sensor_callback(data, sensor_queue, "lidar_16"))
        lidar_8.listen(lambda data: sensor_callback(data, sensor_queue, "lidar_8"))
        lidar_4_1.listen(lambda data: sensor_callback(data, sensor_queue, "lidar_4_1"))
        lidar_4_2.listen(lambda data: sensor_callback(data, sensor_queue, "lidar_4_2"))
        lidar_32.listen(lambda data: sensor_callback(data, sensor_queue, "lidar_32"))
        # lidar01.listen(lambda data: data.save_to_disk('output/%06d.ply' % data.frame))
        sensor_list.append(lidar_16)
        sensor_list.append(lidar_8)
        sensor_list.append(lidar_4_1)
        sensor_list.append(lidar_4_2)
        sensor_list.append(lidar_32)

        traffic_manager = self.carla.client.get_trafficmanager()  # 开启Carla的交通管理器 同步模式为true
        traffic_manager.set_synchronous_mode(True)
        # tm_port = traffic_manager.get_port()
        #20230315 不知道为啥要创建该车辆
        # blueprint_library = self.carla.world.get_blueprint_library()
        # model3_bp = random.choice(BridgeHelper.blueprint_library.filter('vehicle.bmw.*'))
        # spawn_point1 = Transform(Location(x=-18.008675, y=-476.696991, z=40.156261),
        #                         Rotation(pitch=-0.113770, yaw=-14.639460, roll=0.000000))
        # # spawn_point2 = Transform(Location(x=-35.960468, y=-503.584290, z=40.109402),
        # #                          Rotation(pitch=0.362881, yaw=83.129738, roll=0.000000))

        # model31 = self.carla.world.spawn_actor(model3_bp, spawn_point1)
        # # model32 = self.carla.world.spawn_actor(model3_bp, spawn_point2)

    #synchronization_loop()的核心部分，大体分为五步。
    # 主要包括仿真更新、新角色生成、旧角色删除、更新现有角色位置、更新红绿灯状态。
    def tick(self):
        """
        Tick to simulation synchronization
        """
        # print("Tick to simulation synchronization")
        # -----------------
        # sumo-->carla sync
        # -----------------
        
        self.sumo.tick() # 更新上次仿真的结果

        # self.sumo.rou

        # self.sumo.add()

        # Spawning new sumo actors in carla (i.e, not controlled by carla).
        sumo_spawned_actors = self.sumo.spawned_actors - set(self.carla2sumo_ids.values())
        print("sumo.spawned_actors", self.sumo.spawned_actors)
        print("carla2sumo_ids.values()", self.carla2sumo_ids.values())
        print("sumo_spawned_actors", sumo_spawned_actors)

        # 当前时间步中 SUMO插入路网中的车辆id  减去 已经插入CARLA中的车辆id
        for sumo_actor_id in sumo_spawned_actors:
            self.sumo.subscribe(sumo_actor_id)  # 订阅车辆的相关信息（位置车型颜色大小） （sub可以提高检索的效率）
            sumo_actor = self.sumo.get_actor(sumo_actor_id)  # 获取车辆的相关信息

            carla_blueprint = BridgeHelper.get_carla_blueprint(sumo_actor, self.sync_vehicle_color)
            print(type(carla_blueprint), carla_blueprint)
            if carla_blueprint is not None:
                # sumo坐标转换为carla的坐标
                carla_transform = BridgeHelper.get_carla_transform(sumo_actor.transform,
                                                                   sumo_actor.extent)

                carla_actor_id = self.carla.spawn_actor(carla_blueprint, carla_transform)  # carla生成车辆
                if carla_actor_id != INVALID_ACTOR_ID:
                    self.sumo2carla_ids[sumo_actor_id] = carla_actor_id
            else:
                self.sumo.unsubscribe(sumo_actor_id)

        # Destroying sumo arrived actors in carla.
        for sumo_actor_id in self.sumo.destroyed_actors:
            if sumo_actor_id in self.sumo2carla_ids:
                self.carla.destroy_actor(self.sumo2carla_ids.pop(sumo_actor_id))

        # Updating sumo actors in carla.
        for sumo_actor_id in self.sumo2carla_ids:
            carla_actor_id = self.sumo2carla_ids[sumo_actor_id]

            sumo_actor = self.sumo.get_actor(sumo_actor_id)
            carla_actor = self.carla.get_actor(carla_actor_id)

            carla_transform = BridgeHelper.get_carla_transform(sumo_actor.transform,
                                                               sumo_actor.extent)
            if self.sync_vehicle_lights:  # 控制车灯
                carla_lights = BridgeHelper.get_carla_lights_state(carla_actor.get_light_state(),
                                                                   sumo_actor.signals)
            else:
                carla_lights = None

            self.carla.synchronize_vehicle(carla_actor_id, carla_transform, carla_lights)  # 更新车灯

        # Updates traffic lights in carla based on sumo information.
        if self.tls_manager == 'sumo':
            common_landmarks = self.sumo.traffic_light_ids & self.carla.traffic_light_ids
            for landmark_id in common_landmarks:
                sumo_tl_state = self.sumo.get_traffic_light_state(landmark_id)
                carla_tl_state = BridgeHelper.get_carla_traffic_light_state(sumo_tl_state)

                self.carla.synchronize_traffic_light(landmark_id, carla_tl_state)  # 更新信号灯状态

        # -----------------
        # carla-->sumo sync
        # -----------------
        self.carla.tick()

        # Spawning new carla actors (not controlled by sumo)

        carla_spawned_actors = self.carla.spawned_actors - set(self.sumo2carla_ids.values())
        # print("carla.spawned_actors", self.carla.spawned_actors)
        # print("sumo2carla_ids.values()", self.sumo2carla_ids.values())
        # print("carla_spawned_actors", carla_spawned_actors)

        for carla_actor_id in carla_spawned_actors:
            carla_actor = self.carla.get_actor(carla_actor_id)
            type_id = BridgeHelper.get_sumo_vtype(carla_actor)
            color = carla_actor.attributes.get('color', None) if self.sync_vehicle_color else None
            if type_id is not None:
                sumo_actor_id = self.sumo.spawn_actor(type_id, color)
                if sumo_actor_id != INVALID_ACTOR_ID:
                    self.carla2sumo_ids[carla_actor_id] = sumo_actor_id
                    self.sumo.subscribe(sumo_actor_id)

        # Destroying required carla actors in sumo.
        for carla_actor_id in self.carla.destroyed_actors:
            if carla_actor_id in self.carla2sumo_ids:
                self.sumo.destroy_actor(self.carla2sumo_ids.pop(carla_actor_id))

        # Updating carla actors in sumo.
        for carla_actor_id in self.carla2sumo_ids:
            sumo_actor_id = self.carla2sumo_ids[carla_actor_id]

            carla_actor = self.carla.get_actor(carla_actor_id)
            sumo_actor = self.sumo.get_actor(sumo_actor_id)
            # carla坐标转换为sumo的坐标
            sumo_transform = BridgeHelper.get_sumo_transform(carla_actor.get_transform(),
                                                             carla_actor.bounding_box.extent)
            if self.sync_vehicle_lights:
                carla_lights = self.carla.get_actor_light_state(carla_actor_id)
                if carla_lights is not None:
                    sumo_lights = BridgeHelper.get_sumo_lights_state(sumo_actor.signals,
                                                                     carla_lights)
                else:
                    sumo_lights = None
            else:
                sumo_lights = None

            self.sumo.synchronize_vehicle(sumo_actor_id, sumo_transform, sumo_lights)

        # Updates traffic lights in sumo based on carla information.
        if self.tls_manager == 'carla':
            common_landmarks = self.sumo.traffic_light_ids & self.carla.traffic_light_ids
            for landmark_id in common_landmarks:
                carla_tl_state = self.carla.get_traffic_light_state(landmark_id)
                sumo_tl_state = BridgeHelper.get_sumo_traffic_light_state(carla_tl_state)
                # Updates all the sumo links related to this landmark.
                self.sumo.synchronize_traffic_light(landmark_id, sumo_tl_state)

    def close(self):
        """
        Cleans synchronization.
        """
        # Configuring carla simulation in async mode.
        settings = self.carla.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.carla.world.apply_settings(settings)

        # Destroying synchronized actors.
        for carla_actor_id in self.sumo2carla_ids.values():
            self.carla.destroy_actor(carla_actor_id)

        for sumo_actor_id in self.carla2sumo_ids.values():
            self.sumo.destroy_actor(sumo_actor_id)

        # Closing sumo and carla client.
        self.carla.close()
        self.sumo.close()


def synchronization_loop():
    """
    Entry point for sumo-carla co-simulation.
    """
    args = vars(arguments)
    Tdata = ''
    # proj1 = Proj('+proj=utm +zone=50 +ellps=WGS84 +datum=WGS84 +units=m +no_defs')
    proj1 = Proj('+proj=tmerc +lon_0=116.2872229585798 +lat_0=40.04753227284374 +ellps=WGS84')

    #加载自定义地图，world_path为自定义地图地址
    if args['world_path']:
        with open(args['world_path']) as od_file:
            Tdata = od_file.read()
    #使用自定义地图需要修改CarlaSimulation中的__init__()参数及地图获取方式
    #carla仿真器
    carla_simulation = CarlaSimulation(args['carla_host'], args['carla_port'], args['step_length'], Tdata)
    world = carla_simulation.world
    #观察者视角设置
    spectator = world.get_spectator()
    world.debug.draw_string(view_transform.location, 'O', draw_shadow=False,
                            color=carla.Color(r=0, g=255, b=0), life_time=1500,
                            persistent_lines=True)
    spectator.set_transform(carla.Transform(view_transform.location + carla.Location(z=200),
                                            carla.Rotation(pitch=-90, yaw=270)))
    
    #args['sumo_out_file'],原SumoSimulation中的__init__()参数无该项，应该是后续添加的
    #sumo仿真器
    sumo_simulation = SumoSimulation(args['sumo_cfg_file'],  args['step_length'],
                                     args['sumo_host'],
                                     args['sumo_port'], args['sumo_gui'], args['client_order'])

    #sumo仿真器与carla仿真器的时间同步
    synchronization = SimulationSynchronization(sumo_simulation, carla_simulation, args['tls_manager'],
                                                args['sync_vehicle_color'], args['sync_vehicle_lights'])
    stepcount = 0
    type_dic = {'PASSENGER': 4, 'TAXI': 4, 'MOTORCYCLE': 202, 'TRUCK': 603}

    center = [-40.009888, -505.712402]
    with open(r'/home/wanji/carla/Co-Simulation/Sumo/work01/toparse.json') as sim_file:
        sim_data_dic = json.load(sim_file)
    start_frame = world.get_snapshot().frame

    try:
        while True:
            start = time.time()
            #同步更新，关键
            synchronization.tick()
            w_frame = world.get_snapshot().frame
            p_timestamp = world.get_snapshot().timestamp.platform_timestamp
            w_timestamp = get_time_stamp(p_timestamp)
            passframe = w_frame - start_frame
            #发布world时间戳
            print("\nWorld's frame:{0}, time: {1}".format(w_frame, w_timestamp))

            # ==========================================输出e1帧结果===========================================
            sumotime = get_time_stamp(start)
            print("=================" + str(passframe), str(sumotime))
            sumo_ids = synchronization.sumo2carla_ids.keys()
            # for sumoid in sumo_ids:
            #     print(sumoid)
            #     sumo_actor = synchronization.sumo.get_actor(sumoid)
            #     traci.vehicle.setLaneChangeMode(sumoid, 0b000000000000)
            #     courseAngle = sumo_actor.corse
            #     location = sumo_actor.location
            #     lon, lat = traci.simulation.convertGeo(location[0], location[1])
            #     # print(traci.vehicle.getPosition3D(sumoid), traci.vehicle.getAngle(sumoid), traci.vehicle.getSlope(sumoid))
            #     print("old", location, lon, lat, courseAngle)
            
            #     newlongitude = sim_data_dic[sumoid][0][stepcount]
            #     newlatitude = sim_data_dic[sumoid][1][stepcount]
            #     newcourseAngle = sim_data_dic[sumoid][4][stepcount]
            
            #     x3, y3 = traci.simulation.convertGeo(newlongitude, newlatitude, fromGeo=True)
            #     traci.vehicle.moveToXY(sumoid, "", 0, x3, y3, angle=newcourseAngle, keepRoute=2)
            #     print("new", x3, y3, newlongitude, newlatitude, newcourseAngle)

            #     carlaid = synchronization.sumo2carla_ids[sumoid]
            #     carla_actor = synchronization.carla.get_actor(carlaid)
            #     box = carla_actor.bounding_box.extent
            #     Length = round(2.0 * box.x * 100, 2)
            #     Width = round(2.0 * box.y * 100, 2)
            #     Height = round(2.0 * box.z * 100, 2)
            #     carla_transform = carla_actor.get_transform()
            #     x, y, z = carla_transform.location.x, carla_transform.location.y, carla_transform.location.z
            #     # altitude = carla_transform.location.z
            #     # longitude, latitude = proj1(x + 456255.010, 4397808.242 - y, inverse=True)
            #     # longitude, latitude = proj1(x, y, inverse=True)
            #     sumo_actor = synchronization.sumo.get_actor(sumoid)
            #     sumo_type = str(sumo_actor.vclass).split(":")[-1].split(".")[-1]
            #     dis = eucliDist(center, [x, y])
            #     originalType = type_dic[sumo_type]
            #     #     # originalColor = color_dic[sumo_type]
            #     #     # print(originalType, originalColor)
            #     #     # speed = sumo_actor.speed
            #     #     length = sumo_actor.length
            #     #     width = sumo_actor.width
            #     #     height = sumo_actor.height
            #     #     # laneNum = sumo_actor.lanenum
            #     corse = sumo_actor.corse
            #     #     # if originalType == "TRUCK":
            #     if dis < 100:
            #         output_csv = output_csv.append({"frameid": int(w_frame), "carid": int(carlaid),
            #                                         "type": int(originalType),
            #                                         "x": x, "y": y, "z": z,
            #                                         "corseAngle": corse,
            #                                         "Length": Length, "Width": Width,
            #                                         "Height": Height}, ignore_index=True)
            #     print(output_csv)
            #     print(sumoid, longitude,latitude)
            
            #     oneframe["e1FrameParticipant"].append({"id": sumoid, "picLicense": "", "type": originalType,
            #                                               "vehicleColor": originalColor,
            #                                               "licenseColor": '', "longitude": longitude,
            #                                               "latitude": latitude, "altitude": altitude,
            #                                               "courseAngle": corse, "speed": speed,
            #                                               "laneNum": laneNum, "axleNum": '',
            #                                               "length": length, "width": width,
            #                                               "high": height})
            # await sendmsg(str(oneframe).replace("'", '"'))  # 依次给每个客户端都发一条信息
            # # asyntest(oneframe["e1FrameParticipant"])
            # output_json[stepcount] = oneframe
            # ===========================可视化雷达点云和相机图片================================
            # try:
            #     rgbs = []
            #     lidars = []
            #     splicing = []
            #
            #     for i in range(0, len(sensor_list)):
            #         s_frame, s_name, s_data = sensor_queue.get(True, 1.0)
            #         sensor_type = s_name.split('_')[0]
            #         print("    Frame: %d   Sensor: %s" % (s_frame, s_name))
            #
            #         if sensor_type == 'rgb':
            #             rgbs.append(_parse_image_cb(s_data))
            #             # s_data.save_to_disk(
            #             #     save_path + "jpg/" + str(w_timestamp) + "_" + s_name + ".png")
            #         elif sensor_type == 'lidar':
            #             if s_name != 'lidar_32':
            #                 splicing.append(_parse_lidar_cb(s_data))
            #             else:
            #                 lidars.append(_parse_lidar_cb(s_data))
            #                 # points2pcd(save_path + "ply/" + s_frame + ".pcd", _parse_lidar_cb(s_data)[::-1, :])
            #                 # s_data.save_to_disk(save_path + "ply/" + s_name + ".ply")
            #
            #     if splicing:
            #         concat_points = np.concatenate(splicing, axis=0)
            #         # concat_points[:, 1] = [-p for p in concat_points[:, 1]]
            #         # pcd_path = save_path + "pcd/" + str(w_frame) + "_" + "splice" + ".pcd"
            #         # points2pcd(pcd_path, concat_points)
            #
            #         # 仅用来可视化 可注释
            #         rgb = np.concatenate(rgbs, axis=1)[..., :3]
            #         lidar32 = visualize_data(rgb, lidars)
            #         lidarsplice = visualize_data(rgb, [concat_points])
            #         cv2.imshow('rgb_vizs', rgb)
            #         cv2.imshow('lidar_32', lidar32)
            #         cv2.imshow('lidar_splice', lidarsplice)
            #         cv2.waitKey(1)
            #
            #     # if rgb is None or save_path is not None:
            #     #     # 检查是否有各自传感器的文件夹
            #     #     mkdir_folder(save_path)
            #     #     filename = save_path + 'rgb/' + str(w_frame) + '.png'
            #     #     cv2.imwrite(filename, np.array(rgb[..., ::-1]))
            #     #     filename = save_path + 'lidar/' + str(w_frame) + '.npy'
            #     #     np.save(filename, lidar)
            #
            # except Empty:
            #     print("    Some of the sensor information is missed")

            end = time.time()
            elapsed = end - start
            if elapsed < args['step_length']:
                #超时，停止时间同步
                time.sleep(args['step_length'] - elapsed)
            stepcount += 1

    except (KeyboardInterrupt, BaseException):
        logging.info('Cancelled by user.')

    finally:
        logging.info('Cleaning synchronization')
        # for sensor in sensor_list:
        #     sensor.destroy()
        # for actor in actor_list:
        #     actor.destroy()
        synchronization.close()
        # carid_file = json.dumps(output_json, ensure_ascii=False)
        # out_file = open(r'D:\carla_test\Carla_simu\output/taiheqiao_e1_26.json', 'w', encoding='utf8')
        # out_file.write(carid_file)
        print("完成")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--sumo_cfg_file', type=str, help='sumo configuration file',
                           default=r'/home/wanji/carla/Co-Simulation/Sumo/work01/wanji_0701.sumocfg')
                           #default=r'D:\carla_test\Co_simulation\xml_Data\basesta\wanji.sumocfg')
    argparser.add_argument('--sumo_out_file', type=str, help='sumo output file',
                           default=r'/home/wanji/carla/Co-Simulation/Sumo/work01/wanji_0701.output.xml')
    argparser.add_argument('--world_path', type=str, help='sumo output file',
                           default=r'/home/wanji/carla/Co-Simulation/Sumo/work01/wanji_0701.xodr')
    argparser.add_argument('--carla-host',
                           metavar='H',
                           default='127.0.0.1',
                           help='IP of the carla host server (default: 127.0.0.1)')
    argparser.add_argument('--carla-port',
                           metavar='P',
                           default=2000,
                           type=int,
                           help='TCP port to listen to (default: 2000)')
    argparser.add_argument('--sumo-host',
                           metavar='H',
                           default=None,
                           help='IP of the sumo host server (default: 127.0.0.1)')
    argparser.add_argument('--sumo-port',
                           metavar='P',
                           default=None,
                           type=int,
                           help='TCP port to listen to (default: 8813)')
    # argparser.add_argument('--sumo-gui', action='store_true', help='run the gui version of sumo')
    argparser.add_argument('--sumo-gui', default=True)
    argparser.add_argument('--step-length',
                           default=0.1,
                           type=float,
                           help='set fixed delta seconds (default: 0.05s)')
    argparser.add_argument('--client-order',
                           metavar='TRACI_CLIENT_ORDER',
                           default=1,
                           type=int,
                           help='client order number for the co-simulation TraCI connection (default: 1)')
    argparser.add_argument('--sync-vehicle-lights',
                           action='store_true',
                           help='synchronize vehicle lights state (default: False)')
    argparser.add_argument('--sync-vehicle-color',
                           action='store_true',
                           help='synchronize vehicle color (default: False)')
    argparser.add_argument('--sync-vehicle-all',
                           action='store_true',
                           help='synchronize all vehicle properties (default: False)')
    argparser.add_argument('--tls-manager',
                           type=str,
                           choices=['none', 'sumo', 'carla'],
                           help="select traffic light manager (default: none)",
                           default='sumo')
    argparser.add_argument('--debug', action='store_true', help='enable debug messages')

    arguments = argparser.parse_args()

    if arguments.sync_vehicle_all is True:
        arguments.sync_vehicle_lights = True
        arguments.sync_vehicle_color = True
    #
    if arguments.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    output_json = {}
    actor_list, sensor_list = [], []
    sensor_type = ['rgb', 'lidar']
    save_path = r''

    sensor_queue = Queue()
    IM_WIDTH = 256 * 1.0
    IM_HEIGHT = 256 * 1.0

    # view_transform = Transform(Location(x=189.912918, y=-300.235687, z=40.156311),
    #                       Rotation(pitch=90, yaw=90, roll=90))  # 观察视角
    view_transform = Transform(Location(x=241.61057899409823, y=-90.90056100583038, z=45.3),
                               Rotation(pitch=-0.113770, yaw=77.0, roll=0.000000))
    camera01_trans = Transform(Location(x=241.61057899409823, y=-90.90056100583038, z=45.3),
                               Rotation(pitch=-45, yaw=77.0, roll=0.000000))
    camera02_trans = Transform(Location(x=241.61057899409823, y=-90.90056100583038, z=45.3),
                               Rotation(pitch=-45, yaw=77.0, roll=0.000000))

    lidar_trans = Transform(Location(x=241.61057899409823, y=-90.90056100583038, z=42.3),
                               Rotation(pitch=0, yaw=-13, roll=0.000000))

    synchronization_loop()
