#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""Spawn Sumo NPCs vehicles into the simulation"""

# ==================================================================================================
# -- imports ---------------------------------------------------------------------------------------
# ==================================================================================================

import argparse
import json
import logging
import random
import re
import shutil
import tempfile
import time
import carla
import math
from pyproj import Proj
import asyncio
from util.websocketServer import WebSocketServer

import lxml.etree as ET  # pylint: disable=wrong-import-position

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

# ==================================================================================================
# -- imports ---------------------------------------------------------------------------------------
# ==================================================================================================

import sumolib  # pylint: disable=wrong-import-position
import traci  # pylint: disable=wrong-import-position

from sumo_integration.carla_simulation import CarlaSimulation  # pylint: disable=wrong-import-position
from sumo_integration.sumo_simulation import SumoSimulation  # pylint: disable=wrong-import-position

from run_synchronization import SimulationSynchronization  # pylint: disable=wrong-import-position

from util.netconvert_carla import netconvert_carla

# ==================================================================================================
# -- main ------------------------------------------------------------------------------------------
# ==================================================================================================


def write_sumocfg_xml(cfg_file, net_file, vtypes_file, viewsettings_file, additional_traci_clients=0):
    """
    Writes sumo configuration xml file.
    """
    root = ET.Element('configuration')

    input_tag = ET.SubElement(root, 'input')
    ET.SubElement(input_tag, 'net-file', {'value': net_file})
    ET.SubElement(input_tag, 'route-files', {'value': vtypes_file})

    gui_tag = ET.SubElement(root, 'gui_only')
    ET.SubElement(gui_tag, 'gui-settings-file', {'value': viewsettings_file})

    ET.SubElement(root, 'num-clients', {'value': str(additional_traci_clients+1)})

    tree = ET.ElementTree(root)
    tree.write(cfg_file, pretty_print=True, encoding='UTF-8', xml_declaration=True)


def main(args):
    ws_server = WebSocketServer(8083)
    ws_server.start()
    loop = asyncio.get_event_loop()

    # Temporal folder to save intermediate files.
    tmpdir = tempfile.mkdtemp()

    # ----------------
    # carla simulation
    # ----------------
    carla_simulation = CarlaSimulation(args.host, args.port, args.step_length)

    # world = carla_simulation.client.get_world()
    # current_map = world.get_map()

    # xodr_file = os.path.join(tmpdir, current_map.name + '.xodr')
    # current_map.save_to_disk(xodr_file)

    # ---------------
    # sumo simulation
    # ---------------
    # net_file = os.path.join(tmpdir, current_map.name + '.net.xml')
    # netconvert_carla(xodr_file, net_file, guess_tls=True)

    basedir = os.path.dirname(os.path.realpath(__file__))
    net_file = os.path.join(basedir, 'tongji', 'tongji.net.xml')
    cfg_file = os.path.join(basedir, 'tongji', 'tongji.sumocfg')
    vtypes_file = os.path.join(basedir, 'tongji', 'carlavtypes13.rou.xml')
    viewsettings_file = os.path.join(basedir, 'tongji', 'viewsettings.xml')
    write_sumocfg_xml(cfg_file, net_file, vtypes_file, viewsettings_file, args.additional_traci_clients)

    sumo_net = sumolib.net.readNet(net_file)
    sumo_simulation = SumoSimulation(cfg_file,
                                     args.step_length,
                                     host=args.sumo_host,
                                     port=args.sumo_port,
                                     sumo_gui=args.sumo_gui,
                                     client_order=args.client_order)

    # ---------------
    # synchronization
    # ---------------
    synchronization = SimulationSynchronization(sumo_simulation, carla_simulation, args.tls_manager,
                                                args.sync_vehicle_color, args.sync_vehicle_lights)

    try:
        # =================================调整路网中的环境可见度======================================
        # 使得道路中自动生成的交通灯为不可见状态
        env_objs1 = carla_simulation.world.get_environment_objects(carla.CityObjectLabel.TrafficLight)
        env_objs2 = carla_simulation.world.get_environment_objects(carla.CityObjectLabel.Poles)
        env_list = []
        for env_obj in env_objs1:
            env_list.append(env_obj.id)
        for env_obj in env_objs2:
            env_list.append(env_obj.id)
        carla_simulation.world.enable_environment_objects(env_list, False)

        spectator = carla_simulation.world.get_spectator()
        view_transform = carla.Transform(carla.Location(x=-690.8, y=139.2, z=30),
                                         carla.Rotation(pitch=-30.0, yaw=-50.0, roll=0.0))
        spectator.set_transform(view_transform)

        # 控制世界的天气和时间（太阳的位置）
        weather = carla.WeatherParameters(
            cloudiness=0.0,  # 0-100  0是晴朗的天空，100是完全阴天
            precipitation=0.0,  # 0表示没有下雨，100表示大雨
            precipitation_deposits=0.0,  # 0表示道路上没有水坑，100表示道路完全被雨水覆盖
            wind_intensity=0.0,  # 0表示平静，100表示强风，风会影响雨向和树叶
            sun_azimuth_angle=270.0,  # 太阳方位角，0～360
            sun_altitude_angle=13.0,  # 太阳高度角，90是中午，-90是午夜
            fog_density=0.0,  # 0～100表示雾的浓度或厚度，仅影响RGB相机传感器
            fog_distance=0.0,  # 雾开始的距离，单位为米
            wetness=0.0,  # 0～100表示道路湿度百分比，仅影响RGB相机传感器
            fog_falloff=0.0,  # 雾的密度，0至无穷大，0表示雾比空气轻，覆盖整个场景，1表示与空气一样，覆盖正常大小的建筑物
            scattering_intensity=0.0,  # 控制光线对雾的穿透程度
            mie_scattering_scale=0.0,  # 控制光线与花粉或空气等大颗粒的相互作用，导致天气朦胧，光源周围有光晕，0表示无影响
            rayleigh_scattering_scale=0.0331,  # 控制光与空气分子等小粒子的相互作用，取决于光波长，导致白天蓝天或晚上红天
        )
        carla_simulation.world.set_weather(weather)


        # ----------
        # Blueprints
        # ----------
        with open('data/vtypes.json') as f:
            vtypes = json.load(f)['carla_blueprints']
        #
        # blueprints = vtypes.keys()

        # filterv = re.compile(args.filterv)
        # blueprints = list(filter(filterv.search, blueprints))
        #
        # if args.safe:
        #     blueprints = [
        #         x for x in blueprints if vtypes[x]['vClass'] not in ('motorcycle', 'bicycle')
        #     ]
        #     blueprints = [x for x in blueprints if not x.endswith('microlino')]
        #     blueprints = [x for x in blueprints if not x.endswith('carlacola')]
        #     blueprints = [x for x in blueprints if not x.endswith('cybertruck')]
        #     blueprints = [x for x in blueprints if not x.endswith('t2')]
        #     blueprints = [x for x in blueprints if not x.endswith('sprinter')]
        #     blueprints = [x for x in blueprints if not x.endswith('firetruck')]
        #     blueprints = [x for x in blueprints if not x.endswith('ambulance')]

        # if not blueprints:
        #     raise RuntimeError('No blueprints available due to user restrictions.')
        #
        if args.number_of_walkers > 0:
            logging.warning('Pedestrians are not supported yet. No walkers will be spawned.')

        # --------------
        # Spawn vehicles
        # --------------
        # Spawns sumo NPC vehicles.
        sumo_edges = sumo_net.getEdges()

        vehicle_route_id = args.number_of_vehicles

        for i in range(args.number_of_vehicles):
            # type_id = random.choice(blueprints)
            filterv = ['vehicle.dodge.charger_2020', 'vehicle.volkswagen.t2_2021', 'vehicle.ford.ambulance']
            F = [0.6, 0.2, 0.2]
            blueprint_list = random.choices(filterv, weights=F, k=1)
            type_id0 = blueprint_list[0]
            vclass = vtypes[type_id0]['vClass']

            allowed_edges = [e for e in sumo_edges if e.allows(vclass)]
            if allowed_edges:
                edge = random.choice(allowed_edges)

                traci.route.add('route_{}'.format(i), [edge.getID()])
                traci.vehicle.add('sumo_{}'.format(i), 'route_{}'.format(i), typeID=type_id0)
                if type_id0 == 'vehicle.volkswagen.t2_2021':
                    traci.vehicle.setColor('sumo_{}'.format(i), (0, 255, 0))
                elif type_id0 == 'vehicle.ford.ambulance':
                    traci.vehicle.setColor('sumo_{}'.format(i), (0, 0, 255))
            else:
                logging.error(
                    'Could not found a route for %s. No vehicle will be spawned in sumo',
                    type_id0)

        while True:
            start = time.time()

            synchronization.tick()

            if len(traci.vehicle.getIDList()) < args.number_of_vehicles:
                num = args.number_of_vehicles - len(traci.vehicle.getIDList())
                # print(num, ", ", vehicle_route_id, ", ", len(traci.vehicle.getIDList()))
                for i in range(vehicle_route_id, vehicle_route_id+num):
                    filterv = ['vehicle.dodge.charger_2020', 'vehicle.volkswagen.t2_2021', 'vehicle.ford.ambulance']
                    F = [0.6, 0.2, 0.2]
                    blueprint_list = random.choices(filterv, weights=F, k=1)
                    type_id0 = blueprint_list[0]
                    vclass = vtypes[type_id0]['vClass']
                    allowed_edges = [e for e in sumo_edges if e.allows(vclass)]
                    if allowed_edges:
                        edge = random.choice(allowed_edges)
                        traci.route.add('route_{}'.format(i), [edge.getID()])
                        traci.vehicle.add('sumo_{}'.format(i), 'route_{}'.format(i), typeID=type_id0)
                        if type_id0 == 'vehicle.volkswagen.t2_2021':
                            traci.vehicle.setColor('sumo_{}'.format(i), (0, 255, 0))
                        elif type_id0 == 'vehicle.ford.ambulance':
                            traci.vehicle.setColor('sumo_{}'.format(i), (0, 0, 255))
                    else:
                        logging.error(
                            'Could not found a route for %s. No vehicle will be spawned in sumo',
                            type_id0)
                vehicle_route_id = vehicle_route_id+num

            # Updates vehicle routes
            for vehicle_id in traci.vehicle.getIDList():
                sumo_simulation.subscribe(vehicle_id)  # 订阅车辆的相关信息（位置车型颜色大小） （sub可以提高检索的效率）
                sumo_actor = sumo_simulation.get_actor(vehicle_id)
                if sumo_actor.type_id != 'vehicle.lincoln.mkz_2020':
                    route = traci.vehicle.getRoute(vehicle_id)
                    index = traci.vehicle.getRouteIndex(vehicle_id)
                    vclass = traci.vehicle.getVehicleClass(vehicle_id)
                    # print("sumoID:", vehicle_id, ", route:", route, ", index:", index, ", vclass:", vclass)

                    if index == (len(route) - 1):
                        current_edge = sumo_net.getEdge(route[index])
                        available_edges = list(current_edge.getAllowedOutgoing(vclass).keys())
                        # print("sumoID:", vehicle_id, ", route:", available_edges)
                        if available_edges:
                            next_edge = random.choice(available_edges)

                            new_route = [current_edge.getID(), next_edge.getID()]
                            traci.vehicle.setRoute(vehicle_id, new_route)
                elif sumo_actor.color != (255, 0, 0):
                    traci.vehicle.setColor(vehicle_id, (255, 0, 0))

            car_list = list()
            followCar = None
            rule = Proj("+proj=tmerc +lon_0=121.2092870660126 +lat_0=31.292829882838856 +ellps=WGS84")
            all_vehicle_actors = carla_simulation.world.get_actors()
            all_vehicle_actors = [x for x in all_vehicle_actors if x.type_id.startswith('vehicle')]
            if len(all_vehicle_actors) > 0:
                for vehicle in all_vehicle_actors:
                    vehicle_id = vehicle.id
                    vehicle_type = vehicle.type_id
                    vehicle_color = vehicle.attributes['color']
                    # vehicle_color = [int(item) for item in vehicle_color.split(",")]
                    x = vehicle.get_transform().location.x
                    y = vehicle.get_transform().location.y
                    vehicle_angle = 90 - vehicle.get_transform().rotation.yaw
                    v = vehicle.get_velocity()
                    vehicle_speed = 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
                    waypoint = carla_simulation.world.get_map().get_waypoint(vehicle.get_location())
                    lane_id = waypoint.lane_id
                    # 'vehicle.dodge.charger_2020', 'vehicle.volkswagen.t2_2021', 'vehicle.ford.ambulance'
                    # if vehicle_type == 'vehicle.dodge.charger_2020':
                    #     vehicle_type2webGL = 'jiaoche'
                    if vehicle_type == 'vehicle.volkswagen.t2_2021':
                        vehicle_type2webGL = 'zhongba'
                    elif vehicle_type == 'vehicle.ford.ambulance':
                        vehicle_type2webGL = 'xiaohuo'
                    elif vehicle_type == 'vehicle.lincoln.mkz_2020':
                        vehicle_type2webGL = 'jiaoche'
                        followCar = vehicle_id
                    else:
                        vehicle_type2webGL = 'jiaoche'
                    longitude, latitude = rule(x, -y, inverse=True)
                    if x == 0 and y == 0:
                        vehicle.destroy()
                        # vehicles_list.remove(vehicle_id)
                    else:
                        car = {"id": vehicle_id, "ty": vehicle_type2webGL, "color": vehicle_color, "lo": longitude, "la": latitude,
                               "ag": vehicle_angle, "speed": vehicle_speed, "laneNum": lane_id}
                        car_list.append(car)
            if len(car_list) > 0:
                result = {"participants": car_list, "participantNum": len(car_list), "followCar": followCar,
                          "timeStamp": str(time.time()), "globalTimeStamp": str(time.time())}
                # print(result)
                loop.run_until_complete(ws_server.broadcast(str(result).replace("'", '"')))

            end = time.time()
            elapsed = end - start
            if elapsed < args.step_length:
                time.sleep(args.step_length - elapsed)

    except KeyboardInterrupt:
        logging.info('Cancelled by user.')

    finally:
        synchronization.close()

        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)



if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--host',
                           metavar='H',
                           default='127.0.0.1',
                           help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument('-p',
                           '--port',
                           metavar='P',
                           default=2000,
                           type=int,
                           help='TCP port to listen to (default: 2000)')
    argparser.add_argument('--sumo-host',
                           default=None,
                           help='IP of the sumo host server (default: None)')
    argparser.add_argument('--sumo-port',
                           default=None,
                           type=int,
                           help='TCP port to listen to (default: None)')
    argparser.add_argument('-n',
                           '--number-of-vehicles',
                           metavar='N',
                           default=100,
                           type=int,
                           help='number of vehicles (default: 10)')
    argparser.add_argument('-w',
                           '--number-of-walkers',
                           metavar='W',
                           default=100,
                           type=int,
                           help='number of walkers (default: 0)')
    argparser.add_argument('--safe',
                           action='store_true',
                           help='avoid spawning vehicles prone to accidents')
    argparser.add_argument('--filterv',
                           metavar='PATTERN',
                           default='vehicle.*',
                           help='vehicles filter (default: "vehicle.*")')
    argparser.add_argument('--filterw',
                           metavar='PATTERN',
                           default='walker.pedestrian.*',
                           help='pedestrians filter (default: "walker.pedestrian.*")')
    argparser.add_argument('--sumo-gui', action='store_true', help='run the gui version of sumo')
    argparser.add_argument('--step-length',
                           default=0.05,
                           type=float,
                           help='set fixed delta seconds (default: 0.05s)')
    argparser.add_argument('--additional-traci-clients',
                           metavar='TRACI_CLIENTS',
                           default=0,
                           type=int,
                           help='number of additional TraCI clients to wait for (default: 0)')
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
                           default='carla')
    argparser.add_argument('--debug', action='store_true', help='enable debug messages')
    args = argparser.parse_args()

    if args.sync_vehicle_all is True:
        args.sync_vehicle_lights = True
        args.sync_vehicle_color = True

    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    main(args)
