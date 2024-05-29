#!/usr/bin/env python3

import yaml
import numpy as np
from typing import List

# import car model
from bumper_cars.classes.State import State
from bumper_cars.classes.CarModel import CarModel
from bumper_cars.utils import car_utils as utils
from lar_msgs.msg import CarControlStamped, CarStateStamped


# For drawing purposes
import math
from shapely.plotting import plot_polygon
import matplotlib.pyplot as plt
color_dict = {0: 'r', 1: 'b', 2: 'g', 3: 'y', 4: 'm', 5: 'c', 6: 'k', 7: 'tab:orange', 8: 'tab:brown', 9: 'tab:gray', 10: 'tab:olive', 11: 'tab:pink', 12: 'tab:purple', 13: 'tab:red', 14: 'tab:blue', 15: 'tab:green'}


class Controller:
    """
    This class represents the controller for the robot.

    It initializes the controller parameters, subscribes to robot state topics,
    and publishes control commands to the robot. It also implements different
    controller algorithms based on the specified controller type.
    """

    # Initialize class public parameters
    dt = 0.0
    ph = 0.0


    def __init__(self, controller_path :str, car_path = ''):
        
        with open(controller_path, 'r') as openfile:
            yaml_object = yaml.safe_load(openfile)
        
        ### Global variables

        # Simulation params
        self.dt = yaml_object["Controller"]["dt"]
        self.ph = yaml_object["Controller"]["ph"]
        self.show_animation = yaml_object["Simulation"]["show_animation"]

        # Size of map
        self.width_init = yaml_object["Simulation"]["width"]
        self.height_init = yaml_object["Simulation"]["height"]
        self.off_x = yaml_object["Simulation"]["off_x"]
        self.off_y = yaml_object["Simulation"]["off_y"]
        self.boundary_points = np.array([[-self.width_init/2 + self.off_x, -self.height_init/2 + self.off_y],\
                                         [self.width_init/2 + self.off_x, -self.height_init/2 + self.off_y],\
                                         [self.width_init/2 + self.off_x, self.height_init/2 + self.off_y],\
                                         [-self.width_init/2 + self.off_x, self.height_init/2 + self.off_y]])

        # Noise params
        self.add_noise = yaml_object["Controller"]["add_noise"]
        self.noise_scale = yaml_object["Controller"]["noise_scale"]


        self.car_model = CarModel(car_path)

    def set_state(self, state: State) -> None:
        self.car_model.set_state(state)

    def compute_traj(self):
        pass
    
    def set_goal(self, goal: CarControlStamped) -> None:
        raise Exception("Hereditary function doesn't implement 'set_goal'")
    
    def compute_cmd(self, car_list : List[CarStateStamped]) -> CarControlStamped:
        raise Exception("Hereditary function doesn't implement 'compute_cmd'")
    
    def _add_noise(self, car_list : List[CarModel]) -> List[CarModel]:
        raise Exception("Hereditary function doesn't implement '_add_noise'")

    def offset_track(self, off:List[int]):
        ref_x = off[0]
        ref_y = off[1]
        ref_yaw = off[2]

        for i in range(len(self.boundary_points)):
            pos_x = self.boundary_points[i][0]
            pos_y = self.boundary_points[i][1]
            new_x, new_y = utils.transform_point(pos_x,pos_y, 0.0, 0.0,ref_yaw)
            self.boundary_points[i][0] = new_x + ref_x
            self.boundary_points[i][1] = new_y + ref_y

