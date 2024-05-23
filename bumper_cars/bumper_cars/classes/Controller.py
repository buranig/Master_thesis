#!/usr/bin/env python3

import yaml
import numpy as np
from typing import List

# import car model
from bumper_cars.classes.CarModel import CarModel, State
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
        self.boundary_points = np.array([-self.width_init/2+0.154, self.width_init/2+0.154, -self.height_init/2+0.081, self.height_init/2+0.081])

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







    def plot_robot_trajectory(self, x, u, predicted_trajectory, dilated_traj, targets, ax):
        """
        Plots the robot and arrows for visualization.

        Args:
            i (int): Index of the robot.
            x (numpy.ndarray): State vector of shape (4, N), where N is the number of time steps.
            multi_control (numpy.ndarray): Control inputs of shape (2, N).
            targets (list): List of target points.

        """
        plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-", color=color_dict[0])
        plot_polygon(dilated_traj, ax=ax, add_points=False, alpha=0.5, color=color_dict[0])
        # plt.plot(x[0], x[1], "xr")
        # plt.plot(targets[0], targets[1], "x", color=color_dict[0], markersize=15)
        self.plot_robot(x[0], x[1], x[2])
        self.plot_arrow(x[0], x[1], x[2], length=0.5, width=0.05)
        # self.plot_arrow(x[0], x[1], x[2] + u[1], length=0.5, width=0.1)
        self.plot_map()


    def plot_arrow(self, x, y, yaw, length=0.2, width=0.1):  # pragma: no cover
        """
        Plot an arrow.

        Args:
            x (float): X-coordinate of the arrow.
            y (float): Y-coordinate of the arrow.
            yaw (float): Yaw angle of the arrow.
            length (float, optional): Length of the arrow. Defaults to 0.5.
            width (float, optional): Width of the arrow. Defaults to 0.1.
        """
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                head_length=width, head_width=width)
        plt.plot(x, y)

    def plot_robot(self, x, y, yaw):  
        """
        Plot the robot.

        Args:
            x (float): X-coordinate of the robot.
            y (float): Y-coordinate of the robot.
            yaw (float): Yaw angle of the robot.
            i (int): Index of the robot.
        """
        outline = np.array([[-self.car_model.L / 2, self.car_model.L / 2,
                                (self.car_model.L / 2), -self.car_model.L / 2,
                                -self.car_model.L / 2],
                            [self.car_model.width / 2, self.car_model.width / 2,
                                - self.car_model.width / 2, -self.car_model.width / 2,
                                self.car_model.width / 2]])
        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                            [-math.sin(yaw), math.cos(yaw)]])
        outline = (outline.T.dot(Rot1)).T
        outline[0, :] += x
        outline[1, :] += y
        plt.plot(np.array(outline[0, :]).flatten(),
                    np.array(outline[1, :]).flatten(), color_dict[0])

    def plot_map(self):
        """
        Plot the map.
        """
        corner_x = [-self.width_init/2.0 + 0.154, self.width_init/2.0 + 0.154, self.width_init/2.0 + 0.154, -self.width_init/2.0 + 0.154, -self.width_init/2.0 + 0.154]
        corner_y = [self.height_init/2.0 + 0.081, self.height_init/2.0 + 0.081, -self.height_init/2.0 + 0.081, -self.height_init/2.0 + 0.081, self.height_init/2.0 + 0.081]

        plt.plot(corner_x, corner_y)