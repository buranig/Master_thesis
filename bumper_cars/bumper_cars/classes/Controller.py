#!/usr/bin/env python3

import yaml
import numpy as np
from typing import List

# import car model
from bumper_cars.classes.State import State
from bumper_cars.classes.CarModel import CarModel
from bumper_cars.utils import car_utils as utils
from lar_msgs.msg import CarControlStamped, CarStateStamped

class Controller:
    """
    Represents the controller for the robot. Acts as a parent class for all controllers.

    Attributes:
        car_i (int): The index of the car.
        dt (float): The time step for the controller.
        ph (float): The prediction horizon for the controller.
        boundary_points (numpy.ndarray): The boundary points of the map.
        car_model (CarModel): The car model.
    """

    def __init__(self, controller_path :str, car_i = 0):
        """
        Initializes the Controller object.

        Args:
            controller_path (str): The path to the controller configuration file.
            car_i (int, optional): The index of the car. Defaults to 0.
        """
        
        with open(controller_path, 'r') as openfile:
            yaml_object = yaml.safe_load(openfile)
        
        ### Global variables
        self.car_i = car_i

        # Simulation params
        self.dt = yaml_object["Controller"]["dt"]
        self.ph = yaml_object["Controller"]["ph"]
        # Size of map
        width_init = yaml_object["Simulation"]["width"]
        height_init = yaml_object["Simulation"]["height"]
        off_x = yaml_object["Simulation"]["off_x"]
        off_y = yaml_object["Simulation"]["off_y"]
        self.boundary_points = np.array([[-width_init/2 + off_x, -height_init/2 + off_y],\
                                         [width_init/2 + off_x, -height_init/2 + off_y],\
                                         [width_init/2 + off_x, height_init/2 + off_y],\
                                         [-width_init/2 + off_x, height_init/2 + off_y]])

        # Car model with default values
        self.car_model = CarModel(carModel_path = '')

    def set_state(self, state: State) -> None:
        """
        Sets the state of the car.

        Args:
            state (State): The state of the car.
        """
        self.car_model.set_state(state)

    def offset_track(self, off:List[int]):
        """
        Moves the track's boundary points and given the x,y and angular offsets.

        Args:
            off (List[int]): The offset values for x, y, and yaw.
        """
        print("UPDATED TRACK BOUNDARY POINTS")
        ref_x = off[0]
        ref_y = off[1]
        ref_yaw = off[2]

        for i in range(len(self.boundary_points)):
            pos_x = self.boundary_points[i][0]
            pos_y = self.boundary_points[i][1]
            new_x, new_y = utils.transform_point(pos_x,pos_y, 0.0, 0.0,ref_yaw)
            self.boundary_points[i][0] = new_x + ref_x
            self.boundary_points[i][1] = new_y + ref_y
    
    def compute_traj(self):
        """
        On those algorithms that require it, pre-computes the possible trajectories.
        """
        print("Selected algorithm does not implement 'compute_traj' function")
        pass

    def set_traj(self, traj: np.array) -> None:
        """
        Sets the trajectory for the controller.

        Args:
        ----------
            traj (np.array): The trajectory for all cars.
        """
        print("Selected algorithm does not implement 'set_traj' function")
        pass

    def set_goal(self, goal: CarControlStamped) -> None:
        """
        Sets the goal for the controller.

        Args:
            goal (CarControlStamped): The control goal for the car.
        """
        raise Exception("Hereditary function doesn't implement 'set_goal'")
    
    def compute_cmd(self, car_list : List[CarStateStamped]) -> CarControlStamped:
        """
        Computes the safe control command for the car.

        Args:
            car_list (List[CarStateStamped]): The list of current car states.

        Returns:
            CarControlStamped: The control command for the contolled car (car_i).
        """
        raise Exception("Hereditary function doesn't implement 'compute_cmd'")
    