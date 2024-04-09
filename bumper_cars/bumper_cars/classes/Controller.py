#!/usr/bin/env python3

import yaml
import numpy as np
from typing import List

# import car model
from bumper_cars.classes.CarModel import CarModel, State
from lar_msgs.msg import CarControlStamped, CarStateStamped

class Controller:
    """
    This class represents the controller for the robot.

    It initializes the controller parameters, subscribes to robot state topics,
    and publishes control commands to the robot. It also implements different
    controller algorithms based on the specified controller type.
    """

    # Initialize class public parameters
    dt = 0.0
    L_d = 0.0
    max_acc = 0.0
    min_acc = 0.0

    safety = 0.0
    add_noise = 0.0
    noise_scale = 0.0

    # car_model = None


    def __init__(self, controller_path :str, car_path = ''):
        
        with open(controller_path, 'r') as openfile:
            yaml_object = yaml.safe_load(openfile)
        
        # Global variables
        self.dt = yaml_object["Controller"]["dt"]
        self.L_d = yaml_object["Controller"]["L_d"]
        self.max_acc = yaml_object["Controller"]["max_acc"]
        self.min_acc = yaml_object["Controller"]["min_acc"]

        self.safety = yaml_object["Controller"]["safety"]
        self.add_noise = yaml_object["Controller"]["add_noise"]
        self.noise_scale = yaml_object["Controller"]["noise_scale"]

        self.car_model = CarModel(car_path)

    def simulate_input(self, car_cmd: CarControlStamped) -> State:
        return self.car_model.step(car_cmd, self.L_d)

    def set_state(self, state: State) -> None:
        self.car_model.set_state(state)
    
    def set_goal(self, goal: State) -> None:
        raise Exception("Hereditary function doesn't implement 'set_goal'")
    
    def compute_cmd(self, car_list : List[CarStateStamped]) -> CarControlStamped:
        raise Exception("Hereditary function doesn't implement 'compute_cmd'")
    
    def _add_noise(self, car_list : List[CarModel]) -> List[CarModel]:
        raise Exception("Hereditary function doesn't implement '_add_noise'")