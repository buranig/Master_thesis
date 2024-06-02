#!/usr/bin/env python3

import yaml
import math
import numpy as np
import os

from bumper_cars.classes.State import State

from bumper_cars.utils.car_utils import normalize_angle
from lar_msgs.msg import CarControlStamped

class CarModel:
    """
    Represents a car model with various attributes and methods for simulating the car's behavior.

    Attributes:
        state (State): The current state of the car.
        max_steer (float): The maximum steering angle of the car in radians.
        max_speed (float): The maximum speed of the car in meters per second.
        min_speed (float): The minimum speed of the car in meters per second.
        max_acc (float): The maximum acceleration of the car in meters per second squared.
        min_acc (float): The minimum acceleration of the car in meters per second squared.
        acc_gain (float): The acceleration gain of the car, ranging from 0.0 to 1.0.
        width (float): The width of the vehicle's track in meters.
        L (float): The wheel base of the vehicle in meters.
        Lr (float): The distance from the rear axle to the center of mass in meters.
        Cf (float): The cornering stiffness of the front tires in N/rad.
        Cr (float): The cornering stiffness of the rear tires in N/rad.
        Iz (float): The moment of inertia around the vertical axis in kg/m^2.
        m (float): The mass of the car in kg.
        c_a (float): The aerodynamic coefficient of the car.
        c_r1 (float): The friction coefficient of the car.
        next_state (function): The callback function for updating the car's state based on the model type.

    Methods:
        set_state(car_state: State): Sets the state of the car.
        step(cmd: CarControlStamped, dt: float, curr_state: State = None) -> State: Performs a simulation step and returns the updated state of the car.
    """

    # Init state of car
    state = State()
    
    def __init__(self, carModel_path = ''):
        """
        Initializes a CarModel object.

        Args:
            carModel_path (str): The path to the car model configuration file. If not provided, a default path will be used.
        """

        # If path not provided, use known one
        dir_path = os.path.dirname(os.path.realpath(__file__))
        if carModel_path == '':
            carModel_path = dir_path + '/../../config/carModel.yaml' 

        with open(carModel_path, 'r') as openfile:
            yaml_object = yaml.safe_load(openfile)

        # Load car variables from provided file
        self.max_steer = yaml_object["max_steer"] # [rad] max steering angle
        self.max_speed = yaml_object["max_speed"] # [m/s]
        self.min_speed = yaml_object["min_speed"] # [m/s]
        self.max_acc = yaml_object["max_acc"] # [m/ss]
        self.min_acc = yaml_object["min_acc"] # [m/ss]
        self.acc_gain = yaml_object["acc_gain"] # [0.0 - 1.0]

        self.width = yaml_object["width"] # [m] Width of the vehicle's track
        self.L = yaml_object["L"]       # [m] Wheel base of vehicle
        self.Lr = yaml_object["Lr"]          # [m] Distance from rear axle to center of mass
        self.Lf = self.L - self.Lr
        self.Cf = yaml_object["Cf"]     # N/rad
        self.Cr = yaml_object["Cr"]     # N/rad
        self.Iz = yaml_object["Iz"]     # kg/m2
        self.m = yaml_object["m"]       # kg

        # Aerodynamic and friction coefficients
        self.c_a = yaml_object["c_a"]
        self.c_r1 = yaml_object["c_r1"]

        # Linear/nonlinear model
        model_type = yaml_object["model_type"]
        if model_type == 'kinematic':
            self.next_state = self.__kinematic_model_callback
        elif model_type == 'dynamic':
            self.next_state = self.__dynamic_model_callback
        else:
            raise Exception("Wrong value for 'model_type'")
        
    def set_state(self, car_state : State):
        """
        Sets the state of the car.

        Args:
            car_state (State): The new state of the car.
        """            
        self.state.x = car_state.x
        self.state.y = car_state.y
        self.state.v = car_state.v
        self.state.yaw = car_state.yaw
        self.state.omega = car_state.omega

    def step(self, cmd: CarControlStamped, dt: float, curr_state: State = None)-> State:
        """
        Performs a simulation step and returns the updated state of the car.

        Args:
            cmd (CarControlStamped): The control command for the car.
            dt (float): The time step for the simulation.
            curr_state (State, optional): The current state of the car. If not provided, the internal state of the car will be used.

        Returns:
            State: The updated state of the car.
        """
        return self.next_state(cmd, dt, curr_state)


    def __kinematic_model_callback(self, cmd: CarControlStamped, dt: float, curr_state: State = None) -> State:
        """
        The callback function for updating the car's state using a kinematic model.

        Args:
            cmd (CarControlStamped): The control command for the car.
            dt (float): The time step for the simulation.
            curr_state (State, optional): The current state of the car. If not provided, the internal state of the car will be used.

        Returns:
            State: The updated state of the car.
        """
        
        # Simulate given/current state
        old_x = self.state.x if curr_state is None else curr_state.x
        old_y = self.state.y if curr_state is None else curr_state.y
        old_v = self.state.v if curr_state is None else curr_state.v
        old_yaw = self.state.yaw if curr_state is None else curr_state.yaw
        old_omega = self.state.omega if curr_state is None else curr_state.omega

        # Ensure feasible inputs
        mapped_steering = np.interp(cmd.steering, [-1, 1], [-self.max_steer, self.max_steer])
        mapped_throttle = np.interp(cmd.throttle, [-1, 1], [self.min_acc, self.max_acc]) * self.acc_gain

        # State update
        state = State()
        state.x = old_x + old_v * np.cos(old_yaw) * dt
        state.y = old_y + old_v * np.sin(old_yaw) * dt

        state.yaw = old_yaw + old_v / self.L * np.tan(mapped_steering) * dt # WARN: tan is an approximation
        state.yaw = normalize_angle(state.yaw)

        state.v = old_v + mapped_throttle * dt
        state.v = np.clip(state.v, self.min_speed, self.max_speed)

        return state
    
    def __dynamic_model_callback(self, cmd:CarControlStamped, dt: float, curr_state: State = None) -> State:
        """
        The callback function for updating the car's state using a dynamic model.

        Args:
            cmd (CarControlStamped): The control command for the car.
            dt (float): The time step for the simulation.
            curr_state (State, optional): The current state of the car. If not provided, the internal state of the car will be used.

        Returns:
            State: The updated state of the car.
        """

        # Simulate given/current state
        old_x = self.state.x if curr_state is None else curr_state.x
        old_y = self.state.y if curr_state is None else curr_state.y
        old_v = self.state.v if curr_state is None else curr_state.v
        old_yaw = self.state.yaw if curr_state is None else curr_state.yaw
        old_omega = self.state.omega if curr_state is None else curr_state.omega

        # Ensure feasible inputs
        mapped_steering = np.interp(cmd.steering, [-1, 1], [-self.max_steer, self.max_steer])
        mapped_throttle = np.interp(cmd.throttle, [-1, 1], [self.min_acc, self.max_acc]) * self.acc_gain

        beta = math.atan2((self.Lr * math.tan(mapped_steering) / self.L), 1.0)
        vx = old_v * math.cos(beta)
        vy = old_v * math.sin(beta)

        Ffy = -self.Cf * ((vy + self.Lf * old_omega) / (vx + 0.0001) - mapped_steering)
        Fry = -self.Cr * (vy - self.Lr * old_omega) / (vx + 0.0001)
        R_x = self.c_r1 * abs(vx)
        F_aero = self.c_a * vx ** 2 # 
        F_load = F_aero + R_x #
        vx = vx + (mapped_throttle - Ffy * math.sin(mapped_steering) / self.m - F_load / self.m + vy * old_omega) * dt
        vy = vy + (Fry / self.m + Ffy * math.cos(mapped_steering) / self.m - vx * old_omega) * dt

        # State update
        state = State()
        state.omega = old_omega + (Ffy * self.Lf * math.cos(mapped_steering) - Fry * self.Lr) / self.Iz * dt
        
        state.yaw = old_yaw + old_omega * dt
        state.yaw = normalize_angle(state.yaw)

        state.x = old_x + vx * math.cos(old_yaw) * dt - vy * math.sin(old_yaw) * dt
        state.y = old_y + vx * math.sin(old_yaw) * dt + vy * math.cos(old_yaw) * dt

        state.v = math.sqrt(vx ** 2 + vy ** 2)
        state.v = np.clip(state.v, self.min_speed, self.max_speed)

        return state