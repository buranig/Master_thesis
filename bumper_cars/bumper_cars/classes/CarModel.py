#!/usr/bin/env python3

import yaml
import math
import numpy as np
import os
from lar_utils.car_utils import normalize_angle
from lar_msgs.msg import CarControlStamped

class State:
    x = y = v = yaw = omega = 0.0

class CarModel:
    # Init state of car
    state = State()
    
    def __init__(self, carModel_path = ''):

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
        if model_type == 'linear':
            self.next_state = self.__linear_model_callback
        elif model_type == 'nonlinear':
            self.next_state = self.__nonlinear_model_callback
        else:
            raise Exception("Wrong value for 'model_type'")
        
    def set_state(self, car_state : State):
            self.state.x = car_state.x
            self.state.y = car_state.y
            self.state.v = car_state.v
            self.state.yaw = car_state.yaw
            self.state.omega = car_state.omega

    def step(self, cmd: CarControlStamped, dt: float, curr_state: State = None)-> State:
        return self.next_state(cmd, dt, curr_state)


    def __linear_model_callback(self, cmd: CarControlStamped, dt: float, curr_state: State = None) -> State:
        """
        Update the car state using a non-linear kinematic model.

        This function calculates the new state of the car based on the initial state, control inputs, and the time elapsed since the last update.
        It returns the updated state and the current time.

        Args:
            initial_state (FullState): The initial state of the car.
            cmd (CarControlStamped): The control inputs for the car.
            dt (float): Increment of time to simulate for.

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
        cmd.steering = np.clip(cmd.steering, -self.max_steer, self.max_steer)

        # State update
        state = State()
        state.x = old_x + old_v * np.cos(old_yaw) * dt
        state.y = old_y + old_v * np.sin(old_yaw) * dt

        state.yaw = old_yaw + old_v / self.L * np.tan(cmd.steering) * dt # TODO: Tan is approximation
        state.yaw = normalize_angle(state.yaw)

        state.v = old_v + cmd.throttle * dt
        state.v = np.clip(state.v, self.min_speed, self.max_speed)

        return state
    
    def __nonlinear_model_callback(self, cmd:CarControlStamped, dt: float, curr_state: State = None) -> State:
        """
        Update the car state using a nonlinear dynamic model.

        This function calculates the new state of the car based on the current state, control inputs, and the time elapsed since the last update.
        It returns the updated state and the current time.

        Args:
            state (FullState): The current state of the car.
            cmd (ControlInputs): The control inputs for the car.
            dt (float): Increment of time to simulate for.

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
        cmd.steering = np.clip(cmd.steering, -self.max_steer, self.max_steer)

        beta = math.atan2((self.Lr * math.tan(cmd.steering) / self.L), 1.0)
        vx = old_v * math.cos(beta)
        vy = old_v * math.sin(beta)

        Ffy = -self.Cf * ((vy + self.Lf * old_omega) / (vx + 0.0001) - cmd.steering)
        Fry = -self.Cr * (vy - self.Lr * old_omega) / (vx + 0.0001)
        R_x = self.c_r1 * abs(vx)
        F_aero = self.c_a * vx ** 2 # 
        F_load = F_aero + R_x #
        vx = vx + (cmd.throttle - Ffy * math.sin(cmd.steering) / self.m - F_load / self.m + vy * state.omega) * dt
        vy = vy + (Fry / self.m + Ffy * math.cos(cmd.steering) / self.m - vx * state.omega) * dt

        # State update
        state = State()
        state.omega = old_omega + (Ffy * self.Lf * math.cos(cmd.steering) - Fry * self.Lr) / self.Iz * dt
        
        state.yaw = old_yaw + old_omega * dt
        state.yaw = normalize_angle(state.yaw)

        state.x = old_x + vx * math.cos(old_yaw) * dt - vy * math.sin(old_yaw) * dt
        state.y = old_y + vx * math.sin(old_yaw) * dt + vy * math.cos(old_yaw) * dt

        state.v = math.sqrt(vx ** 2 + vy ** 2)
        state.v = np.clip(state.v, self.min_speed, self.max_speed)

        return state