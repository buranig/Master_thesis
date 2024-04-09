#!/usr/bin/env python3

import yaml
import math
import numpy as np
import os
from lar_utils.car_utils import normalize_angle
from lar_msgs.msg import CarControlStamped, CarStateStamped as State

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

        self.L = yaml_object["L"]       # [m] Wheel base of vehicle
        self.Lr = self.L / 2.0          # [m] Assume CG in middle
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
            self.next_state = self._linear_model_callback
        elif model_type == 'nonlinear':
            self.next_state = self._nonlinear_model_callback
        else:
            raise Exception("Wrong value for 'model_type'")
        
    def set_state(self, car_state : State):
            self.state.x = car_state.x
            self.state.y = car_state.y
            self.state.v = car_state.v
            self.state.yaw = car_state.yaw
            self.state.omega = car_state.omega

    def step(self, cmd: CarControlStamped, dt: float)-> State:
        return self.next_state(cmd, dt)

    def _linear_model_callback(self, cmd: CarControlStamped, dt: float) -> State:
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
        state = State()
        state.x = self.state.x
        state.y = self.state.y
        state.v = self.state.v
        state.yaw = self.state.yaw
        state.omega = self.state.omega

        # Ensure feasible inputs
        cmd.steering = np.clip(cmd.steering, -self.max_steer, self.max_steer)

        # State update
        state.x = self.state.x + self.state.v * np.cos(self.state.yaw) * dt
        state.y = self.state.y + self.state.v * np.sin(self.state.yaw) * dt

        state.v = self.state.v + cmd.throttle * dt
        state.v = np.clip(state.v, self.min_speed, self.max_speed)

        state.yaw = self.state.yaw + self.state.v / self.L * np.tan(cmd.steering) * dt
        state.yaw = normalize_angle(state.yaw)

        return state
    
    def _nonlinear_model_callback(self, cmd:CarControlStamped, dt: float) -> State:
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
        state = State()
        state.x = self.state.x
        state.y = self.state.y
        state.v = self.state.v
        state.yaw = self.state.yaw
        state.omega = self.state.omega

        # Ensure feasible inputs
        cmd.steering = np.clip(cmd.steering, -self.max_steer, self.max_steer)

        beta = math.atan2((self.Lr * math.tan(cmd.steering) / self.L), 1.0)
        vx = state.v * math.cos(beta)
        vy = state.v * math.sin(beta)

        Ffy = -self.Cf * ((vy + self.Lf * state.omega) / (vx + 0.0001) - cmd.steering)
        Fry = -self.Cr * (vy - self.Lr * state.omega) / (vx + 0.0001)
        R_x = self.c_r1 * abs(vx)
        F_aero = self.c_a * vx ** 2 # 
        F_load = F_aero + R_x #
        vx = vx + (cmd.throttle - Ffy * math.sin(cmd.steering) / self.m - F_load / self.m + vy * state.omega) * dt
        vy = vy + (Fry / self.m + Ffy * math.cos(cmd.steering) / self.m - vx * state.omega) * dt

        # State update
        state.omega = state.omega + (Ffy * self.Lf * math.cos(cmd.steering) - Fry * self.Lr) / self.Iz * dt
        
        state.yaw = state.yaw + state.omega * dt
        state.yaw = normalize_angle(state.yaw)

        state.x = state.x + vx * math.cos(state.yaw) * dt - vy * math.sin(state.yaw) * dt
        state.y = state.y + vx * math.sin(state.yaw) * dt + vy * math.cos(state.yaw) * dt

        state.v = math.sqrt(vx ** 2 + vy ** 2)
        state.v = np.clip(state.v, self.min_speed, self.max_speed)

        return state