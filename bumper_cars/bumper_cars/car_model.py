#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from custom_message.msg import ControlInputs, State, FullState, MultiControl, MultiState
import numpy as np
import time
import message_filters
import math

# For the parameter file
import pathlib
import json

path = pathlib.Path('/home/giacomo/thesis_ws/src/bumper_cars/params.json')
# Opening JSON file
with open(path, 'r') as openfile:
    # Reading from json file
    json_object = json.load(openfile)

controller_type = json_object["Controller"]["controller_type"]
if controller_type == "DWA":
    max_steer = json_object["DWA"]["max_steer"] # [rad] max steering angle
    max_speed = json_object["DWA"]["max_speed"] # [m/s]
    min_speed = json_object["DWA"]["min_speed"] # [m/s]
else:
    max_steer = json_object["Car_model"]["max_steer"] # [rad] max steering angle
    max_speed = json_object["Car_model"]["max_speed"] # [m/s]
    min_speed = json_object["Car_model"]["min_speed"] # [m/s]
L = json_object["Car_model"]["L"]  # [m] Wheel base of vehicle
Lr = L / 2.0  # [m]
Lf = L - Lr
Cf = json_object["Car_model"]["Cf"]  # N/rad
Cr = json_object["Car_model"]["Cr"] # N/rad
Iz = json_object["Car_model"]["Iz"]  # kg/m2
m = json_object["Car_model"]["m"]  # kg
# Aerodynamic and friction coefficients
c_a = json_object["Car_model"]["c_a"]
c_r1 = json_object["Car_model"]["c_r1"]

robot_num = json_object["robot_num"]
timer_freq = json_object["timer_freq"]

class CarModel(Node):
    """
    Represents the car model and its behavior.

    This class initializes the car model with the given parameters and provides methods for updating the car state based on control inputs.
    It also publishes the car's full state and handles the timer callback for publishing the state periodically.
    """

    def __init__(self):
        super().__init__("robot_model")

        self.declare_parameters(
            namespace='',
            parameters=[
                ('model_type', rclpy.Parameter.Type.STRING_ARRAY),
                ('x0', rclpy.Parameter.Type.DOUBLE_ARRAY),
                ('y0', rclpy.Parameter.Type.DOUBLE_ARRAY),
                ('yaw', rclpy.Parameter.Type.DOUBLE_ARRAY),
                ('v', rclpy.Parameter.Type.DOUBLE_ARRAY),
                ('omega', rclpy.Parameter.Type.DOUBLE_ARRAY)
            ]
        )

        self.x0 = self.get_parameter('x0').get_parameter_value().double_array_value
        self.y0 = self.get_parameter('y0').get_parameter_value().double_array_value
        self.yaw = self.get_parameter('yaw').get_parameter_value().double_array_value
        self.v = self.get_parameter('v').get_parameter_value().double_array_value
        self.omega = self.get_parameter('omega').get_parameter_value().double_array_value
        self.model_type = self.get_parameter('model_type').get_parameter_value().string_array_value

        self.multi_state = MultiState()
        self.old_time = []

        # Initializing the robots TODO add in the launch file parameters the initial control--> here it's hardcoded
        for i in range(robot_num):
            self.multi_state.multiple_state.append(FullState(x=self.x0[i], y=self.y0[i], yaw=self.yaw[i], v=self.v[i], omega=self.omega[i],
                                                                                                delta=0.0, throttle=0.0))
            self.old_time.append(time.time())
            
        # self.get_logger().info("MultiState: " + str(self.multi_state))

        # Initializing the state publishers/subscribers
        self.control_sub = self.create_subscription(MultiControl, '/robot_control', self.general_model_callback, 10)
        self.fullstate_publisher_ = self.create_publisher(MultiState, "robot_fullstate", 60)
        self.get_logger().info("Robots model initialized correctly")

        self.timer = self.create_timer(timer_freq, self.timer_callback)

    def general_model_callback(self, control: MultiControl):
        """
        Callback function for handling general model control inputs.

        This function is called when new control inputs are received.
        It updates the car state based on the model type and the control inputs for each robot.
        """

        for i in range(robot_num):
            if self.model_type[0] == 'linear':
                self.multi_state.multiple_state[i], self.old_time[i] = self.linear_model_callback(self.multi_state.multiple_state[i], control.multi_control[i], self.old_time[i])
                # self.get_logger().info("Speed of robot " + str(i) + ": " + str(self.multi_state.multiple_state[i].v))
            elif self.model_type[0] == 'nonlinear':
                self.multi_state.multiple_state[i], self.old_time[i] = self.nonlinear_model_callback(self.multi_state.multiple_state[i], control.multi_control[i], self.old_time[i])

    def linear_model_callback(self, initial_state: FullState, cmd: ControlInputs, old_time: float):
        """
        Update the car state using a non-linear kinematic model.

        This function calculates the new state of the car based on the initial state, control inputs, and the time elapsed since the last update.
        It returns the updated state and the current time.

        Args:
            initial_state (FullState): The initial state of the car.
            cmd (ControlInputs): The control inputs for the car.
            old_time (float): The time of the last update.

        Returns:
            FullState: The updated state of the car.
            float: The current time.
        """

        if controller_type == "DWA" or controller_type == "LBP" or controller_type == "MPC":
            dt = 0.1
            # dt = time.time() - old_time
        else:
            dt = time.time() - old_time
        
        self.get_logger().info("a: " + str(cmd.throttle) + " delta: " + str(cmd.delta))
            
        state = FullState()
        cmd.delta = np.clip(cmd.delta, -max_steer, max_steer)

        state.x = initial_state.x + initial_state.v * np.cos(initial_state.yaw) * dt
        state.y = initial_state.y + initial_state.v * np.sin(initial_state.yaw) * dt
        
        state.v = initial_state.v + cmd.throttle * dt
        state.v = np.clip(state.v, min_speed, max_speed)
        state.yaw = initial_state.yaw + initial_state.v / L * np.tan(cmd.delta) * dt
        state.yaw = self.normalize_angle(state.yaw)
        state.delta = cmd.delta
        state.throttle = cmd.throttle

        return state, time.time()
    
    def nonlinear_model_callback(self, state: FullState, cmd: ControlInputs, old_time: float):
        """
        Update the car state using a nonlinear dynamic model.

        This function calculates the new state of the car based on the current state, control inputs, and the time elapsed since the last update.
        It returns the updated state and the current time.

        Args:
            state (FullState): The current state of the car.
            cmd (ControlInputs): The control inputs for the car.
            old_time (float): The time of the last update.

        Returns:
            FullState: The updated state of the car.
            float: The current time.
        """

        dt = time.time() - old_time
        cmd.delta = np.clip(cmd.delta, -max_steer, max_steer)

        beta = math.atan2((Lr * math.tan(cmd.delta) / L), 1.0)
        vx = state.v * math.cos(beta)
        vy = state.v * math.sin(beta)

        Ffy = -Cf * ((vy + Lf * state.omega) / (vx + 0.0001) - cmd.delta)
        Fry = -Cr * (vy - Lr * state.omega) / (vx + 0.0001)
        R_x = c_r1 * abs(vx)
        F_aero = c_a * vx ** 2 # 
        F_load = F_aero + R_x #
        state.omega = state.omega + (Ffy * Lf * math.cos(cmd.delta) - Fry * Lr) / Iz * dt
        vx = vx + (cmd.throttle - Ffy * math.sin(cmd.delta) / m - F_load / m + vy * state.omega) * dt
        vy = vy + (Fry / m + Ffy * math.cos(cmd.delta) / m - vx * state.omega) * dt

        state.yaw = state.yaw + state.omega * dt
        state.yaw = self.normalize_angle(state.yaw)

        state.x = state.x + vx * math.cos(state.yaw) * dt - vy * math.sin(state.yaw) * dt
        state.y = state.y + vx * math.sin(state.yaw) * dt + vy * math.cos(state.yaw) * dt

        state.v = math.sqrt(vx ** 2 + vy ** 2)
        state.v = np.clip(state.v, min_speed, max_speed)

        state.delta = cmd.delta
        state.throttle = cmd.throttle

        return state, time.time()
    
    def timer_callback(self):
        """
        Timer callback function for publishing the car's full state.

        This function is called periodically based on the timer frequency.
        It publishes the car's full state.
        """

        self.fullstate_publisher_.publish(self.multi_state)

    def normalize_angle(self, angle):
        """
        Normalize an angle to [-pi, pi].

        Args:
            angle (float): The angle to normalize.

        Returns:
            float: The normalized angle in the range [-pi, pi].
        """
        while angle > np.pi:
            angle -= 2.0 * np.pi

        while angle < -np.pi:
            angle += 2.0 * np.pi

        return angle
        
        
def main(args=None):
    rclpy.init(args=args)
    
    node = CarModel()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()