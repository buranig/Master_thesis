#!/usr/bin/env python3

"""
The intent of this file is to predict the trajectory of a pure pursuit controller given the start, end position and car model
"""

import math
import numpy as np
from custom_message.msg import ControlInputs, State, FullState, Coordinate, Path
import time
import matplotlib.pyplot as plt
from rclpy.node import Node

# For the parameter file
import pathlib
import json

path = pathlib.Path('/home/giacomo/thesis_ws/src/bumper_cars/params.json')
# Opening JSON file
with open(path, 'r') as openfile:
    # Reading from json file
    json_object = json.load(openfile)

max_steer = json_object["Car_model"]["max_steer"] # [rad] max steering angle
max_speed = json_object["Car_model"]["max_speed"] # [m/s]
min_speed = json_object["Car_model"]["min_speed"] # [m/s]
dt = json_object["Controller"]["dt"]  # [s] Time step
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

WB = json_object["Controller"]["WB"] 
L_d = json_object["Controller"]["L_d"] 

debug = False

def predict_trajectory(initial_state: State, target, linear=True):
    """
    Predicts the trajectory of a vehicle from an initial state to a target point.

    Args:
        initial_state (State): The initial state of the vehicle.
        target (tuple): The target point (x, y) to reach.

    Returns:
        Path: The predicted trajectory as a Path object.
    """
    traj = []  # List to store the trajectory points

    # Append the initial state coordinates to the trajectory
    traj.append(Coordinate(x=initial_state.x, y=initial_state.y))

    cmd = ControlInputs()  # Create a ControlInputs object
    old_time = time.time()  # Get the current time

    # Calculate the control inputs for the initial state
    cmd.throttle, cmd.delta = pure_pursuit_steer_control(target, initial_state)

    # Update the state using the linear model and append the new state coordinates to the trajectory
    if linear:
        new_state, old_time = linear_model_callback(initial_state, cmd)
    else:
        new_state, old_time = nonlinear_model_callback(initial_state, cmd)
    traj.append(Coordinate(x=new_state.x, y=new_state.y))

    # Continue predicting the trajectory until the distance between the last point and the target is less than 10
    while dist(point1=(traj[-1].x, traj[-1].y), point2=target) > 0.5:

        # Calculate the control inputs for the new state
        cmd.throttle, cmd.delta = pure_pursuit_steer_control(target, new_state)

        # Update the state using the linear model and append the new state coordinates to the trajectory
        if linear:
            new_state, old_time = linear_model_callback(new_state, cmd)
        else:
            new_state, old_time = nonlinear_model_callback(new_state, cmd)
        traj.append(Coordinate(x=new_state.x, y=new_state.y))

        if debug:
            # Plot the trajectory and other elements for debugging
            plt.cla()
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            
            plot_path(traj)
            plt.plot(initial_state.x, initial_state.y, 'k.')
            plt.plot(target[0], target[1], 'b.')
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.000001)
    
    # Reduce the number of points in the trajectory for efficiency
    # traj = traj[0:-1:5]
    plt.show()

    return Path(path=traj)

def plot_path(path: Path):
        x = []
        y = []
        for coord in path:
            x.append(coord.x)
            y.append(coord.y)
        plt.scatter(x, y, marker='.', s=10)
        plt.scatter(x[0], y[0], marker='x', s=20)

def linear_model_callback(initial_state: State, cmd: ControlInputs):
    """
    Calculates the next state based on the linear model.

    Args:
        initial_state (State): The initial state of the vehicle.
        cmd (ControlInputs): The control inputs for the vehicle.
        old_time (float): The previous time.

    Returns:
        Tuple[State, float]: The next state and the current time.
    """
    state = State()
    cmd.delta = np.clip(cmd.delta, -max_steer, max_steer)

    state.x = initial_state.x + initial_state.v * np.cos(initial_state.yaw) * dt
    state.y = initial_state.y + initial_state.v * np.sin(initial_state.yaw) * dt
    state.yaw = initial_state.yaw + initial_state.v / L * np.tan(cmd.delta) * dt
    state.yaw = normalize_angle(state.yaw)
    state.v = initial_state.v + cmd.throttle * dt
    state.v = np.clip(state.v, min_speed, max_speed)

    return state, time.time()

def nonlinear_model_callback(initial_state: State, cmd: ControlInputs):
    """
    Nonlinear model callback function.

    Args:
        initial_state (State): The initial state of the system.
        cmd (ControlInputs): The control inputs.
        old_time (float): The previous time.

    Returns:
        Tuple[State, float]: The updated state and the current time.
    """

    state = State()

    cmd.delta = np.clip(cmd.delta, -max_steer, max_steer)

    beta = math.atan2((Lr * math.tan(cmd.delta) / L), 1.0)
    vx = initial_state.v * math.cos(beta)
    vy = initial_state.v * math.sin(beta)

    Ffy = -Cf * ((vy + Lf * initial_state.omega) / (vx + 0.0001) - cmd.delta)
    Fry = -Cr * (vy - Lr * initial_state.omega) / (vx + 0.0001)
    R_x = c_r1 * abs(vx)
    F_aero = c_a * vx ** 2
    F_load = F_aero + R_x

    state.omega = initial_state.omega + (Ffy * Lf * math.cos(cmd.delta) - Fry * Lr) / Iz * dt
    vx = vx + (cmd.throttle - Ffy * math.sin(cmd.delta) / m - F_load / m + vy * state.omega) * dt
    vy = vy + (Fry / m + Ffy * math.cos(cmd.delta) / m - vx * state.omega) * dt

    state.yaw = initial_state.yaw + state.omega * dt
    state.yaw = normalize_angle(state.yaw)

    state.x = initial_state.x + vx * math.cos(state.yaw) * dt - vy * math.sin(state.yaw) * dt
    state.y = initial_state.y + vx * math.sin(state.yaw) * dt + vy * math.cos(state.yaw) * dt

    state.v = math.sqrt(vx ** 2 + vy ** 2)
    state.v = np.clip(state.v, min_speed, max_speed)

    return state, time.time()

def pure_pursuit_steer_control(target, pose):
    """
    Calculates the throttle and steering angle for the pure pursuit steering control algorithm.

    Args:
        target (tuple): The target coordinates (x, y) to track.
        pose (Pose): The current pose of the vehicle.

    Returns:
        tuple: A tuple containing the throttle and steering angle (throttle, delta).
    """
        
    alpha = normalize_angle(math.atan2(target[1] - pose.y, target[0] - pose.x) - pose.yaw)

    # this if/else condition should fix the buf of the waypoint behind the car
    if alpha > np.pi/2.0:
        delta = max_steer
    elif alpha < -np.pi/2.0:
        delta = -max_steer
    else:
        # ref: https://www.shuffleai.blog/blog/Three_Methods_of_Vehicle_Lateral_Control.html
        delta = normalize_angle(math.atan2(2.0 * WB *  math.sin(alpha), L_d))

    # decreasing the desired speed when turning
    if delta > math.radians(10) or delta < -math.radians(10):
        desired_speed = 2
    else:
        desired_speed = max_speed

    print(f'Steering angle: {delta} and desired speed: {desired_speed}')
    throttle = 3 * (desired_speed-pose.v)
    return throttle, delta

@staticmethod
def dist(point1, point2):
    """
    Calculate the Euclidean distance between two points.

    Args:
        point1 (tuple): The coordinates of the first point in the form (x1, y1).
        point2 (tuple): The coordinates of the second point in the form (x2, y2).

    Returns:
        float: The Euclidean distance between the two points.
    """
    x1, y1 = point1
    x2, y2 = point2

    x1 = float(x1)
    x2 = float(x2)
    y1 = float(y1)
    y2 = float(y2)

    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return distance

def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].
    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle


def main():
    initial_state = State(x=0.0, y=0.0, yaw=0.0, v=0.0, omega=0.0)
    target = [1, 2]
    # trajectory, tx, ty = predict_trajectory(initial_state, target)
    trajectory = predict_trajectory(initial_state, target, True)
    trajectory1 = predict_trajectory(initial_state, target, False)
    print(len(trajectory.path))
    x = []
    y = []
    for coord in trajectory.path:
    
        x.append(coord.x)
        y.append(coord.y)
    plt.plot(x, y, 'r', label='Linear model')

    x = []
    y = []
    for coord in trajectory1.path:
    
        x.append(coord.x)
        y.append(coord.y)
    plt.plot(x, y, 'b', label='Nonlinear model')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()




