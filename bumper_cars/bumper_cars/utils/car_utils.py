#!/usr/bin/env python3

import numpy as np
from lar_msgs.msg import CarControlStamped, CarStateStamped
from bumper_cars.classes.State import State
from nav_msgs.msg import Path

from typing import List
from scipy.spatial.transform import Rotation as Rot
import random

# For the parameter file
import pathlib
import json
import time
import matplotlib.pyplot as plt
import math

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
    return float(distance)

def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].
    :param angle: (float)
    :return: (float) Angle in radian in [-pidelta, pi]
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle

def rot_mat_2d(angle):
    """
    Create 2D rotation matrix from an angle

    Parameters
    ----------
    angle :

    Returns
    -------
    A 2D rotation matrix

    Examples
    --------
    >>> angle_mod(-4.0)


    """
    return Rot.from_euler('z', angle).as_matrix()[0:2, 0:2]

def longitudinal_velocity(vx:float, vy:float, angle_rad:float):
    """
    Computes the longitudinal velocity given the velocities on the X/Y axis and
    angle 
    """
    dot_prod = vx*math.cos(angle_rad) + vy*math.sin(angle_rad)
    return np.sign(dot_prod) * math.sqrt(vx**2 + vy**2)

def rotateMatrix(a):
    """
    Rotate a matrix by an angle.

    Args:
    ----------
        a (float): Angle in radians.

    Returns:
    ----------
        numpy.ndarray: Rotated matrix.
    """
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

def find_nearest(array, value):
    """
    Find the nearest value in an array.

    Args:
    ----------
        array (numpy.ndarray): Input array.
        value: Value to find.

    Returns:
    ----------
        float: Nearest value in the array.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def carStateStamped_to_State(curr_car: CarStateStamped) -> State:
    car_state = State()
    car_state.x = curr_car.pos_x
    car_state.y = curr_car.pos_y
    car_state.yaw = normalize_angle(curr_car.turn_angle) #turn angle is with respect to the car, not wheel
    car_state.omega = curr_car.turn_rate
    car_state.v = longitudinal_velocity(curr_car.vel_x, curr_car.vel_y, car_state.yaw)
    return car_state


def carStateStamped_to_array(car_list: List[CarStateStamped]) -> np.array:
    """
    Convert a list of CarStateStamped objects to a numpy array.

    Args:
        car_list (List[CarStateStamped]): A list of CarStateStamped objects.

    Returns:
        np.array: A numpy array containing the converted car states.

    """
    car_num = len(car_list)
    car_array = np.empty((car_num, 5))
    for i in range(len(car_list)):
        car_i = car_list[i]
        norm_angle = normalize_angle(car_i.turn_angle)
        curr_v = longitudinal_velocity(car_i.vel_x, car_i.vel_y, norm_angle)
        np_car_i = np.array([car_i.pos_x, car_i.pos_y, norm_angle, curr_v, car_i.turn_rate])
        car_array[i,:] = np_car_i
    return car_array

def pathList_to_array(path_updated: List[bool], path_list: List[Path]) -> np.array:
    """
    Convert a list of Path objects to a numpy array.

    Args:
        path_list (List[Path]): A list of Path objects.

    Returns:
        np.array: A numpy array containing the converted paths.

    """
    path_num = len(path_list)
    path_array = np.empty((path_num, 2))
    for i in range(len(path_list)):
        path_i = path_list[i]
        if path_updated[i]:
            np_path_i = np.array([path_i.poses[0].pose.position.x, path_i.poses[0].pose.position.y])
            path_array[i,:] = np_path_i
    return path_array

def pure_pursuit_steer_control(target, pose, max_steer=0.5, L=1.0, Lf=0.5, max_speed=1.0):
    """
    Calculates the throttle and steering angle for a pure pursuit steering control algorithm.

    Args:
        target (tuple): The coordinates of the target point.
        pose (Pose): The current pose of the vehicle.

    Returns:
        tuple: A tuple containing the throttle and steering angle.

    """
        
    alpha = normalize_angle(math.atan2(target[1] - pose.y, target[0] - pose.x) - pose.yaw)

    # this if/else condition should fix the buf of the waypoint behind the car
    if alpha > np.pi/2.0:
        delta = max_steer
    elif alpha < -np.pi/2.0:
        delta = -max_steer
    else:
        # ref: https://www.shuffleai.blog/blog/Three_Methods_of_Vehicle_Lateral_Control.html
        delta = normalize_angle(math.atan2(2.0 * L *  math.sin(alpha), Lf))

    # decreasing the desired speed when turning
    if delta > math.radians(10) or delta < -math.radians(10):
        desired_speed = 2
    else:
        desired_speed = max_speed

    delta = np.clip(delta, -max_steer, max_steer)
    # delta = delta
    throttle = 3 * (desired_speed-pose.v)
    return throttle, delta

def transform_point(x, y, ref_x, ref_y, ref_yaw):
    # Translate point
    dx = x - ref_x
    dy = y - ref_y

    # Rotate point
    rel_x = np.cos(ref_yaw) * dx - np.sin(ref_yaw) * dy
    rel_y = np.sin(ref_yaw) * dx + np.cos(ref_yaw) * dy

    return rel_x, rel_y

class PID_controller():
    def __init__(self, initial_pos, target, kp = 0.5, ki = 0.0, kd = 0.04):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.target = target
        self.signal = 0
        self.accumulator = 0
        self.last_reading = initial_pos
        
        self.time_bkp = time.time()

        self.sample_rate = 0.1
    
    def compute_input(self, feedback_value):
        error = abs(self.target - feedback_value)

        self.accumulator += error

        self.signal = self.kp * error + self.ki * self.accumulator - self.kd * ( feedback_value - self.last_reading)/(time.time()-self.time_bkp)
        self.time_bkp = time.time()
    
    def update_state(self, feedback_value):
        self.last_reading = feedback_value

    def update_target(self, new_target):
        self.target = new_target