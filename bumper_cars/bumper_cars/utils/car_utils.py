#!/usr/bin/env python3

import numpy as np
from lar_msgs.msg import CarControlStamped, CarStateStamped
from bumper_cars.classes.State import State

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

# def array_to_state(array):
#     """
#     Convert an array to a State object.

#     Args:
#         array (list): The array containing the state values.

#     Returns:
#         State: The State object with the values from the array.
#     """
#     state = State()
#     state.x = array[0]
#     state.y = array[1]
#     state.yaw = array[2]
#     state.v = array[3]
#     return state

# def state_to_array(state: State):
#     """
#     Convert a State object to a numpy array.

#     Args:
#         state (State): The State object to be converted.

#     Returns:
#         numpy.ndarray: The converted numpy array.
#     """
#     array = np.zeros((4,1))
#     array[0,0] = state.x
#     array[1,0] = state.y
#     array[2,0] = state.yaw
#     array[3,0] = state.v

#     return array

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
        a (float): Angle in radians.

    Returns:
        numpy.ndarray: Rotated matrix.
    """
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

def find_nearest(array, value):
    """
    Find the nearest value in an array.

    Args:
        array (numpy.ndarray): Input array.
        value: Value to find.

    Returns:
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
    car_state.v = longitudinal_velocity(curr_car.vel_x, curr_car.vel_y, car_state.omega)
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