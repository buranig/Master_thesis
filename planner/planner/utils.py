#!/usr/bin/env python3

from custom_message.msg import ControlInputs, State, FullState, Coordinate, Path
import numpy as np
# from planner.cubic_spline_planner import *
# from planner.frenet import *
# from planner.predict_traj import *
from scipy.spatial.transform import Rotation as Rot
import random

# For the parameter file
import pathlib
import json
import time
import matplotlib.pyplot as plt
import math
color_dict = {0: 'r', 1: 'b', 2: 'g', 3: 'y', 4: 'm', 5: 'c', 6: 'k', 7: 'tab:orange', 8: 'tab:brown', 9: 'tab:gray', 10: 'tab:olive'}

# TODO: import all this parameters from a config file so that we can easily change them in one place
path = pathlib.Path('/home/giacomo/thesis_ws/src/bumper_cars/params.json')
# Opening JSON file
with open(path, 'r') as openfile:
    # Reading from json file
    json_object = json.load(openfile)

max_steer = json_object["CBF_simple"]["max_steer"]   # [rad] max steering angle
max_speed = json_object["Car_model"]["max_speed"] # [m/s]
min_speed = json_object["Car_model"]["min_speed"]  # [m/s]
car_max_acc = json_object["Controller"]["max_acc"]
car_min_acc = json_object["Controller"]["min_acc"]
dt = json_object["Controller"]["dt"] 
safety_init = json_object["safety"]
width_init = json_object["width"]
height_init = json_object["height"]
min_dist = json_object["min_dist"]
L = json_object["Car_model"]["L"] # [m] Wheel base of vehicle
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

def motion(x, u, dt):
    """
    Motion model for a robot.

    Args:
        x (list): Initial state of the robot [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)].
        u (list): Control inputs [throttle, delta].
        dt (float): Time step.

    Returns:
        list: Updated state of the robot.

    """
    delta = u[1]
    delta = np.clip(delta, -max_steer, max_steer)
    throttle = u[0]
    throttle = np.clip(throttle, car_min_acc, car_max_acc)

    x[0] = x[0] + x[3] * math.cos(x[2]) * dt
    x[1] = x[1] + x[3] * math.sin(x[2]) * dt
    x[2] = x[2] + x[3] / Lr * np.sin(np.arctan2(Lr/L * np.tan(delta),1)) * dt
    # x[2] = x[2] + x[3] / L * math.tan(delta) * dt

    x[2] = normalize_angle(x[2])
    x[3] = x[3] + throttle * dt
    x[3] = np.clip(x[3], min_speed, max_speed)

    return x

def linear_model_callback(initial_state: State, cmd: ControlInputs):
    """
    Computes the next state using a non-linear kinematic model.

    Args:
        initial_state (State): The initial state.
        cmd (ControlInputs): The control inputs.

    Returns:
        State: The next state.
    """

    dt = 0.1
    state = State()
    cmd.delta = np.clip(cmd.delta, -max_steer, max_steer)

    state.x = initial_state.x + initial_state.v * np.cos(initial_state.yaw) * dt
    state.y = initial_state.y + initial_state.v * np.sin(initial_state.yaw) * dt
    state.yaw = initial_state.yaw + initial_state.v / L * np.tan(cmd.delta) * dt
    state.yaw = normalize_angle(state.yaw)
    state.v = initial_state.v + cmd.throttle * dt
    state.v = np.clip(state.v, min_speed, max_speed)

    return state

def nonlinear_model_callback(initial_state: State, cmd: ControlInputs, old_time: float):
    """
    Calculates the state of the system using a nonlinear dynamic model.

    Args:
        initial_state (State): The initial state of the system.
        cmd (ControlInputs): The control inputs for the system.
        old_time (float): The previous time at which the model was updated.

    Returns:
        tuple: A tuple containing the updated state of the system and the current time.
    """

    dt = 0.1
    state = State()
    #dt = time.time() - old_time
    cmd.delta = np.clip(cmd.delta, -max_steer, max_steer)
    cmd.delta = np.clip(cmd.delta, -max_steer, max_steer)

    beta = math.atan2((Lr * math.tan(cmd.delta) / L), 1.0)
    vx = initial_state.v * math.cos(beta)
    vy = initial_state.v * math.sin(beta)

    Ffy = -Cf * ((vy + Lf * initial_state.omega) / (vx + 0.0001) - cmd.delta)
    Fry = -Cr * (vy - Lr * initial_state.omega) / (vx + 0.0001)
    R_x = c_r1 * abs(vx)
    F_aero = c_a * vx ** 2 # 
    F_load = F_aero + R_x #
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
        delta = normalize_angle(math.atan2(2.0 * WB *  math.sin(alpha), Lf))

    # decreasing the desired speed when turning
    if delta > math.radians(10) or delta < -math.radians(10):
        desired_speed = 2
    else:
        desired_speed = max_speed

    delta = np.clip(delta, -max_steer, max_steer)
    # delta = delta
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

def array_to_state(array):
    """
    Convert an array to a State object.

    Args:
        array (list): The array containing the state values.

    Returns:
        State: The State object with the values from the array.
    """
    state = State()
    state.x = array[0]
    state.y = array[1]
    state.yaw = array[2]
    state.v = array[3]
    return state

def state_to_array(state: State):
    """
    Convert a State object to a numpy array.

    Args:
        state (State): The State object to be converted.

    Returns:
        numpy.ndarray: The converted numpy array.
    """
    array = np.zeros((4,1))
    array[0,0] = state.x
    array[1,0] = state.y
    array[2,0] = state.yaw
    array[3,0] = state.v

    return array

def plot_arrow(x, y, yaw, length=0.5, width=0.1, color=None):  # pragma: no cover
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                head_length=width, head_width=width, color=color)
        
def plot_map(width=100, height=100):
        corner_x = [-width/2.0, width/2.0, width/2.0, -width/2.0, -width/2.0]
        corner_y = [height/2.0, height/2.0, -height/2.0, -height/2.0, height/2.0]

        plt.plot(corner_x, corner_y, label="map", linewidth=4)

def plot_path(path: Path):
        x = []
        y = []
        for coord in path.path:
            x.append(coord.x)
            y.append(coord.y)
        plt.scatter(x, y, marker='.', s=10)
        plt.scatter(x[0], y[0], marker='x', s=20)

def normalize_angle_array(angle):
    """
    Normalize an angle to [-pi, pi].
    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    angle[angle[:] > np.pi] -= 2.0 * np.pi

    angle[angle[:] < -np.pi] += 2.0 * np.pi

    return angle

def plot_covariance_ellipse(x, y, cov, chi2=3.0, color="-r", ax=None):
    """
    This function plots an ellipse that represents a covariance matrix. The ellipse is centered at (x, y) and its shape, size and rotation are determined by the covariance matrix.

    Parameters:
    x : float
        The x-coordinate of the center of the ellipse.
    y : float
        The y-coordinate of the center of the ellipse.
    cov : numpy.ndarray
        A 2x2 covariance matrix that determines the shape, size, and rotation of the ellipse.
    chi2 : float, optional
        A scalar value that scales the ellipse size. This value is typically set based on chi-squared distribution quantiles to achieve certain confidence levels (e.g., 3.0 corresponds to ~95% confidence for a 2D Gaussian). Defaults to 3.0.
    color : str, optional
        The color and line style of the ellipse plot, following matplotlib conventions. Defaults to "-r" (a red solid line).
    ax : matplotlib.axes.Axes, optional
        The Axes object to draw the ellipse on. If None (default), a new figure and axes are created.

    Returns:
    None
        This function plots the covariance ellipse on the specified axes.
    """
    eig_val, eig_vec = np.linalg.eig(cov)

    if eig_val[0] >= eig_val[1]:
        big_ind = 0
        small_ind = 1
    else:
        big_ind = 1
        small_ind = 0
    a = math.sqrt(chi2 * eig_val[big_ind])
    b = math.sqrt(chi2 * eig_val[small_ind])
    angle = math.atan2(eig_vec[1, big_ind], eig_vec[0, big_ind])
    plot_ellipse(x, y, a, b, angle, color=color, ax=ax)

def plot_ellipse(x, y, a, b, angle, color="-r", ax=None, **kwargs):
    """
    This function plots an ellipse based on the given parameters.

    Parameters
    ----------
    x : (float) The x-coordinate of the center of the ellipse.
    y : (float) The y-coordinate of the center of the ellipse.
    a : (float) The length of the semi-major axis of the ellipse.
    b : (float) The length of the semi-minor axis of the ellipse.
    angle : (float) The rotation angle of the ellipse, in radians.
    color : (str, optional) The color and line style of the ellipse plot, following matplotlib conventions. Defaults to "-r" (a red solid line).
    ax : (matplotlib.axes.Axes, optional) The Axes object to draw the ellipse on. If None (default), a new figure and axes are created.
    **kwargs: Additional keyword arguments to pass to plt.plot or ax.plot.

    Returns
    ---------
    None. This function plots the ellipse based on the specified parameters.
    """

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)
    px = [a * math.cos(it) for it in t]
    py = [b * math.sin(it) for it in t]
    fx = rot_mat_2d(angle) @ (np.array([px, py]))
    px = np.array(fx[0, :] + x).flatten()
    py = np.array(fx[1, :] + y).flatten()
    if ax is None:
        plt.plot(px, py, color, **kwargs)
    else:
        ax.plot(px, py, color, **kwargs)

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

def samplegrid(width_init, height_init, min_dist, robot_num, safety_init):
    """
    Generate random grid coordinates for robots in a given map.

    Args:
        width_init (float): Initial width of the map.
        height_init (float): Initial height of the map.
        min_dist (float): Minimum distance between robots.
        robot_num (int): Number of robots.
        safety_init (float): Safety border around the map boundaries.

    Returns:
        tuple: A tuple containing the x-coordinates, y-coordinates, yaw angles,
               velocities, angular velocities, and model types for the robots.
    """
    # defining the boundaries
    safety = safety_init # safety border around the map boundaries
    width = width_init - safety
    height = height_init - safety
    # min_dis
    N = int(width/min_dist)
    M = int(height/min_dist)
    x_mesh = np.linspace(-width/2, width/2, N)
    y_mesh = np.linspace(-height/2, height/2, M)

    list = [divmod(i, M) for i in random.sample(range(N * M), robot_num)]
    list = np.array(list)
    x = x_mesh[list[:, 0]]
    y = y_mesh[list[:, 1]]
    yaw = []
    while len(yaw)<robot_num:
        yaw.append(np.radians(random.randint(-180, 180)))

    v = robot_num * [0.0]
    omega = robot_num * [0.0]
    model_type = robot_num * ['linear']
    return x.tolist(), y.tolist(), yaw, v, omega, model_type

def circular_samples(width_init, height_init, R, robot_num, safety_init):
    """
    Generate random grid coordinates for robots in a given map.

    Args:
        width_init (float): Initial width of the map.
        height_init (float): Initial height of the map.
        min_dist (float): Minimum distance between robots.
        robot_num (int): Number of robots.
        safety_init (float): Safety border around the map boundaries.

    Returns:
        tuple: A tuple containing the x-coordinates, y-coordinates, yaw angles,
               velocities, angular velocities, and model types for the robots.
    """
    # defining the boundaries
    safety = safety_init # safety border around the map boundaries
    width = width_init - safety
    height = height_init - safety

    radius = np.sqrt(width**2 + height**2)/2

    assert R < radius and R>=0, "The radius is too large for the map"

    # create robot_num points on a circle and save the angles as yaw
    x = []
    y = []
    yaw = []
    for i in range(robot_num):
        x.append(R*np.cos(2*np.pi*i/robot_num))
        y.append(R*np.sin(2*np.pi*i/robot_num))
        yaw.append(2*np.pi*i/robot_num+np.pi)

    v = robot_num * [0.0]
    omega = robot_num * [0.0]
    model_type = robot_num * ['linear']
    return x, y, yaw, v, omega, model_type

def dist(point1, point2):
    """
    Calculates the Euclidean distance between two points.

    :param point1: (tuple) x, y coordinates of the first point
    :param point2: (tuple) x, y coordinates of the second point
    :return: (float) Euclidean distance between the two points
    """
    x1, y1 = point1
    x2, y2 = point2

    x1 = float(x1)
    x2 = float(x2)
    y1 = float(y1)
    y2 = float(y2)

    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return distance

def update_path(path: Path):
    """
    Updates the path by removing the first waypoint and adding a new random waypoint.

    Removes the first waypoint from the path and adds a new random waypoint within the specified boundaries.
    """
    safety = 2
    path.pop(0)
    path.append(Coordinate(x=float(random.randint(-width_init/2+safety, width_init/2-safety)), y=float(random.randint(-height_init/2+safety, height_init/2-safety))))
    return path

def create_path(len_path=5):
    """
    Creates a random path.

    Generates a random path by creating a list of waypoints within the specified boundaries.
    """
    safety = 2
    path = []
    while len(path)<len_path:
        path.append(Coordinate(x=float(random.randint(-width_init/2+safety, width_init/2-safety)), y=float(random.randint(-height_init/2+safety, height_init/2-safety))))
    return path

def create_seed(len_path=5):
    """
    Creates a random path.

    Generates a random path by creating a list of waypoints within the specified boundaries.
    """
    path = []
    safety = 2
    while len(path)<len_path:
        path.append([float(random.randint(-width_init/2+safety, width_init/2-safety)), float(random.randint(-height_init/2+safety, height_init/2-safety))])
    return path
    
def plot_robot(x, y, yaw, i): 
    """
    Plot the robot.

    Args:
        x (float): X-coordinate of the robot.
        y (float): Y-coordinate of the robot.
        yaw (float): Yaw angle of the robot.
        i (int): Index of the robot.
    """
    outline = np.array([[-L / 2, L / 2,
                            (L / 2), -L / 2,
                            -L / 2],
                        [WB / 2, WB / 2,
                            - WB / 2, -WB / 2,
                            WB / 2]])
    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                        [-math.sin(yaw), math.cos(yaw)]])
    outline = (outline.T.dot(Rot1)).T
    outline[0, :] += x
    outline[1, :] += y
    plt.plot(np.array(outline[0, :]).flatten(),
                np.array(outline[1, :]).flatten(), color_dict[i], label="Robot " + str(i))