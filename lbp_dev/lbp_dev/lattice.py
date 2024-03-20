"""

State lattice planner with model predictive trajectory generator

author: Atsushi Sakai (@Atsushi_twi)

- plookuptable.csv is generated with this script:
https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning
/ModelPredictiveTrajectoryGenerator/lookup_table_generator.py

Ref:

- State Space Sampling of Feasible Motions for High-Performance Mobile Robot
Navigation in Complex Environments
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.187.8210&rep=rep1
&type=pdf

"""
import sys
import os
from matplotlib import pyplot as plt
import numpy as np
import math
import pathlib
from custom_message.msg import State, ControlInputs
from planner import utils as utils
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import lattice_planner as planner
import lattice_motion_model as motion_model

TABLE_PATH = os.path.dirname(os.path.abspath(__file__)) + "/lookup_table.csv"

show_animation = True


def search_nearest_one_from_lookup_table(t_x, t_y, t_yaw, lookup_table):
    mind = float("inf")
    minid = -1

    for (i, table) in enumerate(lookup_table):
        dx = t_x - table[0]
        dy = t_y - table[1]
        dyaw = t_yaw - table[2]
        d = math.sqrt(dx ** 2 + dy ** 2 + dyaw ** 2)
        if d <= mind:
            minid = i
            mind = d

    return lookup_table[minid]


def get_lookup_table(table_path):
    return np.loadtxt(table_path, delimiter=',', skiprows=1)


def generate_path(target_states, k0, v, k=False):
    """
    Generates a path based on the given target states, initial steering angle, velocity, and a flag indicating whether to use a specific value for k.

    Args:
        target_states (list): List of target states [x, y, yaw].
        k0 (float): Initial steering angle.
        v (float): Velocity.
        k (bool, optional): Flag indicating whether to use a specific value for k. Defaults to False.

    Returns:
        list: List of generated paths [x, y, yaw, p, kp].
    """
    
    lookup_table = get_lookup_table(TABLE_PATH)
    result = []

    for state in target_states:
        target = motion_model.State(x=state[0], y=state[1], yaw=state[2])
        initial_state = State(x=0.0, y=0.0, yaw=0.0, v=0.0, omega=0.0)
        cmd = ControlInputs()
        cmd.throttle, cmd.delta = utils.pure_pursuit_steer_control([target.x,target.y], initial_state)
        print(cmd.delta)
        k0 = cmd.delta
        if k:
            print(target.x, target.y, target.yaw)
            km = cmd.delta/2
        else:
            km = 0.0
        init_p = np.array(
            [np.hypot(state[0], state[1]), km, 0.0]).reshape(3, 1)

        x, y, yaw, p, kp = planner.optimize_trajectory(target, k0, init_p, v)

        if x is not None:
            print("find good path")
            result.append(
                [x[-1], y[-1], yaw[-1], float(p[0, 0]), float(p[1, 0]), float(p[2, 0])])

    print("finish path generation")
    return result


def calc_uniform_polar_states(nxy, nh, d, a_min, a_max, p_min, p_max):
    """
    Calculate the uniform polar states based on the given parameters.

    Parameters
    ----------
    nxy : int
        Number of position sampling.
    nh : int
        Number of heading sampling.
    d : float
        Distance of terminal state.
    a_min : float
        Position sampling min angle.
    a_max : float
        Position sampling max angle.
    p_min : float
        Heading sampling min angle.
    p_max : float
        Heading sampling max angle.

    Returns
    -------
    list
        List of uniform polar states.

    """
    angle_samples = [i / (nxy - 1) for i in range(nxy)]
    states = sample_states(angle_samples, a_min, a_max, d, p_max, p_min, nh)

    return states


def calc_biased_polar_states(goal_angle, ns, nxy, nh, d, a_min, a_max, p_min, p_max):
    """
    calc biased state

    :param goal_angle: goal orientation for biased sampling
    :param ns: number of biased sampling
    :param nxy: number of position sampling
    :param nxy: number of position sampling
    :param nh: number of heading sampleing
    :param d: distance of terminal state
    :param a_min: position sampling min angle
    :param a_max: position sampling max angle
    :param p_min: heading sampling min angle
    :param p_max: heading sampling max angle
    :return: states list
    """

    asi = [a_min + (a_max - a_min) * i / (ns - 1) for i in range(ns - 1)]
    cnav = [math.pi - abs(i - goal_angle) for i in asi]

    cnav_sum = sum(cnav)
    cnav_max = max(cnav)

    # normalize
    cnav = [(cnav_max - cnav[i]) / (cnav_max * ns - cnav_sum)
            for i in range(ns - 1)]

    csumnav = np.cumsum(cnav)
    di = []
    li = 0
    for i in range(nxy):
        for ii in range(li, ns - 1):
            if ii / ns >= i / (nxy - 1):
                di.append(csumnav[ii])
                li = ii - 1
                break

    states = sample_states(di, a_min, a_max, d, p_max, p_min, nh)

    di = np.degrees(di)
    bins = np.arange(min(di), max(di), 2)
    n, bins, patches = plt.hist(di, bins=bins)
    plt.show()
    return states


def calc_lane_states(l_center, l_heading, l_width, v_width, d, nxy):
    """

    calc lane states

    :param l_center: lane lateral position
    :param l_heading:  lane heading
    :param l_width:  lane width
    :param v_width: vehicle width
    :param d: longitudinal position
    :param nxy: sampling number
    :return: state list
    """
    xc = d
    yc = l_center

    states = []
    for i in range(nxy):
        delta = -0.5 * (l_width - v_width) + \
            (l_width - v_width) * i / (nxy - 1)
        xf = xc - delta * math.sin(l_heading)
        yf = yc + delta * math.cos(l_heading)
        yawf = l_heading
        states.append([xf, yf, yawf])

    return states


def sample_states(angle_samples, a_min, a_max, d, p_max, p_min, nh):
    """
    Generate a list of states based on the given parameters.

    Args:
        angle_samples (list): List of angle samples.
        a_min (float): Minimum angle value.
        a_max (float): Maximum angle value.
        d (float): Distance value.
        p_max (float): Maximum yaw value.
        p_min (float): Minimum yaw value.
        nh (int): Number of yaw samples.

    Returns:
        list: List of states, each represented as [xf, yf, yawf].
    """
    states = []
    for i in angle_samples:
        a = a_min + (a_max - a_min) * i

        for j in range(nh):
            xf = d * math.cos(a)
            yf = d * math.sin(a)
            if nh == 1:
                yawf = (p_max - p_min) / 2 + a
            else:
                yawf = p_min + (p_max - p_min) * j / (nh - 1) + a
            states.append([xf, yf, yawf])

    return states


def uniform_terminal_state_sampling_test1():
    v = 1.0
    k0 = 0.0
    nxy = 5
    nh = 3
    d = 3
    a_min = - np.deg2rad(45.0)
    a_max = np.deg2rad(45.0)
    p_min = - np.deg2rad(45.0)
    p_max = np.deg2rad(45.0)
    states = calc_uniform_polar_states(nxy, nh, d, a_min, a_max, p_min, p_max)
    result = generate_path(states, k0, v)

    for table in result:
        xc, yc, yawc, kp = motion_model.generate_trajectory(
            table[3], table[4], table[5], k0, v)

        if show_animation:
            plt.plot(xc, yc, "-r")

    if show_animation:
        plt.grid(True)
        plt.axis("equal")
        plt.show()

    print("Done")


def uniform_terminal_state_sampling_test2():
    v = 1.0
    k0 = 0.1
    nxy = 6
    nh = 3
    d = 20
    a_min = - np.deg2rad(-10.0)
    a_max = np.deg2rad(45.0)
    p_min = - np.deg2rad(20.0)
    p_max = np.deg2rad(20.0)
    states = calc_uniform_polar_states(nxy, nh, d, a_min, a_max, p_min, p_max)
    result = generate_path(states, k0,v)

    for table in result:
        xc, yc, yawc, kp = motion_model.generate_trajectory(
            table[3], table[4], table[5], k0,v)

        if show_animation:
            plt.plot(xc, yc, "-r")

    if show_animation:
        plt.grid(True)
        plt.axis("equal")
        plt.show()

    print("Done")


def biased_terminal_state_sampling_test1():
    k0 = 0.0
    nxy = 30
    nh = 2
    d = 20
    a_min = np.deg2rad(-45.0)
    a_max = np.deg2rad(45.0)
    p_min = - np.deg2rad(20.0)
    p_max = np.deg2rad(20.0)
    ns = 100
    goal_angle = np.deg2rad(0.0)
    states = calc_biased_polar_states(
        goal_angle, ns, nxy, nh, d, a_min, a_max, p_min, p_max)
    result = generate_path(states, k0)

    for table in result:
        xc, yc, yawc, kp = motion_model.generate_trajectory(
            table[3], table[4], table[5], k0)
        if show_animation:
            plt.plot(xc, yc, "-r")

    if show_animation:
        plt.grid(True)
        plt.axis("equal")
        plt.show()


def biased_terminal_state_sampling_test2():
    k0 = np.deg2rad(10.0)
    nxy = 20
    nh = 1
    d = 20
    a_min = np.deg2rad(0.0)
    a_max = np.deg2rad(45.0)
    p_min = - np.deg2rad(20.0)
    p_max = np.deg2rad(20.0)
    ns = 100
    goal_angle = np.deg2rad(45.0)
    states = calc_biased_polar_states(
        goal_angle, ns, nxy, nh, d, a_min, a_max, p_min, p_max)
    result = generate_path(states, k0)
    
    yaw_samples = []
    for table in result:
        xc, yc, yawc, kp = motion_model.generate_trajectory(
            table[3], table[4], table[5], k0)
        yaw_samples.append(yawc[-1])

        if show_animation:
            plt.plot(xc, yc, "-r")

    if show_animation:
        a = np.arange(0,d,0.1)
        plt.plot(a, a*np.sin(goal_angle), "-b")
        plt.plot(a, a*np.sin(a_min), "-k")
        plt.plot(a, a*np.sin(a_max), "-k")
        plt.grid(True)
        plt.axis("equal")
        plt.show()
    
    yaw_samples = np.degrees(yaw_samples)
    bins = np.arange(min(yaw_samples), max(yaw_samples), 0.5)
    n, bins, patches = plt.hist(yaw_samples, bins=bins)
    plt.show()


def lane_state_sampling_test1():
    k0 = 0.0

    l_center = 10.0
    l_heading = np.deg2rad(0.0)
    l_width = 3.0
    v_width = 1.0
    d = 10
    nxy = 5
    states = calc_lane_states(l_center, l_heading, l_width, v_width, d, nxy)
    result = generate_path(states, k0)

    if show_animation:
        plt.close("all")

    for table in result:
        x_c, y_c, yaw_c, kp = motion_model.generate_trajectory(
            table[3], table[4], table[5], k0)

        if show_animation:
            plt.plot(x_c, y_c, "-r")

    if show_animation:
        plt.grid(True)
        plt.axis("equal")
        plt.show()


def main():
    planner.show_animation = show_animation
    uniform_terminal_state_sampling_test1()
    # uniform_terminal_state_sampling_test2()
    # biased_terminal_state_sampling_test1()
    # biased_terminal_state_sampling_test2()
    # lane_state_sampling_test1()


if __name__ == '__main__':
    main()