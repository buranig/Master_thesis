"""

Model trajectory generator

author: Atsushi Sakai(@Atsushi_twi)

"""

import math
import matplotlib.pyplot as plt
import numpy as np
import sys
import pathlib
path_planning_dir = pathlib.Path(__file__).parent.parent
sys.path.append(str(path_planning_dir))

import lattice_motion_model as motion_model
from custom_message.msg import State, ControlInputs
from planner import utils as utils

# optimization parameter
max_iter = 100
# TODO: reduce step sizes with the iterations
h = np.array([0.3, 0.02, 0.02]).T  # parameter sampling distance
cost_th = 0.12

show_animation = False


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):  # pragma: no cover
    """
    Plot arrow
    """
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              fc=fc, ec=ec, head_width=width, head_length=width)
    plt.plot(x, y)
    plt.plot(0, 0)


def calc_diff(target, x, y, yaw):
    d = np.array([target.x - x[-1],
                  target.y - y[-1],
                  motion_model.pi_2_pi(target.yaw - yaw[-1])])

    return d


def calc_j(target, p, h, k0, v):
    """
    Calculate the Jacobian matrix J for a given target and state vector p.

    Args:
        target (list): List of target coordinates [x, y, yaw].
        p (numpy.ndarray): Optimization parameters vector [s, km, kf].
        h (numpy.ndarray): Step sizes for numerical differentiation.
        k0 (float): Curvature of the motion model.
        v (float): Velocity of the motion model.

    Returns:
        numpy.ndarray: Jacobian matrix J.

    """
    xp, yp, yawp = motion_model.generate_last_state(
        p[0, 0] + h[0], p[1, 0], p[2, 0], k0, v)
    dp = calc_diff(target, [xp], [yp], [yawp])
    xn, yn, yawn = motion_model.generate_last_state(
        p[0, 0] - h[0], p[1, 0], p[2, 0], k0, v)
    dn = calc_diff(target, [xn], [yn], [yawn])
    d1 = np.array((dp - dn) / (2.0 * h[0])).reshape(3, 1)

    xp, yp, yawp = motion_model.generate_last_state(
        p[0, 0], p[1, 0] + h[1], p[2, 0], k0, v)
    dp = calc_diff(target, [xp], [yp], [yawp])
    xn, yn, yawn = motion_model.generate_last_state(
        p[0, 0], p[1, 0] - h[1], p[2, 0], k0, v)
    dn = calc_diff(target, [xn], [yn], [yawn])
    d2 = np.array((dp - dn) / (2.0 * h[1])).reshape(3, 1)

    xp, yp, yawp = motion_model.generate_last_state(
        p[0, 0], p[1, 0], p[2, 0] + h[2], k0, v)
    dp = calc_diff(target, [xp], [yp], [yawp])
    xn, yn, yawn = motion_model.generate_last_state(
        p[0, 0], p[1, 0], p[2, 0] - h[2], k0, v)
    dn = calc_diff(target, [xn], [yn], [yawn])
    d3 = np.array((dp - dn) / (2.0 * h[2])).reshape(3, 1)

    J = np.hstack((d1, d2, d3))

    return J


def selection_learning_param(dp, p, k0, target, v):
    """
    Selects the learning parameter 'a' that minimizes the cost function.

    Args:
        dp (float): The change in parameter 'p'.
        p (float): The current value of parameter 'p'.
        k0 (float): The value of parameter 'k0'.
        target (float): The target value.
        v (float): The value of parameter 'v'.

    Returns:
        float: The selected value of parameter 'a'.
    """

    mincost = float("inf")
    mina = 1.0
    maxa = 2.0
    da = 0.5

    for a in np.arange(mina, maxa, da):
        tp = p + a * dp
        xc, yc, yawc = motion_model.generate_last_state(
            tp[0], tp[1], tp[2], k0, v)
        dc = calc_diff(target, [xc], [yc], [yawc])
        cost = np.linalg.norm(dc)

        if cost <= mincost and a != 0.0:
            mina = a
            mincost = cost

    #  print(mincost, mina)
    #  input()

    return mina


def show_trajectory(target, xc, yc):  # pragma: no cover
    plt.clf()
    plot_arrow(target.x, target.y, target.yaw)
    plt.plot(xc, yc, "-r")
    plt.axis("equal")
    plt.grid(True)
    plt.pause(0.1)


def optimize_trajectory(target, k0, p, v):
    """
    Optimize the trajectory to reach the target position.

    Args:
        target (tuple): The target position (x, y, yaw).
        k0 (float): The initial curvature.
        p (numpy.ndarray): The initial trajectory parameters.
        v (float): The velocity.

    Returns:
        tuple: The optimized trajectory (xc, yc, yawc, p, kp).

    Raises:
        LinAlgError: If the path calculation encounters a linear algebra error.
    """
    
    for i in range(max_iter):
        xc, yc, yawc, kp = motion_model.generate_trajectory(p[0, 0], p[1, 0], p[2, 0], k0, v)
        dc = np.array(calc_diff(target, xc, yc, yawc)).reshape(3, 1)

        cost = np.linalg.norm(dc)
        if cost <= cost_th:
            print("path is ok cost is:" + str(cost))
            break

        J = calc_j(target, p, h, k0, v)
        try:
            dp = - np.linalg.pinv(J) @ dc
        except np.linalg.linalg.LinAlgError:
            print("cannot calc path LinAlgError")
            xc, yc, yawc, p = None, None, None, None
            break
        alpha = selection_learning_param(dp, p, k0, target, v)

        p += alpha * np.array(dp)
        # print(p.T)

        if show_animation:  # pragma: no cover
            show_trajectory(target, xc, yc)
    else:
        xc, yc, yawc, p = None, None, None, None
        print("cannot calc path")

    return xc, yc, yawc, p, kp


def optimize_trajectory_demo():  # pragma: no cover

    # target = motion_model.State(x=5.0, y=2.0, yaw=np.deg2rad(00.0))
    target = motion_model.State(x=1.0, y=-3.0, yaw=np.deg2rad(-90.0))
    # target = motion_model.State(19.87806172474534, 2.205144454953997, 0.4595476973074064)
    k0 = 0.0
    v  = 1.0

    initial_state = State(x=0.0, y=0.0, yaw=0.0, v=0.0, omega=0.0)
    cmd = ControlInputs()  # Create a ControlInputs object
    cmd.throttle, cmd.delta = utils.pure_pursuit_steer_control([target.x,target.y], initial_state)
    print(cmd.delta)
    k0 = cmd.delta
    print(target.x, target.y, target.yaw)
    # init_p = np.array([6.0, 0.0, 0.0]).reshape(3, 1)
    init_p = np.array(
            [np.hypot(target.y, target.x), cmd.delta/2, 0.0]).reshape(3, 1)
    # init_p = np.array([5.84663478, 0.20309538, 0.68336985]).reshape(3, 1)

    x, y, yaw, p, kp = optimize_trajectory(target, k0, init_p, v)

    if show_animation:
        show_trajectory(target, x, y)
        plot_arrow(target.x, target.y, target.yaw)
        plt.axis("equal")
        plt.grid(True)
        plt.show()


def main():  # pragma: no cover
    print(__file__ + " start!!")
    optimize_trajectory_demo()


if __name__ == '__main__':
    main()