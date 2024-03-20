import math
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from planner import utils as utils  

# motion parameter
L = 1.0  # wheel base
ds = 0.1  # course distance
dt = 0.1
# TODO: remove hardcoded parts

# v = 0.5 # velocity [m/s]


class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v


def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def update(state, v, delta, dt):
    state.v = v
    delta = np.clip(delta, -np.radians(45), np.radians(45))
    state.x = state.x + state.v * math.cos(state.yaw) * dt
    state.y = state.y + state.v * math.sin(state.yaw) * dt
    state.yaw = state.yaw + state.v / L * math.tan(delta) * dt
    state.yaw = utils.normalize_angle(state.yaw)
    # state.yaw = pi_2_pi(state.yaw)

    return state

def generate_trajectory(s, km, kf, k0, v):
    """
    Generate a trajectory based on the given parameters.

    Args:
        s (float): The distance to be covered.
        km (float): The middle curvature.
        kf (float): The final curvature.
        k0 (float): The initial curvature.
        v (float): The velocity.

    Returns:
        tuple: A tuple containing the x-coordinates, y-coordinates, yaw angles, and curvature values of the generated trajectory.
    """

    # n = s / ds
    time = s / v  # [s]
    n = time / dt


    if isinstance(time, type(np.array([]))):
        time = time[0]
    if isinstance(km, type(np.array([]))):
        km = km[0]
    if isinstance(kf, type(np.array([]))):
        kf = kf[0]

    tk = np.array([0.0, time / 2.0, time])
    kk = np.array([k0, km, kf])
    t = np.arange(0.0, time, time / n)
    fkp = interp1d(tk, kk, kind="quadratic")
    kp = [float(fkp(ti)) for ti in t]
    # dt = abs(float(time / n))

    #  plt.plot(t, kp)
    #  plt.show()

    state = State()
    x, y, yaw = [state.x], [state.y], [state.yaw]

    for ikp in kp:
        state = update(state, v, ikp, dt)
        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)

    return x, y, yaw, kp


def generate_last_state(s, km, kf, k0, v):
    """
    Generates the last state of the motion model based on the given parameters.

    Args:
        s (float): The distance traveled.
        km (float): The middle curvature.
        kf (float): The final curvature.
        k0 (float): The initial curvature.
        v (float): The velocity.

    Returns:
        tuple: A tuple containing the x-coordinate, y-coordinate, and yaw angle of the last state.
    """

    n = s / ds
    time = abs(s / v)  # [s]

    if isinstance(n, type(np.array([]))):
        n = n[0]
    if isinstance(time, type(np.array([]))):
        time = time[0]
    if isinstance(km, type(np.array([]))):
        km = km[0]
    if isinstance(kf, type(np.array([]))):
        kf = kf[0]

    tk = np.array([0.0, time / 2.0, time])
    kk = np.array([k0, km, kf])
    t = np.arange(0.0, time, time / n)
    fkp = interp1d(tk, kk, kind="quadratic")
    kp = [fkp(ti) for ti in t]
    dt = time / n

    # plt.plot(t, kp)
    # plt.show()

    state = State()

    _ = [update(state, v, ikp, dt) for ikp in kp]

    return state.x, state.y, state.yaw