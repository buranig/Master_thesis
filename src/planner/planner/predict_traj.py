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
from planner import utils as utils

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
max_acc = json_object["Controller"]["max_acc"] # [m/ss]
min_acc = json_object["Controller"]["min_acc"] # [m/ss]
dt = json_object["Controller"]["dt"]  # [s] Time step
L = json_object["Car_model"]["L"]  # [m] Wheel base of vehicle
Lr = L / 2.0  # [m]
Lf = L - Lr
# Cf = json_object["Car_model"]["Cf"]  # N/rad
# Cr = json_object["Car_model"]["Cr"] # N/rad
# Iz = json_object["Car_model"]["Iz"]  # kg/m2
# m = json_object["Car_model"]["m"]  # kg
# # Aerodynamic and friction coefficients
# c_a = json_object["Car_model"]["c_a"]
# c_r1 = json_object["Car_model"]["c_r1"]


# dt = 0.1
# Lr = 1.62 #L / 2.0  # [m]
# Lf = 1.04 #L - Lr
# L = Lr+Lf  # [m] Wheel base of vehicle
Cf = 50000 #1600.0 * 2.0  # N/rad
Cr = Cf #5650 #1700.0 * 2.0  # N/rad
Iz = 4192.0  # kg*m2
m = 1395.0  # kg
# # Aerodynamic and friction coefficients
c_a = 1.36
c_r1 = 0.02
dt = 0.1

WB = json_object["Controller"]["WB"] 
L_d = json_object["Controller"]["L_d"] 

debug = True

color_dict = {
    0: "r",
    1: "b",
    2: "g",
    3: "y",
    4: "m",
    5: "c",
    6: "k",
    7: "tab:orange",
    8: "tab:brown",
    9: "tab:gray",
    10: "tab:olive",
    11: "tab:pink",
    12: "tab:purple",
    13: "tab:red",
    14: "tab:blue",
    15: "tab:green",
}

class Dynamic_params:
    def __init__(self):
        self.Iz = Iz
        self.m = m
        self.c_a = c_a
        self.c_r1 = c_r1
        self.Cf = Cf
        self.Cr = Cr

        

def predict_trajectory(initial_state: State, target, dynamic_params: Dynamic_params, model='linear', ax=None):
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
    traj.append((initial_state.x, initial_state.y))

    cmd = ControlInputs()  # Create a ControlInputs object
    old_time = time.time()  # Get the current time

    # Calculate the control inputs for the initial state
    cmd.throttle, cmd.delta = pure_pursuit_steer_control(target, initial_state)

    # Update the state using the linear model and append the new state coordinates to the trajectory
    if model == 'linear':
            new_state, old_time = linear_model_callback(initial_state, cmd)
    elif model == 'linear_improved':
            new_state, old_time = linear_model_improved_callback(initial_state, cmd)
    else:
        new_state, old_time = nonlinear_model_callback_stable(initial_state, cmd, dyn_param=dynamic_params)
    traj.append((new_state.x, new_state.y))

    # Continue predicting the trajectory until the utils.distance between the last point and the target is less than 10
    while utils.dist(point1=(traj[-1][0], traj[-1][1]), point2=target) > 0.5:

        # Calculate the control inputs for the new state
        cmd.throttle, cmd.delta = pure_pursuit_steer_control(target, new_state)

        # Update the state using the linear model and append the new state coordinates to the trajectory
        if model == 'linear':
            new_state, old_time = linear_model_callback(new_state, cmd)
        elif model == 'linear_improved':
            new_state, old_time = linear_model_improved_callback(new_state, cmd)
        else:
            new_state, old_time = nonlinear_model_callback_stable(new_state, cmd, dyn_param=dynamic_params)
        traj.append((new_state.x, new_state.y))

        if debug:
            # Plot the trajectory and other elements for debugging
            plt.cla()
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            
            plt.plot(initial_state.x, initial_state.y, 'k.')
            utils.plot_robot(new_state.x, new_state.y, new_state.yaw, 0)
            utils.plot_arrow(new_state.x, new_state.y, new_state.yaw, length=1, width=0.5, color='k')
            utils.plot_arrow(new_state.x, new_state.y, new_state.yaw + cmd.delta, length=1, width=0.5, color='k')
            plt.plot(target[0], target[1], 'b.')
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.000001)
    
    # Reduce the number of points in the trajectory for efficiency
    # traj = traj[0:-1:5]
    if debug:
       plt.show()

    return traj

def predict_trajectory_horizon(initial_state: State, target, linear=True, horizon=2):
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
    traj.append((initial_state.x, initial_state.y))

    cmd = ControlInputs()  # Create a ControlInputs object
    old_time = time.time()  # Get the current time

    # Calculate the control inputs for the initial state
    cmd.throttle, cmd.delta = pure_pursuit_steer_control(target, initial_state)

    # Update the state using the linear model and append the new state coordinates to the trajectory
    if linear:
        new_state, old_time = linear_model_callback(initial_state, cmd)
    else:
        new_state, old_time = nonlinear_model_callback(initial_state, cmd)
    traj.append((new_state.x, new_state.y))

    # Continue predicting the trajectory until the utils.distance between the last point and the target is less than 10
    t = 0

    while t < horizon:

        # Calculate the control inputs for the new state
        cmd.throttle, cmd.delta = pure_pursuit_steer_control(target, new_state)

        # Update the state using the linear model and append the new state coordinates to the trajectory
        if linear:
            new_state, old_time = linear_model_callback(new_state, cmd)
        else:
            new_state, old_time = nonlinear_model_callback(new_state, cmd)
        traj.append((new_state.x, new_state.y))

        t += dt

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

    return traj

def plot_path(path: Path):
        x = []
        y = []
        for coord in path:
            x.append(coord[0])
            y.append(coord[1])
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
    state.yaw = utils.normalize_angle(state.yaw)
    state.v = initial_state.v + cmd.throttle * dt
    state.v = np.clip(state.v, min_speed, max_speed)

    return state, time.time()

def linear_model_improved_callback(initial_state: State, cmd: ControlInputs):
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
    cmd.trhottle = np.clip(cmd.throttle, min_acc, max_acc)
    beta = math.atan2(Lr / (Lf + Lr) * math.tan(cmd.delta), 1.0)

    state.x = initial_state.x + initial_state.v * np.cos(initial_state.yaw + beta) * dt
    state.y = initial_state.y + initial_state.v * np.sin(initial_state.yaw + beta) * dt
    state.yaw = initial_state.yaw + initial_state.v / L * np.tan(cmd.delta) * dt
    state.yaw = utils.normalize_angle(state.yaw)
    state.v = initial_state.v + cmd.throttle * dt
    state.v = np.clip(state.v, min_speed, max_speed)

    return state, time.time()

def nonlinear_model_callback(initial_state: State, cmd: ControlInputs, dyn_param: Dynamic_params):
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

    # alpha_f = cmd.delta - math.atan2((vy + Lf * initial_state.omega), vx)
    # alpha_r = -math.atan2((vy - Lr * initial_state.omega), vx)

    # Ffy = utils.pacejka_magic_formula(alpha_f)
    # Fry = utils.pacejka_magic_formula(alpha_r)

    Ffy = -dyn_param.Cf * (np.arctan2((vy + Lf * initial_state.omega), vx) - cmd.delta)
    Fry = -dyn_param.Cr * np.arctan2((vy - Lr * initial_state.omega),vx)
    R_x = dyn_param.c_r1 * abs(vx)
    F_aero = dyn_param.c_a * vx ** 2
    F_load = F_aero + R_x

    state.omega = initial_state.omega + (Ffy * Lf * math.cos(cmd.delta) - Fry * Lr) / Iz * dt
    vx = vx + (cmd.throttle - Ffy * math.sin(cmd.delta) / m - F_load / m + vy * state.omega) * dt
    vy = vy + (Fry / m + Ffy * math.cos(cmd.delta) / m - vx * state.omega) * dt

    state.yaw = initial_state.yaw + state.omega * dt
    state.yaw = utils.normalize_angle(state.yaw)

    state.v = math.sqrt(vx ** 2 + vy ** 2)
    state.v = np.clip(state.v, min_speed, max_speed)

    state.x = initial_state.x + vx * math.cos(state.yaw) * dt - vy * math.sin(state.yaw) * dt
    state.y = initial_state.y + vx * math.sin(state.yaw) * dt + vy * math.cos(state.yaw) * dt


    return state, time.time()

def nonlinear_model_callback_stable(initial_state: State, cmd: ControlInputs, dyn_param: Dynamic_params):
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
    u = initial_state.v * math.cos(beta)
    v = initial_state.v * math.sin(beta)

    kf = -dyn_param.Cf
    kr = -dyn_param.Cr

    state.x = initial_state.x + u * math.cos(initial_state.yaw) * dt - v * math.sin(initial_state.yaw) * dt
    state.y = initial_state.y + u * math.sin(initial_state.yaw) * dt + v * math.cos(initial_state.yaw) * dt
    state.yaw = initial_state.yaw + initial_state.omega * dt
    u = u + cmd.throttle * dt
    v = (m*u*v+dt*(Lf*kf-Lr*kr)*initial_state.omega - dt*kf*cmd.delta*u - dt*m*u**2*initial_state.omega)/(m*u - dt*(kf+kr))
    state.v = math.sqrt(u**2 + v**2)
    state.v = np.clip(state.v, min_speed, max_speed)
    state.omega = (Iz*u*initial_state.omega+ dt*(Lf*kf-Lr*kr)*v - dt*Lf*kf*cmd.delta*u)/(Iz*u - dt*(Lf**2*kf+Lr**2*kr))

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
        
    alpha = utils.normalize_angle(math.atan2(target[1] - pose.y, target[0] - pose.x) - pose.yaw)

    # this if/else condition should fix the buf of the waypoint behind the car
    if alpha > np.pi/2.0:
        alpha = np.pi - alpha
        desired_speed = -2
        # delta = max_steer
    elif alpha < -np.pi/2.0:
        alpha = -np.pi - alpha
        desired_speed = -2
        # delta = -max_steer
    else:
        desired_speed = 2
        # ref: https://www.shuffleai.blog/blog/Three_Methods_of_Vehicle_Lateral_Control.html
    
    delta = utils.normalize_angle(math.atan2(2.0 * WB *  math.sin(alpha), L_d))

    # decreasing the desired speed when turning
    # if delta > math.radians(10) or delta < -math.radians(10):
    #     desired_speed = 2
    # else:
    #     desired_speed = max_speed

    # print(f'Steering angle: {delta} and desired speed: {desired_speed}')
    throttle = 3 * (desired_speed-pose.v)
    return throttle, delta

def main():
    initial_state = State(x=0.0, y=0.0, yaw=0.0, v=0.0, omega=0.0)
    target = [5, 0]
    # trajectory, tx, ty = predict_trajectory(initial_state, target)
    trajectory = predict_trajectory(initial_state, target, 'linear')

    fontsize = 10
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = fontsize

    print(len(trajectory))

    # Plot Kinematic model Trajectory
    x_kin = []
    y_kin = []
    for coord in trajectory:
    
        x_kin.append(coord[0])
        y_kin.append(coord[1])
    
    trajectory = predict_trajectory(initial_state, target, 'linear_improved')

    fontsize = 10
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = fontsize

    print(len(trajectory))

    # Plot Improved Kinematic model Trajectory
    x_imp = []
    y_imp= []
    for coord in trajectory:
    
        x_imp.append(coord[0])
        y_imp.append(coord[1])

    # Plot Dynamic model Trajectory
    lb = 10000
    ub = 100000
    step = 10000
    Cf_range = np.linspace(lb, ub, int((ub-lb)/step))
    dyn_param = Dynamic_params()

    fig = plt.figure(0, figsize=(10, 10))
    ax = fig.add_subplot(111)
    for i, Cf_i in enumerate(Cf_range):
        dyn_param.Cf = Cf_i
        dyn_param.Cr = Cf_i
        trajectory1 = predict_trajectory(initial_state=initial_state, target=target, model='non_linear', dynamic_params=dyn_param, ax=ax)

        # print(f'Cf: {dyn_param.Cf}')
        x = []
        y = []
        for coord in trajectory1:
        
            x.append(coord[0])
            y.append(coord[1])
        plt.plot(x, y, color_dict[i], label='Dynamic model, m: ' + str(round(dyn_param.m,1)) + ' kg, Iz: ' + str(round(dyn_param.Iz,2)) + ' kg*m2, Cf: ' + str(dyn_param.Cf))

    plt.plot(target[0], target[1], 'kx', label='Target', markersize=20)
    utils.plot_arrow(initial_state.x, initial_state.y, initial_state.yaw, length=L/2, width=0.7, label='Initial State')
    plt.plot(x_kin, y_kin, 'r', label='Kinematic model')
    plt.plot(x_imp, y_imp, '--y', label='Improved kinematic model')
    plt.axis("equal")
    plt.xlabel("x [m]", fontdict={'size': fontsize, 'family': 'serif'})
    plt.ylabel("y [m]", fontdict={'size': fontsize, 'family': 'serif'})
    plt.title('Kinematic vs. Dynamic model comparison, m:' + str(dyn_param.m), fontdict={'size': fontsize, 'family': 'serif'})
    plt.legend()   
    plt.grid(True) 
    print('Saving plot to /home/giacomo/Immagini/system_identification/kinematic_vs_dynamic_model_Cf.png\n')
    plt.savefig('/home/giacomo/Immagini/system_identification/kinematic_vs_dynamic_model_Cf.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot Dynamic model Trajectory as funtion of m
    lb = 50
    ub = 150
    step = 10
    m_range = np.linspace(lb, ub, int((ub-lb)/step))
    dyn_param = Dynamic_params()

    fig = plt.figure(0, figsize=(10, 10))

    for i, m_i in enumerate(m_range):
        dyn_param.m = m_i
        dyn_param.Iz = 1/12 * m_i * (L**2 + WB**2)
        trajectory1 = predict_trajectory(initial_state=initial_state, target=target, model='non_linear', dynamic_params=dyn_param)

        x = []
        y = []
        for coord in trajectory1:
        
            x.append(coord[0])
            y.append(coord[1])
        plt.plot(x, y, color_dict[i], label='Dynamic model, m: ' + str(round(dyn_param.m,1)) + ' kg, Iz: ' + str(round(dyn_param.Iz,2)) + ' kg*m2')

    plt.plot(target[0], target[1], 'kx', label='Target', markersize=20)
    utils.plot_arrow(initial_state.x, initial_state.y, initial_state.yaw, length=L/2, width=0.7, label='Initial State')
    plt.plot(x_kin, y_kin, 'r', label='Kinematic model')  
    plt.plot(x_imp, y_imp, '--y', label='Improved kinematic model')
    plt.axis("equal")
    plt.xlabel("x [m]", fontdict={'size': fontsize, 'family': 'serif'})
    plt.ylabel("y [m]", fontdict={'size': fontsize, 'family': 'serif'})
    plt.title('Kinematic vs. Dynamic model comparison, Cf: ' + str(dyn_param.Cf), fontdict={'size': fontsize, 'family': 'serif'})
    plt.legend()   
    plt.grid(True) 
    # save the plot
    print('Saving plot to /home/giacomo/Immagini/system_identification/kinematic_vs_dynamic_model_Iz.png\n')
    plt.savefig('/home/giacomo/Immagini/system_identification/kinematic_vs_dynamic_model_Iz.png', dpi=300, bbox_inches='tight')

    plt.show()

    # Plot Dynamic model Trajectory as function of c_a
    lb = 0.5
    ub = 2
    step = 0.1
    c_a_range = np.linspace(lb, ub, int((ub-lb)/step))
    dyn_param = Dynamic_params()
    dyn_param.m = 150
    dyn_param.Iz = 1/12 * dyn_param.m * (L**2 + WB**2)
    
    fig = plt.figure(0, figsize=(10, 10))

    for i, c_a_i in enumerate(c_a_range):
        dyn_param.c_a = c_a_i
        trajectory1 = predict_trajectory(initial_state=initial_state, target=target, model='non_linear', dynamic_params=dyn_param)

        # print(f'c_a: {dyn_param.c_a}')
        x = []
        y = []
        for coord in trajectory1:
        
            x.append(coord[0])
            y.append(coord[1])
        plt.plot(x, y, color_dict[i], label='Dynamic model, c_a: ' + str(dyn_param.c_a))

    plt.plot(target[0], target[1], 'kx', label='Target', markersize=20)    
    utils.plot_arrow(initial_state.x, initial_state.y, initial_state.yaw, length=L/2, width=0.7, label='Initial State')
    plt.plot(x_kin, y_kin, 'r', label='Kinematic model')
    plt.plot(x_imp, y_imp, '--y', label='Improved kinematic model')
    plt.axis("equal")
    plt.xlabel("x [m]", fontdict={'size': fontsize, 'family': 'serif'})
    plt.ylabel("y [m]", fontdict={'size': fontsize, 'family': 'serif'})
    plt.title('Kinematic vs. Dynamic model comparison, m:' + str(dyn_param.m) + ', Cf: ' + str(dyn_param.Cf), fontdict={'size': fontsize, 'family': 'serif'})
    plt.legend()
    plt.grid(True)
    # save the plot
    print('Saving plot to /home/giacomo/Immagini/system_identification/kinematic_vs_dynamic_model_c_a.png\n')
    plt.savefig('/home/giacomo/Immagini/system_identification/kinematic_vs_dynamic_model_c_a.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot Dynamic model Trajectory as function of c_r1
    lb = 0.01
    ub = 0.1
    step = 0.01
    c_r1_range = np.linspace(lb, ub, int((ub-lb)/step))
    dyn_param = Dynamic_params()
    dyn_param.m = 150
    dyn_param.Iz = 1/12 * dyn_param.m * (L**2 + WB**2)

    fig = plt.figure(0, figsize=(10, 10))

    for i, c_r1_i in enumerate(c_r1_range):
        dyn_param.c_r1 = c_r1_i
        trajectory1 = predict_trajectory(initial_state=initial_state, target=target, model='non_linear', dynamic_params=dyn_param)

        # print(f'c_r1: {dyn_param.c_r1}')
        x = []
        y = []
        for coord in trajectory1:
        
            x.append(coord[0])
            y.append(coord[1])
        plt.plot(x, y, color_dict[i], label='Dynamic model, c_r1: ' + str(dyn_param.c_r1))
    
    plt.plot(target[0], target[1], 'kx', label='Target', markersize=20)
    utils.plot_arrow(initial_state.x, initial_state.y, initial_state.yaw, length=L/2, width=0.7, label='Initial State')
    plt.plot(x_kin, y_kin, 'r', label='Kinematic model')
    plt.plot(x_imp, y_imp, '--y', label='Improved kinematic model')
    plt.axis("equal")
    plt.xlabel("x [m]", fontdict={'size': fontsize, 'family': 'serif'})
    plt.ylabel("y [m]", fontdict={'size': fontsize, 'family': 'serif'})
    plt.title('Kinematic vs. Dynamic model comparison, m:' + str(dyn_param.m) + ', Cf: ' + str(dyn_param.Cf), fontdict={'size': fontsize, 'family': 'serif'})
    plt.legend()
    plt.grid(True)
    # save the plot
    print('Saving plot to /home/giacomo/Immagini/system_identification/kinematic_vs_dynamic_model_c_r1.png\n')
    plt.savefig('/home/giacomo/Immagini/system_identification/kinematic_vs_dynamic_model_c_r1.png', dpi=300, bbox_inches='tight')
    plt.show()

    
if __name__ == "__main__":
    main()




