import numpy as np
import matplotlib.pyplot as plt
import math
import time
import os
import json
import pathlib
import dwa_dev.DWA as DWA
import lbp_dev.LBP as LBP
import cbf_dev.CBF_simple as CBF
import cbf_dev.C3BF as C3BF
import mpc_dev.MPC as MPC
import planner.utils as utils
from custom_message.msg import Coordinate
from shapely.geometry import Point, LineString

path = pathlib.Path('/home/giacomo/thesis_ws/src/bumper_cars/params.json')
# Opening JSON file
with open(path, 'r') as openfile:
    # Reading from json file
    json_object = json.load(openfile)

robot_num = json_object["robot_num"]
safety_init = json_object["safety"]
width_init = json_object["width"]
height_init = json_object["height"]
min_dist = json_object["min_dist"]
robot_num = json_object["robot_num"]
show_animation = True
go_to_goal_bool = True
iterations = 1000

color_dict = {0: 'r', 1: 'b', 2: 'g', 3: 'y', 4: 'm', 5: 'c', 6: 'k'}

def dwa_sim(seed):

    dt = json_object["DWA"]["dt"] # [s] Time tick for motion prediction
    predict_time = json_object["DWA"]["predict_time"] # [s]
    dilation_factor = json_object["DWA"]["dilation_factor"]

    """
    Main function that controls the execution of the program.

    Steps:
    1. Initialize the necessary variables and arrays.
    2. Generate initial robot states and trajectories.
    3. Initialize paths, targets, and dilated trajectories.
    4. Start the main loop for a specified number of iterations.
    5. Update targets and robot states for each robot.
    6. Calculate the right input using the Dynamic Window Approach method.
    7. Predict the future trajectory using the calculated input.
    8. Check if the goal is reached for each robot.
    9. Plot the robot trajectories and the map.
    11. Break the loop if the goal is reached for any robot.
    12. Print "Done" when the loop is finished.
    13. Plot the final trajectories if animation is enabled.
    """
    print("DWA start!!")
    break_flag = False
    
    # Step 2: Sample initial values for x0, y, yaw, v, omega, and model_type
    initial_state = seed['initial_position']
    x0 = initial_state['x']
    y = initial_state['y']
    yaw = initial_state['yaw']
    v = initial_state['v']

    # Step 3: Create an array x with the initial values
    x = np.array([x0, y, yaw, v])
    u = np.zeros((2, robot_num))

    trajectory = np.zeros((x.shape[0]+u.shape[0], robot_num, 1))
    trajectory[:, :, 0] = np.concatenate((x,u))

    predicted_trajectory = dict.fromkeys(range(robot_num),np.zeros([int(predict_time/dt), 3]))
    for i in range(robot_num):
        predicted_trajectory[i] = np.full((int(predict_time/dt), 3), x[0:3,i])

    # Step 4: Create paths for each robot
    traj = seed['trajectories']
    paths = [[Coordinate(x=traj[str(idx)][i][0], y=traj[str(idx)][i][1]) for i in range(len(traj[str(idx)]))] for idx in range(robot_num)]

    # Step 5: Extract the target coordinates from the paths
    targets = [[path[0].x, path[0].y] for path in paths]

    # Step 6: Create dilated trajectories for each robot
    dilated_traj = []
    for i in range(robot_num):
        dilated_traj.append(Point(x[0, i], x[1, i]).buffer(dilation_factor, cap_style=3))
    
    fig = plt.figure(1, dpi=90)
    ax = fig.add_subplot(111)
    
    # Step 7: Create an instance of the DWA_algorithm class
    dwa = DWA.DWA_algorithm(paths, safety_init, width_init, height_init,
                        min_dist, paths, targets, dilated_traj, predicted_trajectory, ax)
    
    for z in range(iterations):
        plt.cla()
        plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
        
        if go_to_goal_bool:
            x, u, break_flag = dwa.go_to_goal(x, u, break_flag)
        else:
            x, u, break_flag = dwa.run_dwa(x, u, break_flag)
        trajectory = np.dstack([trajectory, np.concatenate((x,u))])
            
        utils.plot_map(width=width_init, height=height_init)
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)

        if break_flag:
            break
    
    # plt.close()
    print("Done")
    if show_animation:
        for i in range(robot_num):
            plt.plot(trajectory[0, i, :], trajectory[1, i, :], "-"+color_dict[i])
        plt.pause(0.0001)
        plt.show()

    return trajectory, dwa.computational_time 

def mpc_sim(seed):
    """
    Main function for controlling multiple robots using Model Predictive Control (MPC).

    Steps:
    1. Initialize the necessary variables and parameters.
    2. Create an instance of the ModelPredictiveControl class.
    3. Set the initial state and control inputs.
    4. Generate the reference trajectory for each robot.
    5. Plot the initial positions and reference trajectory.
    6. Set the bounds and constraints for the MPC.
    7. Initialize the predicted trajectory for each robot.
    8. Enter the main control loop:
        - Check if the distance between the current position and the target is less than 5.
            - If yes, update the path and target.
        - Perform MPC control for each robot.
        - Plot the robot trajectory.
        - Update the predicted trajectory.
        - Plot the map and pause for visualization.
    """
    print("MPC start!!")
    break_flag = False

    # MPC initialization
    mpc = MPC.ModelPredictiveControl([], [])
    
    initial_state = seed['initial_position']
    x0 = initial_state['x']
    y = initial_state['y']
    yaw = initial_state['yaw']
    v = initial_state['v']
    x = np.array([x0, y, yaw, v])
    num_inputs = 2
    u = np.zeros([mpc.horizon*num_inputs, robot_num])

    trajectory = np.zeros((x.shape[0] + num_inputs, robot_num, 1))
    trajectory[:, :, 0] = np.concatenate((x,u[:2]))

    # Generate reference trajectory
    traj = seed['trajectories']
    cx = []
    cy = []
    ref = []

    for i in range(robot_num):
        x_buf = []
        y_buf = []
        for idx in range(len(traj[str(i)])):
            x_buf.append(traj[str(i)][idx][0])
            y_buf.append(traj[str(i)][idx][1])
        cx.append(x_buf)
        cy.append(y_buf)
        ref.append([cx[i][0], cy[i][0]])

    # Usage:
    bounds, constraints = MPC.set_bounds_and_constraints(mpc)
    
    predicted_trajectory = dict.fromkeys(range(robot_num),np.zeros([mpc.horizon, x.shape[0]]))
    for i in range(robot_num):
        predicted_trajectory[i] = np.full((mpc.horizon, 4), x[:,i])
    
    # input [throttle, steer (delta)]
    fig = plt.figure(1, dpi=90)
    ax = fig.add_subplot(111)

    # mpc = MPC_algorithm(cx, cy, ref, mpc, bounds, constraints, predicted_trajectory)
    mpc.cx = cx
    mpc.cy = cy
    mpc.ref = ref
    mpc.bounds = bounds
    mpc.constraints = constraints
    mpc.predicted_trajectory = predicted_trajectory

    for z in range(iterations):
        plt.cla()
        plt.gcf().canvas.mpl_connect('key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

        if go_to_goal_bool:
            x, u, break_flag = mpc.go_to_goal(x, u, break_flag)
        else:
            x, u = mpc.run_mpc(x, u)
        trajectory = np.dstack([trajectory, np.concatenate((x,u[:2]))])

        plt.title('MPC 2D')
        utils.plot_map(width=width_init, height=height_init)
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)

        if break_flag:
            break
    
    print("Done")
    if show_animation:
        for i in range(robot_num):
            plt.plot(trajectory[0, i, :], trajectory[1, i, :], "-"+color_dict[i])
        plt.pause(0.0001)
        plt.show()

    return trajectory, mpc.computational_time

def c3bf_sim(seed):
    """
    Main function for controlling multiple robots using Model Predictive Control (MPC).

    Steps:
    1. Initialize the necessary variables and parameters.
    2. Create an instance of the ModelPredictiveControl class.
    3. Set the initial state and control inputs.
    4. Generate the reference trajectory for each robot.
    5. Plot the initial positions and reference trajectory.
    6. Set the bounds and constraints for the MPC.
    7. Initialize the predicted trajectory for each robot.
    8. Enter the main control loop:
        - Check if the distance between the current position and the target is less than 5.
            - If yes, update the path and target.
        - Perform 3CBF control for each robot.
        - Plot the robot trajectory.
        - Update the predicted trajectory.
        - Plot the map and pause for visualization.
    """
    print("3CBF start!!")
    break_flag = False

    # Step 2: Sample initial values for x0, y, yaw, v, omega, and model_type
    initial_state = seed['initial_position']
    x0 = initial_state['x']
    y = initial_state['y']
    yaw = initial_state['yaw']
    v = initial_state['v']

    # Step 3: Create an array x with the initial values
    x = np.array([x0, y, yaw, v])
    u = np.zeros((2, robot_num))

    trajectory = np.zeros((x.shape[0]+u.shape[0], robot_num, 1))
    trajectory[:, :, 0] = np.concatenate((x,u))
    
    # Step 4: Create paths for each robot
    traj = seed['trajectories']
    paths = [[Coordinate(x=traj[str(idx)][i][0], y=traj[str(idx)][i][1]) for i in range(len(traj[str(idx)]))] for idx in range(robot_num)]

    # Step 5: Extract the target coordinates from the paths
    targets = [[path[0].x, path[0].y] for path in paths]

    c3bf = C3BF.C3BF_algorithm(targets, paths)
    # Step 8: Perform the simulation for the specified number of iterations
    for z in range(iterations):
        plt.cla()
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        
        if go_to_goal_bool:
            x, u, break_flag = c3bf.go_to_goal(x, break_flag)
        else:
            x, u, break_flag = c3bf.run_3cbf(x, break_flag)
        trajectory = np.dstack([trajectory, np.concatenate((x,u))])
        
        C3BF.plot_map(width=width_init, height=height_init)
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)

        if break_flag:
            break

    print("Done")
    if show_animation:
        for i in range(robot_num):
            plt.plot(trajectory[0, i, :], trajectory[1, i, :], "-"+color_dict[i])
        plt.pause(0.0001)
        plt.show()

    return trajectory, c3bf.computational_time

def cbf_sim(seed):
    """
    Main function for controlling multiple robots using Model Predictive Control (MPC).

    Steps:
    1. Initialize the necessary variables and parameters.
    2. Create an instance of the ModelPredictiveControl class.
    3. Set the initial state and control inputs.
    4. Generate the reference trajectory for each robot.
    5. Plot the initial positions and reference trajectory.
    6. Set the bounds and constraints for the MPC.
    7. Initialize the predicted trajectory for each robot.
    8. Enter the main control loop:
        - Check if the distance between the current position and the target is less than 5.
            - If yes, update the path and target.
        - Perform 3CBF control for each robot.
        - Plot the robot trajectory.
        - Update the predicted trajectory.
        - Plot the map and pause for visualization.
    """
    print("CBF start!!")
    break_flag = False

    # Step 2: Sample initial values for x0, y, yaw, v, omega, and model_type
    initial_state = seed['initial_position']
    x0 = initial_state['x']
    y = initial_state['y']
    yaw = initial_state['yaw']
    v = initial_state['v']

    # Step 3: Create an array x with the initial values
    x = np.array([x0, y, yaw, v])
    u = np.zeros((2, robot_num))

    trajectory = np.zeros((x.shape[0]+u.shape[0], robot_num, 1))
    trajectory[:, :, 0] = np.concatenate((x,u))
    
    # Step 4: Create paths for each robot
    traj = seed['trajectories']
    paths = [[Coordinate(x=traj[str(idx)][i][0], y=traj[str(idx)][i][1]) for i in range(len(traj[str(idx)]))] for idx in range(robot_num)]

    # Step 5: Extract the target coordinates from the paths
    targets = [[path[0].x, path[0].y] for path in paths]

    cbf = CBF.CBF_algorithm(targets, paths)
    # Step 8: Perform the simulation for the specified number of iterations
    for z in range(iterations):
        plt.cla()
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        
        if go_to_goal_bool:
            x, u, break_flag = cbf.go_to_goal(x, break_flag)
        else:
            x, u, break_flag = cbf.run_cbf(x, break_flag) 
        trajectory = np.dstack([trajectory, np.concatenate((x,u))])
        
        CBF.plot_map(width=width_init, height=height_init)
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)

        if break_flag:
            break
    
    print("Done")
    if show_animation:
        for i in range(robot_num):
            plt.plot(trajectory[0, i, :], trajectory[1, i, :], "-"+color_dict[i])
        plt.pause(0.0001)
        plt.show()

    return trajectory, cbf.computational_time

def lbp_sim(seed):
    dt = json_object["LBP"]["dt"] # [s] Time tick for motion prediction
    predict_time = json_object["LBP"]["predict_time"] # [s]
    dilation_factor = json_object["LBP"]["dilation_factor"]

    """
    This function runs the main loop for the LBP algorithm.
    It initializes the necessary variables, updates the robot state, and plots the robot trajectory.

    The simulation if this function has a variable amout of robots robot_num defined in the parameter file.
    THis is the core a reference implementation of the LBP algorithm with random generation of goals that are updated when 
    the robot reaches the current goal.
    """
    print("LBP start!!")

    break_flag = False

    # Step 2: Sample initial values for x0, y, yaw, v, omega, and model_type
    initial_state = seed['initial_position']
    x0 = initial_state['x']
    y = initial_state['y']
    yaw = initial_state['yaw']
    v = initial_state['v']

    # Step 3: Create an array x with the initial values
    x = np.array([x0, y, yaw, v])
    u = np.zeros((2, robot_num))

    trajectory = np.zeros((x.shape[0]+u.shape[0], robot_num, 1))
    trajectory[:, :, 0] = np.concatenate((x,u))

    predicted_trajectory = dict.fromkeys(range(robot_num),np.zeros([int(predict_time/dt), 3]))
    for i in range(robot_num):
        predicted_trajectory[i] = np.full((int(predict_time/dt), 3), x[0:3,i])

    # Step 4: Create paths for each robot
    traj = seed['trajectories']
    paths = [[Coordinate(x=traj[str(idx)][i][0], y=traj[str(idx)][i][1]) for i in range(len(traj[str(idx)]))] for idx in range(robot_num)]

    # Step 5: Extract the target coordinates from the paths
    targets = [[path[0].x, path[0].y] for path in paths]

    # Step 6: Create dilated trajectories for each robot
    dilated_traj = []
    for i in range(robot_num):
        dilated_traj.append(Point(x[0, i], x[1, i]).buffer(dilation_factor, cap_style=3))

    u_hist = dict.fromkeys(range(robot_num),[0]*int(predict_time/dt))
    fig = plt.figure(1, dpi=90)
    ax = fig.add_subplot(111)
    
    lbp = LBP.LBP_algorithm(x, predicted_trajectory, robot_num, safety_init, 
                        width_init, height_init, min_dist, paths, targets, dilated_traj,
                        predicted_trajectory, ax, u_hist)
    
    for z in range(iterations):
        plt.cla()
        plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
        
        if go_to_goal_bool:
            x, u, break_flag = lbp.go_to_goal(x, u, break_flag)
        else:
            x, u, break_flag = lbp.run_lbp(x, u, break_flag)
        trajectory = np.dstack([trajectory, np.concatenate((x,u))])

        utils.plot_map(width=width_init, height=height_init)
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)

        if break_flag:
            break

    print("Done")
    if show_animation:
        for i in range(robot_num):
            plt.plot(trajectory[0, i, :], trajectory[1, i, :], "-"+color_dict[i])
        plt.pause(0.0001)
        plt.show()
    
    return trajectory, lbp.computational_time

def main():
    # Load the seed from a file
    filename = '/home/giacomo/thesis_ws/src/seed_1.json'
    # filename = '/home/giacomo/thesis_ws/src/circular_seed_0.json'
    with open(filename, 'r') as file:
        seed = json.load(file)

    dwa_trajectory, dwa_computational_time = dwa_sim(seed)   
    print(f"DWA average computational time: {sum(dwa_computational_time) / len(dwa_computational_time)}\n")

    mpc_trajectory, mpc_computational_time = mpc_sim(seed)
    print(f"MPC average computational time: {sum(mpc_computational_time) / len(mpc_computational_time)}\n")

    c3bf_trajectory, c3bf_computational_time = c3bf_sim(seed)
    print(f"C3BF average computational time: {sum(c3bf_computational_time) / len(c3bf_computational_time)}\n")

    cbf_trajectory, cbf_computational_time = cbf_sim(seed)
    print(f"CBF average computational time: {sum(cbf_computational_time) / len(cbf_computational_time)}\n")
    
    lbp_trajectory, lbp_computational_time = lbp_sim(seed)
    print(f"LBP average computational time: {sum(lbp_computational_time) / len(lbp_computational_time)}\n")

if __name__ == '__main__':
    main()