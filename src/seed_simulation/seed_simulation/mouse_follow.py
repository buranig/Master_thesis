# from pynput.mouse import Listener
import matplotlib.pyplot as plt
import numpy as np
import math
# For the parameter file
import pathlib
import json
from custom_message.msg import Coordinate
from shapely.geometry import Point
import planner.utils as utils

from lbp_dev import LBP as LBP
from dwa_dev import DWA as DWA
from mpc_dev import MPC as MPC
from cbf_dev import C3BF as C3BF
from cbf_dev import CBF_simple as CBF
from test_ps4_controller import PS4Controller
from custom_message.msg import ControlInputs, MultiControl, Coordinate
from shapely.geometry import Point, LineString

# for debugging
from numpy import cos, sin

path = pathlib.Path('/home/giacomo/thesis_ws/src/bumper_cars/params.json')
# Opening JSON file
with open(path, 'r') as openfile:
    # Reading from json file
    json_object = json.load(openfile)

dt = json_object["Controller"]["dt"] # [s] Time tick for motion prediction
predict_time = json_object["LBP"]["predict_time"] # [s]
dilation_factor = json_object["LBP"]["dilation_factor"]

# robot_num = json_object["robot_num"]
width_init = json_object["width"]
height_init = json_object["height"]

show_animation = True
check_collision_bool = False
color_dict = {0: 'r', 1: 'b', 2: 'g', 3: 'y', 4: 'm', 5: 'c', 6: 'k'}

# Simple mouse click function to store coordinates
def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    if ix is None or iy is None:
        return

    # assign global variable to access outside of function
    global coords
    coords.pop(0)
    coords.append((ix, iy))
    print(coords[-1])

    return

coords = [(0,0)]

def main_lbp(seed, robot_num):
    """
    This function runs the main loop for the LBP algorithm.
    It initializes the necessary variables, updates the robot state, and plots the robot trajectory.

    The simulation if this function has a variable amout of robots robot_num defined in the parameter file.
    THis is the core a reference implementation of the LBP algorithm with random generation of goals that are updated when 
    the robot reaches the current goal.
    """
    print(__file__ + " start!!")
    iterations = 3000
    break_flag = False
    global coords
    coords = [(0,0)]
    fig = plt.figure(1, dpi=90, figsize=(10,10))
    ax = fig.add_subplot(111)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    # Step 2: Sample initial values for x0, y, yaw, v, omega, and model_type
    initial_state = seed['initial_position']
    x0 = initial_state['x']
    y = initial_state['y']
    yaw = initial_state['yaw']
    v = initial_state['v']

    # Step 3: Create an array x with the initial values
    x = np.array([x0, y, yaw, v])
    u = np.zeros((2, robot_num))

    trajectory = np.zeros((x.shape[0], robot_num, 1))
    trajectory[:, :, 0] = x

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

    coords = [(10,10)]
    for i in range(robot_num):
        dilated_traj.append(Point(x[0, i], x[1, i]).buffer(dilation_factor, cap_style=3))

    u_hist = dict.fromkeys(range(robot_num),[0]*int(predict_time/dt))
    # fig = plt.figure(1, dpi=90)
    # ax = fig.add_subplot(111)
    
    lbp = LBP.LBP_algorithm(predicted_trajectory, paths, targets, dilated_traj,
                        predicted_trajectory, ax, u_hist, robot_num=robot_num)
    
    for z in range(iterations):
        plt.cla()
        plt.xlim(-width_init, width_init)
        plt.ylim(-height_init, height_init)
        
        plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
        
        x, u, break_flag = lbp.run_lbp(x, u, break_flag)
        # piloting the first robot
        lbp.targets[0] = (coords[-1][0], coords[-1][1])

        trajectory = np.dstack([trajectory, x])

        utils.plot_map(width=width_init, height=height_init)
        plt.plot(coords[-1][0], coords[-1][1], 'k', marker='o', markersize=20)
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)

        if break_flag:
            break

    print("Done")
    if show_animation:
        for i in range(robot_num):
            LBP.plot_robot(x[0, i], x[1, i], x[2, i], i)
            LBP.plot_arrow(x[0, i], x[1, i], x[2, i] + u[1, i], length=3, width=0.5)
            LBP.plot_arrow(x[0, i], x[1, i], x[2, i], length=1, width=0.5)
            plt.plot(trajectory[0, i, :], trajectory[1, i, :], "-"+color_dict[i])
        plt.pause(0.0001)
        plt.show()

def main_dwa(seed, robot_num):
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
    
    print(__file__ + " start!!")
    iterations = 3000
    break_flag = False

    ps4 = PS4Controller()
    ps4.init()
    
    x0, y, yaw, v, omega, model_type = utils.samplegrid(width_init, height_init, robot_num)
    
    # Step 3: Create an array x with the initial values
    x = np.array([x0, y, yaw, v, omega])
    u = np.zeros((2, robot_num))

    trajectory = np.zeros((x.shape[0], robot_num, 1))
    trajectory[:, :, 0] = x

    predicted_trajectory = dict.fromkeys(range(robot_num),np.zeros([int(predict_time/dt), 4]))
    for i in range(robot_num):
        predicted_trajectory[i] = np.full((int(predict_time/dt), 4), x[0:4,i])

    # Step 4: Create paths for each robot
    traj = seed['trajectories']
    paths = [[Coordinate(x=traj[str(idx)][i][0], y=traj[str(idx)][i][1]) for i in range(len(traj[str(idx)]))] for idx in range(robot_num)]

    # Step 5: Extract the target coordinates from the paths
    targets = [[path[0].x, path[0].y] for path in paths]

    # Step 6: Create dilated trajectories for each robot
    dilated_traj = []
    for i in range(robot_num):
        dilated_traj.append(Point(x[0, i], x[1, i]).buffer(dilation_factor, cap_style=3))
    
    
    fig = plt.figure(1, dpi=90, figsize=(10,10))
    ax = fig.add_subplot(111)

    u_hist = dict.fromkeys(range(robot_num),[[0,0] for _ in range(int(predict_time/dt))])

    dwa = DWA.DWA_algorithm(robot_num, paths, paths, targets, dilated_traj, predicted_trajectory, ax, u_hist)
        
    for z in range(iterations):
        plt.cla()
        plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
        
        for i in range(robot_num):
            # Step 9: Check if the distance between the current position and the target is less than 5
            if utils.dist(point1=(x[0,i], x[1,i]), point2=targets[i]) < 3:
                # Perform some action when the condition is met
                dwa.paths[i] = utils.update_path(paths[i])
                dwa.targets[i] = (paths[i][0].x, paths[i][0].y)
            
            ob = [dwa.dilated_traj[idx] for idx in range(len(dwa.dilated_traj)) if idx != i]

            if i != 0:
                dwa.goal_input[0,i], dwa.goal_input[1,i] = utils.pure_pursuit_steer_control(dwa.targets[i], utils.array_to_state(x[:, i]))
            else:
                dwa.goal_input[0,i], dwa.goal_input[1,i] = ps4.control_input()
                utils.plot_arrow(x[0, i], x[1, i], x[2, i] + dwa.goal_input[1,i], length=3, width=0.5, color='b')
            
            u[:, i], dwa.predicted_trajectory[i], dwa.u_hist[i] = dwa.dwa_control(x[:, i], ob, i)
            dwa.dilated_traj[i] = LineString(zip(dwa.predicted_trajectory[i][:, 0], dwa.predicted_trajectory[i][:, 1])).buffer(dilation_factor, cap_style=3)
            
            u, x = dwa.check_collision(x,u,i)
            # x, u = dwa.update_robot_state(x, u, dt, i)
            x[:, i] = utils.nonlinear_model_numpy_stable(x[:, i], u[:, i], dt)


            if show_animation:
                utils.plot_robot_trajectory(x, u, dwa.predicted_trajectory, dwa.dilated_traj, dwa.targets, dwa.ax, i)

        utils.plot_map(width=width_init, height=height_init)
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)

        if break_flag:
            break

    print("Done")
    if show_animation:
        for i in range(robot_num):
            plt.plot(trajectory[0, i, :], trajectory[1, i, :], "-r")
        plt.pause(0.0001)
        plt.show()

def main_mpc(seed, robot_num):
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
    print(__file__ + " start!!")

    # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    iterations = 3000
    break_flag = False
    global coords
    coords = [(0,0)]
    fig = plt.figure(1, dpi=90, figsize=(10,10))
    ax = fig.add_subplot(111)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    dl = 3
    
    initial_state = seed['initial_position']
    x0 = initial_state['x']
    y = initial_state['y']
    yaw = initial_state['yaw']
    v = initial_state['v']
    x = np.array([x0, y, yaw, v])

    # MPC initialization
    mpc = MPC.ModelPredictiveControl([], [], x=x, robot_num=robot_num)

    num_inputs = 2
    u = np.zeros([mpc.horizon*num_inputs, robot_num])

    trajectory = np.zeros((x.shape[0], robot_num, 1))
    trajectory[:, :, 0] = x

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
    
    predicted_trajectory = dict.fromkeys(range(robot_num),np.zeros([mpc.horizon, x.shape[0]]))
    for i in range(robot_num):
        predicted_trajectory[i] = np.full((mpc.horizon, 4), x[:,i])

    # mpc = MPC_algorithm(cx, cy, ref, mpc, bounds, constraints, predicted_trajectory)
    mpc.cx = cx
    mpc.cy = cy
    mpc.ref = ref

    for z in range(iterations):
        plt.cla()
        plt.gcf().canvas.mpl_connect('key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

        x, u, break_flag = mpc.run_mpc(x, u, break_flag)
        mpc.ref[0][0] = coords[-1][0]
        mpc.ref[0][1] = coords[-1][1]

        trajectory = np.dstack([trajectory, x])

        plt.plot(coords[-1][0], coords[-1][1], 'k', marker='o', markersize=20)
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
            MPC.plot_robot(x[0, i], x[1, i], x[2, i], i)
            MPC.plot_arrow(x[0, i], x[1, i], x[2, i] + u[1, i], length=3, width=0.5)
            MPC.plot_arrow(x[0, i], x[1, i], x[2, i], length=1, width=0.5)
            plt.plot(trajectory[0, i, :], trajectory[1, i, :], "-"+color_dict[i])
        plt.pause(0.0001)
        plt.show()

def main_c3bf(seed, robot_num):
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
    iterations = 3000
    break_flag = False
    
    ps4 = PS4Controller()
    ps4.init()

    fig = plt.figure(1, dpi=90, figsize=(10,10))
    ax = fig.add_subplot(111)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    x0, y, yaw, v, omega, model_type = utils.samplegrid(width_init, height_init, robot_num)

    # Step 3: Create an array x with the initial values
    x = np.array([x0, y, yaw, v, omega])
    u = np.zeros((2, robot_num))

    trajectory = np.zeros((x.shape[0]+u.shape[0], robot_num, 1))
    trajectory[:, :, 0] = np.concatenate((x,u))
    
    # Step 4: Create paths for each robot
    traj = seed['trajectories']
    paths = [[Coordinate(x=traj[str(idx)][i][0], y=traj[str(idx)][i][1]) for i in range(len(traj[str(idx)]))] for idx in range(robot_num)]

    # Step 5: Extract the target coordinates from the paths
    targets = [[path[0].x, path[0].y] for path in paths]

    c3bf = C3BF.C3BF_algorithm(targets, paths, robot_num=robot_num, ax=ax)
    # Step 8: Perform the simulation for the specified number of iterations

    collision_avoidance = False

    for z in range(iterations):
        plt.cla()
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        for i in range(robot_num):
            # Step 9: Check if the distance between the current position and the target is less than 5
            if utils.dist(point1=(x[0,i], x[1,i]), point2=targets[i]) < 3:
                # Perform some action when the condition is met
                c3bf.paths[i] = utils.update_path(paths[i])
                c3bf.targets[i] = (paths[i][0].x, paths[i][0].y)
            
            cmd = ControlInputs()

            if collision_avoidance:
                x = c3bf.check_collision(x, i)
            x1 = utils.array_to_state(x[:, i])

            if i != 0:
                cmd.throttle, cmd.delta = utils.pure_pursuit_steer_control(c3bf.targets[i], x1)
                c3bf.dxu[0, i], c3bf.dxu[1, i] = cmd.throttle, cmd.delta
                c3bf.C3BF(i, x)
            
            else:
                c3bf.dxu[0, i], c3bf.dxu[1, i] = ps4.control_input()
                utils.plot_arrow(x[0, i], x[1, i], x[2, i] + c3bf.dxu[1, i], length=3, width=0.5, color='b')

            c3bf.C3BF(i, x)
            x[:, i] = utils.nonlinear_model_numpy_stable(x[:, i], c3bf.dxu[:, i])
            utils.plot_robot_and_arrows(i, x, c3bf.dxu, c3bf.targets, c3bf.ax)

        utils.plot_map(width=width_init, height=height_init)
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)

        if break_flag:
            break

    print("Done")
    if show_animation:
        for i in range(robot_num):
            utils.plot_robot(x[0, i], x[1, i], x[2, i], i)
            utils.plot_arrow(x[0, i], x[1, i], x[2, i] + u[1, i], length=3, width=0.5)
            utils.plot_arrow(x[0, i], x[1, i], x[2, i], length=1, width=0.5)
            plt.plot(trajectory[0, i, :], trajectory[1, i, :], "-"+color_dict[i])
        plt.pause(0.0001)
        plt.show()

def main_cbf(seed, robot_num):
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
    iterations = 3000
    break_flag = False
    
    ps4 = PS4Controller()
    ps4.init()

    fig = plt.figure(1, dpi=90, figsize=(10,10))
    ax = fig.add_subplot(111)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    x0, y, yaw, v, omega, model_type = utils.samplegrid(width_init, height_init, robot_num)

    # Step 3: Create an array x with the initial values
    x = np.array([x0, y, yaw, v, omega])
    u = np.zeros((2, robot_num))

    trajectory = np.zeros((x.shape[0]+u.shape[0], robot_num, 1))
    trajectory[:, :, 0] = np.concatenate((x,u))
    
    # Step 4: Create paths for each robot
    traj = seed['trajectories']
    paths = [[Coordinate(x=traj[str(idx)][i][0], y=traj[str(idx)][i][1]) for i in range(len(traj[str(idx)]))] for idx in range(robot_num)]

    # Step 5: Extract the target coordinates from the paths
    targets = [[path[0].x, path[0].y] for path in paths]

    cbf = CBF.CBF_algorithm(targets, paths, robot_num=robot_num, ax=ax)
    # Step 8: Perform the simulation for the specified number of iterations

    collision_avoidance = False

    for z in range(iterations):
        plt.cla()
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        for i in range(robot_num):
            # Step 9: Check if the distance between the current position and the target is less than 5
            if utils.dist(point1=(x[0,i], x[1,i]), point2=targets[i]) < 3:
                # Perform some action when the condition is met
                cbf.paths[i] = utils.update_path(paths[i])
                cbf.targets[i] = (paths[i][0].x, paths[i][0].y)
            
            cmd = ControlInputs()

            if collision_avoidance:
                x = cbf.check_collision(x, i)
            x1 = utils.array_to_state(x[:, i])

            if i != 0:
                cmd.throttle, cmd.delta = utils.pure_pursuit_steer_control(cbf.targets[i], x1)
                cbf.dxu[0, i], cbf.dxu[1, i] = cmd.throttle, cmd.delta
                cbf.CBF(i, x)
            
            else:
                cbf.dxu[0, i], cbf.dxu[1, i], collision_avoidance = ps4.control_input()
                if collision_avoidance:
                    cbf.CBF(i, x)

            x[:, i] = utils.nonlinear_model_numpy_stable(x[:, i], cbf.dxu[:, i])
            utils.plot_robot_and_arrows(i, x, cbf.dxu, cbf.targets, cbf.ax)

        utils.plot_map(width=width_init, height=height_init)
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)

        if break_flag:
            break

    print("Done")
    if show_animation:
        for i in range(robot_num):
            CBF.plot_robot(x[0, i], x[1, i], x[2, i], i)
            utils.plot_arrow(x[0, i], x[1, i], x[2, i] + u[1, i], length=3, width=0.5)
            utils.plot_arrow(x[0, i], x[1, i], x[2, i], length=1, width=0.5)
            plt.plot(trajectory[0, i, :], trajectory[1, i, :], "-"+color_dict[i])
        plt.pause(0.0001)
        plt.show()

if __name__ == '__main__':
     # Load the seed from a file
    filename = '/home/giacomo/thesis_ws/src/seeds/seed_11.json'
    # filename = '/home/giacomo/thesis_ws/src/circular_seed_0.json'
    with open(filename, 'r') as file:
        seed = json.load(file)

    robot_num = seed['robot_num']

    main_c3bf(seed, robot_num)
    # main_cbf(seed, robot_num) 