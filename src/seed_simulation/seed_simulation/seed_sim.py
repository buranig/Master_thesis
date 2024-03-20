import numpy as np
import matplotlib.pyplot as plt
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
import pandas as pd
from data_process import DataProcessor
import os
import matplotlib

matplotlib.use("Qt5Agg")
mgr = plt.get_current_fig_manager()
mgr.full_screen_toggle()
py = mgr.canvas.height()
px = mgr.canvas.width()+1000
mgr.window.close()

path = pathlib.Path('/home/giacomo/thesis_ws/src/bumper_cars/params.json')
# Opening JSON file
with open(path, 'r') as openfile:
    # Reading from json file
    json_object = json.load(openfile)

# robot_num = json_object["robot_num"]
width_init = json_object["width"]
height_init = json_object["height"]
show_animation = json_object["show_animation"]
add_noise = json_object["add_noise"]
noise_scale_param = json_object["noise_scale_param"]
go_to_goal_bool = True
iterations = 700

color_dict = {0: 'r', 1: 'b', 2: 'g', 3: 'y', 4: 'm', 5: 'c', 6: 'k', 7: 'tab:orange', 8: 'tab:brown', 9: 'tab:gray', 10: 'tab:olive', 11: 'tab:pink', 12: 'tab:purple', 13: 'tab:red', 14: 'tab:blue', 15: 'tab:green'}

def mypause(interval):
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return

def dwa_sim(seed, robot_num):

    dt = json_object["Controller"]["dt"] # [s] Time tick for motion prediction
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
    
    if show_animation:
        plt.ion()
        fig = plt.figure(1, dpi=90)
        figManager = plt.get_current_fig_manager()
        figManager.window.move(px, 0)
        # figManager.window.showMaximized()
        # figManager.window.setFocus()
        ax = fig.add_subplot(111)
        plt.show(block=False)
    else:
        ax = None
    
    u_hist = dict.fromkeys(range(robot_num),[[0,0] for _ in range(int(predict_time/dt))])    
    # Step 7: Create an instance of the DWA_algorithm class
    dwa = DWA.DWA_algorithm(robot_num, paths, paths, targets, dilated_traj, predicted_trajectory, ax, u_hist)
    
    for z in range(iterations):
        
        if show_animation:
            plt.cla()
            plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
            
        if go_to_goal_bool:
            x, u, break_flag = dwa.go_to_goal(x, u, break_flag)
        else:
            x, u, break_flag = dwa.run_dwa(x, u, break_flag)
        trajectory = np.dstack([trajectory, np.concatenate((x,u))])


        if show_animation:  
            utils.plot_map(width=width_init, height=height_init)
            plt.axis("equal")
            plt.grid(True)
            # plt.pause(0.0001)
            mypause(0.0001)

        if break_flag:
            break
    
    print("Done")
    if show_animation:
        for i in range(robot_num):
            DWA.plot_robot(x[0, i], x[1, i], x[2, i], i)
            DWA.plot_arrow(x[0, i], x[1, i], x[2, i] + u[1, i], length=3, width=0.5)
            DWA.plot_arrow(x[0, i], x[1, i], x[2, i], length=1, width=0.5)
            plt.plot(trajectory[0, i, :], trajectory[1, i, :], "-", color=color_dict[i])
        mypause(0.0001)
        plt.show(block=False)
        mypause(3)
        plt.close()

    return trajectory, dwa.computational_time

def mpc_sim(seed, robot_num):
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
    
    initial_state = seed['initial_position']
    x0 = initial_state['x']
    y = initial_state['y']
    yaw = initial_state['yaw']
    v = initial_state['v']
    x = np.array([x0, y, yaw, v])

    mpc = MPC.ModelPredictiveControl(obs_x=[], obs_y=[], x=x, robot_num=robot_num)

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

    if show_animation:
        plt.ion()
        fig = plt.figure(1, dpi=90)
        figManager = plt.get_current_fig_manager()
        figManager.window.move(px, 0)
        # figManager.window.showMaximized()
        # figManager.window.setFocus()
        ax = fig.add_subplot(111)
        plt.show(block=False)

    # mpc = MPC_algorithm(cx, cy, ref, mpc, bounds, constraints, predicted_trajectory)
    mpc.cx = cx
    mpc.cy = cy
    mpc.ref = ref

    for z in range(iterations):
        if show_animation:
            plt.cla()
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])

        if go_to_goal_bool:
            x, u, break_flag = mpc.go_to_goal(x, u, break_flag)
        else:
            x, u, break_flag = mpc.run_mpc(x, u, break_flag)
        trajectory = np.dstack([trajectory, np.concatenate((x,u[:2]))])

        if show_animation:
            plt.title('MPC 2D')
            utils.plot_map(width=width_init, height=height_init)
            plt.axis("equal")
            plt.grid(True)
            mypause(0.0001)

        if break_flag:
            break
    
    print("Done")
    if show_animation:
        for i in range(robot_num):
            MPC.plot_robot(x[0, i], x[1, i], x[2, i], i)
            MPC.plot_arrow(x[0, i], x[1, i], x[2, i] + u[1, i], length=3, width=0.5)
            MPC.plot_arrow(x[0, i], x[1, i], x[2, i], length=1, width=0.5)
            plt.plot(trajectory[0, i, :], trajectory[1, i, :], "-", color=color_dict[i])
        mypause(0.0001)
        plt.show(block=False)
        mypause(3)
        plt.close()

    return trajectory, mpc.computational_time

def c3bf_sim(seed, robot_num):
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

    if show_animation:
        plt.ion()
        fig = plt.figure(1, dpi=90)
        figManager = plt.get_current_fig_manager()
        figManager.window.move(px, 0)
        # figManager.window.showMaximized()
        # figManager.window.setFocus()
        ax = fig.add_subplot(111)
        plt.show(block=False)

    trajectory = np.zeros((x.shape[0]+u.shape[0], robot_num, 1))
    trajectory[:, :, 0] = np.concatenate((x,u))
    
    # Step 4: Create paths for each robot
    traj = seed['trajectories']
    paths = [[Coordinate(x=traj[str(idx)][i][0], y=traj[str(idx)][i][1]) for i in range(len(traj[str(idx)]))] for idx in range(robot_num)]

    # Step 5: Extract the target coordinates from the paths
    targets = [[path[0].x, path[0].y] for path in paths]

    c3bf = C3BF.C3BF_algorithm(targets, paths, robot_num=robot_num)
    # Step 8: Perform the simulation for the specified number of iterations
    for z in range(iterations):
        if show_animation:
            plt.cla()
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
        
        if go_to_goal_bool:
            x, break_flag = c3bf.go_to_goal(x, break_flag)
        else:
            x, break_flag = c3bf.run_3cbf(x, break_flag)
        trajectory = np.dstack([trajectory, np.concatenate((x, c3bf.dxu))])
        
        if show_animation:
            utils.plot_map(width=width_init, height=height_init)
            plt.axis("equal")
            plt.grid(True)
            mypause(0.0001)

        if break_flag:
            break

    print("Done")
    if show_animation:
        for i in range(robot_num):
            C3BF.plot_robot(x[0, i], x[1, i], x[2, i], i)
            utils.plot_arrow(x[0, i], x[1, i], x[2, i] + u[1, i], length=3, width=0.5)
            utils.plot_arrow(x[0, i], x[1, i], x[2, i], length=1, width=0.5)
            plt.plot(trajectory[0, i, :], trajectory[1, i, :], "-", color=color_dict[i])
        mypause(0.0001)
        plt.show(block=False)
        mypause(3)
        plt.close()

    return trajectory, c3bf.computational_time, c3bf.solver_failure

def cbf_sim(seed, robot_num):
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

    if show_animation:
        plt.ion()
        fig = plt.figure(1, dpi=90)
        figManager = plt.get_current_fig_manager()
        figManager.window.move(px, 0)
        # figManager.window.showMaximized()
        # figManager.window.setFocus()
        ax = fig.add_subplot(111)
        plt.show(block=False)

    trajectory = np.zeros((x.shape[0]+u.shape[0], robot_num, 1))
    trajectory[:, :, 0] = np.concatenate((x,u))
    
    # Step 4: Create paths for each robot
    traj = seed['trajectories']
    paths = [[Coordinate(x=traj[str(idx)][i][0], y=traj[str(idx)][i][1]) for i in range(len(traj[str(idx)]))] for idx in range(robot_num)]

    # Step 5: Extract the target coordinates from the paths
    targets = [[path[0].x, path[0].y] for path in paths]

    cbf = CBF.CBF_algorithm(targets, paths, robot_num=robot_num)
    # Step 8: Perform the simulation for the specified number of iterations
    for z in range(iterations):
        if show_animation:
            plt.cla()
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
        
        if go_to_goal_bool:
            x, break_flag = cbf.go_to_goal(x, break_flag)
        else:
            x, break_flag = cbf.run_cbf(x, break_flag) 
            
        trajectory = np.dstack([trajectory, np.concatenate((x, cbf.dxu))])
        
        if show_animation:
            utils.plot_map(width=width_init, height=height_init)
            plt.axis("equal")
            plt.grid(True)
            mypause(0.0001)

        if break_flag:
            break
    
    print("Done")
    if show_animation:
        for i in range(robot_num):
            CBF.plot_robot(x[0, i], x[1, i], x[2, i], i)
            utils.plot_arrow(x[0, i], x[1, i], x[2, i] + u[1, i], length=3, width=0.5)
            utils.plot_arrow(x[0, i], x[1, i], x[2, i], length=1, width=0.5)
            plt.plot(trajectory[0, i, :], trajectory[1, i, :], "-", color=color_dict[i])
        mypause(0.0001)
        plt.show(block=False)
        mypause(3)
        plt.close()

    return trajectory, cbf.computational_time, cbf.solver_failure

def lbp_sim(seed, robot_num):
    dt = json_object["Controller"]["dt"]
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

    if show_animation:
        plt.ion()
        fig = plt.figure(1, dpi=90)
        figManager = plt.get_current_fig_manager()
        figManager.window.move(px, 0)
        # figManager.window.showMaximized()
        # figManager.window.setFocus()
        ax = fig.add_subplot(111)
        plt.show(block=False)
    else: 
        ax = None
    
    lbp = LBP.LBP_algorithm(predicted_trajectory, paths, targets, dilated_traj,
                        predicted_trajectory, ax, u_hist, robot_num=robot_num)
    
    for z in range(iterations):
        if z%100 == 0:
            print(f"iteration: {z}")

        if show_animation:
            plt.cla()
            plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
            
        if go_to_goal_bool:
            x, u, break_flag = lbp.go_to_goal(x, u, break_flag)
        else:
            # add noise: gaussians zero mean different variances ~50cm for position and ~5deg for orientation
            x, u, break_flag = lbp.run_lbp(x, u, break_flag)
        trajectory = np.dstack([trajectory, np.concatenate((x,u))])

        if show_animation:
            utils.plot_map(width=width_init, height=height_init)
            plt.axis("equal")
            plt.grid(True)
            mypause(0.0001)

        if break_flag:
            break

    print("Done")
    if show_animation:
        for i in range(robot_num):
            LBP.plot_robot(x[0, i], x[1, i], x[2, i], i)
            LBP.plot_arrow(x[0, i], x[1, i], x[2, i] + u[1, i], length=3, width=0.5)
            LBP.plot_arrow(x[0, i], x[1, i], x[2, i], length=1, width=0.5)
            plt.plot(trajectory[0, i, :], trajectory[1, i, :], "-", color=color_dict[i])
        mypause(0.0001)
        plt.show(block=False)
        mypause(3)
        plt.close()
    
    return trajectory, lbp.computational_time

def main():
    # Load the seed from a file
    path = pathlib.Path('/home/giacomo/thesis_ws/src/seeds/')
    dir_list = os.listdir(path)
    # dir_list = ['circular_seed_54.json', 'circular_seed_55.json', 'circular_seed_56.json']

    csv_file = '/home/giacomo/thesis_ws/src/seed_simulation/seed_simulation/seed_sim.csv'
    df = pd.read_csv(csv_file)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Analyze the results
    for filename in dir_list:
        
        if 'circular' not in filename:
            continue

        seed_path = path / filename
        with open(seed_path, 'r') as file:
            print(f"Loading seed from {seed_path}\n")
            seed = json.load(file)

        robot_num = seed['robot_num']

        assert robot_num == len(seed['initial_position']['x']), "The number of robots in the seed file does not match the number of robots in the seed file"
        
        data_process = DataProcessor(robot_num, file_name=filename, seed=seed)
        data = []

        if filename in list(df["File Name"]):   
            print(f"File {filename} already exists in the csv file, checking for single methods...\n")      
            df_temp = df.loc[df["File Name"] == filename]
            if 'DWA' not in list(df_temp["Method"]) or (noise_scale_param not in list(df_temp["Noise Scaling"]) and add_noise):
                dwa_trajectory, dwa_computational_time = dwa_sim(seed, robot_num)   
                print(f"DWA average computational time: {sum(dwa_computational_time) / len(dwa_computational_time)}\n")
                dwa_data = data_process.post_process_simultation(dwa_trajectory, dwa_computational_time, method='DWA')
                data.append(dwa_data)
                plt.close()
            else: 
                print(f"\tDWA already executed for {filename}, for noise scaling {noise_scale_param}.\n")

            if 'MPC' not in list(df_temp["Method"]) or (noise_scale_param not in list(df_temp["Noise Scaling"]) and add_noise):
                mpc_trajectory, mpc_computational_time = mpc_sim(seed, robot_num)
                print(f"MPC average computational time: {sum(mpc_computational_time) / len(mpc_computational_time)}\n")
                mpc_data = data_process.post_process_simultation(mpc_trajectory, mpc_computational_time, method="MPC")
                data.append(mpc_data)
                plt.close()
            else:
                print(f"\tMPC already executed for {filename}, for noise scaling {noise_scale_param}.\n")
            
            if 'C3BF' not in list(df_temp["Method"]) or (noise_scale_param not in list(df_temp["Noise Scaling"]) and add_noise):
                c3bf_trajectory, c3bf_computational_time, c3bf_solver_failure = c3bf_sim(seed, robot_num)
                print(f"C3BF average computational time: {sum(c3bf_computational_time) / len(c3bf_computational_time)}\n")
                c3bf_data = data_process.post_process_simultation(c3bf_trajectory, c3bf_computational_time, method='C3BF', 
                                                                solver_failure=c3bf_solver_failure)
                data.append(c3bf_data)
                plt.close()
            else:
                print(f"\tC3BF already executed for {filename}, for noise scaling {noise_scale_param}.\n")

            if 'CBF' not in list(df_temp["Method"]) or (noise_scale_param not in list(df_temp["Noise Scaling"]) and add_noise):
                cbf_trajectory, cbf_computational_time, cbf_solver_failure = cbf_sim(seed, robot_num)
                print(f"CBF average computational time: {sum(cbf_computational_time) / len(cbf_computational_time)}\n")
                cbf_data = data_process.post_process_simultation(cbf_trajectory, cbf_computational_time, method="CBF", 
                                                                solver_failure=cbf_solver_failure)
                data.append(cbf_data)
                plt.close()
            else:
                print(f"\tCBF already executed for {filename}, for noise scaling {noise_scale_param}.\n")

            if 'LBP' not in list(df_temp["Method"]) or (noise_scale_param not in list(df_temp["Noise Scaling"]) and add_noise):
                lbp_trajectory, lbp_computational_time = lbp_sim(seed, robot_num)
                print(f"LBP average computational time: {sum(lbp_computational_time) / len(lbp_computational_time)}\n")
                lbp_data = data_process.post_process_simultation(lbp_trajectory, lbp_computational_time, method="LBP")
                data.append(lbp_data)
                plt.close()
            else:
                print(f"\tLBP already executed for {filename}, for noise scaling {noise_scale_param}.\n")

        else:
            print(f"File {filename}, for noise scaling {noise_scale_param} does not exist in the csv file, executing all methods...\n")
            dwa_trajectory, dwa_computational_time = dwa_sim(seed, robot_num)   
            print(f"DWA average computational time: {sum(dwa_computational_time) / len(dwa_computational_time)}\n")
            dwa_data = data_process.post_process_simultation(dwa_trajectory, dwa_computational_time, method='DWA')
            data.append(dwa_data)
            plt.close()

            mpc_trajectory, mpc_computational_time = mpc_sim(seed, robot_num)
            print(f"MPC average computational time: {sum(mpc_computational_time) / len(mpc_computational_time)}\n")
            mpc_data = data_process.post_process_simultation(mpc_trajectory, mpc_computational_time, method="MPC")
            data.append(mpc_data)
            plt.close()
        
            c3bf_trajectory, c3bf_computational_time, c3bf_solver_failure = c3bf_sim(seed, robot_num)
            print(f"C3BF average computational time: {sum(c3bf_computational_time) / len(c3bf_computational_time)}\n")
            c3bf_data = data_process.post_process_simultation(c3bf_trajectory, c3bf_computational_time, method='C3BF', 
                                                            solver_failure=c3bf_solver_failure)
            data.append(c3bf_data)
            plt.close()

            cbf_trajectory, cbf_computational_time, cbf_solver_failure = cbf_sim(seed, robot_num)
            print(f"CBF average computational time: {sum(cbf_computational_time) / len(cbf_computational_time)}\n")
            cbf_data = data_process.post_process_simultation(cbf_trajectory, cbf_computational_time, method="CBF", 
                                                            solver_failure=cbf_solver_failure)
            data.append(cbf_data)
            plt.close()

            lbp_trajectory, lbp_computational_time = lbp_sim(seed, robot_num)
            print(f"LBP average computational time: {sum(lbp_computational_time) / len(lbp_computational_time)}\n")
            lbp_data = data_process.post_process_simultation(lbp_trajectory, lbp_computational_time, method="LBP")
            data.append(lbp_data)
            plt.close()

        df1 = pd.DataFrame(data)
        frames = [df, df1]
        df = pd.concat(frames, ignore_index=True)

        # Save the results to a csv file
        df = data_process.remove_df_duplicates(df)
        df.to_csv(csv_file)

        print(f"Saving File {filename} simulations to csv file: {csv_file}")

if __name__ == '__main__':
    main()