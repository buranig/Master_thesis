import matplotlib.pyplot as plt
import numpy as np
import math
from enum import Enum
# For the parameter file
import pathlib
import json
from shapely.geometry import Point, LineString
from shapely.plotting import plot_polygon, plot_line
from dwa_dev import DWA as DWA
import planner.utils as utils

color_dict = {0: 'r', 1: 'b', 2: 'g', 3: 'y', 4: 'm', 5: 'c', 6: 'k'}

path = pathlib.Path('/home/giacomo/thesis_ws/src/bumper_cars/params.json')
# Opening JSON file
with open(path, 'r') as openfile:
    # Reading from json file
    json_object = json.load(openfile)

max_steer = json_object["DWA"]["max_steer"] # [rad] max steering angle
max_speed = json_object["DWA"]["max_speed"] # [m/s]
min_speed = json_object["DWA"]["min_speed"] # [m/s]
a_resolution = json_object["DWA"]["a_resolution"] # [m/s²]
v_resolution = json_object["DWA"]["v_resolution"] # [m/s]
delta_resolution = math.radians(json_object["DWA"]["delta_resolution"])# [rad/s]
max_acc = json_object["DWA"]["max_acc"] # [m/ss]
min_acc = json_object["DWA"]["min_acc"] # [m/ss]
dt = json_object["Controller"]["dt"] # [s] Time tick for motion prediction
predict_time = json_object["DWA"]["predict_time"] # [s]
to_goal_cost_gain = json_object["DWA"]["to_goal_cost_gain"]
speed_cost_gain = json_object["DWA"]["speed_cost_gain"]
obstacle_cost_gain = json_object["DWA"]["obstacle_cost_gain"]
robot_stuck_flag_cons = json_object["DWA"]["robot_stuck_flag_cons"]
dilation_factor = json_object["DWA"]["dilation_factor"]
L = json_object["Car_model"]["L"]  # [m] Wheel base of vehicle
Lr = L / 2.0  # [m]
Lf = L - Lr
Cf = json_object["Car_model"]["Cf"]  # N/rad
Cr = json_object["Car_model"]["Cr"] # N/rad
Iz = json_object["Car_model"]["Iz"]  # kg/m2
m = json_object["Car_model"]["m"]  # kg
# Aerodynamic and friction coefficients
c_a = json_object["Car_model"]["c_a"]
c_r1 = json_object["Car_model"]["c_r1"]
WB = json_object["Controller"]["WB"] # Wheel base
robot_num = json_object["robot_num"]
safety_init = json_object["safety"]
width_init = json_object["width"]
height_init = json_object["height"]
N=3
save_flag = True
show_animation = True
plot_flag = False
timer_freq = json_object["timer_freq"]
kinematic = True

class RobotType(Enum):
    circle = 0
    rectangle = 1

robot_type = RobotType.rectangle
robot_radius = 1.0

def calc_trajectory(x_init, u, dt):
    """
    calc trajectory
    """
    x = np.array(x_init)
    traj = np.array(x)
    time = 0.0
    while time <= predict_time:
        if kinematic:
            x = utils.motion(x, u, dt)
        else:
            x = utils.nonlinear_model_numpy_stable(x, u, dt)
        traj = np.vstack((traj, x))
        time += dt
        if x[3]>max_speed or x[3]<min_speed:
            print(x[3])
    return traj

def calc_dynamic_window():
    """
    calculation dynamic window based on current state x
    motion model
    initial state [x(m), y(m), yaw(rad), v(m/s), delta(rad)]
    """
    # Dynamic window from robot specification
    Vs = [min_acc, max_acc,
          -max_steer, max_steer]  # [m/s^2, m/s^2, rad/s, rad/s]

    dw = [Vs[0], Vs[1], Vs[2], Vs[3]]
    
    return dw

def generate_trajectories_normal(x_init):
    """
    Generate trajectories
    """
    dw = calc_dynamic_window()
    traj = []
    u_total = []
    for a in np.arange(dw[0], dw[1]+a_resolution, a_resolution):
    
        for delta in np.arange(dw[2]/2, dw[3]/2+delta_resolution/2, delta_resolution/6):
            u = np.array([a, delta])
            traj.append(calc_trajectory(x_init, u, dt))
            u_total.append(u)

        for delta in np.arange(dw[2], dw[2]/2, delta_resolution):
            u = np.array([a, delta])
            traj.append(calc_trajectory(x_init, u, dt))
            u_total.append(u)

        for delta in np.arange(dw[3]/2+delta_resolution, dw[3]+delta_resolution, delta_resolution):
            u = np.array([a, delta])
            traj.append(calc_trajectory(x_init, u, dt))
            u_total.append(u)

    return traj, u_total

def generate_trajectories(x_init):
    """
    Generate trajectories
    """
    dw = calc_dynamic_window()
    traj = []
    u_total = []
    for a in np.arange(dw[0], dw[1]+a_resolution, a_resolution):
    
        for delta in np.arange(dw[2], dw[3]+delta_resolution, delta_resolution):
            u = np.array([a, delta])
            traj.append(calc_trajectory(x_init, u, dt))
            u_total.append(u)

    return traj, u_total

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def main():
    print(__file__ + " start!!")
    print("\n######################STARTING THE TRAJECTORY GENERATION######################\n")
    if kinematic:
        print("###################### KINEMATIC MODEL ######################")
    else:
        print("###################### DYNAMIC MODEL ######################")
    
    print("Max speed: ", max_speed)
    print("Min speed: ", min_speed)
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 11  
    if save_flag:
        complete_trajectories = {}

        # initial state [x(m), y(m), yaw(rad), v(m/s)]
        for v in np.arange(min_speed, max_speed+v_resolution, v_resolution):
            print(v)

            if kinematic:
                x_init = np.array([0.0, 0.0, 0.0, v])
            else:
                x_init = np.array([0.0, 0.0, 0.0, v, 0.0])

            traj, u_total = generate_trajectories(x_init)
            # traj, u_total = generate_trajectories_normal(x_init)
            
            if plot_flag:
                plt.plot(x_init[0], x_init[1], "xr")
                for trajectory in traj:
                    # for stopping simulation with the esc key.
                    plt.gcf().canvas.mpl_connect(
                        'key_release_event',
                        lambda event: [exit(0) if event.key == 'escape' else None])
                    plt.plot(trajectory[:, 0], trajectory[:, 1], "-r")
                    plt.grid(True)
                    plt.axis("equal")

                plt.show()

                # fig = plt.figure(1, dpi=90)
                # ax = fig.add_subplot(111)
                # traj = np.array(traj)
                # for i in range(len(traj)):
                #     line = LineString(zip(traj[i, :, 0], traj[i, :, 1]))
                #     dilated = line.buffer(dilation_factor, cap_style=3, join_style=3)
                #     plot_line(line, ax=ax, add_points=False, linewidth=3)
                #     plot_polygon(dilated, ax=ax, add_points=False, alpha=0.5)
                # plt.show()

            traj = np.array(traj)
            temp2 = {}
            for j in range(len(traj)):
                temp2[u_total[j][0]] = {}
            
            for i in range(len(traj)):
                line = LineString(zip(traj[i, :, 0], traj[i, :, 1]))
                dilated = line.buffer(dilation_factor, cap_style=3, join_style=3)
                coords = []
                for idx in range(len(dilated.exterior.coords)):
                    coords.append([dilated.exterior.coords[idx][0], dilated.exterior.coords[idx][1]])
                temp2[u_total[i][0]][u_total[i][1]] = traj[i, :, :].tolist()
            complete_trajectories[v] = temp2

        if plot_flag:
            plt.xlabel("x [m]", fontdict={'size': 11, 'family': 'serif'})
            plt.ylabel("y [m]", fontdict={'size': 11, 'family': 'serif'})
            plt.title('DWA Trajectories Set')
            plt.grid(True)
            plt.axis("equal")
            plt.show()
        # print(complete_trajectories)
        
        # saving the complete trajectories to a csv file
        if kinematic:
            with open('src/dwa_dev/trajectories.json', 'w') as file:
                json.dump(complete_trajectories, file, indent=4)
            print("\nThe JSON data has been written to 'src/dwa_dev/trajectories.json'")
            
        else:
            with open('src/dwa_dev/dynamic_trajectories.json', 'w') as file:
                json.dump(complete_trajectories, file, indent=4)
            print("\nThe JSON data has been written to 'src/dwa_dev/dynamic_trajectories.json'")


    # reading the first element of data.jason, rotating and traslating the geometry to and arbitrary position and plotting it
    if kinematic:
        with open('src/dwa_dev/trajectories.json', 'r') as file:
            data = json.load(file)
    else:
        with open('src/dwa_dev/dynamic_trajectories.json', 'r') as file:
            data = json.load(file)

    fig = plt.figure(1, dpi=90)
    ax = fig.add_subplot(111)
    for v in np.arange(min_speed, max_speed, v_resolution):
        x_init = np.array([0.0, 0.0, np.radians(90.0), v])
        dw = calc_dynamic_window()
        for a in np.arange(dw[0], dw[1]+a_resolution, a_resolution):
            delta_keys = list(data[str(v)][str(a)].keys())
            for delta in delta_keys:
                # print(v, a, delta)
                geom = data[str(v)][str(a)][str(delta)]
                geom = np.array(geom)
                newgeom = (geom[:, 0:2]) @ utils.rotateMatrix(np.radians(-45)) + [10,10]
                geom = LineString(zip(geom[:, 0], geom[:, 1]))
                newgeom = LineString(zip(newgeom[:, 0], newgeom[:, 1]))
                plot_line(geom, ax=ax, add_points=False, linewidth=3)
                plot_line(newgeom, ax=ax, add_points=False, linewidth=3)
                
                # geom = data[str(v)][str(a)][str(delta)]
                # geom = LineString(zip(geom[i, :, 0], geom[i, :, 1]))
                # newgeom = (geom) @ rotateMatrix(np.radians(-45)) + [10,10]
                # plot_polygon(Polygon(geom), ax=ax, add_points=False, alpha=0.5)
                # plot_polygon(Polygon(newgeom), ax=ax, add_points=False, alpha=0.5)
    plt.show()
    #########################################################################################################

    print("Starting the simulation!")
    iterations = 3000
    N=2
    break_flag = False
    v = np.arange(min_speed, max_speed, v_resolution)

    # x = np.array([[0, 20, 15], [0, 0, 20], [0, np.pi, -np.pi/2], [0, 0, 0]])
    # goal = np.array([[30, 0, 15], [10, 10, 0]])
    x = np.array([[0, 10], [0, 0], [0, np.pi], [0, 0], [0, 0]])
    goal = np.array([[10, 0], [10, 10]])
    u = np.zeros((2, N))
    
    # create a trajcetory array to store the trajectory of the N robots
    trajectory = np.zeros((x.shape[0], N, 1))
    # append the firt state to the trajectory
    trajectory[:, :, 0] = x

    predicted_trajectory = dict.fromkeys(range(2),np.zeros([int(predict_time/dt), 4]))
    for i in range(2):
        predicted_trajectory[i] = np.full((int(predict_time/dt), 4), x[0:4,i])

    paths = [[goal[0,idx], goal[1,idx]] for idx in range(N)]

    # Step 5: Extract the target coordinates from the paths
    targets = [[goal[0,idx], goal[1,idx]] for idx in range(N)]

    # Step 6: Create dilated trajectories for each robot
    dilated_traj = []
    for i in range(N):
        dilated_traj.append(Point(x[0, i], x[1, i]).buffer(dilation_factor, cap_style=3))
    
    fig = plt.figure(1, dpi=90, figsize=(10,10))
    ax = fig.add_subplot(111)

    u_hist = dict.fromkeys(range(N),[[0,0] for _ in range(int(predict_time/dt))])    
    dwa = DWA.DWA_algorithm(N, paths, paths, targets, dilated_traj, predicted_trajectory, ax, u_hist)
    
    for z in range(iterations):
        plt.cla()
        plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
        
        x, u, break_flag = dwa.go_to_goal(x, u, break_flag)
        trajectory = np.dstack([trajectory, x])
            
        utils.plot_map(width=width_init, height=height_init)
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)

        if break_flag:
            break
    
    # plt.close()
    print("Done")
    if show_animation:
        for i in range(N):
            utils.plot_robot(x[0, i], x[1, i], x[2, i], i)
            utils.plot_arrow(x[0, i], x[1, i], x[2, i] + u[1, i], length=3, width=0.5)
            utils.plot_arrow(x[0, i], x[1, i], x[2, i], length=1, width=0.5)
            plt.plot(trajectory[0, i, :], trajectory[1, i, :], "-"+color_dict[i])
        plt.pause(0.0001)
        plt.show()


        
        

if __name__ == '__main__':
    main()