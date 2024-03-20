"""

Lookup Table generation for model predictive trajectory generator

author: Atsushi Sakai

"""
import sys
import pathlib
path_planning_dir = pathlib.Path(__file__).parent.parent
sys.path.append(str(path_planning_dir))

from matplotlib import pyplot as plt
import numpy as np
import math
from planner import utils as utils

import lattice_motion_model as motion_model

from lattice import calc_uniform_polar_states, generate_path
import json

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
predict_time = 3 # json_object["DWA"]["predict_time"] # [s]
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
robot_num = json_object["robot_num"]
timer_freq = json_object["timer_freq"]

with open('/home/giacomo/thesis_ws/src/trajectories.json', 'r') as file:
    data1 = json.load(file)

def calc_states_list(max_yaw=np.deg2rad(-30.0)):

    x = np.arange(1.0, 8.0, 1.0)
    y = np.arange(0.0, 4.0, 1.0)
    yaw = np.arange(-max_yaw, max_yaw, max_yaw)

    states = []
    for iyaw in yaw:
        for iy in y:
            for ix in x:
                states.append([ix, iy, iyaw])
    print("n_state:", len(states))

    return states


def search_nearest_one_from_lookup_table(tx, ty, tyaw, lookup_table):
    mind = float("inf")
    minid = -1

    for (i, table) in enumerate(lookup_table):

        dx = tx - table[0]
        dy = ty - table[1]
        dyaw = tyaw - table[2]
        d = math.sqrt(dx ** 2 + dy ** 2 + dyaw ** 2)
        if d <= mind:
            minid = i
            mind = d

    # print(minid)

    return lookup_table[minid]

def save_lookup_table(file_name, table):
    np.savetxt(file_name, np.array(table),
               fmt='%s', delimiter=",", header="x,y,yaw,s,km,kf", comments="")

    print("lookup table file is saved as " + file_name)

def calc_trajectory(x_init, u, dt):
    """
    calc trajectory
    """
    x = np.array(x_init)
    traj = np.array(x)
    time = 0.0
    while time <= predict_time:
        x = utils.motion(x, u, dt)
        traj = np.vstack((traj, x))
        time += dt
        if x[3]>max_speed or x[3]<min_speed:
            print(x[3])
    return traj

def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].
    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi
    while angle < -np.pi:
        angle += 2.0 * np.pi
    return angle

def generate_lookup_table():
    temp = {}
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 11
    for v in np.arange(0.5, 2.0+0.5, 0.5):
        temp[v] = {}
        k0 = 0.0
        nxy = 5
        nh = 3
        d = v*predict_time
        print(f'distance: {d}')

        if v == 0.5:
            angle = 45
            a_min = - np.deg2rad(angle)
            a_max = np.deg2rad(angle)
            p_min = - np.deg2rad(angle)
            p_max = np.deg2rad(angle)
        else:
            angle = 60
            a_min = - np.deg2rad(angle)
            a_max = np.deg2rad(angle)
            p_min = - np.deg2rad(angle)
            p_max = np.deg2rad(angle)
        states = calc_uniform_polar_states(nxy, nh, d, a_min, a_max, p_min, p_max)
        result = generate_path(states, k0, v)

        i = 0
        for table in result:
            xc, yc, yawc, kp = motion_model.generate_trajectory(
                table[3], table[4], table[5], k0, v)
            for id, element in enumerate(kp): kp[id] = np.clip(element, -max_steer, max_steer) # clipping elements withing feasible bounds
            temp[v][i] = {}
            temp[v][i]['ctrl'] = list(kp)
            temp[v][i]['x'] = xc
            temp[v][i]['y'] = yc
            temp[v][i]['yaw'] = yawc
            i +=1

            if show_animation:
                plt.plot(xc, yc, "-r")
        
        if v==1.0:
            target = [[1.0, 3.0, np.deg2rad(90.0)],
                      [1.0, -3.0, np.deg2rad(-90.0)],
                      [1.5, 3.0, np.deg2rad(90.0)],
                      [1.5, -3.0, np.deg2rad(-90.0)]]
            result = generate_path(target, k0, v, k=True)
            i = 0
            for table in result:
                xc, yc, yawc, kp = motion_model.generate_trajectory(
                    table[3], table[4], table[5], k0, v)
                for id, element in enumerate(kp): kp[id] = np.clip(element, -max_steer, max_steer) # clipping elements withing feasible bounds
                temp[v][i] = {}
                temp[v][i]['ctrl'] = list(kp)
                temp[v][i]['x'] = xc
                temp[v][i]['y'] = yc
                temp[v][i]['yaw'] = yawc
                i +=1

                if show_animation:
                    plt.plot(xc, yc, "-r")

        if show_animation:
            plt.xlabel("x [m]", fontdict={'size': 11, 'family': 'serif'})
            plt.ylabel("y [m]", fontdict={'size': 11, 'family': 'serif'})
            plt.title('LBP Trajectory Generation')
            plt.grid(True)
            plt.axis("equal")
            plt.show()

        print("Done")
    print("finish lookup table generation")

    for v in np.arange(-1.0, 0.5, 0.5):
        x_init = [0.0, 0.0, 0.0, v]
        i = 0
        temp[v] = {}
        for delta in np.arange(-max_steer, max_steer+delta_resolution, delta_resolution):
            u = [0.0, delta]
            traj = calc_trajectory(x_init, u, dt)
            # plt.plot(traj[:, 0], traj[:, 1])
            xc = traj[:, 0]
            yc = traj[:, 1]
            yawc = traj[:, 2]
            kp = [delta]*len(xc)

            temp[v][i] = {}
            temp[v][i]['ctrl'] = list(kp)
            temp[v][i]['x'] = list(xc)
            temp[v][i]['y'] = list(yc)
            temp[v][i]['yaw'] = list(yawc)
            i +=1
            # print(f'len: {len(xc)}')

    for id, info in temp.items():
        print(f'\nV: {id}')
        for id2, info2 in info.items():
            print(f"\nN°: {id2}, lenght: {np.degrees(info2['ctrl'])}")
            # plot_polygon(info.buffer(0.5, cap_style=3, join_style=3))
            # plot_line(LineString(zip(info2['x'], info2['y'])))
            plt.plot(info2['x'], info2['y'], 'r')
    plt.show()

    # saving the complete trajectories to a csv file
    with open('src/lbp_dev/lbp_dev/LBP.json', 'w') as file:
        json.dump(temp, file, indent=4)

    print("\nThe JSON data has been written to 'src/lbp_dev/lbp_dev/LBP.json'")

    # states = calc_states_list(max_yaw=np.deg2rad(-35.0))
    # k0 = 0.0

    # # x, y, yaw, s, km, kf
    # lookup_table = [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]

    
    # temp2 = {}
    # i = 0

    # for state in states:
    #     best_p = search_nearest_one_from_lookup_table(
    #         state[0], state[1], state[2], lookup_table)

    #     if state[0] == 1.0: print(state)
    #     target = motion_model.State(x=state[0], y=state[1], yaw=state[2])
    #     init_p = np.array(
    #         [np.hypot(state[0], state[1])*1.5, best_p[4], best_p[5]]).reshape(3, 1)

    #     x, y, yaw, p, kp = trajectory_generator.optimize_trajectory(target,
    #                                                             k0, init_p)

    #     if x is not None:
    #         print("find good path")
    #         lookup_table.append(
    #             [x[-1], y[-1], yaw[-1], float(p[0, 0]), float(p[1, 0]), float(p[2, 0])])
            
    #         line = LineString(zip(x, y))
    #         line1 = LineString(zip(x, -np.array(y)))
    #         # dilated = line.buffer(1.5, cap_style=3, join_style=3)
    #         # coords = []
    #         # for idx in range(len(dilated.exterior.coords)):
    #         #     coords.append([dilated.exterior.coords[idx][0], dilated.exterior.coords[idx][1]])
    #         temp2[i] = {}
    #         temp2[i]['ctrl'] = list(kp)
    #         temp2[i]['x'] = x
    #         temp2[i]['y'] = y
    #         temp2[i]['yaw'] = yaw
    #         i +=1
    #         temp2[i] = {}
    #         temp2[i]['ctrl'] = list(-np.array(kp))
    #         temp2[i]['x'] = x
    #         temp2[i]['y'] = list(-np.array(y))
    #         temp2[i]['yaw'] = list(-np.array(yaw))
    #         i+=1

    # print("finish lookup table generation")

    # for id, info in temp2.items():
    #     print(f"\nV: {id}, lenght: {np.degrees(info['ctrl'])}")
    #     # plot_polygon(info.buffer(0.5, cap_style=3, join_style=3))
    #     plot_line(LineString(zip(info['x'], info['y'])))
    # plt.show()
    # # saving the complete trajectories to a csv file
    # with open('src/lattice_planning/LBP.json', 'w') as file:
    #     json.dump(temp2, file, indent=4)

    # print("\nThe JSON data has been written to 'data.json'")

    # save_lookup_table("src/lattice_planning/LBP.csv", lookup_table)

    # for table in lookup_table:
    #     x_c, y_c, yaw_c, kp = motion_model.generate_trajectory(
    #         table[3], table[4], table[5], k0)
    #     plt.plot(x_c, y_c, "-r")
    #     x_c, y_c, yaw_c, kp = motion_model.generate_trajectory(
    #         table[3], -table[4], -table[5], k0)
    #     plt.plot(x_c, y_c, "-r")

    # plt.grid(True)
    # plt.axis("equal")
    # plt.show()

    # print("Done")


def main():
    generate_lookup_table()


if __name__ == '__main__':
    main()