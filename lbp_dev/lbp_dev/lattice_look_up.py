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

import lattice_planner as trajectory_generator,\
    lattice_motion_model as motion_model

from shapely.geometry import Point, Polygon, LineString
from shapely.plotting import plot_polygon, plot_line
import json


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


def generate_lookup_table():
    states = calc_states_list(max_yaw=np.deg2rad(-35.0))
    k0 = 0.0

    # x, y, yaw, s, km, kf
    lookup_table = [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]

    
    temp2 = {}
    temp = {}
    i = 0

    for state in states:
        best_p = search_nearest_one_from_lookup_table(
            state[0], state[1], state[2], lookup_table)

        if state[0] == 1.0: print(state)
        target = motion_model.State(x=state[0], y=state[1], yaw=state[2])
        init_p = np.array(
            [np.hypot(state[0], state[1])*1.5, best_p[4], best_p[5]]).reshape(3, 1)

        x, y, yaw, p, kp = trajectory_generator.optimize_trajectory(target,
                                                                k0, init_p)

        if x is not None:
            print("find good path")
            lookup_table.append(
                [x[-1], y[-1], yaw[-1], float(p[0, 0]), float(p[1, 0]), float(p[2, 0])])
            
            line = LineString(zip(x, y))
            line1 = LineString(zip(x, -np.array(y)))
            # dilated = line.buffer(1.5, cap_style=3, join_style=3)
            # coords = []
            # for idx in range(len(dilated.exterior.coords)):
            #     coords.append([dilated.exterior.coords[idx][0], dilated.exterior.coords[idx][1]])
            temp2[i] = {}
            temp2[i]['ctrl'] = list(kp)
            temp2[i]['x'] = x
            temp2[i]['y'] = y
            temp2[i]['yaw'] = yaw
            i +=1
            temp2[i] = {}
            temp2[i]['ctrl'] = list(-np.array(kp))
            temp2[i]['x'] = x
            temp2[i]['y'] = list(-np.array(y))
            temp2[i]['yaw'] = list(-np.array(yaw))
            i+=1

    print("finish lookup table generation")

    for id, info in temp2.items():
        print(f"\nV: {id}, lenght: {np.degrees(info['ctrl'])}")
        # plot_polygon(info.buffer(0.5, cap_style=3, join_style=3))
        plot_line(LineString(zip(info['x'], info['y'])))
    plt.show()
    # saving the complete trajectories to a csv file
    with open('src/lattice_planning/LBP.json', 'w') as file:
        json.dump(temp2, file, indent=4)

    print("\nThe JSON data has been written to 'data.json'")

    save_lookup_table("src/lattice_planning/LBP.csv", lookup_table)

    for table in lookup_table:
        x_c, y_c, yaw_c, kp = motion_model.generate_trajectory(
            table[3], table[4], table[5], k0)
        plt.plot(x_c, y_c, "-r")
        x_c, y_c, yaw_c, kp = motion_model.generate_trajectory(
            table[3], -table[4], -table[5], k0)
        plt.plot(x_c, y_c, "-r")

    plt.grid(True)
    plt.axis("equal")
    plt.show()

    print("Done")


def main():
    generate_lookup_table()


if __name__ == '__main__':
    main()