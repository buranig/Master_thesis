import numpy as np
import math
# For the parameter file
import json
import yaml
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from shapely.plotting import plot_polygon, plot_line
from ament_index_python.packages import get_package_share_directory

# Write directory
dir_path = os.path.dirname(os.path.realpath(__file__))
dest_path = dir_path + "/../config/trajectories.json"




package_path = get_package_share_directory('bumper_cars')
yaml_controller = package_path + '/config/controller.yaml'  # Adjust the path to your YAML file
yaml_carModel = package_path + '/config/carModel.yaml'  # Adjust the path to your YAML file

# Opening YAML file
with open(yaml_controller, 'r') as openfile:
    # Reading from yaml file
    yaml_dwa = yaml.safe_load(openfile)

# Opening YAML file
with open(yaml_carModel, 'r') as openfile:
    # Reading from yaml file
    yaml_carModel = yaml.safe_load(openfile)

a_resolution = yaml_dwa["DWA"]["a_resolution"] # [m/s²]
v_resolution = yaml_dwa["DWA"]["v_resolution"] # [m/s]
delta_resolution = math.radians(yaml_dwa["DWA"]["delta_resolution"])# [rad/s]


max_steer = yaml_carModel["max_steer"] # [rad] max steering angle
max_speed = yaml_carModel["max_speed"] # [m/s]
min_speed = yaml_carModel["min_speed"] # [m/s]
max_acc = yaml_carModel["max_acc"] # [m/ss]
min_acc = yaml_carModel["min_acc"] # [m/ss]


def calc_dynamic_window():
    """
    Calculate the dynamic window based on the current state.
    Args:
        x (list): Current state [x(m), y(m), yaw(rad), v(m/s), delta(rad)].
    Returns:
        list: Dynamic window [min_throttle, max_throttle, min_steer, max_steer].
    """

    Vs = [min_acc, max_acc,
        -max_steer, max_steer]

    dw = [Vs[0], Vs[1], Vs[2], Vs[3]]

    return dw

def rotateMatrix(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

def main():
    package_path = get_package_share_directory('dwa_dev')

    # reading the first element of data.jason, rotating and traslating the geometry to and arbitrary position and plotting it
    with open(package_path + '/config/trajectories.json', 'r') as file:
        data = json.load(file)

    fig = plt.figure(1, dpi=90)
    ax = fig.add_subplot(111)
    vel_list = list(data.keys())
    i = [0, -1]
    for v in tqdm(range(2)):
        v = float(vel_list[i[v]])
        x_init = np.array([0.0, 0.0, np.radians(90.0), v])
        dw = calc_dynamic_window()
        for a in np.arange(dw[0], dw[1]+a_resolution, a_resolution):
            delta_list = data[str(v)][str(a)].keys()
            for delta in delta_list:
            # for delta in np.arange(dw[2], dw[3]+delta_resolution, delta_resolution):
                # print(v, a, delta)
                geom = data[str(v)][str(a)][str(delta)]
                geom = np.array(geom)
                newgeom = (geom[:, 0:2]) @ rotateMatrix(np.radians(-45)) + [3,3]
                geom = LineString(zip(geom[:, 0], geom[:, 1]))
                newgeom = LineString(zip(newgeom[:, 0], newgeom[:, 1]))
                plot_line(geom, ax=ax, add_points=False, linewidth=3)
                # plot_line(newgeom, ax=ax, add_points=False, linewidth=3)
    plt.axis('equal')
    plt.show()
    #########################################################################################################        

if __name__ == '__main__':
    main()