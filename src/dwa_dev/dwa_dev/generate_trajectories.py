import numpy as np
import math
# For the parameter file
import json
import os

# Write directory
dir_path = os.path.dirname(os.path.realpath(__file__))
dest_path = dir_path + "/../config/trajectories.json"

# DWA config params
dwa_path = dir_path + '/../config/dwa_config.json'

# Vehicle config params
model_path = dir_path + '/../../bumper_cars/params.json'

json_object = {}

# Read JSON file
with open(dwa_path, 'r') as openfile:
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
L = json_object["Car_model"]["L"]  # [m] Wheel base of vehicle
# Lr = L / 2.0  # [m]
# Lf = L - Lr
# Cf = json_object["Car_model"]["Cf"]  # N/rad
# Cr = json_object["Car_model"]["Cr"] # N/rad
# Iz = json_object["Car_model"]["Iz"]  # kg/m2
# m = json_object["Car_model"]["m"]  # kg
# # Aerodynamic and friction coefficients
# c_a = json_object["Car_model"]["c_a"]
# c_r1 = json_object["Car_model"]["c_r1"]
# WB = json_object["Controller"]["WB"] # Wheel base
# robot_num = json_object["robot_num"]
# safety_init = json_object["safety"]
# width_init = json_object["width"]
# height_init = json_object["height"]
# N=3

def motion(x, u, dt):
    """
    motion model
    initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    """
    delta = u[1]
    delta = np.clip(delta, -max_steer, max_steer)
    throttle = u[0]

    x[0] = x[0] + x[3] * math.cos(x[2]) * dt
    x[1] = x[1] + x[3] * math.sin(x[2]) * dt
    x[2] = x[2] + x[3] / L * math.tan(delta) * dt
    x[3] = x[3] + throttle * dt
    x[2] = normalize_angle(x[2])
    x[3] = np.clip(x[3], min_speed, max_speed)

    return x

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


def calc_trajectory(x_init, u, dt):
    """
    calc trajectory
    """
    x = np.array(x_init)
    traj = np.array(x)
    time = 0.0
    while time <= predict_time:
        x = motion(x, u, dt)
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
          -max_steer, max_steer]
    
    
    # Dynamic window from motion model
    # Vd = [x[3] - config.max_acc*0.1,
    #       x[3] + config.max_acc*0.1,
    #       -max_steer,
    #       max_steer]
    
    # #  [min_throttle, max_throttle, min_steer, max_steer]
    # dw = [min(Vs[0], Vd[0]), min(Vs[1], Vd[1]), min(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    dw = [Vs[0], Vs[1], Vs[2], Vs[3]]
    
    return dw

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

def main():
    print(__file__ + " start!!")

    complete_trajectories = {}

    # initial state [x(m), y(m), yaw(rad), v(m/s)]
    for v in np.arange(min_speed, max_speed, v_resolution):
        x_init = np.array([0.0, 0.0, np.radians(90.0), v])
        traj, u_total = generate_trajectories(x_init)

        traj = np.array(traj)
        temp2 = {}
        for j in range(len(traj)):
            temp2[u_total[j][0]] = {}
        for i in range(len(traj)):
            temp2[u_total[i][0]][u_total[i][1]] = traj[i, :, :].tolist()
        complete_trajectories[v] = temp2
        
    # saving the complete trajectories to a csv file
    with open(dest_path, 'w') as file:
        json.dump(complete_trajectories, file, indent=4)

    print("\nThe JSON data has been written to: "+dest_path)
                
if __name__ == '__main__':
    main()


# ROS Wrapper
    
import rclpy
from rclpy.node import Node

class TestParams(Node):
    def __init__(self):
        global dest_path, model_path, \
        max_steer,max_speed,min_speed,a_resolution,v_resolution, \
        delta_resolution,max_acc,min_acc,dt,predict_time,L
        super().__init__('traj_gen')
        self.declare_parameter('car_json', model_path)
        self.declare_parameter('traj_json', dest_path)

        dwa_path = self.get_parameter('car_json').value
        dest_path = self.get_parameter('traj_json').value

        # Reload JSON file
        with open(dwa_path, 'r') as openfile:
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
        L = json_object["Car_model"]["L"]  # [m] Wheel base of vehicle
        # Lr = L / 2.0  # [m]
        # Lf = L - Lr
        # Cf = json_object["Car_model"]["Cf"]  # N/rad
        # Cr = json_object["Car_model"]["Cr"] # N/rad
        # Iz = json_object["Car_model"]["Iz"]  # kg/m2
        # m = json_object["Car_model"]["m"]  # kg
        # # Aerodynamic and friction coefficients
        # c_a = json_object["Car_model"]["c_a"]
        # c_r1 = json_object["Car_model"]["c_r1"]
        # WB = json_object["Controller"]["WB"] # Wheel base
        # robot_num = json_object["robot_num"]
        # safety_init = json_object["safety"]
        # width_init = json_object["width"]
        # height_init = json_object["height"]
        # N=3
    def main(self):
        main()

def main_ros(args=None):
    rclpy.init(args=args)
    node = TestParams()
    node.main()
    node.destroy_node()
    rclpy.shutdown()