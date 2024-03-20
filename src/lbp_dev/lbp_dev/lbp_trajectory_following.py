import json
import numpy as np
import pathlib
import math
from LBP import normalize_angle
from shapely.geometry import Point, Polygon, LineString
import matplotlib.pyplot as plt
from planner import utils as utils

path = pathlib.Path('/home/giacomo/thesis_ws/src/bumper_cars/params.json')
# Opening JSON file
with open(path, 'r') as openfile:
    # Reading from json file
    json_object = json.load(openfile)

max_steer = json_object["LBP"]["max_steer"] # [rad] max steering angle
max_speed = json_object["LBP"]["max_speed"] # [m/s]
min_speed = json_object["LBP"]["min_speed"] # [m/s]
v_resolution = json_object["LBP"]["v_resolution"] # [m/s]
delta_resolution = math.radians(json_object["LBP"]["delta_resolution"])# [rad/s]
max_acc = json_object["LBP"]["max_acc"] # [m/ss]
min_acc = json_object["LBP"]["min_acc"] # [m/ss]
dt = json_object["Controller"]["dt"] # [s] Time tick for motion prediction
predict_time = json_object["LBP"]["predict_time"] # [s]
to_goal_cost_gain = json_object["LBP"]["to_goal_cost_gain"]
speed_cost_gain = json_object["LBP"]["speed_cost_gain"]
obstacle_cost_gain = json_object["LBP"]["obstacle_cost_gain"]
robot_stuck_flag_cons = json_object["LBP"]["robot_stuck_flag_cons"]
dilation_factor = json_object["LBP"]["dilation_factor"]
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

robot_num = json_object["robot_num"]
timer_freq = json_object["timer_freq"]

show_animation = True
v_ref = 2.0 # [m/s] reference speed

# Load trajectory and control inputs from LBP.json file
with open('src/lbp_dev/lbp_dev/LBP.json', 'r') as file:
    data = json.load(file)

def interp0(x, xp, yp):
    """Zeroth order hold interpolation w/ same
    (base)   signature  as numpy.interp.
    For report plotting purposes."""

    def func(x0):
        if x0 <= xp[0]:
            return yp[0]
        if x0 >= xp[-1]:
            return yp[-1]
        k = 0
        while x0 > xp[k]:
            k += 1
        return yp[k-1]
    
    if isinstance(x,float):
        return func(x)
    elif isinstance(x, list):
        return [func(x) for x in x]
    elif isinstance(x, np.ndarray):
        return np.asarray([func(x) for x in x])
    else:
        raise TypeError('argument must be float, list, or ndarray')

def plot_arrow(x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)

def plot_robot(x, y, yaw):  # pragma: no cover
        outline = np.array([[-L / 2, L / 2,
                             (L / 2), -L / 2,
                             -L / 2],
                            [WB / 2, WB / 2,
                             - WB / 2, -WB / 2,
                             WB / 2]])
        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                         [-math.sin(yaw), math.cos(yaw)]])
        outline = (outline.T.dot(Rot1)).T
        outline[0, :] += x
        outline[1, :] += y
        plt.plot(np.array(outline[0, :]).flatten(),
                 np.array(outline[1, :]).flatten(), "-k")


# Implement kinematic bicycle model to follow the trajectory
def main():
    v = 2.0 # [m/s] reference speed
    x = [0.0, 0.0, 0.0, 0.0] #[x, y, yaw, v]   
    ref = data[str(v)]['10']
    trajectory = np.zeros((len(ref['x']),3))
    trajectory[:,0] = ref['x']
    trajectory[:,1] = ref['y']
    trajectory[:,2] = ref['yaw']
    control_inputs = ref['ctrl']

    
    fig = plt.figure(1, dpi=90)
    ax = fig.add_subplot(111)
    fontsize = 20
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = fontsize
    a_buf = []

    for i in range(len(control_inputs)):
        # Get current state and control input
        delta = control_inputs[i]
        a = (v-x[3])/dt
        a = np.clip(a, -5, 5)
        a_buf.append(a)
        u = [a, delta]

        # Apply control input to the kinematic bicycle model
        # TODO: Implement the kinematic bicycle model logic here
        x = utils.motion(x, u, dt)

        # Print the current state and control input
        print(f"Control Input: {u}")
        print(f'Current State: {x[0:3]}, reference: {trajectory[i,:]}')
        print(f'Current Speed: {x[3]}\n')
        
        # plot the trajectory, the robot and the yaw
        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        
        plt.plot(trajectory[:, 0], trajectory[:, 1], "-g")
        # plot_polygon(dilated_traj[i], ax=ax, add_points=False, alpha=0.5)
        plt.plot(x[0], x[1], "xr")
        plt.plot(trajectory[-1, 0], trajectory[-1, 1], "xb")
        plot_robot(x[0], x[1], x[2])
        plot_arrow(x[0], x[1], x[2], length=1, width=0.2)
        plot_arrow(x[0], x[1], x[2]+u[1], length=1, width=0.2)
  
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)

    plt.show()

    # Create two subplots and unpack the output array immediately
    t = np.arange(0, 3.1, 0.1)
    t2 = np.linspace(0, 3.1, 1000)
    fig = plt.figure(2, dpi=90, figsize=(14, 5))
    # fig, (ax1, ax2, ax3) = plt.subplots(2, 2, sharex=True)
    fig.suptitle('LBP Trajectory Following')
    ax1 = fig.add_subplot(221)
    # ax1.plot(t, a_buf, label='Acceleration')
    ax1.scatter(t, a_buf, color='b', s=20)
    ax1.plot(t2,interp0(t2,t,a_buf),'r',label='Discrete Acceleration')
    ax1.set_title('Acceleration')
    ax1.set_ylim(min_acc-1, max_acc+1)
    ax1.set_ylabel("Acceleration", fontdict={'size': fontsize, 'family': 'serif'})
    ax1.grid(True)
    ax1.legend()
    plt.scatter
    ax2 = fig.add_subplot(223)
    ax2.set_title('Steering Inputs')
    # ax2.plot(t, control_inputs, label='Control Inputs')
    ax2.scatter(t, control_inputs, color='b', s=20)
    ax2.plot(t2,interp0(t2,t,control_inputs),'r', label='Discrete Control Inputs')
    ax2.scatter([0, (len(control_inputs)-1)/20, (len(control_inputs)-1)/10], [control_inputs[0], control_inputs[int(len(control_inputs)/2)], control_inputs[-1]], color='r', marker='x', s=90,label='Checkpoints for interpolation')
    ax2.set_xlabel('time (s)')
    ax2.set_ylim(-max_steer-0.1, max_steer+0.1)
    ax2.set_xlabel("Time (s)", fontdict={'size': fontsize, 'family': 'serif'})
    ax2.set_ylabel("Steering", fontdict={'size': fontsize, 'family': 'serif'})
    ax2.grid(True)
    ax2.legend()

    ax3 = fig.add_subplot(122)
    ax3.plot(trajectory[:, 0], trajectory[:, 1], "-g", label='Reference Trajectory')
    ax3.set_xlabel("x (m)", fontdict={'size': fontsize, 'family': 'serif'})
    ax3.set_ylabel("y (m)", fontdict={'size': fontsize, 'family': 'serif'})
    ax3.set_title('Reference Trajectory')
    ax3.grid(True)
    ax3.legend()
    plt.show()


if __name__ == "__main__":
    main()