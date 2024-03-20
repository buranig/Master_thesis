import matplotlib.pyplot as plt
import numpy as np
import math
from enum import Enum
# For the parameter file
import pathlib
import json
from custom_message.msg import Coordinate
from shapely.geometry import Point, Polygon, LineString
from shapely import intersection, distance
from shapely.plotting import plot_polygon, plot_line
import planner.utils as utils
# for debugging
import time

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
heading_cost_gain = json_object["LBP"]["heading_cost_gain"]
robot_stuck_flag_cons = json_object["LBP"]["robot_stuck_flag_cons"]
dilation_factor = json_object["LBP"]["dilation_factor"]

L = json_object["Car_model"]["L"]  # [m] Wheel base of vehicle
Lr = L / 2.0  # [m]
Lf = L - Lr

WB = json_object["Controller"]["WB"] # Wheel base
robot_num = json_object["robot_num"]
safety_init = json_object["safety"]
width_init = json_object["width"]
height_init = json_object["height"]
min_dist = json_object["min_dist"]
to_goal_stop_distance = json_object["to_goal_stop_distance"]
update_dist = 2
N=3

show_animation = json_object["show_animation"]
boundary_points = np.array([-width_init/2, width_init/2, -height_init/2, height_init/2])
check_collision_bool = False
add_noise = json_object["add_noise"]
noise_scale_param = json_object["noise_scale_param"]
np.random.seed(1)

color_dict = {0: 'r', 1: 'b', 2: 'g', 3: 'y', 4: 'm', 5: 'c', 6: 'k', 7: 'tab:orange', 8: 'tab:brown', 9: 'tab:gray', 10: 'tab:olive', 11: 'tab:pink', 12: 'tab:purple', 13: 'tab:red', 14: 'tab:blue', 15: 'tab:green'}

with open('/home/giacomo/thesis_ws/src/lbp_dev/lbp_dev/LBP.json', 'r') as file:
    data = json.load(file)

with open('/home/giacomo/thesis_ws/src/seeds/circular_seed_11.json', 'r') as file:
    seed = json.load(file)

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

def rotateMatrix(a):
    """
    Rotate a 2D matrix by the given angle.

    Parameters:
    a (float): The angle of rotation in radians.

    Returns:
    numpy.ndarray: The rotated matrix.
    """
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

def lbp_control(x, goal, ob, u_buf, trajectory_buf):
    """
    Calculates the control input, trajectory, and control history for the LBP algorithm.

    Args:
        x (tuple): Current state of the system.
        goal (tuple): Goal state.
        ob (list): List of obstacles.
        u_buf (list): Buffer for storing control inputs.
        trajectory_buf (list): Buffer for storing trajectories.

    Returns:
        tuple: Control input, trajectory, and control history.
    """
    v_search = calc_dynamic_window(x)
    u, trajectory, u_history = calc_control_and_trajectory(x, v_search, goal, ob, u_buf, trajectory_buf)
    return u, trajectory, u_history

def predict_trajectory(x_init, a, delta):
    """
    Predicts the trajectory of an object given the initial state, acceleration, and steering angle.

    Parameters:
    x_init (list): The initial state of the object [x, y, theta].
    a (float): The acceleration of the object.
    delta (float): The steering angle of the object.

    Returns:
    numpy.ndarray: The predicted trajectory of the object.
    """

    x = np.array(x_init)
    trajectory = np.array(x)
    time = 0
    while time < predict_time:
        x = utils.motion(x, [a, delta], dt)
        trajectory = np.vstack((trajectory, x))
        time += dt
    return trajectory

def calc_dynamic_window(x):
    """
    Calculates the dynamic window for velocity search based on the current state.

    Args:
        x (list): Current state of the system [x, y, theta, v]

    Returns:
        list: List of possible velocities within the dynamic window
    """
    v_poss = np.arange(min_speed, max_speed+v_resolution, v_resolution)
    v_achiv = [x[3] + min_acc*dt, x[3] + max_acc*dt]

    v_search = []

    for v in v_poss:
        if v >= v_achiv[0] and v <= v_achiv[1]:
            v_search.append(v)
    
    return v_search

def calc_control_and_trajectory(x, v_search, goal, ob, u_buf, trajectory_buf):
    """
    Calculates the final input with LBP method.

    Args:
        x (list): The current state of the system.
        dw (float): The dynamic window.
        goal (list): The goal position.
        ob (list): The obstacle positions.
        u_buf (list): The buffer of control inputs.
        trajectory_buf (list): The buffer of trajectories.

    Returns:
        tuple: A tuple containing the best control input, the best trajectory, and the control input history.
    """

    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])

    # Calculate the cost of each possible trajectory and return the minimum
    for v in v_search:
        dict = data[str(v)]
        for id, info in dict.items():

            # old_time = time.time()
            geom = np.zeros((len(info['x']),3))
            geom[:,0] = info['x']
            geom[:,1] = info['y']
            geom[:,2] = info['yaw']
            geom[:,0:2] = (geom[:,0:2]) @ rotateMatrix(-x[2]) + [x[0],x[1]]
            
            geom[:,2] = geom[:,2] + x[2] #bringing also the yaw angle in the new frame
            
            # trajectory = predict_trajectory(x_init, a, delta)
            trajectory = geom
            # calc cost

            # TODO: small bug when increasing the factor too much for the to_goal_cost_gain
            to_goal_cost = to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
            
            if v <= 0.0:
                speed_cost = 30
            else:
                speed_cost = 0.0
                
            ob_cost = obstacle_cost_gain * calc_obstacle_cost(trajectory, ob)
            heading_cost = heading_cost_gain * calc_to_goal_heading_cost(trajectory, goal)
            final_cost = to_goal_cost + ob_cost + heading_cost + speed_cost 
            
            # search minimum trajectory
            if min_cost >= final_cost:
                min_cost = final_cost

                # interpolate the control inputs
                a = (v-x[3])/dt

                # print(f'v: {v}, id: {id}')
                # print(f"Control seq. {len(info['ctrl'])}")
                best_u = [a, info['ctrl'][1]]
                best_trajectory = trajectory
                u_history = info['ctrl'].copy()

    # Calculate cost of the previous best trajectory and compare it with that of the new trajectories
    # If the cost of the previous best trajectory is lower, use the previous best trajectory
    
    #TODO: this section has a small bug due to popping elements from the buffer, it gets to a point where there 
    # are no more elements in the buffer to use
    if len(u_buf) > 4:
        u_buf.pop(0)
    
        trajectory_buf = trajectory_buf[1:]

        to_goal_cost = to_goal_cost_gain * calc_to_goal_cost(trajectory_buf, goal)
        # speed_cost = speed_cost_gain * np.sign(trajectory[-1, 3]) * trajectory[-1, 3]
        ob_cost = obstacle_cost_gain * calc_obstacle_cost(trajectory_buf, ob)
        heading_cost = heading_cost_gain * calc_to_goal_heading_cost(trajectory_buf, goal)
        final_cost = to_goal_cost + ob_cost + heading_cost #+ speed_cost 

        if min_cost >= final_cost:
            min_cost = final_cost
            best_u = [0, u_buf[1]]
            best_trajectory = trajectory_buf
            u_history = u_buf

    elif min_cost == np.inf:
        # emergency stop
        print("Emergency stop")
        if x[3]>0:
            best_u = [min_acc, 0]
        else:
            best_u = [max_acc, 0]
        best_trajectory = np.array([x[0:3], x[0:3]])
        u_history = [min_acc, 0]


    return best_u, best_trajectory, u_history

def calc_obstacle_cost(trajectory, ob):
    """
    Calculate the obstacle cost for a given trajectory.

    Parameters:
    trajectory (numpy.ndarray): The trajectory to calculate the obstacle cost for.
    ob (list): List of obstacles.

    Returns:
    float: The obstacle cost for the trajectory.
    """
    min_distance = np.inf

    line = LineString(zip(trajectory[:, 0], trajectory[:, 1]))
    
    minxp = min(abs(width_init/2-trajectory[:, 0]))
    minxn = min(abs(-width_init/2-trajectory[:, 0]))
    minyp = min(abs(height_init/2-trajectory[:, 1]))
    minyn = min(abs(-height_init/2-trajectory[:, 1]))
    min_distance = min(minxp, minxn, minyp, minyn)
    dilated = line.buffer(dilation_factor, cap_style=3)

    x = trajectory[:, 0]
    y = trajectory[:, 1]

    # check if the trajectory is out of bounds
    if any(element < -width_init/2+WB or element > width_init/2-WB for element in x):
        return np.inf
    if any(element < -height_init/2+WB or element > height_init/2-WB for element in y):
        return np.inf

    if ob:
        for obstacle in ob:
            if dilated.intersects(obstacle):
                return np.inf # collision        
            elif distance(dilated, obstacle) < min_distance:
                min_distance = distance(dilated, obstacle)
                
        return 1/distance(dilated, obstacle)
    else:
        return 0.0

def calc_to_goal_cost(trajectory, goal):
    """
    Calculate the cost to reach the goal from the last point in the trajectory.

    Args:
        trajectory (numpy.ndarray): The trajectory as a 2D array of shape (n, 2), where n is the number of points.
        goal (tuple): The goal coordinates as a tuple (x, y).

    Returns:
        float: The cost to reach the goal from the last point in the trajectory.
    """
    # dx = goal[0] - trajectory[-1, 0]
    # dy = goal[1] - trajectory[-1, 1]

    # cost = math.hypot(dx, dy)

    # TODO: this causes bug when we use old traj because when popping elements we reduce the length and it may be less than 5
    # dx = goal[0] - trajectory[5, 0]
    # dy = goal[1] - trajectory[5, 1]

    # cost += math.hypot(dx, dy)

    dx = goal[0] - trajectory[:, 0]
    dy = goal[1] - trajectory[:, 1]

    cost = min(np.sqrt(dx**2+dy**2))

    return cost

def calc_to_goal_heading_cost(trajectory, goal):
    """
    Calculate the cost to reach the goal based on the heading angle difference.

    Args:
        trajectory (numpy.ndarray): The trajectory array containing the x, y, and heading values.
        goal (tuple): The goal coordinates (x, y).

    Returns:
        float: The cost to reach the goal based on the heading angle difference.
    """
    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]

    # either using the angle difference or the distance --> if we want to use the angle difference, we need to normalize the angle before taking the difference
    error_angle = normalize_angle(math.atan2(dy, dx))
    cost_angle = error_angle - normalize_angle(trajectory[-1, 2])
    cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

    return cost

def plot_arrow(x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)

def plot_robot(x, y, yaw, i):  
    """
    Plot the robot.

    Args:
        x (float): X-coordinate of the robot.
        y (float): Y-coordinate of the robot.
        yaw (float): Yaw angle of the robot.
        i (int): Index of the robot.
    """
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
                np.array(outline[1, :]).flatten(), color_dict[i])

def update_targets(paths, targets, x, i):
    """
    Update the targets based on the current position and distance threshold.

    Args:
        paths (list): List of paths.
        targets (list): List of target positions.
        x (numpy.ndarray): Current position.
        i (int): Index of the target to update.

    Returns:
        tuple: Updated paths and targets.
    """
    if utils.dist(point1=(x[0, i], x[1, i]), point2=targets[i]) < update_dist:
        paths[i] = utils.update_path(paths[i])
        targets[i] = (paths[i][0].x, paths[i][0].y)

    return paths, targets

def initialize_paths_targets_dilated_traj(x):
    """
    Initializes paths, targets, and dilated_traj based on the initial positions.

    Args:
        x (numpy.ndarray): Input array containing x and y coordinates.

    Returns:
        tuple: A tuple containing paths, targets, and dilated_traj.
            - paths (list): A list of paths.
            - targets (list): A list of target coordinates.
            - dilated_traj (list): A list of dilated trajectories.
    """
    paths = []
    targets = []
    dilated_traj = []

    for i in range(robot_num):
        dilated_traj.append(Point(x[0, i], x[1, i]).buffer(dilation_factor, cap_style=3))
        paths.append(utils.create_path())
        targets.append([paths[i][0].x, paths[i][0].y])

    return paths, targets, dilated_traj

def update_robot_state(x, u, dt, targets, dilated_traj, u_hist, predicted_trajectory, i):
    """
    Update the state of a robot in a multi-robot system.

    Args:
        x (numpy.ndarray): Current state of all robots.
        u (numpy.ndarray): Current control inputs of all robots.
        dt (float): Time step.
        targets (list): List of target positions for each robot.
        dilated_traj (list): List of dilated trajectories for each robot.
        u_hist (list): List of control input histories for each robot.
        predicted_trajectory (list): List of predicted trajectories for each robot.
        i (int): Index of the robot to update.

    Returns:
        tuple: Updated state, control inputs, predicted trajectories, and control input histories of all robots.
    """
    x1 = x[:, i]
    ob = [dilated_traj[idx] for idx in range(len(dilated_traj)) if idx != i]
    if add_noise:
        noise = np.concatenate([np.random.normal(0, 0.21*noise_scale_param, 2).reshape(1, 2), np.random.normal(0, np.radians(5)*noise_scale_param, 1).reshape(1,1), np.random.normal(0, 0.2*noise_scale_param, 1).reshape(1,1)], axis=1)
        noisy_pos = x1 + noise[0]
        u1, predicted_trajectory1, u_history = lbp_control(noisy_pos, targets[i], ob, u_hist[i], predicted_trajectory[i])
        plt.plot(noisy_pos[0], noisy_pos[1], "x", color=color_dict[i], markersize=10)
    else:
        u1, predicted_trajectory1, u_history = lbp_control(x1, targets[i], ob, u_hist[i], predicted_trajectory[i])
    dilated_traj[i] = LineString(zip(predicted_trajectory1[:, 0], predicted_trajectory1[:, 1])).buffer(dilation_factor, cap_style=3)
   
    # Collision check
    if check_collision_bool:
        if any([utils.dist([x1[0], x1[1]], [x[0, idx], x[1, idx]]) < WB for idx in range(robot_num) if idx != i]): raise Exception('Collision')

    x1 = utils.motion(x1, u1, dt)
    x[:, i] = x1
    u[:, i] = u1

    # u, x = self.check_collision(x, u, i)

    predicted_trajectory[i] = predicted_trajectory1
    u_hist[i] = u_history

    return x, u, predicted_trajectory, u_hist

def plot_robot_trajectory(x, u, predicted_trajectory, dilated_traj, targets, ax, i):
    """
    Plots the robot and arrows for visualization.

    Args:
        i (int): Index of the robot.
        x (numpy.ndarray): State vector of shape (4, N), where N is the number of time steps.
        multi_control (numpy.ndarray): Control inputs of shape (2, N).
        targets (list): List of target points.

    """
    plt.plot(predicted_trajectory[i][:, 0], predicted_trajectory[i][:, 1], "-", color=color_dict[i])
    plot_polygon(dilated_traj[i], ax=ax, add_points=False, alpha=0.5, color=color_dict[i])
    # plt.plot(x[0, i], x[1, i], "xr")
    plt.plot(targets[i][0], targets[i][1], "x", color=color_dict[i])
    plot_robot(x[0, i], x[1, i], x[2, i], i)
    plot_arrow(x[0, i], x[1, i], x[2, i], length=1, width=0.5)
    plot_arrow(x[0, i], x[1, i], x[2, i] + u[1, i], length=3, width=0.5)

def check_goal_reached(x, targets, i, distance=0.5):
    """
    Check if the given point has reached the goal.

    Args:
        x (numpy.ndarray): Array of x-coordinates.
        targets (list): List of target points.
        i (int): Index of the current point.

    Returns:
        bool: True if the point has reached the goal, False otherwise.
    """
    dist_to_goal = math.hypot(x[0, i] - targets[i][0], x[1, i] - targets[i][1])
    if dist_to_goal <= distance:
        print("Goal!!")
        return True
    return False

class LBP_algorithm():
    def __init__(self, trajectories, paths, targets, dilated_traj, predicted_trajectory, ax, u_hist, robot_num=robot_num):
        self.trajectories = trajectories
        self.paths = paths
        self.targets = targets
        self.dilated_traj = dilated_traj
        self.predicted_trajectory = predicted_trajectory
        self.ax = ax
        self.u_hist = u_hist
        self.robot_num = robot_num
        self.reached_goal = [False]*robot_num
        self.computational_time = []

    def run_lbp(self, x, u, break_flag):
        for i in range(self.robot_num):
            
            # self.paths, self.targets = update_targets(self.paths, self.targets, x, i)
            if utils.dist(point1=(x[0,i], x[1,i]), point2=self.targets[i]) < update_dist:
                # Perform some action when the condition is met
                self.paths[i].pop(0)
                if not self.paths[i]:
                    print("Path complete")
                    return x, u, True
                
                self.targets[i] = (self.paths[i][0].x, self.paths[i][0].y)

            t_prev = time.time()
            x, u, self.predicted_trajectory, self.u_hist = update_robot_state(x, u, dt, self.targets, self.dilated_traj, self.u_hist, self.predicted_trajectory, i)
            self.computational_time.append(time.time()-t_prev)

            if check_goal_reached(x, self.targets, i):
                break_flag = True

            if show_animation:
                plot_robot_trajectory(x, u, self.predicted_trajectory, self.dilated_traj, self.targets, self.ax, i)
        return x, u, break_flag
    
    def go_to_goal(self, x, u, break_flag):
        for i in range(self.robot_num):
            # Step 9: Check if the distance between the current position and the target is less than 5
            if not self.reached_goal[i]:        
                # If goal is reached, stop the robot
                if check_goal_reached(x, self.targets, i, distance=to_goal_stop_distance):
                    u[:, i] = np.zeros(2)
                    x[3, i] = 0
                    self.reached_goal[i] = True
                else:
                    t_prev = time.time()
                    x, u, self.predicted_trajectory, self.u_hist = update_robot_state(x, u, dt, self.targets, self.dilated_traj, self.u_hist, self.predicted_trajectory, i)
                    self.computational_time.append(time.time()-t_prev)

                u, x = self.check_collision(x, u, i) 
            
            # else:
            #     print(u[:, i])
            # If we want the robot to disappear when it reaches the goal, indent one more time
            if all(self.reached_goal):
                break_flag = True

            if show_animation:
                plot_robot_trajectory(x, u, self.predicted_trajectory, self.dilated_traj, self.targets, self.ax, i)
        return x, u, break_flag
    
    def check_collision(self, x, u, i):
        """
        Checks for collision between the robot at index i and other robots.

        Args:
            x (numpy.ndarray): State vector of shape (4, N), where N is the number of time steps.
            i (int): Index of the robot to check collision for.

        Raises:
            Exception: If collision is detected.

        """
        if x[0,i]>=boundary_points[1]-WB/2 or x[0,i]<=boundary_points[0]+WB/2 or x[1,i]>=boundary_points[3]-WB or x[1,i]<=boundary_points[2]+WB/2:
            if check_collision_bool:
                raise Exception('Collision')
            else:
                print("Collision detected")
                self.reached_goal[i] = True
                u[:, i] = np.zeros(2)
                x[3, i] = 0

        for idx in range(self.robot_num):
            if idx == i:
                continue
            if utils.dist([x[0,i], x[1,i]], [x[0, idx], x[1, idx]]) <= WB/2:
                if check_collision_bool:
                    raise Exception('Collision')
                else:
                    print("Collision detected")
                    self.reached_goal[i] = True
                    u[:, i] = np.zeros(2)
                    x[3, i] = 0
        
        return u, x

def main():
    """
    This is the main function that controls the execution of the program.

    The simulation if this function has a fixed amout of robots N. This main() is mainly used for debugging purposes.

    It initializes the necessary variables, sets the targets, and updates the robot states.
    The function also plots the robot trajectory and checks if the goal is reached.

    Before using this main comment out the robot_num variable at the begginning of the file and
    put the robot_num value equal to the value of the N variable.
    """
    print(__file__ + " start!!")
    # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    iterations = 3000
    break_flag = False

    x = np.array([[-7, 7, 0.0], [0, 0, 7], [0, np.pi, -np.pi/2], [0, 0, 0]])
    u = np.array([[0, 0, 0], [0, 0, 0]])
    targets = [[7,7],[-7,7],[0.0,0.0]]

    # create a trajcetory array to store the trajectory of the N robots
    trajectory = np.zeros((x.shape[0], N, 1))
    # append the firt state to the trajectory
    trajectory[:, :, 0] = x

    predicted_trajectory = dict.fromkeys(range(N),np.zeros([int(predict_time/dt), 3]))
    for i in range(N):
        predicted_trajectory[i][:, 0:3] = x[0:3, i]
    u_hist = dict.fromkeys(range(N),[0]*int(predict_time/dt))

    dilated_traj = []
    for i in range(N):
        dilated_traj.append(Point(x[0, i], x[1, i]).buffer(dilation_factor, cap_style=3))

    fig = plt.figure(1, dpi=90, figsize=(10,10))
    ax = fig.add_subplot(111)
    for z in range(iterations):
        plt.cla()
        plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
        
        for i in range(N):

            x, u, predicted_trajectory, u_hist = update_robot_state(x, u, dt, targets, dilated_traj, u_hist, predicted_trajectory, i)

            trajectory = np.dstack([trajectory, x])

            if check_goal_reached(x, targets, i):
                break_flag = True

            if show_animation:
                plot_robot_trajectory(x, u, predicted_trajectory, dilated_traj, targets, ax, i)

        utils.plot_map(width=width_init, height=height_init)
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)

        if break_flag:
            break

    print("Done")
    if show_animation:
        for i in range(N):
            plt.plot(trajectory[0, i, :], trajectory[1, i, :], "-r")
        plt.pause(0.0001)
        plt.show()

def main1():
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

    x, y, yaw, v, omega, model_type = utils.samplegrid(width_init, height_init, min_dist, robot_num, safety_init)
    x = np.array([x, y, yaw, v])
    u = np.zeros((2, robot_num))

    trajectory = np.zeros((x.shape[0], robot_num, 1))
    trajectory[:, :, 0] = x

    predicted_trajectory = dict.fromkeys(range(robot_num),np.zeros([int(predict_time/dt), 3]))
    for i in range(robot_num):
        predicted_trajectory[i] = np.full((int(predict_time/dt), 3), x[0:3,i])
    
    u_hist = dict.fromkeys(range(robot_num),[0]*int(predict_time/dt))

    paths, targets, dilated_traj = initialize_paths_targets_dilated_traj(x)

    fig = plt.figure(1, dpi=90, figsize=(10,10))
    ax = fig.add_subplot(111)

    for z in range(iterations):
        plt.cla()
        plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
        
        for i in range(robot_num):
            
            paths, targets = update_targets(paths, targets, x, i)

            x, u, predicted_trajectory, u_hist = update_robot_state(x, u, dt, targets, dilated_traj, u_hist, predicted_trajectory, i)

            trajectory = np.dstack([trajectory, x])

            if check_goal_reached(x, targets, i):
                break_flag = True

            if show_animation:
                plot_robot_trajectory(x, u, predicted_trajectory, dilated_traj, targets, ax, i)

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

def main2():
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
    for i in range(robot_num):
        dilated_traj.append(Point(x[0, i], x[1, i]).buffer(dilation_factor, cap_style=3))

    u_hist = dict.fromkeys(range(robot_num),[0]*int(predict_time/dt))
    fig = plt.figure(1, dpi=90, figsize=(10,10))
    ax = fig.add_subplot(111)

    for z in range(iterations):
        plt.cla()
        plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
        
        for i in range(robot_num):
            
            # Step 9: Check if the distance between the current position and the target is less than 5
            if utils.dist(point1=(x[0,i], x[1,i]), point2=targets[i]) < 5:
                # Perform some action when the condition is met
                paths[i].pop(0)
                if not paths[i]:
                    print("Path complete")
                    return
                targets[i] = (paths[i][0].x, paths[i][0].y)

            x, u, predicted_trajectory, u_hist = update_robot_state(x, u, dt, targets, dilated_traj, u_hist, predicted_trajectory, i)

            trajectory = np.dstack([trajectory, x])

            if check_goal_reached(x, targets, i):
                break_flag = True

            if show_animation:
                plot_robot_trajectory(x, u, predicted_trajectory, dilated_traj, targets, ax, i)

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

def main_seed():
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

    # Step 2: Sample initial values for x0, y, yaw, v, omega, and model_type
    initial_state = seed['initial_position']
    x0 = initial_state['x']
    y = initial_state['y']
    yaw = initial_state['yaw']
    v = initial_state['v']

    assert robot_num == len(seed['initial_position']['x']), "The number of robots in the seed file does not match the number of robots in the seed file"
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
    for i in range(robot_num):
        dilated_traj.append(Point(x[0, i], x[1, i]).buffer(dilation_factor, cap_style=3))

    u_hist = dict.fromkeys(range(robot_num),[0]*int(predict_time/dt))
    fig = plt.figure(1, dpi=90, figsize=(10,10))
    ax = fig.add_subplot(111)
    
    lbp = LBP_algorithm(predicted_trajectory, paths, targets, dilated_traj,
                        predicted_trajectory, ax, u_hist)
    
    for z in range(iterations):
        plt.cla()
        plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
        
        # x, u, break_flag = lbp.run_lbp(x, u, break_flag)
        x, u, break_flag = lbp.go_to_goal(x, u, break_flag)

        trajectory = np.dstack([trajectory, x])

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

if __name__ == '__main__':
    # main1()
    main_seed()

    