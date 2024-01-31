import matplotlib.pyplot as plt
import numpy as np
import math
import planner.utils as utils
# For the parameter file
import pathlib
import json
from custom_message.msg import Coordinate
from shapely.geometry import Point, LineString
from shapely import distance
from shapely.plotting import plot_polygon
import time

path = pathlib.Path('/home/giacomo/thesis_ws/src/bumper_cars/params.json')
# Opening JSON file
with open(path, 'r') as openfile:
    # Reading from json file
    json_object = json.load(openfile)

max_steer = json_object["DWA"]["max_steer"] # [rad] max steering angle
max_speed = json_object["DWA"]["max_speed"] # [m/s]
min_speed = json_object["DWA"]["min_speed"] # [m/s]
v_resolution = json_object["DWA"]["v_resolution"] # [m/s]
delta_resolution = math.radians(json_object["DWA"]["delta_resolution"])# [rad/s]
max_acc = json_object["DWA"]["max_acc"] # [m/ss]
min_acc = json_object["DWA"]["min_acc"] # [m/ss]
dt = json_object["DWA"]["dt"] # [s] Time tick for motion prediction
predict_time = json_object["DWA"]["predict_time"] # [s]
to_goal_cost_gain = json_object["DWA"]["to_goal_cost_gain"]
speed_cost_gain = json_object["DWA"]["speed_cost_gain"]
obstacle_cost_gain = json_object["DWA"]["obstacle_cost_gain"]
heading_cost_gain = json_object["DWA"]["heading_cost_gain"]
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
L_d = json_object["Controller"]["L_d"]  # [m] look-ahead distance
robot_num = json_object["robot_num"]
safety_init = json_object["safety"]
width_init = json_object["width"]
height_init = json_object["height"]
min_dist = json_object["min_dist"]
update_dist = 2
# N=3

robot_num = json_object["robot_num"]
timer_freq = json_object["timer_freq"]

show_animation = True
check_collision_bool = False

color_dict = {0: 'r', 1: 'b', 2: 'g', 3: 'y', 4: 'm', 5: 'c', 6: 'k'}

with open('/home/giacomo/thesis_ws/src/trajectories.json', 'r') as file:
    data = json.load(file)

with open('/home/giacomo/thesis_ws/src/seed_1.json', 'r') as file:
    seed = json.load(file)

def motion(x, u, dt):
    """
    Motion model.

    Args:
        x (list): Current state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)].
        u (list): Control inputs (throttle, delta).
        dt (float): Time tick for motion prediction.

    Returns:
        list: Updated state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)].
    """
    delta = u[1]
    delta = np.clip(delta, -max_steer, max_steer)
    throttle = u[0]

    x[0] = x[0] + x[3] * math.cos(x[2]) * dt
    x[1] = x[1] + x[3] * math.sin(x[2]) * dt
    x[2] = x[2] + x[3] / L * math.tan(delta) * dt
    x[2] = normalize_angle(x[2])
    x[3] = x[3] + throttle * dt
    x[3] = np.clip(x[3], min_speed, max_speed)

    return x

def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].

    Args:
        angle (float): Angle in radians.

    Returns:
        float: Angle in radians in the range [-pi, pi].
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi
    while angle < -np.pi:
        angle += 2.0 * np.pi
    return angle

def rotateMatrix(a):
    """
    Rotate a matrix by an angle.

    Args:
        a (float): Angle in radians.

    Returns:
        numpy.ndarray: Rotated matrix.
    """
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

def find_nearest(array, value):
    """
    Find the nearest value in an array.

    Args:
        array (numpy.ndarray): Input array.
        value: Value to find.

    Returns:
        float: Nearest value in the array.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def calc_dynamic_window(x):
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

def predict_trajectory(x_init, a, delta):
    """
    Predict the trajectory with an input.

    Args:
        x_init (list): Initial state [x(m), y(m), yaw(rad), v(m/s), delta(rad)].
        a (float): Throttle input.
        delta (float): Steering input.

    Returns:
        numpy.ndarray: Predicted trajectory.
    """
    x = np.array(x_init)
    trajectory = np.array(x)
    time = 0
    while time < predict_time:
        x = motion(x, [a, delta], dt)
        trajectory = np.vstack((trajectory, x))
        time += dt
    return trajectory

def calc_obstacle_cost(trajectory, ob):
    """
    Calculate the obstacle cost.

    Args:
        trajectory (numpy.ndarray): Trajectory.
        ob (list): List of obstacles.

    Returns:
        float: Obstacle cost.
    """
    line = LineString(zip(trajectory[:, 0], trajectory[:, 1]))
    dilated = line.buffer(dilation_factor, cap_style=3)

    min_distance = np.inf

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
                return 100000 # collision        
            elif distance(dilated, obstacle) < min_distance:
                min_distance = distance(dilated, obstacle)
                
        return 1/distance(dilated, obstacle)
    else:
        return 0.0

def calc_to_goal_cost(trajectory, goal):
    """
    Calculate the cost to the goal.

    Args:
        trajectory (numpy.ndarray): Trajectory.
        goal (list): Goal position [x(m), y(m)].

    Returns:
        float: Cost to the goal.
    """
    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]

    # either using the angle difference or the distance --> if we want to use the angle difference, we need to normalize the angle before taking the difference
    # error_angle = math.atan2(dy, dx)
    # cost_angle = error_angle - trajectory[-1, 2]
    # cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

    cost = math.hypot(dx, dy)
    return cost

def calc_to_goal_heading_cost(trajectory, goal):
    """
    Calculate the cost to the goal with angle difference.

    Args:
        trajectory (numpy.ndarray): Trajectory.
        goal (list): Goal position [x(m), y(m)].

    Returns:
        float: Cost to the goal with angle difference.
    """
    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]

    # either using the angle difference or the distance --> if we want to use the angle difference, we need to normalize the angle before taking the difference
    error_angle = normalize_angle(math.atan2(dy, dx))
    cost_angle = error_angle - normalize_angle(trajectory[-1, 2])
    cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

    return cost

def plot_arrow(x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
    """
    Plot an arrow.

    Args:
        x (float): X-coordinate of the arrow.
        y (float): Y-coordinate of the arrow.
        yaw (float): Yaw angle of the arrow.
        length (float, optional): Length of the arrow. Defaults to 0.5.
        width (float, optional): Width of the arrow. Defaults to 0.1.
    """
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

def plot_map():
    """
    Plot the map.
    """
    corner_x = [-width_init/2.0, width_init/2.0, width_init/2.0, -width_init/2.0, -width_init/2.0]
    corner_y = [height_init/2.0, height_init/2.0, -height_init/2.0, -height_init/2.0, height_init/2.0]

    plt.plot(corner_x, corner_y)

def update_targets(paths, targets, x, i):
    """
    Update the targets for the paths.

    Args:
        paths (list): List of paths.
        targets (list): List of targets.
        x (numpy.ndarray): Current state.
        i (int): Index of the robot.

    Returns:
        tuple: Tuple containing the updated paths and targets.
    """
    if utils.dist(point1=(x[0, i], x[1, i]), point2=targets[i]) < 5:
        paths[i] = utils.update_path(paths[i])
        targets[i] = (paths[i][0].x, paths[i][0].y)

    return paths, targets

def initialize_paths_targets_dilated_traj(x):
    """
    Initialize the paths, targets, and dilated trajectory.

    Args:
        x (numpy.ndarray): Current states.

    Returns:
        tuple: Tuple containing the paths, targets, and dilated trajectory.
    """
    paths = []
    targets = []
    dilated_traj = []

    for i in range(robot_num):
        dilated_traj.append(Point(x[0, i], x[1, i]).buffer(dilation_factor, cap_style=3))
        paths.append(utils.create_path())
        targets.append([paths[i][0].x, paths[i][0].y])

    return paths, targets, dilated_traj

def plot_robot_trajectory(x, u, predicted_trajectory, dilated_traj, targets, ax, i):
    """
    Plots the robot and arrows for visualization.

    Args:
        i (int): Index of the robot.
        x (numpy.ndarray): State vector of shape (4, N), where N is the number of time steps.
        multi_control (numpy.ndarray): Control inputs of shape (2, N).
        targets (list): List of target points.

    """
    plt.plot(predicted_trajectory[i][:, 0], predicted_trajectory[i][:, 1], "-"+color_dict[i])
    plot_polygon(dilated_traj[i], ax=ax, add_points=False, alpha=0.5, color=color_dict[i])
    # plt.plot(x[0, i], x[1, i], "xr")
    plt.plot(targets[i][0], targets[i][1], "x"+color_dict[i], markersize=15)
    plot_robot(x[0, i], x[1, i], x[2, i], i)
    plot_arrow(x[0, i], x[1, i], x[2, i], length=1, width=0.5)
    plot_arrow(x[0, i], x[1, i], x[2, i] + u[1, i], length=3, width=0.5)

def check_goal_reached(x, targets, i, distance=0.5):
    """
    Check if the robot has reached the goal.

    Parameters:
    x (numpy.ndarray): Robot's current position.
    targets (list): List of target positions.
    i (int): Index of the current target.

    Returns:
    bool: True if the robot has reached the goal, False otherwise.
    """
    dist_to_goal = math.hypot(x[0, i] - targets[i][0], x[1, i] - targets[i][1])
    if dist_to_goal <= distance:
        print("Goal!!")
        return True
    return False

class DWA_algorithm():
    def __init__(self, trajectories, safety, width, height, min_dist,
                paths, targets, dilated_traj, predicted_trajectory, ax):
        self.trajectories = trajectories
        self.safety = safety
        self.width = width
        self.height = height
        self.min_dist = min_dist
        self.paths = paths
        self.targets = targets
        self.dilated_traj = dilated_traj
        self.predicted_trajectory = predicted_trajectory
        self.ax = ax
        self.reached_goal = [False]*robot_num
        self.computational_time = []
        self.old_time = [time.time()]*robot_num

    def run_dwa(self, x, u, break_flag):
        for i in range(robot_num):
            # Step 9: Check if the distance between the current position and the target is less than 5
            if utils.dist(point1=(x[0,i], x[1,i]), point2=self.targets[i]) < update_dist:
                # Perform some action when the condition is met
                self.paths[i].pop(0)
                if not self.paths[i]:
                    print("Path complete")
                    return x, u, True
                self.targets[i] = (self.paths[i][0].x, self.paths[i][0].y)

            t_prev = time.time()
            x, u, self.predicted_trajectory, self.old_time[i] = self.update_robot_state(x, u, dt, self.targets, self.dilated_traj, self.predicted_trajectory, i, self.old_time[i])
            self.computational_time.append(time.time()-t_prev)

            if check_goal_reached(x, self.targets, i):
                break_flag = True

            if show_animation:
                plot_robot_trajectory(x, u, self.predicted_trajectory, self.dilated_traj, self.targets, self.ax, i)
        return x, u, break_flag
    
    def go_to_goal(self, x, u, break_flag):
        for i in range(robot_num):
            # Step 9: Check if the distance between the current position and the target is less than 5
            if not self.reached_goal[i]:                
                # If goal is reached, stop the robot
                if check_goal_reached(x, self.targets, i, distance=1):
                    u[:, i] = np.zeros(2)
                    x[3, i] = 0
                    self.reached_goal[i] = True
                else:
                    # If goal is not reached, update the robot's state
                    time_prev = time.time()
                    x, u, self.predicted_trajectory = self.update_robot_state(x, u, dt, self.targets, self.dilated_traj, self.predicted_trajectory, i)
                    self.computational_time.append(time.time()-time_prev)
                    
            # print(f"Speed of robot {i}: {x[3, i]}")
            
            # If we want the robot to disappear when it reaches the goal, indent one more time
            if show_animation:
                plot_robot_trajectory(x, u, self.predicted_trajectory, self.dilated_traj, self.targets, self.ax, i)
        if all(self.reached_goal):
                break_flag = True
        return x, u, break_flag
    
    def update_robot_state(self, x, u, dt, targets, dilated_traj, predicted_trajectory, i, old_time):
        """
        Update the state of a robot based on its current state, control input, and environment information.

        Args:
            x (numpy.ndarray): Current state of the robot.
            u (numpy.ndarray): Control input for the robot.
            dt (float): Time step.
            targets (list): List of target positions for each robot.
            dilated_traj (list): List of dilated trajectories for each robot.
            predicted_trajectory (list): List of predicted trajectories for each robot.
            i (int): Index of the robot.

        Returns:
            tuple: Updated state, control input, and predicted trajectory.

        Raises:
            Exception: If a collision is detected.

        """
        x1 = x[:, i]
        ob = [dilated_traj[idx] for idx in range(len(dilated_traj)) if idx != i]
        u1, predicted_trajectory1 = self.dwa_control(x1, targets[i], ob)
        dilated_traj[i] = LineString(zip(predicted_trajectory1[:, 0], predicted_trajectory1[:, 1])).buffer(dilation_factor, cap_style=3)

        # Collision check
        if check_collision_bool:
            if any([utils.dist([x1[0], x1[1]], [x[0, idx], x[1, idx]]) < WB for idx in range(robot_num) if idx != i]): raise Exception('Collision')

        print(time.time()-old_time)
        x1 = motion(x1, u1, dt=time.time()-old_time)
        x[:, i] = x1
        u[:, i] = u1
        predicted_trajectory[i] = predicted_trajectory1

        return x, u, predicted_trajectory, time.time()
    
    def dwa_control(self, x, goal, ob):
        """
        Dynamic Window Approach control.

        Args:
            x (list): Current state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)].
            goal (list): Goal position [x(m), y(m)].
            ob (list): List of obstacles.

        Returns:
            tuple: Tuple containing the control inputs (throttle, delta) and the trajectory.
        """
        dw = calc_dynamic_window(x)
        u, trajectory = self.calc_control_and_trajectory(x, dw, goal, ob)
        return u, trajectory
    
    def calc_control_and_trajectory(self, x, dw, goal, ob):
        """
        Calculate the final input with the dynamic window.

        Args:
            x (list): Current state [x(m), y(m), yaw(rad), v(m/s), delta(rad)].
            dw (list): Dynamic window [min_throttle, max_throttle, min_steer, max_steer].
            goal (list): Goal position [x(m), y(m)].
            ob (list): List of obstacles.

        Returns:
            tuple: Tuple containing the control inputs (throttle, delta) and the trajectory.
        """
        min_cost = float("inf")
        best_u = [0.0, 0.0]
        best_trajectory = np.array([x])

        # evaluate all trajectory with sampled input in dynamic window
        nearest = find_nearest(np.arange(min_speed, max_speed, v_resolution), x[3])

        for a in np.arange(dw[0], dw[1]+v_resolution, v_resolution):
            for delta in np.arange(dw[2], dw[3]+delta_resolution, delta_resolution):

                # old_time = time.time()
                geom = data[str(nearest)][str(a)][str(delta)]
                geom = np.array(geom)
                geom[:,0:2] = (geom[:,0:2]) @ rotateMatrix(np.radians(90)-x[2]) + [x[0],x[1]]
                # print(time.time()-old_time)
                geom[:,2] = geom[:,2] + x[2] - np.pi/2 #bringing also the yaw angle in the new frame

                # trajectory = predict_trajectory(x_init, a, delta)
                trajectory = geom
                # calc cost

                to_goal_cost = to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
                # speed_cost = speed_cost_gain * (max_speed - trajectory[-1, 3])
                ob_cost = obstacle_cost_gain * calc_obstacle_cost(trajectory, ob)
                # heading_cost = heading_cost_gain * calc_to_goal_heading_cost(trajectory, goal)
                final_cost = to_goal_cost + ob_cost # + heading_cost #+ speed_cost 
                
                # search minimum trajectory
                if min_cost >= final_cost:
                    min_cost = final_cost
                    best_u = [a, delta]
                    best_trajectory = trajectory
                    if abs(best_u[0]) < robot_stuck_flag_cons \
                            and abs(x[2]) < robot_stuck_flag_cons:
                        # to ensure the robot do not get stuck in
                        # best v=0 m/s (in front of an obstacle) and
                        # best omega=0 rad/s (heading to the goal with
                        # angle difference of 0)
                        best_u[1] = -max_steer
        # print(time.time()-old_time)
        return best_u, best_trajectory

def main():
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
    
    x0, y, yaw, v, omega, model_type = utils.samplegrid(width_init, height_init, min_dist, robot_num, safety_init)
    x = np.array([x0, y, yaw, v])
    u = np.zeros((2, robot_num))

    trajectory = np.zeros((x.shape[0], robot_num, 1))
    trajectory[:, :, 0] = x

    predicted_trajectory = dict.fromkeys(range(robot_num),np.zeros([int(predict_time/dt), 3]))
    for i in range(robot_num):
        predicted_trajectory[i] = np.full((int(predict_time/dt), 3), x[0:3,i])

    paths, targets, dilated_traj = initialize_paths_targets_dilated_traj(x)
    
    fig = plt.figure(1, dpi=90)
    ax = fig.add_subplot(111)

    dwa = DWA_algorithm(paths, safety_init, width_init, height_init,
                        min_dist, paths, targets, dilated_traj, predicted_trajectory, ax)
    
    for z in range(iterations):
        plt.cla()
        plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
        
        for i in range(robot_num):
            
            paths, targets = update_targets(paths, targets, x, i)

            x, u, predicted_trajectory = dwa.update_robot_state(x, u, dt, targets, dilated_traj, predicted_trajectory, i)

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

def main1():
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
    
    fig = plt.figure(1, dpi=90)
    ax = fig.add_subplot(111)

    dwa = DWA_algorithm(paths, safety_init, width_init, height_init,
                        min_dist, paths, targets, dilated_traj, predicted_trajectory, ax)
    
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

            x, u, predicted_trajectory = dwa.update_robot_state(x, u, dt, targets, dilated_traj, predicted_trajectory, i)

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
    
    fig = plt.figure(1, dpi=90)
    ax = fig.add_subplot(111)
    
    # Step 7: Create an instance of the DWA_algorithm class
    dwa = DWA_algorithm(paths, safety_init, width_init, height_init,
                        min_dist, paths, targets, dilated_traj, predicted_trajectory, ax)
    
    for z in range(iterations):
        plt.cla()
        plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
        
        x, u, break_flag = dwa.run_dwa(x, u, break_flag)
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
    main_seed()
    # main()