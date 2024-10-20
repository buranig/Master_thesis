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
import time
import pickle
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
path = pathlib.Path(dir_path + '/../../bumper_cars/params.json')

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
dt = json_object["Controller"]["dt"] # [s] Time tick for motion prediction
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
robot_num = 15 #json_object["robot_num"]
safety_init = json_object["safety"]
width_init = json_object["width"]
height_init = json_object["height"]
min_dist = json_object["min_dist"]
to_goal_stop_distance = json_object["to_goal_stop_distance"]
emergency_brake_distance = json_object["DWA"]["emergency_brake_distance"]
update_dist = 2
# N=3

show_animation = json_object["show_animation"]
boundary_points = np.array([-width_init/2, width_init/2, -height_init/2, height_init/2])
check_collision_bool = False
add_noise = json_object["add_noise"]
noise_scale_param = json_object["noise_scale_param"]
linear_model = json_object["linear_model"]
np.random.seed(1)

color_dict = {0: 'r', 1: 'b', 2: 'g', 3: 'y', 4: 'm', 5: 'c', 6: 'k', 7: 'tab:orange', 8: 'tab:brown', 9: 'tab:gray', 10: 'tab:olive', 11: 'tab:pink', 12: 'tab:purple', 13: 'tab:red', 14: 'tab:blue', 15: 'tab:green'}
if linear_model:
    with open(dir_path + '/../trajectories.json', 'r') as file:
        data = json.load(file)
else:
    with open(dir_path + '/../dynamic_trajectories.json', 'r') as file:
        data = json.load(file)

with open(dir_path + '/../../seeds/seed_7.json', 'r') as file:
    seed = json.load(file)

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
        x = utils.motion(x, [a, delta], dt)
        trajectory = np.vstack((trajectory, x))
        time += dt
    return trajectory

def calc_reference_input_cost(input, goal_input, i):
    """
    Calculate the reference input cost.

    Args:
        input (list): Input [throttle, delta].

    Returns:
        float: Reference input cost.
    """
    da = input[0]-goal_input[0,i]
    ddelta = input[1]-goal_input[1,i]

    cost = np.hypot(da, ddelta)
    return cost

def calc_obstacle_cost(trajectory, ob):
    """
    Calculate the obstacle cost.

    Args:
        trajectory (numpy.ndarray): Trajectory.
        ob (list): List of obstacles.

    Returns:
        float: Obstacle cost.
    """
    minxp = min(abs(width_init/2-trajectory[:, 0]))
    minxn = min(abs(-width_init/2-trajectory[:, 0]))
    minyp = min(abs(height_init/2-trajectory[:, 1]))
    minyn = min(abs(-height_init/2-trajectory[:, 1]))
    min_distance = min(minxp, minxn, minyp, minyn)

    cost = 1/min_distance

    line = LineString(zip(trajectory[:, 0], trajectory[:, 1]))
    dilated = line.buffer(dilation_factor, cap_style=1)

    x = trajectory[:, 0]
    y = trajectory[:, 1]

    # check if the trajectory is out of bounds
    if any(element < -width_init/2+WB/2 or element > width_init/2-WB/2 for element in x):
        return np.inf
    if any(element < -height_init/2+WB/2 or element > height_init/2-WB/2 for element in y):
        return np.inf

    if ob:
        for obstacle in ob:
            if dilated.intersects(obstacle):
                return np.inf # collision        
            elif distance(dilated, obstacle) < min_distance:
                min_distance = distance(dilated, obstacle)

        cost += 1/min_distance    
        return cost
    else:
        return cost

def calc_to_goal_cost(trajectory, goal):
    """
    Calculate the cost to the goal.

    Args:
        trajectory (numpy.ndarray): Trajectory.
        goal (list): Goal position [x(m), y(m)].

    Returns:
        float: Cost to the goal.
    """
    # dx = goal[0] - trajectory[-1, 0]
    # dy = goal[1] - trajectory[-1, 1]

    # cost = math.hypot(dx, dy)

    dx = goal[0] - trajectory[:, 0]
    dy = goal[1] - trajectory[:, 1]

    cost = min(np.sqrt(dx**2+dy**2))
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
    error_angle = utils.normalize_angle(math.atan2(dy, dx))
    cost_angle = error_angle - utils.normalize_angle(trajectory[-1, 2])
    cost_angle = abs(cost_angle)

    return cost_angle

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

class DWA_algorithm():
    """
    Class representing the Dynamic Window Approach algorithm.

    Args:
        robot_num (int): Number of robots.
        trajectories (list): List of trajectories for each robot.
        paths (list): List of paths for each robot.
        targets (list): List of target positions for each robot.
        dilated_traj (list): List of dilated trajectories for each robot.
        predicted_trajectory (list): List of predicted trajectories for each robot.
        ax (matplotlib.axes.Axes): Axes object for plotting.
        u_hist (list): List of control inputs history for each robot.

    Attributes:
        trajectories (list): List of trajectories for each robot.
        robot_num (int): Number of robots.
        paths (list): List of paths for each robot.
        targets (list): List of target positions for each robot.
        dilated_traj (list): List of dilated trajectories for each robot.
        predicted_trajectory (list): List of predicted trajectories for each robot.
        ax (matplotlib.axes.Axes): Axes object for plotting.
        u_hist (list): List of control inputs history for each robot.
        reached_goal (list): List of flags indicating if each robot has reached its goal.
        computational_time (list): List of computational times for each iteration.

    Methods:
        run_dwa: Runs the DWA algorithm.
        go_to_goal: Moves the robots towards their respective goals.
        update_robot_state: Updates the state of a robot based on its current state, control input, and environment information.
        dwa_control: Dynamic Window Approach control.
        calc_control_and_trajectory: Calculates the final input with the dynamic window.
    """

    def __init__(self, robot_num, trajectories, paths, targets, dilated_traj, predicted_trajectory, ax, u_hist):
        self.trajectories = trajectories
        self.robot_num = robot_num
        self.paths = paths
        self.targets = targets
        self.dilated_traj = dilated_traj
        self.predicted_trajectory = predicted_trajectory
        self.ax = ax
        self.u_hist = u_hist
        self.reached_goal = [False]*robot_num
        self.computational_time = []
        self.solver_failure = 0
        self.goal_input = np.zeros((2, robot_num))
        self.time_bkp = time.time()

    def run_dwa(self, x, u, break_flag):
        """
        Runs the DWA algorithm.

        Args:
            x (numpy.ndarray): Current state of the robots.
            u (numpy.ndarray): Control input for the robots.
            break_flag (bool): Flag indicating if the algorithm should stop.

        Returns:
            tuple: Updated state, control input, and break flag.

        """
        for i in range(self.robot_num):
            if not self.reached_goal[i]:
                # Step 9: Check if the distance between the current position and the target is less than 5
                if utils.dist(point1=(x[0,i], x[1,i]), point2=self.targets[i]) < update_dist:
                    # Perform some action when the condition is met
                    self.paths[i].pop(0)
                    if not self.paths[i]:
                        print(f"Path complete for vehicle {i}!")
                        u[:, i] = np.zeros(2)
                        x[3, i] = 0
                        self.reached_goal[i] = True
                        self.dilated_traj[i] = Point(x[0, i], x[1, i]).buffer(L/2, cap_style=3)
                        self.predicted_trajectory[i] = np.array([x[0:5, i]]*int(predict_time/dt))
                    else: 
                        self.targets[i] = (self.paths[i][0].x, self.paths[i][0].y)

                else:
                    t_prev = time.time()
                    x, u = self.update_robot_state(x, u, dt, i)
                    self.computational_time.append(time.time()-t_prev)

            if show_animation:
                utils.plot_robot_trajectory(x, u, self.predicted_trajectory, self.dilated_traj, self.targets, self.ax, i)
        
        if all(self.reached_goal):
                break_flag = True
        return x, u, break_flag
    
    def go_to_goal(self, x, u, break_flag):
        """
        Moves the robots towards their respective goals.

        Args:
            x (numpy.ndarray): Current state of the robots.
            u (numpy.ndarray): Control input for the robots.
            break_flag (bool): Flag indicating if the algorithm should stop.

        Returns:
            tuple: Updated state, control input, and break flag.

        """
        for i in range(self.robot_num):
            # Step 9: Check if the distance between the current position and the target is less than 5
            if not self.reached_goal[i]:                
                # If goal is reached, stop the robot
                if self.check_goal_reached(x, i, distance=to_goal_stop_distance):
                    u[:, i] = np.zeros(2)
                    x[3, i] = 0
                    self.reached_goal[i] = True
                else:
                    # If goal is not reached, update the robot's state
                    time_prev = time.time()
                    x, u = self.update_robot_state(x, u, dt, i)
                    self.computational_time.append(time.time()-time_prev)
                    
            # print(f"Speed of robot {i}: {x[3, i]}")
            
            # If we want the robot to disappear when it reaches the goal, indent one more time
            if show_animation:
                utils.plot_robot_trajectory(x, u, self.predicted_trajectory, self.dilated_traj, self.targets, self.ax, i)
        if all(self.reached_goal):
                break_flag = True
        return x, u, break_flag
    
    def random_harem(self, x, u, break_flag):
        """
        Runs the DWA algorithm.

        Args:
            x (numpy.ndarray): Current state of the robots.
            u (numpy.ndarray): Control input for the robots.
            break_flag (bool): Flag indicating if the algorithm should stop.

        Returns:
            tuple: Updated state, control input, and break flag.

        """
        for i in range(self.robot_num):
            if not self.reached_goal[i]:
                # Step 9: Check if the distance between the current position and the target is less than 5
                if time.time()-self.time_bkp > 30:
                    self.targets = utils.update_targets(x, self.targets)
                    self.time_bkp = time.time()

                else:
                    self.targets[i] = [x[0, self.targets[i][2]], x[1, self.targets[i][2]], self.targets[i][2]]
                    t_prev = time.time()
                    x, u = self.update_robot_state(x, u, dt, i)
                    self.computational_time.append(time.time()-t_prev)

            if show_animation:
                utils.plot_robot_trajectory(x, u, self.predicted_trajectory, self.dilated_traj, self.targets, self.ax, i)
        
        if all(self.reached_goal):
                break_flag = True
        return x, u, break_flag
    
    def update_robot_state(self, x, u, dt, i):
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
        ob = [self.dilated_traj[idx] for idx in range(len(self.dilated_traj)) if idx != i]
        if add_noise:
            noise = np.concatenate([np.random.normal(0, 0.21*noise_scale_param, 2).reshape(1, 2), np.random.normal(0, np.radians(5)*noise_scale_param, 1).reshape(1,1), np.random.normal(0, 0.2*noise_scale_param, 1).reshape(1,1), np.random.normal(0, 0.2*noise_scale_param, 1).reshape(1,1)], axis=1)
            noisy_pos = x1 + noise[0]

            self.goal_input[0, i], self.goal_input[1,i] = utils.pure_pursuit_steer_control(self.targets[i], utils.array_to_state(noisy_pos))
            u1, predicted_trajectory1, u_history = self.dwa_control(noisy_pos, ob, i)
            plt.plot(noisy_pos[0], noisy_pos[1], "x", color=color_dict[i], markersize=10)
        else:
            self.goal_input[0, i], self.goal_input[1,i] = utils.pure_pursuit_steer_control(self.targets[i], utils.array_to_state(x1))
            u1, predicted_trajectory1, u_history = self.dwa_control(x1, ob, i)
        self.dilated_traj[i] = LineString(zip(predicted_trajectory1[:, 0], predicted_trajectory1[:, 1])).buffer(dilation_factor, cap_style=3)

        # Collision check
        if linear_model:
            x1 = utils.motion(x1, u1, dt)
        else:
            x1 = utils.nonlinear_model_numpy_stable(x1, u1, dt)
        # x1 = utils.nonlinear_model_numpy_stable(x1, u1, dt)
        x[:, i] = x1
        u[:, i] = u1
        
        u, x = self.check_collision(x, u, i)
        
        self.predicted_trajectory[i] = predicted_trajectory1
        self.u_hist[i] = u_history
        
        return x, u
    
    def dwa_control(self, x, ob, i):
            """
            Dynamic Window Approach control.

            This method implements the Dynamic Window Approach control algorithm.
            It takes the current state, obstacles, and the index of the current iteration as input.
            It calculates the dynamic window, control inputs, trajectory, and control history.
            The control inputs are returned as a tuple (throttle, delta), and the trajectory and control history are also returned.

            Args:
                x (list): Current state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)].
                ob (list): List of obstacles.
                i (int): Index of the current iteration.

            Returns:
                tuple: Tuple containing the control inputs (throttle, delta) and the trajectory.
            """
            dw = calc_dynamic_window()
            u, trajectory, u_history = self.calc_control_and_trajectory(x, dw, ob, i)
            return u, trajectory, u_history
    
    def calc_control_and_trajectory(self, x, dw, ob, i):
            """
            Calculate the final input with the dynamic window.

            Args:
                x (list): Current state [x(m), y(m), yaw(rad), v(m/s), delta(rad)].
                dw (list): Dynamic window [min_throttle, max_throttle, min_steer, max_steer].
                ob (list): List of obstacles.
                i (int): Index of the target.

            Returns:
                tuple: Tuple containing the control inputs (throttle, delta) and the trajectory.
            """
            min_cost = float("inf")
            best_u = [0.0, 0.0]
            best_trajectory = np.array([x])
            u_buf = self.u_hist[i]
            goal = self.targets[i] 
            trajectory_buf = self.predicted_trajectory[i]

            # evaluate all trajectory with sampled input in dynamic window
            nearest = find_nearest(np.arange(min_speed, max_speed+v_resolution, v_resolution), x[3])

            for a in np.arange(dw[0], dw[1]+v_resolution, v_resolution):
                delta_keys = data[str(nearest)][str(a)].keys()
                for delta in delta_keys:
                    delta = float(delta)
                    # old_time = time.time()
                    geom = data[str(nearest)][str(a)][str(delta)]
                    geom = np.array(geom)
                    geom[:,0:2] = (geom[:,0:2]) @ utils.rotateMatrix(-x[2]) + [x[0],x[1]]
                    # print(time.time()-old_time)
                    geom[:,2] = geom[:,2] + x[2] #bringing also the yaw angle in the new frame

                    # trajectory = predict_trajectory(x_init, a, delta)
                    trajectory = geom
                    # calc cost

                    ob_cost = obstacle_cost_gain * calc_obstacle_cost(trajectory, ob)
                    # to_goal_cost = to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
                    # if trajectory[-1, 3] <= 0.0:
                    #     speed_cost = 10
                    # else:
                    #     speed_cost = 0.0
                    # final_cost = to_goal_cost + ob_cost + speed_cost # + heading_cost #+ speed_cost 

                    reference_input_cost = calc_reference_input_cost([a, float(delta)], self.goal_input, i)
                    final_cost = reference_input_cost + ob_cost

                    # search minimum trajectory
                    if min_cost >= final_cost:
                        min_cost = final_cost
                        best_u = [a, delta]
                        best_trajectory = trajectory
                        u_history = [[a, delta] for _ in range(len(trajectory-1))]

            # print(time.time()-old_time)
            if len(u_buf) > emergency_brake_distance:              
                u_buf.pop(0)

                trajectory_buf = trajectory_buf[1:]
                # Even when following the old trajectory, we need to update it to the position of the robot
                trajectory_buf[:,0:2] -= trajectory_buf[1,0:2]
                trajectory_buf[:,0:2] = (trajectory_buf[:,0:2]) @ utils.rotateMatrix(utils.normalize_angle(-x[2]+trajectory_buf[0,2]))
                trajectory_buf[:,0:2] += x[0:2]
                trajectory_buf[:,2] += utils.normalize_angle(x[2]-trajectory_buf[0,2])

                ob_cost = obstacle_cost_gain * calc_obstacle_cost(trajectory_buf, ob)
                # to_goal_cost = to_goal_cost_gain * calc_to_goal_cost(trajectory_buf, goal)
                # final_cost = to_goal_cost + ob_cost #+ speed_cost # + heading_cost #+ speed_cost 

                reference_input_cost = calc_reference_input_cost([u_buf[0][0], u_buf[0][1]], self.goal_input, i)
                final_cost = reference_input_cost + ob_cost
                
                if min_cost >= final_cost:
                    min_cost = final_cost
                    best_u = u_buf[0]
                    best_trajectory = trajectory_buf
                    u_history = u_buf

            if min_cost == np.inf:
                # emergency stop
                print(f"Emergency stop for vehicle {i}")
                self.solver_failure += 1
                # if x[3]>0:
                #     best_u = [min_acc, 0]
                # else:
                #     best_u = [max_acc, 0]
                
                best_u = [(0-x[3])/dt, 0]
                best_u[0] = np.clip(best_u[0], min_acc, max_acc)

                best_trajectory = np.array([x[0:3]]*int(predict_time/dt))
                u_history = [best_u]*int(predict_time/dt)

            return best_u, best_trajectory, u_history
    
    def check_collision(self, x, u, i):
        """
        Checks for collision between the robot at index i and other robots.

        Args:
            x (numpy.ndarray): State vector of shape (4, N), where N is the number of time steps.
            i (int): Index of the robot to check collision for.

        Raises:
            Exception: If collision is detected.

        """
        if x[0,i]>= boundary_points[1]-WB/2 or x[0,i]<= boundary_points[0]+WB/2 or x[1,i]>=boundary_points[3]-WB/2 or x[1,i]<=boundary_points[2]+WB/2:
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
    
    def check_goal_reached(self, x, i, distance=0.5):
        """
        Check if the given point has reached the goal.

        Args:
            x (numpy.ndarray): Array of x-coordinates.
            targets (list): List of target points.
            i (int): Index of the current point.

        Returns:
            bool: True if the point has reached the goal, False otherwise.
        """
        dist_to_goal = math.hypot(x[0, i] - self.targets[i][0], x[1, i] - self.targets[i][1])
        if dist_to_goal <= distance:
            print(f"Vehicle {i} reached goal!")
            self.dilated_traj[i] = Point(x[0, i], x[1, i]).buffer(L/2, cap_style=3)
            self.predicted_trajectory[i] = np.array([x[0:5, i]]*int(predict_time/dt))
            return True
        return False

def random_harem():
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
    iterations = 500
    break_flag = False
    
    # Step 2: Sample initial values for x0, y, yaw, v, omega, and model_type
    initial_state = seed['initial_position']
    x0 = initial_state['x']
    y = initial_state['y']
    yaw = initial_state['yaw']
    v = initial_state['v']
    omega = [0.0]*len(initial_state['x'])

    # Step 3: Create an array x with the initial values
    x = np.array([x0, y, yaw, v, omega])
    u = np.zeros((2, robot_num))

    trajectory = np.zeros((x.shape[0]+u.shape[0], robot_num, 1))
    trajectory[:, :, 0] = np.concatenate((x,u))

    predicted_trajectory = dict.fromkeys(range(robot_num),np.zeros([int(predict_time/dt), 4]))
    for i in range(robot_num):
        predicted_trajectory[i] = np.full((int(predict_time/dt), 4), x[0:4,i])

    # Step 4: Create paths for each robot
    traj = seed['trajectories']
    paths = [[Coordinate(x=traj[str(idx)][i][0], y=traj[str(idx)][i][1]) for i in range(len(traj[str(idx)]))] for idx in range(robot_num)]

    # Step 5: Extract the target coordinates from the paths
    targets = [[x[0,0], x[1, 0], 0] for path in paths]
    targets = utils.update_targets(x, targets)

    # Step 6: Create dilated trajectories for each robot
    dilated_traj = []
    for i in range(robot_num):
        dilated_traj.append(Point(x[0, i], x[1, i]).buffer(dilation_factor, cap_style=3))
    
    u_hist = dict.fromkeys(range(robot_num),[[0,0] for _ in range(int(predict_time/dt))])
    fig = plt.figure(1, dpi=90, figsize=(10,10))
    ax = fig.add_subplot(111)
    
    # Step 7: Create an instance of the DWA_algorithm class
    dwa = DWA_algorithm(robot_num, paths, paths, targets, dilated_traj, predicted_trajectory, ax, u_hist)

    predicted_trajectory = {}
    targets = {}

    for z in range(iterations):
        plt.cla()
        plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
        
        x, u, break_flag = dwa.random_harem(x, u, break_flag)
        # x, u, break_flag = dwa.go_to_goal(x, u, break_flag)
        trajectory = np.dstack([trajectory, np.concatenate((x,u))])


        predicted_trajectory[z] = {}
        targets[z] = {}
        for i in range(robot_num):
            predicted_trajectory[z][i] = dwa.predicted_trajectory[i]
            targets[z][i] = dwa.targets[i]

            
        utils.plot_map(width=width_init, height=height_init)
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)

        if break_flag:
            break

    print("Done")
    if show_animation:
        for i in range(robot_num):
            plt.plot(trajectory[0, i, :], trajectory[1, i, :], "-", color=color_dict[i])
        plt.pause(0.0001)
        plt.show()

    print("Saving the trajectories to /dwa_dev/dwa_dev/DWA_trajectories_harem.pkl\n")
    with open(dir_path + '/DWA_trajectories_harem.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([trajectory, targets], f) 
    print("Saving the trajectories to /dwa_dev/dwa_dev/DWA_dilated_traj_harem.pkl")
    with open(dir_path + '/DWA_dilated_traj_harem.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([predicted_trajectory], f)  


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
    omega = [0.0]*len(initial_state['x'])

    # Step 3: Create an array x with the initial values
    x = np.array([x0, y, yaw, v, omega])
    u = np.zeros((2, robot_num))

    trajectory = np.zeros((x.shape[0]+u.shape[0], robot_num, 1))
    trajectory[:, :, 0] = np.concatenate((x,u))

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
    
    u_hist = dict.fromkeys(range(robot_num),[[0,0] for _ in range(int(predict_time/dt))])
    fig = plt.figure(1, dpi=90, figsize=(10,10))
    ax = fig.add_subplot(111)
    
    # Step 7: Create an instance of the DWA_algorithm class
    dwa = DWA_algorithm(robot_num, paths, paths, targets, dilated_traj, predicted_trajectory, ax, u_hist)
    
    predicted_trajectory = {}

    for z in range(iterations):
        plt.cla()
        plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
        
        x, u, break_flag = dwa.run_dwa(x, u, break_flag)
        # x, u, break_flag = dwa.go_to_goal(x, u, break_flag)
        trajectory = np.dstack([trajectory, np.concatenate((x,u))])


        predicted_trajectory[z] = {}
        for i in range(robot_num):
            predicted_trajectory[z][i] = dwa.predicted_trajectory[i]

            
        utils.plot_map(width=width_init, height=height_init)
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)

        if break_flag:
            break

    print("Done")
    if show_animation:
        for i in range(robot_num):
            plt.plot(trajectory[0, i, :], trajectory[1, i, :], "-", color=color_dict[i])
        plt.pause(0.0001)
        plt.show()
    
    print("Saving the trajectories to /dwa_dev/dwa_dev/DWA_trajectories.pkl\n")
    with open('/home/giacomo/thesis_ws/src/dwa_dev/dwa_dev/DWA_trajectories.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([trajectory, targets], f) 
    print("Saving the trajectories to /dwa_dev/dwa_dev/DWA_dilated_traj.pkl")
    with open('/home/giacomo/thesis_ws/src/dwa_dev/dwa_dev/DWA_dilated_traj.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([predicted_trajectory], f)  

       
if __name__ == '__main__':
    # main_seed()
    random_harem()
    # main()
