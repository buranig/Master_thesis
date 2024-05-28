import matplotlib.pyplot as plt
import numpy as np
import math
from enum import Enum
# For the parameter file
import pathlib
import json
from custom_message.msg import Coordinate
from shapely.geometry import Point, LineString
from shapely import distance
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
robot_num = 11 #json_object["robot_num"]
safety_init = json_object["safety"]
width_init = json_object["width"]
height_init = json_object["height"]
min_dist = json_object["min_dist"]
to_goal_stop_distance = json_object["to_goal_stop_distance"]
emergency_brake_distance = json_object["LBP"]["emergency_brake_distance"]
update_dist = 2
N=3

show_animation = json_object["show_animation"]
boundary_points = np.array([-width_init/2, width_init/2, -height_init/2, height_init/2])
check_collision_bool = False
add_noise = json_object["add_noise"]
noise_scale_param = json_object["noise_scale_param"]
linear_model = json_object["linear_model"]
np.random.seed(1)

color_dict = {0: 'r', 1: 'b', 2: 'g', 3: 'y', 4: 'm', 5: 'c', 6: 'k', 7: 'tab:orange', 8: 'tab:brown', 9: 'tab:gray', 10: 'tab:olive', 11: 'tab:pink', 12: 'tab:purple', 13: 'tab:red', 14: 'tab:blue', 15: 'tab:green'}

with open('/home/giacomo/thesis_ws/src/lbp_dev/lbp_dev/LBP.json', 'r') as file:
    data = json.load(file)

with open('/home/giacomo/thesis_ws/src/seeds/circular_seed_30.json', 'r') as file:
    seed = json.load(file)

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
    error_angle = utils.normalize_angle(math.atan2(dy, dx))
    cost_angle = error_angle - utils.normalize_angle(trajectory[-1, 2])
    cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

    return cost

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
        self.solver_failure = 0

    def run_lbp(self, x, u, break_flag):
        """
        Runs the LBP (Loop-based Path Planning) algorithm for the given robot configuration.

        Args:
            x (numpy.ndarray): The current state of the robots.
            u (numpy.ndarray): The control inputs for the robots.
            break_flag (bool): A flag indicating whether the algorithm should break or continue.

        Returns:
            numpy.ndarray: The updated state of the robots.
            numpy.ndarray: The updated control inputs for the robots.
            bool: The updated break flag.
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
                    else: 
                        self.targets[i] = (self.paths[i][0].x, self.paths[i][0].y)
                else:
                    t_prev = time.time()
                    x, u = self.update_robot_state(x, u, dt, i)
                    self.computational_time.append(time.time()-t_prev)
                
                u, x = self.check_collision(x, u, i) 

            if show_animation:
                utils.plot_robot_trajectory(x, u, self.predicted_trajectory, self.dilated_traj, self.targets, self.ax, i)

        if all(self.reached_goal):
            break_flag = True

        return x, u, break_flag
    
    def go_to_goal(self, x, u, break_flag):
        for i in range(self.robot_num):
            # Step 9: Check if the distance between the current position and the target is less than 5
            if not self.reached_goal[i]:        
                # If goal is reached, stop the robot
                if self.check_goal_reached(x, i, distance=to_goal_stop_distance):
                    u[:, i] = np.zeros(2)
                    x[3, i] = 0
                    self.reached_goal[i] = True
                    # print(f"Vehicle {i} reached goal!!")
                else:
                    t_prev = time.time()
                    x, u = self.update_robot_state(x, u, dt, i)
                    self.computational_time.append(time.time()-t_prev)

                u, x = self.check_collision(x, u, i) 
            
            # else:
            #     print(u[:, i])
            # If we want the robot to disappear when it reaches the goal, indent one more time
            if all(self.reached_goal):
                break_flag = True

            if show_animation:
                utils.plot_robot_trajectory(x, u, self.predicted_trajectory, self.dilated_traj, self.targets, self.ax, i)
        return x, u, break_flag
    
    def update_robot_state(self, x, u, dt, i):
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
        ob = [self.dilated_traj[idx] for idx in range(len(self.dilated_traj)) if idx != i]
        if add_noise:
            noise = np.concatenate([np.random.normal(0, 0.21*noise_scale_param, 2).reshape(1, 2), np.random.normal(0, np.radians(5)*noise_scale_param, 1).reshape(1,1), np.random.normal(0, 0.2*noise_scale_param, 1).reshape(1,1), np.random.normal(0, 0.2*noise_scale_param, 1).reshape(1,1)], axis=1)
            noisy_pos = x1 + noise[0]
            u1, predicted_trajectory1, u_history = self.lbp_control(noisy_pos, ob, i)
            plt.plot(noisy_pos[0], noisy_pos[1], "x", color=color_dict[i], markersize=10)
        else:
            u1, predicted_trajectory1, u_history = self.lbp_control(x1, ob, i)
        self.dilated_traj[i] = LineString(zip(predicted_trajectory1[:, 0], predicted_trajectory1[:, 1])).buffer(dilation_factor, cap_style=3)
    
        # Collision check
        if check_collision_bool:
            if any([utils.dist([x1[0], x1[1]], [x[0, idx], x[1, idx]]) < WB for idx in range(robot_num) if idx != i]): raise Exception('Collision')

        if linear_model:
            x1 = utils.motion(x1, u1, dt)
        else:
            x1 = utils.nonlinear_model_numpy_stable(x1, u1, dt)
        x[:, i] = x1
        u[:, i] = u1

        # u, x = self.check_collision(x, u, i)

        self.predicted_trajectory[i] = predicted_trajectory1
        self.u_hist[i] = u_history

        return x, u
    
    def lbp_control(self, x, ob, i):
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
        u, trajectory, u_history = self.calc_control_and_trajectory(x, v_search, ob, i)
        return u, trajectory, u_history
    
    def calc_control_and_trajectory(self, x, v_search, ob, i):
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
        u_history = {}
        u_buf = self.u_hist[i]
        goal = self.targets[i] 
        trajectory_buf = self.predicted_trajectory[i]

        # Calculate the cost of each possible trajectory and return the minimum
        for v in v_search:
            dict = data[str(v)]
            for id, info in dict.items():

                # old_time = time.time()
                geom = np.zeros((len(info['x']),4))
                geom[:,0] = info['x']
                geom[:,1] = info['y']
                geom[:,2] = info['yaw']
                geom[:,3] = info['v']
                geom[:,0:2] = (geom[:,0:2]) @ utils.rotateMatrix(-x[2]) + [x[0],x[1]]
                
                geom[:,2] = geom[:,2] + x[2] #bringing also the yaw angle in the new frame
                
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
                    u_history['ctrl'] = info['ctrl'].copy()
                    u_history['v_goal'] = v

        # Calculate cost of the previous best trajectory and compare it with that of the new trajectories
        # If the cost of the previous best trajectory is lower, use the previous best trajectory
        
        #TODO: this section has a small bug due to popping elements from the buffer, it gets to a point where there 
        # are no more elements in the buffer to use
        if len(u_buf['ctrl']) > emergency_brake_distance:
            u_buf['ctrl'].pop(0)
            
            trajectory_buf = trajectory_buf[1:]
            # Even when following the old trajectory, we need to update it to the position of the robot
            trajectory_buf[:,0:2] -= trajectory_buf[1,0:2]
            trajectory_buf[:,0:2] = (trajectory_buf[:,0:2]) @ utils.rotateMatrix(utils.normalize_angle(-x[2]+trajectory_buf[0,2]))
            trajectory_buf[:,0:2] += x[0:2]
            trajectory_buf[:,2] += utils.normalize_angle(x[2]-trajectory_buf[0,2])

            # trajectory_buf[:,2] = trajectory_buf[:,2] + x[2]

            to_goal_cost = to_goal_cost_gain * calc_to_goal_cost(trajectory_buf, goal)
            # speed_cost = speed_cost_gain * np.sign(trajectory[-1, 3]) * trajectory[-1, 3]
            ob_cost = obstacle_cost_gain * calc_obstacle_cost(trajectory_buf, ob)
            heading_cost = heading_cost_gain * calc_to_goal_heading_cost(trajectory_buf, goal)
            final_cost = to_goal_cost + ob_cost + heading_cost #+ speed_cost 

            if min_cost >= final_cost:
                min_cost = final_cost
                best_u = [(u_buf['v_goal']-x[3])/dt, u_buf['ctrl'][1]]
                # best_u = [0, u_buf['ctrl'][1]]
                best_trajectory = trajectory_buf
                u_history['ctrl'] = u_buf['ctrl']

        elif min_cost == np.inf:
            # emergency stop
            print(f"Emergency stop for vehicle {i}")
            self.solver_failure += 1
            best_u = [(0-x[3])/dt, 0]
            best_u[0] = np.clip(best_u[0], min_acc, max_acc)
            
            best_trajectory = utils.predict_trajectory(x, best_u[0], best_u[1], predict_time)
            # best_trajectory = np.array([x[0:3], x[0:3]]*int(predict_time/dt))
            u_history['ctrl'] = [0.0]*int(predict_time/dt)
            u_history['v_goal'] = 0.0


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
            self.predicted_trajectory[i] = np.array([x[0:-1, i]]*int(predict_time/dt))
            return True
        return False

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
    omega = [0.0]*len(initial_state['x'])

    assert robot_num == len(seed['initial_position']['x']), "The number of robots in the seed file does not match the number of robots in the seed file"
    # Step 3: Create an array x with the initial values
    x = np.array([x0, y, yaw, v, omega])
    u = np.zeros((2, robot_num))

    trajectory = np.zeros((x.shape[0], robot_num, 1))
    trajectory[:, :, 0] = x

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

    u_hist = dict.fromkeys(range(robot_num),{'ctrl': [0]*int(predict_time/dt), 'v_goal': 0})
    fig = plt.figure(1, dpi=90, figsize=(10,10))
    ax = fig.add_subplot(111)
    
    lbp = LBP_algorithm(predicted_trajectory, paths, targets, dilated_traj,
                        predicted_trajectory, ax, u_hist)
    
    for z in range(iterations):
        plt.cla()
        plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
        
        x, u, break_flag = lbp.run_lbp(x, u, break_flag)
        # x, u, break_flag = lbp.go_to_goal(x, u, break_flag)

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
            plt.plot(trajectory[0, i, :], trajectory[1, i, :], "-", color=color_dict[i])
        plt.pause(0.0001)
        plt.show()

if __name__ == '__main__':
    # main()
    main_seed()

    