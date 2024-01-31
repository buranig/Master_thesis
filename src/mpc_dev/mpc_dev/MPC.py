import numpy as np
import mpc_dev.cubic_spline_planner as cubic_spline_planner
import math
import random
import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.optimize import minimize, NonlinearConstraint
import time
import planner.utils as utils

import pathlib
import json

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
# N=3

timer_freq = json_object["timer_freq"]

show_animation = True
debug = False
check_collision_bool = False

color_dict = {0: 'r', 1: 'b', 2: 'g', 3: 'y', 4: 'm', 5: 'c', 6: 'k'}

# Simulator options.
options = {}
options['FIG_SIZE'] = [8,8]
options['OBSTACLES'] = True

with open('/home/giacomo/thesis_ws/src/seed_1.json', 'r') as file:
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

class ModelPredictiveControl:
    """
    Class representing a Model Predictive Control (MPC) system.

    Attributes:
        horizon (int): The prediction horizon.
        dt (float): The time step.
        reference1 (list): The reference or set point the controller will achieve.
        reference2 (list): An optional second reference point.
        x_obs (list): The x-coordinates of the obstacles.
        y_obs (list): The y-coordinates of the obstacles.
        initial_state (list): The initial state of the system.
        safety_radius (float): The safety radius for obstacle avoidance.

    Methods:
        plant_model: Computes the next state of the system based on the current state and control inputs.
        cost_function: Computes the cost associated with a given control sequence.
        propagation1: Propagates the system state in the x-direction based on a given control sequence.
        propagation2: Propagates the system state in the y-direction based on a given control sequence.
        propagation3: Computes the distance between the system state and the obstacles based on a given control sequence.
    """

    def __init__(self, obs_x, obs_y,  cx=None, cy=None, ref=None, bounds=None, constraints=None, predicted_trajectory=None):
        self.horizon = 8
        self.dt = 0.2

        # self.L = 2.5 # Car base [m]

        # Reference or set point the controller will achieve.
        self.reference1 = [10, 0, 0]
        self.reference2 = None

        self.x_obs = obs_x
        self.y_obs = obs_y

        self.initial_state = None
        self.safety_radius = 1.5

        self.cx = cx
        self.cy = cy
        self.ref = ref
        self.bounds = bounds
        self.constraints = constraints
        self.predicted_trajectory = predicted_trajectory
        self.reached_goal = [False]*robot_num
        self.computational_time = []
        self.old_time = [time.time()]*robot_num

    def plant_model(self, prev_state, dt, pedal, steering):
        """
        Computes the next state of the system based on the current state and control inputs.

        Args:
            prev_state (list): The current state of the system.
            dt (float): The time step.
            pedal (float): The control input for acceleration.
            steering (float): The control input for steering.

        Returns:
            list: The next state of the system.
        """
        x_t = prev_state[0]
        y_t = prev_state[1]
        psi_t = prev_state[2]
        v_t = prev_state[3]

        x_t += np.cos(psi_t) * v_t * dt
        y_t += np.sin(psi_t) * v_t * dt

        a_t = pedal
        v_t += a_t * dt #- v_t/25
        v_t = np.clip(v_t, min_speed, max_speed)

        psi_t += v_t * dt * np.tan(steering)/L
        psi_t = normalize_angle(psi_t)

        return [x_t, y_t, psi_t, v_t]

    def cost_function(self, u, *args):
        """
        Computes the cost associated with a given control sequence.

        Args:
            u (list): The control sequence.
            args (tuple): Additional arguments (state, reference).

        Returns:
            float: The cost.
        """
        state = args[0]
        ref = args[1]
        cost = 0.0

        for i in range(self.horizon):
            speed = state[3]
            heading = state[2]

            state = self.plant_model(state, self.dt, u[i*2], u[i*2 + 1])

            distance_to_goal = np.sqrt((ref[0] - state[0])**2 + (ref[1] - state[1])**2)

            # Position cost
            # cost +=  distance_to_goal

            # Obstacle cost
            for z in range(len(self.x_obs)-1):
                distance_to_obstacle = np.sqrt((self.x_obs[z] - state[0])**2 + (self.y_obs[z] - state[1])**2)
                # if any(distance_to_obstacle < 3):
                if distance_to_obstacle < 5:
                    cost += 100 #np.inf/distance_to_obstacle

            # Heading cost
            cost += 10 * (heading - state[2])**2
            # cost += 10 * state[2]

            # negative speed cost
            # cost += -5 * np.sign(speed) * 3 * speed

            # Acceleration cost
            # if abs(u[2*i]) > 0.2:
            #     cost += (speed - state[3])**2

        # Final heading and position cost        
        # cost +=  4 * (normalize_angle(ref[2]) - normalize_angle(state[2]))**2
        distance_to_goal = np.sqrt((ref[0] - state[0])**2 + (ref[1] - state[1])**2)
        cost += 5*distance_to_goal
        print(f'cost: {cost}, distance to goal: {distance_to_goal}')
        return cost
    
    def cost_function2(self,u, *args):
        state = args[0]
        ref = args[1]
        cost = 0.0

        for i in range(self.horizon):
            speed = state[3]
            heading = state[2]

            state = self.plant_model(state, self.dt, u[i*2], u[i*2 + 1])

            distance_to_goal = np.sqrt((ref[0] - state[0])**2 + (ref[1] - state[1])**2)

            # Position cost
            cost +=  distance_to_goal

            # Obstacle cost
            for z in range(len(self.x_obs)-1):
                distance_to_obstacle = np.sqrt((self.x_obs[z] - state[0])**2 + (self.y_obs[z] - state[1])**2)
                if distance_to_obstacle < 3:
                    cost += 40/distance_to_obstacle

            # Heading cost
            cost += 10 * (heading - state[2])**2

            # cost +=  2 * (ref[2] - state[2])**2

            # negative speed cost
            cost += -10 * np.sign(speed) * 3 * speed

            # Acceleration cost
            if abs(u[2*i]) > 0.2:
                cost += (speed - state[3])**2

        cost += (state[3])**2
        distance_to_goal = np.sqrt((ref[0] - state[0])**2 + (ref[1] - state[1])**2)
        cost += 100*distance_to_goal
        return cost

    def cost_function3(self,u, *args):
        """
        Define the cost function for the MPC controller. Composed of a stage cost calculated in
        the for loop and a terminal cost, calculated at the end on the loop.
        The cost is based on the input sequence and also the way the states are propagated.

        Args:
            self
            u: control sequence used to calculate the state sequence

        Returns:
            cost: total cost of the control sequence
        """
        state = args[0]
        ref = args[1]
        cost = 0.0

        for i in range(self.horizon):
            speed = state[3]
            heading = state[2]

            state = self.plant_model(state, self.dt, u[i*2], u[i*2 + 1])

            distance_to_goal = np.sqrt((ref[0] - state[0])**2 + (ref[1] - state[1])**2)

            # Position cost
            cost +=  distance_to_goal

            # Obstacle cost
            # for z in range(len(self.x_obs)-1):
            #     distance_to_obstacle = np.sqrt((self.x_obs[z] - state[0])**2 + (self.y_obs[z] - state[1])**2)
            #     if distance_to_obstacle < 3:
            #         cost += 40/distance_to_obstacle

            # Heading cost
            cost += 10 * (heading - state[2])**2

            cost +=  2 * (ref[2] - state[2])**2

            # negative speed cost
            cost += -10 * np.sign(speed) * 3 * speed

            # Acceleration cost
            if abs(u[2*i]) > 0.2:
                cost += (speed - state[3])**2

        cost += (state[3])**2
        distance_to_goal = np.sqrt((ref[0] - state[0])**2 + (ref[1] - state[1])**2)
        cost += 100*distance_to_goal
        return cost
    
    def seed_cost(self,u, *args):
        """
        Define the cost function for the MPC controller. Composed of a stage cost calculated in
        the for loop and a terminal cost, calculated at the end on the loop.
        The cost is based on the input sequence and also the way the states are propagated.

        Args:
            self
            u: control sequence used to calculate the state sequence

        Returns:
            cost: total cost of the control sequence
        """
        state = args[0]
        ref = args[1]
        cost = 0.0

        for i in range(self.horizon):
            speed = state[3]
            heading = state[2]

            state = self.plant_model(state, self.dt, u[i*2], u[i*2 + 1])

            distance_to_goal = np.sqrt((ref[0] - state[0])**2 + (ref[1] - state[1])**2)

            # Position cost
            cost +=  distance_to_goal

            # Obstacle cost
            # for z in range(len(self.x_obs)-1):
            #     distance_to_obstacle = np.sqrt((self.x_obs[z] - state[0])**2 + (self.y_obs[z] - state[1])**2)
            #     if distance_to_obstacle < 3:
            #         cost += 40/distance_to_obstacle

            # Heading cost
            cost += 10 * (heading - state[2])**2

            # cost +=  2 * (ref[2] - state[2])**2

            # negative speed cost
            cost += -10 * np.sign(speed) * 3 * speed

            # Acceleration cost
            if abs(u[2*i]) > 0.2:
                cost += (speed - state[3])**2

        cost += (state[3])**2
        distance_to_goal = np.sqrt((ref[0] - state[0])**2 + (ref[1] - state[1])**2)
        cost += 100*distance_to_goal
        return cost

    def propagation1(self, u):
        """
        Propagates the system state in the x-direction based on a given control sequence.
        This function is used to check wether the states are propagated outside the boundaries
        for a given control sequence u.

        Args:
            u (list): The control sequence.

        Returns:
            numpy.ndarray: The system state in the x-direction.
        """
        state = [self.initial_state]

        for i in range(self.horizon):
            state.append(self.plant_model(state[-1], self.dt, u[2*i], u[2*i + 1]))
        return np.array(state)[:,0]
    
    def propagation2(self, u):
        """
        Propagates the system state in the y-direction based on a given control sequence.
        This function is used to check wether the states are propagated outside the boundaries
        for a given control sequence u.

        Args:
            u (list): The control sequence.

        Returns:
            numpy.ndarray: The system state in the y-direction.
        """
        state = [self.initial_state]

        for i in range(self.horizon):
            state.append(self.plant_model(state[-1], self.dt, u[2*i], u[2*i + 1]))
        return np.array(state)[:,1]
    
    def propagation3(self, u):
        """
        Computes the distance between the system state and the obstacles based on a given control sequence.

        Args:
            u (list): The control sequence.

        Returns:
            numpy.ndarray: The distances between the system state and the obstacles.
        """
        state = self.initial_state
        distance = []

        for t in range(self.horizon):
            state = self.plant_model(state, self.dt, u[2*t], u[2*t + 1])
            for i in range(len(self.x_obs)):
                distance.append((state[0] - self.x_obs[i])**2 + (state[1] - self.y_obs[i])**2-self.safety_radius**2)

        return np.array(distance)
    
    def run_mpc(self, x, u):
        for i in range(robot_num):
            start_time = time.time()
            if dist([x[0, i], x[1, i]], point2=self.ref[i]) < 5:
                self.cx[i].pop(0)
                self.cy[i].pop(0)
                if not self.cx[i]:
                    print("Path complete")
                    return
                self.ref[i][0] = self.cx[i][0]
                self.ref[i][1] = self.cy[i][0]

            # cx, cy, ref = update_paths(i, x, cx, cy, cyaw, target_ind, ref, dl)
            t_prev = time.time()
            x, u, self.predicted_trajectory = self.mpc_control(i, x, u, self.bounds, self.constraints, self.ref, self.predicted_trajectory, self.seed_cost)
            self.computational_time.append(time.time() - t_prev)

            if debug:
                print('Robot ' + str(i+1) + ' of ' + str(robot_num) + '   Time ' + str(round(time.time() - start_time,5)))

            plot_robot_seed(x, u, self.predicted_trajectory, self.ref, i) 
        
        return x, u
    
    def go_to_goal(self, x, u, break_flag):
        for i in range(robot_num):
            if not self.reached_goal[i]:                
                # If goal is reached, stop the robot
                if check_goal_reached(x, self.ref, i, distance=1):
                    # u[:, i] = np.zeros(2)
                    x[3, i] = 0
                    self.reached_goal[i] = True
                else:
                    t_prev = time.time()
                    x, u, self.predicted_trajectory = self.mpc_control(i, x, u, self.bounds, self.constraints, self.ref, self.predicted_trajectory, self.seed_cost)
                    self.computational_time.append(time.time() - t_prev)
                    
            # If we want the robot to disappear when it reaches the goal, indent one more time
            if all(self.reached_goal):
                break_flag = True

            if show_animation:
                plot_robot_seed(x, u, self.predicted_trajectory, self.ref, i)
        return x, u, break_flag
    
    def mpc_control(self, i, x, u, bounds, constraints, ref, predicted_trajectory, cost_function):
        x1 = x[:, i]
        u1 = u[:,i]
        u1 = np.delete(u1,0)
        u1 = np.delete(u1,0)
        u1 = np.append(u1, u1[-2])
        u1 = np.append(u1, u1[-2])

        self.update_obstacles(i, x1, x, predicted_trajectory)        

        self.initial_state = x1

        # MPC control
        u_solution = minimize(cost_function, u1, (x1, ref[i]),
                            method='SLSQP',
                            bounds=bounds,
                            constraints=constraints,
                            tol = 1e-1)
        
        u1 = u_solution.x
        x1 = self.plant_model(x1, time.time() - self.old_time[i], u1[0], u1[1])
        print(time.time() - self.old_time[i]) 
        self.old_time[i] = time.time() 
        x[:, i] = x1
        u[:, i] = u1
        predicted_state = np.array([x1])

        for j in range(1, self.horizon):
            predicted = self.plant_model(predicted_state[-1], self.dt, u1[2*j], u1[2*j+1])
            predicted_state = np.append(predicted_state, np.array([predicted]), axis=0)
        predicted_trajectory[i] = predicted_state

        return x, u, predicted_trajectory
    
    def update_obstacles(self, i, x1, x, predicted_trajectory):
        """
        Update the obstacles for the model predictive control (MPC) algorithm.

        Args:
            mpc (MPC): The MPC object.
            i (int): The index of the current robot.
            x1 (list): The position of the current robot.
            x (ndarray): The positions of all robots.
            predicted_trajectory (list): The predicted trajectories of all robots.

        Raises:
            Exception: If a collision is detected.

        Returns:
            None
        """
        self.x_obs = []
        self.y_obs = []
        for idx in range(robot_num):
            if idx == i:
                continue
            if check_collision_bool:
                if dist([x1[0], x1[1]], [x[0, idx], x[1, idx]]) < 1:
                    raise Exception('Collision')
            
            next_robot_state = predicted_trajectory[idx]
            self.x_obs.append(next_robot_state[0:-1:5, 0])
            self.y_obs.append(next_robot_state[0:-1:5, 1])
        self.x_obs = [item for sublist in self.x_obs for item in sublist]
        self.y_obs = [item for sublist in self.y_obs for item in sublist]

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

def get_switch_back_course(dl):
    ax = [0.0, 30.0, 6.0, 20.0, 35.0]
    ay = [0.0, 0.0, 20.0, 35.0, 20.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)
    ax = [35.0, 10.0, 0.0, 0.0]
    ay = [20.0, 30.0, 5.0, 0.0]
    cx2, cy2, cyaw2, ck2, s2 = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)
    cyaw2 = [i - math.pi for i in cyaw2]
    cx.extend(cx2)
    cy.extend(cy2)
    cyaw.extend(cyaw2)
    ck.extend(ck2)

    return cx, cy, cyaw, ck

def dist(point1, point2):
    """
    Calculates the Euclidean distance between two points.

    :param point1: (tuple) x, y coordinates of the first point
    :param point2: (tuple) x, y coordinates of the second point
    :return: (float) Euclidean distance between the two points
    """
    x1, y1 = point1
    x2, y2 = point2

    x1 = float(x1)
    x2 = float(x2)
    y1 = float(y1)
    y2 = float(y2)

    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return distance

def get_straight_course(start, goal, dl):
    """
    Generates a straight course between the given start and goal points.

    Args:
        start (tuple): The coordinates of the start point (x, y).
        goal (tuple): The coordinates of the goal point (x, y).
        dl (float): The desired spacing between waypoints.

    Returns:
        tuple: A tuple containing the x-coordinates (cx), y-coordinates (cy),
               yaw angles (cyaw), and curvature values (ck) of the generated course.
    """
    ax = [start[0], goal[0]]
    ay = [start[1], goal[1]]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)
    return cx, cy, cyaw, ck
    
def update_paths(i, x, cx, cy, cyaw, target_ind, ref, dl):
    """
    Update the paths of the robots based on their current positions and target indices.

    Args:
        i (int): Index of the robot.
        x (numpy.ndarray): Array of robot positions.
        cx (list): List of x-coordinates of the path for each robot.
        cy (list): List of y-coordinates of the path for each robot.
        cyaw (list): List of yaw angles of the path for each robot.
        target_ind (list): List of target indices for each robot.
        ref (list): List of reference points for each robot.
        dl (float): Length of each path segment.

    Returns:
        tuple: Updated cx, cy, and ref lists.
    """
    x1 = x[:, i]
    # Updating the paths of the robots
    if (target_ind[i] < len(cx[i])-1):
        if dist([x1[0], x1[1]], [cx[i][target_ind[i]], cy[i][target_ind[i]]]) < 4:
            target_ind[i]+=1
            ref[i][0] = cx[i][target_ind[i]]
            ref[i][1] = cy[i][target_ind[i]]
            ref[i][2] = cyaw[i][target_ind[i]]
    elif (target_ind[i] == len(cx[i])-1):
        target_ind[i] = 0
        cx.pop(i)
        cy.pop(i)
        cyaw.pop(i)
        sample_point = (float(random.randint(-width_init/2, width_init/2)), float(random.randint(-height_init/2, height_init/2)))

        cx1, cy1, cyaw1, ck1 = get_straight_course(start=(x[0, i], x[1, i]), goal=(sample_point[0], sample_point[1]), dl=dl)
        cx.insert(i, cx1)
        cy.insert(i, cy1)
        cyaw.insert(i, cyaw1)
        
        ref[i] = [cx[i][target_ind[i]], cy[i][target_ind[i]], cyaw[i][target_ind[i]]]
    
    return cx, cy, ref

def generate_reference_trajectory(x, dl):
    cx = []
    cy = []
    cyaw = []
    ref = []
    target_ind = [0] * robot_num
    for i in range(robot_num):
        sample_point = (float(random.randint(-width_init/2, width_init/2)), float(random.randint(-height_init/2, height_init/2)))
        
        cx1, cy1, cyaw1, ck1 = get_straight_course(start=(x[0, i], x[1, i]), goal=(sample_point[0], sample_point[1]), dl=dl)
        cx.append(cx1)
        cy.append(cy1)
        cyaw.append(cyaw1)
        ref.append([cx[i][target_ind[i]], cy[i][target_ind[i]], cyaw[i][target_ind[i]]])

        if debug:
            plt.plot(x[0, i], x[1, i], "xr")
            plt.plot(cx[i], cy[i], "-r", label="course")

    if debug:
        plt.show()
    return cx, cy, cyaw, ref, target_ind

def set_bounds_and_constraints(mpc):
        bounds = []
        # Set bounds for inputs bounded optimization.
        for i in range(mpc.horizon):
            bounds += [[min_acc, max_acc]]
            bounds += [[-max_steer, max_steer]]

        constraint1 = NonlinearConstraint(fun=mpc.propagation1, lb=-width_init/2 + mpc.safety_radius, ub=width_init/2 - mpc.safety_radius)
        constraint2 = NonlinearConstraint(fun=mpc.propagation2, lb=-height_init/2 + mpc.safety_radius, ub=height_init/2 - mpc.safety_radius)
        constraint3 = NonlinearConstraint(fun=mpc.propagation3, lb=0, ub=np.inf)
        constraints = [constraint1, constraint2, constraint3]

        return bounds, constraints

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
    
def plot_robot_trajectory(x, u, cx, cy, predicted_trajectory, targets, i):
    plt.plot(predicted_trajectory[i][:, 0], predicted_trajectory[i][:, 1], "-"+color_dict[i])
    # plt.plot(x[0, i], x[1, i], "xr")
    plt.plot(targets[i][0], targets[i][1], "x"+color_dict[i])
    plt.plot(cx[i], cy[i], "--"+color_dict[i], label="course")
    plot_robot(x[0, i], x[1, i], x[2, i], i)
    plot_arrow(x[0, i], x[1, i], x[2, i], length=1, width=0.5)
    plot_arrow(x[0, i], x[1, i], x[2, i] + u[1, i], length=3, width=0.5)

def plot_robot_seed(x, u, predicted_trajectory, targets, i):
    plt.plot(predicted_trajectory[i][:, 0], predicted_trajectory[i][:, 1], "-"+color_dict[i])
    plt.plot(targets[i][0], targets[i][1], "x"+color_dict[i])
    plot_robot(x[0, i], x[1, i], x[2, i], i)
    plot_arrow(x[0, i], x[1, i], x[2, i], length=1, width=0.5)
    plot_arrow(x[0, i], x[1, i], x[2, i] + u[1, i], length=3, width=0.5)

def main():
    """
    Main function for controlling multiple robots using Model Predictive Control (MPC).

    Steps:
    1. Initialize the necessary variables and parameters.
    2. Create an instance of the ModelPredictiveControl class.
    3. Set the initial state and control inputs.
    4. Generate the reference trajectory for each robot.
    5. Plot the initial positions and reference trajectory.
    6. Set the bounds and constraints for the MPC.
    7. Initialize the predicted trajectory for each robot.
    8. Enter the main control loop:
        - Check if the distance between the current position and the target is less than 5.
            - If yes, update the path and target.
        - Perform MPC control for each robot.
        - Plot the robot trajectory.
        - Update the predicted trajectory.
        - Plot the map and pause for visualization.
    """
    print(__file__ + " start!!")

    # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    iterations = 3000
    break_flag = False
    dl = 3
    
    # MPC initialization
    mpc = ModelPredictiveControl(obs_x=[], obs_y=[])
    
    x0, y, yaw, v, omega, model_type = utils.samplegrid(width_init, height_init, min_dist, robot_num, safety_init)
    x = np.array([x0, y, yaw, v])
    num_inputs = 2
    u = np.zeros([mpc.horizon*num_inputs, robot_num])

    # Generate reference trajectory
    cx, cy, cyaw, ref, target_ind = generate_reference_trajectory(x, dl)

    # Usage:
    bounds, constraints = set_bounds_and_constraints(mpc)
    
    predicted_trajectory = dict.fromkeys(range(robot_num),np.zeros([mpc.horizon, x.shape[0]]))
    for i in range(robot_num):
        predicted_trajectory[i] = np.full((mpc.horizon, 4), x[:,i])
    
    # input [throttle, steer (delta)]
    fig = plt.figure(1, dpi=90)
    ax = fig.add_subplot(111)

    for z in range(iterations):
        plt.cla()
        plt.gcf().canvas.mpl_connect('key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
        for i in range(robot_num):
            start_time = time.time()
            cx, cy, ref = update_paths(i, x, cx, cy, cyaw, target_ind, ref, dl)
            x, u, predicted_trajectory = mpc.mpc_control(i, x, u, bounds, constraints, ref, predicted_trajectory, mpc.cost_function3)
            
            if debug:
                print('Robot ' + str(i+1) + ' of ' + str(robot_num) + '   Time ' + str(round(time.time() - start_time,5)))

            plot_robot_trajectory(x, u, cx, cy, predicted_trajectory, ref, i)    
        
        plt.title('MPC 2D')
        utils.plot_map(width=width_init, height=height_init)
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)

def main1():
    print(__file__ + " start!!")

    dl = 1.0  
    cx, cy, cyaw, ck = get_switch_back_course(dl)

    for i in range(len(cyaw)):
        cyaw[i] = normalize_angle(cyaw[i])

    initial_state = np.array([cx[0], cy[0], cyaw[0], 0.0])
    
    # sim_run(options, ModelPredictiveControl, initial_state, cx, cy, cyaw, ck)

    # Simulator Options
    FIG_SIZE = options['FIG_SIZE'] # [Width, Height]
    OBSTACLES = options['OBSTACLES']

    mpc = ModelPredictiveControl(obs_x=cx[5:-1:12], obs_y=cy[5:-1:12])

    num_inputs = 2
    u = np.zeros(mpc.horizon*num_inputs)

    # Set bounds for inputs bounded optimization.
    bounds = []
    for i in range(mpc.horizon):
        bounds += [[min_acc, max_acc]]
        bounds += [[-max_steer, max_steer]]
    
    target_ind = 1
    ref = [cx[target_ind], cy[target_ind], cyaw[target_ind]]

    state_i = np.array([initial_state])
    u_i = np.array([[0,0]])
    sim_total = 1000
    predict_info = [state_i]

    # Total Figure
    fig = plt.figure(figsize=(FIG_SIZE[0], FIG_SIZE[1]))
    gs = gridspec.GridSpec(8,8)

    # Elevator plot settings.
    ax = fig.add_subplot(gs[:8, :8])

    plt.xlim(-3, 17)
    ax.set_ylim([-3, 17])
    plt.xticks(np.arange(0,11, step=2))
    plt.yticks(np.arange(0,11, step=2))
    plt.title('MPC 2D')

    for i in range(1,sim_total+1):
        u = np.delete(u,0)
        u = np.delete(u,0)
        u = np.append(u, u[-2])
        u = np.append(u, u[-2])
        start_time = time.time()

        # explore possibility of iterative MPC: for z in range(3):
        # Non-linear optimization.
        u_solution = minimize(mpc.cost_function2, u, (state_i[-1], ref),
                                method='SLSQP',
                                bounds=bounds,
                                tol = 1e-2)
        print('Step ' + str(i) + ' of ' + str(sim_total) + '   Time ' + str(round(time.time() - start_time,5)))
        u = u_solution.x
        y = mpc.plant_model(state_i[-1], mpc.dt, u[0], u[1])
        if (target_ind < len(cx)-1):
            if dist([y[0], y[1]], [cx[target_ind], cy[target_ind]]) < 4:
                target_ind+=1
                ref[0] = cx[target_ind]
                ref[1] = cy[target_ind]
                ref[2] = cyaw[target_ind]

        predicted_state = np.array([y])
        for j in range(1, mpc.horizon):
            if u[2*j]>max_acc or u[2*j]<min_acc:
                print('Acceleration out of bounds')
                break
            elif u[2*j+1]>max_steer or u[2*j+1]<-max_steer:
                print('Steering out of bounds')
                break
            predicted = mpc.plant_model(predicted_state[-1], mpc.dt, u[2*j], u[2*j+1])
            predicted_state = np.append(predicted_state, np.array([predicted]), axis=0)
        predict_info += [predicted_state]
        state_i = np.append(state_i, np.array([y]), axis=0)
        # print(f'yaw angle: {y[2]}')
        # print(f'speed: {y[3]}')
        # print(f'ref yaw angle: {ref[2]}')
        u_i = np.append(u_i, np.array([(u[0], u[1])]), axis=0)

        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
        if OBSTACLES:
            for zz in range(len(mpc.x_obs)):
                patch_obs = mpatches.Circle((mpc.x_obs[zz], mpc.y_obs[zz]),0.5)
                ax.add_patch(patch_obs)
        utils.plot_robot(state_i[i,0], state_i[i,1], state_i[i,2])
        utils.plot_robot(ref[0],ref[1],ref[2])
        plt.plot(cx, cy, "-r", label="course")
        plt.plot(predicted_state[:,0], predicted_state[:,1])
        plt.xlim(-10, 40)
        plt.ylim(-10, 40)
        plt.title('MPC 2D')
        plt.grid(True)
        plt.pause(0.0001)

# Uncomment the following line to run the main1() function.
# robot_num = 2
def main2():
    """
    Main function for controlling multiple robots using Model Predictive Control (MPC).

    Disclaimer: Please make sure to overwrite the robot number to 2 when using this function.

    Steps:
    1. Initialize the necessary variables and parameters.
    2. Create an instance of the ModelPredictiveControl class.
    3. Set the initial state and control inputs.
    4. Generate the reference trajectory for each robot.
    5. Plot the initial positions and reference trajectory.
    6. Set the bounds and constraints for the MPC.
    7. Initialize the predicted trajectory for each robot.
    8. Enter the main control loop:
        - Perform MPC control for each robot.
        - Plot the robot trajectory.
        - Update the predicted trajectory.
        - Plot the map and pause for visualization.
    """
    print(__file__ + " start!!")
    # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    iterations = 3000
    break_flag = False
    dl = 2
    
    mpc = ModelPredictiveControl(obs_x=[], obs_y=[])

    # x0, y, yaw, v, omega, model_type = utils.samplegrid(width_init, height_init, min_dist, robot_num, safety_init)
    # x = np.array([x0, y, yaw, v])
    x = np.array([[0, 10], [0, 0], [0, np.pi], [0,0]])
    num_inputs = 2
    u = np.zeros([mpc.horizon*num_inputs, robot_num])

    ref = [[10, 0, 0], [0, 0, -np.pi]]
    cx = []
    cy = []
    cyaw = []
    for i in range(robot_num):
        cx1, cy1, cyaw1, ck1 = get_straight_course(start=(x[0, i], x[1, i]), goal=(ref[i][0], ref[i][1]), dl=dl)

        cx.append(cx1)
        cy.append(cy1)
        cyaw.append(cyaw1)
        
        plt.plot(x[0, i], x[1, i], "xr")
        plt.plot(cx[i], cy[i], "-r", label="course")
    plt.show()

    # Usage:
    bounds, constraints = set_bounds_and_constraints(mpc)

    predicted_trajectory = dict.fromkeys(range(robot_num),np.zeros([mpc.horizon, x.shape[0]]))
    for i in range(robot_num):
        predicted_trajectory[i] = np.full((mpc.horizon, 4), x[:,i])
    
    # input [throttle, steer (delta)]
    fig = plt.figure(1, dpi=90)
    ax = fig.add_subplot(111)

    for z in range(iterations):
        plt.cla()
        plt.gcf().canvas.mpl_connect('key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
        
        for i in range(robot_num):
            x, u, predicted_trajectory = mpc.mpc_control(i, x, u, bounds, constraints, ref, predicted_trajectory, mpc.cost_function3)
            
            plot_robot_trajectory(x, u, cx, cy, predicted_trajectory, ref, i)
            
        plt.title('MPC 2D')
        utils.plot_map(width=width_init, height=height_init)
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)

def main3():
    """
    Main function for controlling multiple robots using Model Predictive Control (MPC).

    Steps:
    1. Initialize the necessary variables and parameters.
    2. Create an instance of the ModelPredictiveControl class.
    3. Set the initial state and control inputs.
    4. Generate the reference trajectory for each robot.
    5. Plot the initial positions and reference trajectory.
    6. Set the bounds and constraints for the MPC.
    7. Initialize the predicted trajectory for each robot.
    8. Enter the main control loop:
        - Check if the distance between the current position and the target is less than 5.
            - If yes, update the path and target.
        - Perform MPC control for each robot.
        - Plot the robot trajectory.
        - Update the predicted trajectory.
        - Plot the map and pause for visualization.
    """
    print(__file__ + " start!!")

    # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    iterations = 3000
    break_flag = False
    dl = 3
    
    # MPC initialization
    mpc = ModelPredictiveControl(obs_x=[], obs_y=[])
    
    initial_state = seed['initial_position']
    x0 = initial_state['x']
    y = initial_state['y']
    yaw = initial_state['yaw']
    v = initial_state['v']
    x = np.array([x0, y, yaw, v])
    num_inputs = 2
    u = np.zeros([mpc.horizon*num_inputs, robot_num])

    # Generate reference trajectory
    traj = seed['trajectories']
    cx = []
    cy = []
    ref = []

    for i in range(robot_num):
        x_buf = []
        y_buf = []
        for idx in range(len(traj[str(i)])):
            x_buf.append(traj[str(i)][idx][0])
            y_buf.append(traj[str(i)][idx][1])
        cx.append(x_buf)
        cy.append(y_buf)
        ref.append([cx[i][0], cy[i][0]])

    # Usage:
    bounds, constraints = set_bounds_and_constraints(mpc)
    
    predicted_trajectory = dict.fromkeys(range(robot_num),np.zeros([mpc.horizon, x.shape[0]]))
    for i in range(robot_num):
        predicted_trajectory[i] = np.full((mpc.horizon, 4), x[:,i])
    
    # input [throttle, steer (delta)]
    fig = plt.figure(1, dpi=90)
    ax = fig.add_subplot(111)

    for z in range(iterations):
        plt.cla()
        plt.gcf().canvas.mpl_connect('key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
        for i in range(robot_num):
            start_time = time.time()
            if dist([x[0, i], x[1, i]], point2=ref[i]) < 5:
                cx[i].pop(0)
                cy[i].pop(0)
                if not cx[i]:
                    print("Path complete")
                    return
                ref[i][0] = cx[i][0]
                ref[i][1] = cy[i][0]

            # cx, cy, ref = update_paths(i, x, cx, cy, cyaw, target_ind, ref, dl)
            x, u, predicted_trajectory = mpc.mpc_control(i, x, u, bounds, constraints, ref, predicted_trajectory, mpc.seed_cost)
            
            if debug:
                print('Robot ' + str(i+1) + ' of ' + str(robot_num) + '   Time ' + str(round(time.time() - start_time,5)))

            plot_robot_seed(x, u, predicted_trajectory, ref, i)    
        
        plt.title('MPC 2D')
        utils.plot_map(width=width_init, height=height_init)
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)

def main_seed():
    """
    Main function for controlling multiple robots using Model Predictive Control (MPC).

    Steps:
    1. Initialize the necessary variables and parameters.
    2. Create an instance of the ModelPredictiveControl class.
    3. Set the initial state and control inputs.
    4. Generate the reference trajectory for each robot.
    5. Plot the initial positions and reference trajectory.
    6. Set the bounds and constraints for the MPC.
    7. Initialize the predicted trajectory for each robot.
    8. Enter the main control loop:
        - Check if the distance between the current position and the target is less than 5.
            - If yes, update the path and target.
        - Perform MPC control for each robot.
        - Plot the robot trajectory.
        - Update the predicted trajectory.
        - Plot the map and pause for visualization.
    """
    print(__file__ + " start!!")

    # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    iterations = 3000
    break_flag = False
    dl = 3
    
    # MPC initialization
    mpc = ModelPredictiveControl([], [])
    
    initial_state = seed['initial_position']
    x0 = initial_state['x']
    y = initial_state['y']
    yaw = initial_state['yaw']
    v = initial_state['v']
    x = np.array([x0, y, yaw, v])
    num_inputs = 2
    u = np.zeros([mpc.horizon*num_inputs, robot_num])

    trajectory = np.zeros((x.shape[0], robot_num, 1))
    trajectory[:, :, 0] = x

    # Generate reference trajectory
    traj = seed['trajectories']
    cx = []
    cy = []
    ref = []

    for i in range(robot_num):
        x_buf = []
        y_buf = []
        for idx in range(len(traj[str(i)])):
            x_buf.append(traj[str(i)][idx][0])
            y_buf.append(traj[str(i)][idx][1])
        cx.append(x_buf)
        cy.append(y_buf)
        ref.append([cx[i][0], cy[i][0]])

    # Usage:
    bounds, constraints = set_bounds_and_constraints(mpc)
    
    predicted_trajectory = dict.fromkeys(range(robot_num),np.zeros([mpc.horizon, x.shape[0]]))
    for i in range(robot_num):
        predicted_trajectory[i] = np.full((mpc.horizon, 4), x[:,i])
    
    # input [throttle, steer (delta)]
    fig = plt.figure(1, dpi=90)
    ax = fig.add_subplot(111)

    # mpc = MPC_algorithm(cx, cy, ref, mpc, bounds, constraints, predicted_trajectory)
    mpc.cx = cx
    mpc.cy = cy
    mpc.ref = ref
    mpc.bounds = bounds
    mpc.constraints = constraints
    mpc.predicted_trajectory = predicted_trajectory

    for z in range(iterations):
        plt.cla()
        plt.gcf().canvas.mpl_connect('key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

        x, u = mpc.run_mpc(x, u)
        trajectory = np.dstack([trajectory, x])

        plt.title('MPC 2D')
        utils.plot_map(width=width_init, height=height_init)
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)

if __name__ == '__main__':
    main_seed()
    # main3()