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

max_steer = json_object["MPC"]["max_steer"] # [rad] max steering angle
max_speed = json_object["MPC"]["max_speed"] # [m/s]
min_speed = json_object["MPC"]["min_speed"] # [m/s]
max_acc = json_object["MPC"]["max_acc"] # [m/ss]
min_acc = json_object["MPC"]["min_acc"] # [m/ss]
dt = json_object["Controller"]["dt"] # [s] Time tick for motion prediction
horizon = json_object["MPC"]["horizon"] # [s] Time horizon for motion prediction
dt_pred = json_object["MPC"]["dt_pred"] # [s] Time tick for motion prediction
safety_radius = json_object["MPC"]["safety_radius"] # [m] Safety radius for obstacle avoidance

L = json_object["Car_model"]["L"]  # [m] Wheel base of vehicle
Lr = L / 2.0  # [m]
Lf = L - Lr  # [m]
WB = json_object["Controller"]["WB"] # Wheel base
robot_num = json_object["robot_num"]
safety_init = json_object["safety"]
width_init = json_object["width"]
height_init = json_object["height"]
min_dist = json_object["min_dist"]

show_animation = True
debug = False
boundary_points = np.array([-width_init/2, width_init/2, -height_init/2, height_init/2])
check_collision_bool = False
add_noise = False 
np.random.seed(1)

color_dict = {0: 'r', 1: 'b', 2: 'g', 3: 'y', 4: 'm', 5: 'c', 6: 'k'}

# Simulator options.
options = {}
options['FIG_SIZE'] = [8,8]
options['OBSTACLES'] = True

with open('/home/giacomo/thesis_ws/src/seeds/circular_seed_10.json', 'r') as file:
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
        x_obs (list): The x-coordinates of the obstacles.
        y_obs (list): The y-coordinates of the obstacles.
        initial_state (list): The initial state of the system.
        safety_radius (float): The safety radius for obstacle avoidance.
        cx (list): The x-coordinates of the path for each robot.
        cy (list): The y-coordinates of the path for each robot.
        ref (list): The reference points for each robot.
        bounds (list): The bounds for the optimization problem.
        constraints (list): The constraints for the optimization problem.
        predicted_trajectory (list): The predicted trajectories of all robots.
        reached_goal (list): A list of flags indicating whether each robot has reached the goal.
        computational_time (list): The computational time for each iteration of the MPC controller.

    Methods:
        plant_model: Computes the next state of the system based on the current state and control inputs.
        cost_function: Computes the cost associated with a given control sequence.
        cost_function2: Computes the cost associated with a given control sequence.
        cost_function3: Computes the cost associated with a given control sequence.
        seed_cost: Computes the cost associated with a given control sequence.
        propagation1: Propagates the system state in the x-direction based on a given control sequence.
        propagation2: Propagates the system state in the y-direction based on a given control sequence.
        propagation3: Computes the distance between the system state and the obstacles based on a given control sequence.
        run_mpc: Runs the MPC controller for a given number of iterations.
        go_to_goal: Moves the robot to the goal position.
        mpc_control: Computes the control inputs for the MPC controller.
        update_obstacles: Update the obstacles for the model predictive control (MPC) algorithm.

    """

    def __init__(self, obs_x, obs_y, x, robot_num=robot_num, cx=None, cy=None, ref=None, bounds=None, constraints=None):
        self.horizon = horizon
        self.dt = dt_pred

        self.x_obs = obs_x
        self.y_obs = obs_y

        self.initial_state = None
        self.safety_radius = safety_radius
        self.robot_num = robot_num

        self.cx = cx
        self.cy = cy
        self.ref = ref
        self.bounds = bounds
        self.constraints = constraints
        self.predicted_trajectory = dict.fromkeys(range(robot_num),np.zeros([self.horizon, x.shape[0]]))
        for i in range(robot_num):
            self.predicted_trajectory[i] = np.full((self.horizon, 4), x[:,i])

        self.reached_goal = [False]*robot_num
        self.computational_time = []
        self.bounds, self.constraints = self.set_bounds_and_constraints()

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
        pedal = np.clip(pedal, min_acc, max_acc)
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

    def cost_function3(self, u, *args):
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
            cost +=  5 * distance_to_goal

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

        # cost += (state[3])**2
        distance_to_goal = np.sqrt((ref[0] - state[0])**2 + (ref[1] - state[1])**2)
        cost += 0.2*distance_to_goal
        return cost
    
    def seed_cost(self, u, *args):
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
    
    def set_bounds_and_constraints(self):
            """
            Set the bounds and constraints for the optimization problem.

            Returns:
                bounds (list): List of bounds for the inputs.
                constraints (list): List of constraints for the optimization problem.
            """
                
            bounds = []
            constraints = []
            # Set bounds for inputs bounded optimization.
            for i in range(self.horizon):
                bounds += [[min_acc, max_acc]]
                bounds += [[-max_steer, max_steer]]

            constraint1 = NonlinearConstraint(fun=self.propagation1, lb=-width_init/2 + self.safety_radius, ub=width_init/2 - self.safety_radius)
            constraint2 = NonlinearConstraint(fun=self.propagation2, lb=-height_init/2 + self.safety_radius, ub=height_init/2 - self.safety_radius)
            if len(self.x_obs) > 0 or len(self.y_obs) > 0:
                constraint3 = NonlinearConstraint(fun=self.propagation3, lb=0, ub=np.inf)
                constraints = [constraint1, constraint2, constraint3]
            else:
                constraints = [constraint1, constraint2]

            return bounds, constraints

    def run_mpc(self, x, u, break_flag):
        for i in range(self.robot_num):
            start_time = time.time()
            if dist([x[0, i], x[1, i]], point2=self.ref[i]) < 2:
                self.cx[i].pop(0)
                self.cy[i].pop(0)
                if not self.cx[i]:
                    print("Path complete")
                    return x, u, True
                
                self.ref[i][0] = self.cx[i][0]
                self.ref[i][1] = self.cy[i][0]

            # cx, cy, ref = update_paths(i, x, cx, cy, cyaw, target_ind, ref, dl)
            t_prev = time.time()
            x, u = self.mpc_control(i, x, u, self.ref, self.seed_cost)
            self.computational_time.append(time.time() - t_prev)

            if debug:
                print('Robot ' + str(i+1) + ' of ' + str(self.robot_num) + '   Time ' + str(round(time.time() - start_time,5)))

            plot_robot_seed(x, u, self.predicted_trajectory, self.ref, i) 
        
        return x, u, break_flag
    
    def go_to_goal(self, x, u, break_flag):
        for i in range(self.robot_num):
            self.check_collision(x, u, i)
            if not self.reached_goal[i]:                
                # If goal is reached, stop the robot
                if check_goal_reached(x, self.ref, i, distance=1.5):
                    # u[:, i] = np.zeros(2)
                    x[3, i] = 0
                    u[:,i] = 0
                    self.reached_goal[i] = True
                else:
                    t_prev = time.time()
                    x, u = self.mpc_control(i, x, u, self.ref, self.seed_cost)
                    self.computational_time.append(time.time() - t_prev)
                    
            # If we want the robot to disappear when it reaches the goal, indent one more time
            if all(self.reached_goal):
                break_flag = True

            if show_animation:
                plot_robot_seed(x, u, self.predicted_trajectory, self.ref, i)
        return x, u, break_flag
    
    def mpc_control(self, i, x, u, ref, cost_function):
        """
        Perform model predictive control (MPC) for a given time step.

        Args:
            i (int): The current time step.
            x (numpy.ndarray): The state vector.
            u (numpy.ndarray): The control vector.
            bounds (list): The bounds on the control inputs.
            constraints (list): The constraints on the control inputs.
            ref (numpy.ndarray): The reference trajectory.
            predicted_trajectory (numpy.ndarray): The predicted trajectory.
            cost_function (function): The cost function to be minimized.

        Returns:
            tuple: A tuple containing the updated state vector, control vector, and predicted trajectory.

        """
        x1 = x[:, i]
        u1 = u[:,i]
        u1 = np.delete(u1,0)
        u1 = np.delete(u1,0)
        u1 = np.append(u1, u1[-2])
        u1 = np.append(u1, u1[-2])  

        if add_noise:
            noise = np.concatenate([np.random.normal(0, 0.3, 2).reshape(1, 2), np.random.normal(0, np.radians(5), 1).reshape(1,1), np.zeros((1,1))], axis=1)
            noisy_pos = x1 + noise[0]
            plt.plot(noisy_pos[0], noisy_pos[1], "x" + color_dict[i], markersize=10)
            self.update_obstacles(i, noisy_pos, x, self.predicted_trajectory) 
            self.bounds, self.constraints = self.set_bounds_and_constraints()
            self.initial_state = noisy_pos
            u_solution = minimize(cost_function, u1, (noisy_pos, ref[i]),
                            method='SLSQP',
                            bounds=self.bounds,
                            constraints=self.constraints,
                            tol = 1e-1)
        else:
            self.update_obstacles(i, x1, x, self.predicted_trajectory) 
            self.bounds, self.constraints = self.set_bounds_and_constraints()
            self.initial_state = x1
            u_solution = minimize(cost_function, u1, (x1, ref[i]),
                            method='SLSQP',
                            bounds=self.bounds,
                            constraints=self.constraints,
                            tol = 1e-1)
               
        u1 = u_solution.x
        if u1[0]>max_acc or u1[0]<min_acc:
            print(f'Acceleration out of bounds: {u1[0]}')
            u1[0] = np.clip(u1[0], min_acc, max_acc)
        x1 = self.plant_model(x1, dt, u1[0], u1[1])
        x[:, i] = x1
        u[:, i] = u1
       
        if add_noise:
            predicted_state = np.array([noisy_pos])

            for j in range(1, self.horizon):
                predicted = self.plant_model(predicted_state[-1], self.dt, u1[2*j], u1[2*j+1])
                predicted_state = np.append(predicted_state, np.array([predicted]), axis=0)
        else:
            predicted_state = np.array([x1])

            for j in range(1, self.horizon):
                predicted = self.plant_model(predicted_state[-1], self.dt, u1[2*j], u1[2*j+1])
                predicted_state = np.append(predicted_state, np.array([predicted]), axis=0)
        self.predicted_trajectory[i] = predicted_state

        return x, u
    
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

        for idx in range(self.robot_num):
            if idx == i:
                continue
            
            next_robot_state = predicted_trajectory[idx]
            self.x_obs.append(next_robot_state[0:-1:5, 0])
            self.y_obs.append(next_robot_state[0:-1:5, 1])
        self.x_obs = [item for sublist in self.x_obs for item in sublist]
        self.y_obs = [item for sublist in self.y_obs for item in sublist]

    def check_collision(self, x, u, i):
        """
        Checks for collision between the robot at index i and other robots.

        Args:
            x (numpy.ndarray): State vector of shape (4, N), where N is the number of time steps.
            i (int): Index of the robot to check collision for.

        Raises:
            Exception: If collision is detected.

        """
        if x[0,i]>=boundary_points[1]-WB or x[0,i]<=boundary_points[0]+WB or x[1,i]>=boundary_points[3]-WB or x[1,i]<=boundary_points[2]+WB:
            if check_collision_bool:
                raise Exception('Collision')
            else:
                print("Collision detected")
                x[3, i] = 0
                u[:,i] = 0
                self.reached_goal[i] = True

        for idx in range(self.robot_num):
            if idx == i:
                continue
            if utils.dist([x[0,i], x[1,i]], [x[0, idx], x[1, idx]]) <= WB:
                if check_collision_bool:
                    raise Exception('Collision')
                else:
                    print("Collision detected")
                    x[3, i] = 0
                    u[:,i] = 0
                    self.reached_goal[i] = True

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
    plt.plot(predicted_trajectory[i][:, 0], predicted_trajectory[i][:, 1], "-"+color_dict[i], label="Trajectory")
    # plt.plot(x[0, i], x[1, i], "xr")
    plt.plot(targets[i][0], targets[i][1], "x"+color_dict[i], label="Target")
    plt.plot(cx[i], cy[i], "--"+color_dict[i], label="Course")
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

    x0, y, yaw, v, omega, model_type = utils.samplegrid(width_init, height_init, min_dist, robot_num, safety_init)
    x = np.array([x0, y, yaw, v])
    # MPC initialization
    mpc = ModelPredictiveControl(obs_x=[], obs_y=[], x=x, robot_num=robot_num)
    
    num_inputs = 2
    u = np.zeros([mpc.horizon*num_inputs, robot_num])

    # Generate reference trajectory
    cx, cy, cyaw, ref, target_ind = generate_reference_trajectory(x, dl)
    
    # input [throttle, steer (delta)]
    fig = plt.figure(1, dpi=90, figsize=(10,10))
    ax = fig.add_subplot(111)

    for z in range(iterations):
        plt.cla()
        plt.gcf().canvas.mpl_connect('key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
        for i in range(robot_num):
            start_time = time.time()
            cx, cy, ref = update_paths(i, x, cx, cy, cyaw, target_ind, ref, dl)
            x, u = mpc.mpc_control(i, x, u, ref, mpc.cost_function3)
            
            if debug:
                print('Robot ' + str(i+1) + ' of ' + str(robot_num) + '   Time ' + str(round(time.time() - start_time,5)))

            plot_robot_trajectory(x, u, cx, cy, mpc.predicted_trajectory, ref, i)    
        
        plt.title('MPC 2D')
        utils.plot_map(width=width_init, height=height_init)
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)

def main1():
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

    # x0, y, yaw, v, omega, model_type = utils.samplegrid(width_init, height_init, min_dist, robot_num, safety_init)
    # x = np.array([x0, y, yaw, v])
    x = np.array([[0.0], [0.0], [0.0], [0.0]])

    mpc = ModelPredictiveControl(obs_x=[3], obs_y=[0.01], x=x, robot_num=x.shape[1])

    num_inputs = 2
    u = np.zeros([mpc.horizon*num_inputs, x.shape[1]])

    ref = [[7.5, 0, 0]]
    cx = []
    cy = []
    cyaw = []
    for i in range(x.shape[1]):
        cx1, cy1, cyaw1, ck1 = get_straight_course(start=(x[0, i], x[1, i]), goal=(ref[i][0], ref[i][1]), dl=dl)

        cx.append(cx1)
        cy.append(cy1)
        cyaw.append(cyaw1)
        
        plt.plot(x[0, i], x[1, i], "xr")
        plt.plot(cx[i], cy[i], "-r", label="course")
    plt.show()
    
    # input [throttle, steer (delta)]
    fig = plt.figure(1, dpi=90, figsize=(12.5,6))
    ax = fig.add_subplot(111)
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 20
    
    z = 0
    # while z <3000:
    plt.cla()
    plt.gcf().canvas.mpl_connect('key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
    for i in range(x.shape[1]):
        x1 = x[:, i]
        u1 = u[:,i]
        u1 = np.delete(u1,0)
        u1 = np.delete(u1,0)
        u1 = np.append(u1, u1[-2])
        u1 = np.append(u1, u1[-2])  

        # self.update_obstacles(i, x1, x, self.predicted_trajectory) 
        mpc.initial_state = x1
        u_solution = minimize(mpc.cost_function3, u1, (x1, ref[i]),
                        method='SLSQP',
                        bounds=mpc.bounds,
                        constraints=mpc.constraints,
                        tol = 1e-3)
            
        u1 = u_solution.x
        # x1 = mpc.plant_model(x1, dt, u1[0], u1[1])
        x[:, i] = x1
        u[:, i] = u1
        print(f'Acceleration: {u1[0]}')

        predicted_state = np.array([x1])

        for j in range(1, mpc.horizon):
            predicted = mpc.plant_model(predicted_state[-1], mpc.dt, u1[2*j], u1[2*j+1])
            predicted_state = np.append(predicted_state, np.array([predicted]), axis=0)
        mpc.predicted_trajectory[i] = predicted_state

        plot_robot_trajectory(x, u, cx, cy, mpc.predicted_trajectory, ref, i)
        for idx in range(len(mpc.x_obs)):
            plt.plot(mpc.x_obs[idx], mpc.y_obs[idx], "ok", markersize=30)
        
    z += 1

    plt.xlim(-2, 10.5)
    plt.ylim(-3, 3)
    plt.legend()
    plt.grid(True)
    plt.xlabel("x [m]", fontdict={'size': 20, 'family': 'serif'})
    plt.ylabel("y [m]", fontdict={'size': 20, 'family': 'serif'})
    plt.title('MPC Trajectory Generation')
    # plt.grid(True)
    # plt.pause(0.0001)
    plt.show()
    u1 = u_solution.x
    # x1 = self.plant_model(x1, dt, u1[0], u1[1])
    x[:, i] = x1
    u[:, i] = u1

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

    # x0, y, yaw, v, omega, model_type = utils.samplegrid(width_init, height_init, min_dist, robot_num, safety_init)
    # x = np.array([x0, y, yaw, v])
    x = np.array([[0, 10], [0, 0], [0, np.pi], [0,0]])

    mpc = ModelPredictiveControl(obs_x=[], obs_y=[], x=x, robot_num=x.shape[1])

    num_inputs = 2
    u = np.zeros([mpc.horizon*num_inputs, x.shape[1]])

    ref = [[10, 0, 0], [0, 0, -np.pi]]
    cx = []
    cy = []
    cyaw = []
    for i in range(x.shape[1]):
        cx1, cy1, cyaw1, ck1 = get_straight_course(start=(x[0, i], x[1, i]), goal=(ref[i][0], ref[i][1]), dl=dl)

        cx.append(cx1)
        cy.append(cy1)
        cyaw.append(cyaw1)
        
        plt.plot(x[0, i], x[1, i], "xr")
        plt.plot(cx[i], cy[i], "-r", label="course")
    plt.show()
    
    # input [throttle, steer (delta)]
    fig = plt.figure(1, dpi=90, figsize=(10,10))
    ax = fig.add_subplot(111)

    for z in range(iterations):
        plt.cla()
        plt.gcf().canvas.mpl_connect('key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
        
        for i in range(x.shape[1]):
            x, u = mpc.mpc_control(i, x, u, ref, mpc.cost_function3)
            
            plot_robot_trajectory(x, u, cx, cy, mpc.predicted_trajectory, ref, i)
            
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
    
    initial_state = seed['initial_position']
    x0 = initial_state['x']
    y = initial_state['y']
    yaw = initial_state['yaw']
    v = initial_state['v']
    x = np.array([x0, y, yaw, v])

    mpc = ModelPredictiveControl(obs_x=[], obs_y=[], x=x, robot_num=robot_num)


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
    
    # input [throttle, steer (delta)]
    fig = plt.figure(1, dpi=90, figsize=(10,10))
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
            x, u= mpc.mpc_control(i, x, u, ref, mpc.seed_cost)
            
            if debug:
                print('Robot ' + str(i+1) + ' of ' + str(robot_num) + '   Time ' + str(round(time.time() - start_time,5)))

            plot_robot_seed(x, u, mpc.predicted_trajectory, ref, i)    
        
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
    
    
    initial_state = seed['initial_position']
    x0 = initial_state['x']
    y = initial_state['y']
    yaw = initial_state['yaw']
    v = initial_state['v']
    x = np.array([x0, y, yaw, v])

    mpc = ModelPredictiveControl(obs_x=[], obs_y=[], x=x, robot_num=robot_num)

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
    
    # input [throttle, steer (delta)]
    fig = plt.figure(1, dpi=90, figsize=(10,10))
    ax = fig.add_subplot(111)

    mpc.cx = cx
    mpc.cy = cy
    mpc.ref = ref

    for z in range(iterations):
        plt.cla()
        plt.gcf().canvas.mpl_connect('key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

        # x, u, break_flag = mpc.run_mpc(x, u, break_flag)
        x, u, break_flag = mpc.go_to_goal(x, u, break_flag)
        trajectory = np.dstack([trajectory, x])

        plt.title('MPC 2D')
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
    # main2()