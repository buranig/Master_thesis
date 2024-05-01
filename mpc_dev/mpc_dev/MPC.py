import numpy as np
import mpc_dev.cubic_spline_planner as cubic_spline_planner
import math
import random

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.optimize import minimize, NonlinearConstraint
import time
import os

from bumper_cars.classes.CarModel import State, CarModel
from bumper_cars.classes.Controller import Controller
from lar_utils import car_utils as utils
from lar_msgs.msg import CarControlStamped, CarStateStamped
from typing import List


# For the parameter file
import yaml
import json

np.random.seed(1)

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

class MPC_algorithm(Controller):
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

    dir_path = os.path.dirname(os.path.realpath(__file__))

    def __init__(self, controller_path:str, robot_num = 1):
        super().__init__(controller_path)
        self.robot_num = robot_num - 1
        self.dilated_traj = []
        self.predicted_trajectory = None
        self.u_hist = []

    # def __init__(self, obs_x, obs_y, x, robot_num=robot_num, cx=None, cy=None, ref=None, bounds=None, constraints=None):

        # Opening YAML file
        with open(controller_path, 'r') as openfile:
            # Reading from yaml file
            yaml_object = yaml.safe_load(openfile)

        self.horizon = horizon
        self.dt = dt_pred

        self.x_obs = []
        self.y_obs = []

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

        self.max_steer = yaml_object["MPC"]["max_steer"] # [rad] max steering angle
        self.max_speed = yaml_object["MPC"]["max_speed"] # [m/s]
        self.min_speed = yaml_object["MPC"]["min_speed"] # [m/s]
        self.max_acc = yaml_object["MPC"]["max_acc"] # [m/ss]
        self.min_acc = yaml_object["MPC"]["min_acc"] # [m/ss]
        self.dt = yaml_object["Controller"]["dt"] # [s] Time tick for motion prediction
        self.horizon = yaml_object["MPC"]["horizon"] # [s] Time horizon for motion prediction
        self.dt_pred = yaml_object["MPC"]["dt_pred"] # [s] Time tick for motion prediction
        self.safety_radius = yaml_object["MPC"]["safety_radius"] # [m] Safety radius for obstacle avoidance

        self.L = yaml_object["Car_model"]["L"]  # [m] Wheel base of vehicle
        self.Lr = yaml_object["Car_model"]["Lr"]  # [m]
        self.Lf = self.L - self.Lr  # [m]
        self.WB = yaml_object["Controller"]["WB"] # Wheel base
        self.robot_num = yaml_object["robot_num"]
        self.safety_init = yaml_object["safety"]
        self.width_init = yaml_object["width"]
        self.height_init = yaml_object["height"]
        self.min_dist = yaml_object["min_dist"]
        self.to_goal_stop_distance = yaml_object["to_goal_stop_distance"]

        self.show_animation = yaml_object["show_animation"]
        self.debug = False
        self.boundary_points = np.array([-self.width_init/2, self.width_init/2, -self.height_init/2, self.height_init/2])
        self.check_collision_bool = False
        self.add_noise = yaml_object["add_noise"]
        self.noise_scale_param = yaml_object["noise_scale_param"]

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
            noise = np.concatenate([np.random.normal(0, 0.21*noise_scale_param, 2).reshape(1, 2), np.random.normal(0, np.radians(5)*noise_scale_param, 1).reshape(1,1), np.random.normal(0, 0.2*noise_scale_param, 1).reshape(1,1)], axis=1)
            noisy_pos = x1 + noise[0]
            plt.plot(noisy_pos[0], noisy_pos[1], "x", color=color_dict[i], markersize=10)
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
        # x1 = self.plant_model(x1, dt, u1[0], u1[1])
        x1 = utils.motion(x1, u1, dt)
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
