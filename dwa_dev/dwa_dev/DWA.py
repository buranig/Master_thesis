import matplotlib.pyplot as plt
import numpy as np
import math
from bumper_cars.classes.Controller import Controller
# from bumper_cars.include.Controller import Controller
# from bumper_cars import Controller
from lar_utils import car_utils as utils
# For the parameter file
import pathlib
import yaml
import json

from shapely.geometry import Point, LineString
from shapely import distance
import time
import os



class DWA_algorithm(Controller):
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
        u_hist (list): List of control inputs history for each robot.
        reached_goal (list): List of flags indicating if each robot has reached its goal.
        computational_time (list): List of computational times for each iteration.

    Methods:
        run_dwa: Runs the DWA algorithm.
        update_robot_state: Updates the state of a robot based on its current state, control input, and environment information.
        dwa_control: Dynamic Window Approach control.
        calc_control_and_trajectory: Calculates the final input with the dynamic window.
    """

    def __init__(self, controller_path, robot_num):
        super().__init__(self, controller_path)
        self.robot_num = robot_num
        self.targets = None
        self.dilated_traj = None
        self.predicted_trajectory = None
        self.u_hist = []

        # Opening YAML file
        with open(controller_path, 'r') as openfile:
            # Reading from yaml file
            yaml_object = yaml.safe_load(openfile)

        self.max_steer = yaml_object["DWA"]["max_steer"] # [rad] max steering angle
        self.max_speed = yaml_object["DWA"]["max_speed"] # [m/s]
        self.min_speed = yaml_object["DWA"]["min_speed"] # [m/s]
        self.v_resolution = yaml_object["DWA"]["v_resolution"] # [m/s]
        self.delta_resolution = math.radians(yaml_object["DWA"]["delta_resolution"])# [rad/s]
        self.max_acc = yaml_object["DWA"]["max_acc"] # [m/ss]
        self.min_acc = yaml_object["DWA"]["min_acc"] # [m/ss]
        self.dt = yaml_object["Controller"]["dt"] # [s] Time tick for motion prediction
        self.predict_time = yaml_object["DWA"]["predict_time"] # [s]
        self.to_goal_cost_gain = yaml_object["DWA"]["to_goal_cost_gain"]
        self.speed_cost_gain = yaml_object["DWA"]["speed_cost_gain"]
        self.obstacle_cost_gain = yaml_object["DWA"]["obstacle_cost_gain"]
        self.heading_cost_gain = yaml_object["DWA"]["heading_cost_gain"]
        self.robot_stuck_flag_cons = yaml_object["DWA"]["robot_stuck_flag_cons"]
        self.dilation_factor = yaml_object["DWA"]["dilation_factor"]
        self.L = yaml_object["Car_model"]["L"]  # [m] Wheel base of vehicle
        self.Lr = self.L / 2.0  # [m]
        self.Lf = self.L - self.Lr
        self.Cf = yaml_object["Car_model"]["Cf"]  # N/rad
        self.Cr = yaml_object["Car_model"]["Cr"] # N/rad
        self.Iz = yaml_object["Car_model"]["Iz"]  # kg/m2
        self.m = yaml_object["Car_model"]["m"]  # kg
        
        # Aerodynamic and friction coefficients
        self.c_a = yaml_object["Car_model"]["c_a"]
        self.c_r1 = yaml_object["Car_model"]["c_r1"]
        self.WB = yaml_object["Controller"]["WB"] # Wheel base
        self.L_d = yaml_object["Controller"]["L_d"]  # [m] look-ahead distance
        self.robot_num = yaml_object["robot_num"]
        self.safety_init = yaml_object["safety"]
        self.width_init = yaml_object["width"]
        self.height_init = yaml_object["height"]
        self.min_dist = yaml_object["min_dist"]
        self.to_goal_stop_distance = yaml_object["to_goal_stop_distance"]
        self.update_dist = 2
        

        self.robot_num = yaml_object["robot_num"]
        self.timer_freq = yaml_object["timer_freq"]

        self.show_animation = yaml_object["show_animation"]
        self.boundary_points = np.array([-self.width_init/2, self.width_init/2, -self.height_init/2, self.height_init/2])
        self.check_collision_bool = False
        self.add_noise = yaml_object["add_noise"]
        self.noise_scale_param = yaml_object["noise_scale_param"]
        np.random.seed(1)

        
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(dir_path + '/../config/trajectories.json', 'r') as file:
            self.trajs = json.load(file)

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

        x, u = self.update_robot_state(x, u, self.dt, i) # State, Input, deltaT, robotNum

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
        if self.add_noise:
            noise = np.concatenate([np.random.normal(0, 0.21*self.noise_scale_param, 2).reshape(1, 2), np.random.normal(0, np.radians(5)*self.noise_scale_param, 1).reshape(1,1), np.random.normal(0, 0.2*self.noise_scale_param, 1).reshape(1,1)], axis=1)
            noisy_pos = x1 + noise[0]
            u1, predicted_trajectory1, u_history = self.dwa_control(noisy_pos, ob, i)
        else:
            u1, predicted_trajectory1, u_history = self.dwa_control(x1, ob, i)
        self.dilated_traj[i] = LineString(zip(predicted_trajectory1[:, 0], predicted_trajectory1[:, 1])).buffer(self.dilation_factor, cap_style=3)

        # Collision check
        x1 = utils.motion(x1, u1, dt)
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
            dw = self.calc_dynamic_window()
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
            nearest = utils.find_nearest(np.arange(self.min_speed, self.max_speed+self.v_resolution, self.v_resolution), x[3])

            for a in np.arange(dw[0], dw[1]+self.v_resolution, self.v_resolution):
                for delta in np.arange(dw[2], dw[3]+self.delta_resolution, self.delta_resolution):

                    # old_time = time.time()
                    geom = self.trajs[str(nearest)][str(a)][str(delta)]
                    geom = np.array(geom)
                    geom[:,0:2] = (geom[:,0:2]) @ utils.rotateMatrix(np.radians(90)-x[2]) + [x[0],x[1]]
                    # print(time.time()-old_time)
                    geom[:,2] = geom[:,2] + x[2] - np.pi/2 #bringing also the yaw angle in the new frame

                    # trajectory = predict_trajectory(x_init, a, delta)
                    trajectory = geom
                    # calc cost

                    to_goal_cost = self.to_goal_cost_gain * self.calc_to_goal_cost(trajectory, goal)
                    speed_cost = self.speed_cost_gain * (self.max_speed - trajectory[-1, 3])
                    # if trajectory[-1, 3] <= 0.0:
                    #     speed_cost = 5
                    # else:
                    #     speed_cost = 0.0
                    ob_cost = self.obstacle_cost_gain * self.calc_obstacle_cost(trajectory, ob)
                    heading_cost = self.heading_cost_gain * self.calc_to_goal_heading_cost(trajectory, goal)
                    final_cost = to_goal_cost + ob_cost + speed_cost #+ heading_cost #+ speed_cost

                    # search minimum trajectory
                    if min_cost >= final_cost:
                        min_cost = final_cost
                        best_u = [a, delta]
                        best_trajectory = trajectory
                        u_history = [[a, delta] for _ in range(len(trajectory-1))]

                        # Shouldn't get stuck anyways
                        # if abs(best_u[0]) < robot_stuck_flag_cons \
                        #         and abs(x[2]) < robot_stuck_flag_cons:
                        #     # to ensure the robot do not get stuck in
                        #     # best v=0 m/s (in front of an obstacle) and
                        #     # best omega=0 rad/s (heading to the goal with
                        #     # angle difference of 0)
                        #     best_u[1] = -max_steer
                        #     best_trajectory = trajectory
                        #     u_history = [delta]*len(trajectory)
            # print(time.time()-old_time)
            if len(u_buf) > 4:              
                u_buf.pop(0)
                trajectory_buf = trajectory_buf[1:]

            #     to_goal_cost = to_goal_cost_gain * calc_to_goal_cost(trajectory_buf, goal)
            #     # speed_cost = speed_cost_gain * (max_speed - trajectory[-1, 3])
            #     ob_cost = obstacle_cost_gain * calc_obstacle_cost(trajectory_buf, ob)
            #     heading_cost = heading_cost_gain * calc_to_goal_heading_cost(trajectory_buf, goal)
            #     final_cost = to_goal_cost + ob_cost + heading_cost #+ speed_cost

                if min_cost >= final_cost:
                    min_cost = final_cost
                    best_u = u_buf[0]
                    best_trajectory = trajectory_buf
                    u_history = u_buf

            elif min_cost == np.inf:
                # emergency stop
                print("Emergency stop")
                if x[3]>0:
                    best_u = [self.min_acc, 0]
                else:
                    best_u = [self.max_acc, 0]
                best_trajectory = np.array([x[0:3], x[0:3]])
                u_history = [self.min_acc, 0]

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
        if x[0,i]>= self.boundary_points[1]-self.WB/2 or x[0,i]<= self.boundary_points[0]+self.WB/2 or x[1,i]>=self.boundary_points[3]-self.WB/2 or x[1,i]<=self.boundary_points[2]+self.WB/2:
            if self.check_collision_bool:
                raise Exception('Collision')
            else:
                print("Collision detected")
                u[:, i] = np.zeros(2)
                x[3, i] = 0


        for idx in range(self.robot_num):
            if idx == i:
                continue
            if utils.dist([x[0,i], x[1,i]], [x[0, idx], x[1, idx]]) <= self.WB/2:
                if self.check_collision_bool:
                    raise Exception('Collision')
                else:
                    print("Collision detected")
                    u[:, i] = np.zeros(2)
                    x[3, i] = 0
        return u, x

    def calc_dynamic_window(self):
        """
        Calculate the dynamic window based on the current state.

        Args:
            x (list): Current state [x(m), y(m), yaw(rad), v(m/s), delta(rad)].

        Returns:
            list: Dynamic window [min_throttle, max_throttle, min_steer, max_steer].
        """

        Vs = [self.min_acc, self.max_acc,
            -self.max_steer, self.max_steer]

        dw = [Vs[0], Vs[1], Vs[2], Vs[3]]

        return dw

    def predict_trajectory(self, x_init, a, delta):
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
        while time < self.predict_time:
            x = utils.motion(x, [a, delta], self.dt)
            trajectory = np.vstack((trajectory, x))
            time += self.dt
        return trajectory

    def calc_obstacle_cost(self,trajectory, ob):
        """
        Calculate the obstacle cost.

        Args:
            trajectory (numpy.ndarray): Trajectory.
            ob (list): List of obstacles.

        Returns:
            float: Obstacle cost.
        """
        minxp = min(abs(self.width_init/2-trajectory[:, 0]))
        minxn = min(abs(-self.width_init/2-trajectory[:, 0]))
        minyp = min(abs(self.height_init/2-trajectory[:, 1]))
        minyn = min(abs(-self.height_init/2-trajectory[:, 1]))
        min_distance = min(minxp, minxn, minyp, minyn)

        line = LineString(zip(trajectory[:, 0], trajectory[:, 1]))
        dilated = line.buffer(self.dilation_factor, cap_style=3)

        x = trajectory[:, 0]
        y = trajectory[:, 1]

        # check if the trajectory is out of bounds
        if any(element < -self.width_init/2+self.WB/2 or element > self.width_init/2-self.WB/2 for element in x):
            return np.inf
        if any(element < -self.height_init/2+self.WB/2 or element > self.height_init/2-self.WB/2 for element in y):
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