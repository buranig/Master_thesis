import matplotlib.pyplot as plt
import numpy as np
import math

from bumper_cars.classes.CarModel import State, CarModel
from bumper_cars.classes.Controller import Controller
from lar_utils import car_utils as utils
from lar_msgs.msg import CarControlStamped, CarStateStamped
from typing import List


# For the parameter file
import pathlib
import yaml
import json

from shapely.geometry import Point, LineString
from shapely import distance
import time
import os


# For drawing purposes
from shapely.plotting import plot_polygon

color_dict = {0: 'r', 1: 'b', 2: 'g', 3: 'y', 4: 'm', 5: 'c', 6: 'k', 7: 'tab:orange', 8: 'tab:brown', 9: 'tab:gray', 10: 'tab:olive', 11: 'tab:pink', 12: 'tab:purple', 13: 'tab:red', 14: 'tab:blue', 15: 'tab:green'}

fig = plt.figure(1, dpi=90, figsize=(10,10))
ax= fig.add_subplot(111)



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

    robot_num = None
    targets = None
    dilated_traj = None
    predicted_trajectory = None
    u_hist = None

    max_steer = None
    max_speed = None
    min_speed = None
    v_resolution = None
    delta_resolution = None
    max_acc = None
    min_acc = None
    dt = None
    predict_time = None
    to_goal_cost_gain = None
    speed_cost_gain = None
    obstacle_cost_gain = None
    heading_cost_gain = None
    robot_stuck_flag_cons = None
    dilation_factor = None
    L = None
    Lr = None
    Lf = None
    Cf = None
    Cr = None
    Iz = None
    m = None
    
    # Aerodynamic and friction coefficients
    c_a = None
    c_r1 = None
    WB = None
    L_d = None
    safety_init = None
    width_init = None
    height_init = None
    min_dist = None
    to_goal_stop_distance = None
    update_dist = None
    
    timer_freq = None

    show_animation = None
    boundary_points = None
    check_collision_bool = None
    add_noise = None
    noise_scale_param = None
    

    def __init__(self, controller_path:str, robot_num = 1):
        super().__init__(controller_path)
        self.robot_num = robot_num - 1
        self.dilated_traj = []
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
        self.safety_init = yaml_object["Controller"]["safety"]
        self.width_init = yaml_object["width"]
        self.height_init = yaml_object["height"]
        self.min_dist = yaml_object["min_dist"]
        self.to_goal_stop_distance = yaml_object["to_goal_stop_distance"]
        self.update_dist = 2
        

        self.timer_freq = yaml_object["timer_freq"]

        self.show_animation = yaml_object["show_animation"]
        self.boundary_points = np.array([-self.width_init/2, self.width_init/2, -self.height_init/2, self.height_init/2])
        self.check_collision_bool = False
        self.add_noise = yaml_object["Controller"]["add_noise"]
        self.noise_scale_param = yaml_object["Controller"]["noise_scale"]
        np.random.seed(1)

        # Read pre-computed trajectories
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(dir_path + '/../config/trajectories.json', 'r') as file:
            self.trajs = json.load(file)

    ################# PUBLIC METHODS
    
    def compute_cmd(self, car_list : List[CarStateStamped]) -> CarControlStamped:
        # Init empty command
        car_cmd = CarControlStamped()

        # Update current state of all cars
        self.curr_state = self._carStateStamped_to_array(car_list)
        if self.dilated_traj == []:
            self._initialize_paths_targets_dilated_traj()

        # Compute control
        u, trajectory, u_history = self.calc_control_and_trajectory(self.curr_state[self.robot_num])
        self.dilated_traj[self.robot_num] = LineString(zip(trajectory[:, 0], trajectory[:, 1])).buffer(self.dilation_factor, cap_style=3)
        # print("Current_state: " + str(self.curr_state[self.robot_num]))

        car_cmd.throttle = u[0]
        car_cmd.steering = u[1]

        # print(u, trajectory, u_history)


        
        plt.cla()
        plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
        self.plot_robot_trajectory(self.curr_state[self.robot_num], u, trajectory, self.dilated_traj[self.robot_num], [self.goal.x, self.goal.y], ax)
        
            
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)

        return car_cmd

    def calc_control_and_trajectory(self, x):
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
            # u_buf = self.u_hist[self.robot_num]
            # trajectory_buf = self.predicted_trajectory[self.robot_num]
            ob = [self.dilated_traj[idx] for idx in range(len(self.dilated_traj)) if idx != self.robot_num]
            dw = self._calc_dynamic_window()

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

                    to_goal_cost = self.to_goal_cost_gain * self._calc_to_goal_cost(trajectory)
                    speed_cost = self.speed_cost_gain * (self.max_speed - trajectory[-1, 3])
                    # if trajectory[-1, 3] <= 0.0:
                    #     speed_cost = 5
                    # else:
                    #     speed_cost = 0.0
                    ob_cost = self.obstacle_cost_gain * self._calc_obstacle_cost(trajectory, ob)
                    heading_cost = self.heading_cost_gain * self._calc_to_goal_heading_cost(trajectory)
                    final_cost = to_goal_cost +  speed_cost + ob_cost #+ heading_cost #+ speed_cost 
                    # print("COSTS: ", to_goal_cost, speed_cost, ob_cost)
                    # search minimum trajectory
                    if min_cost >= final_cost:
                        min_cost = final_cost
                        best_u = [a, delta]
                        best_trajectory = trajectory
                        u_history = [[a, delta] for _ in range(len(trajectory-1))]

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

    def set_goal(self, goal: State):
        self.goal = goal
        

    ################## PRIVATE METHODS

    def _initialize_paths_targets_dilated_traj(self):
        """
        Initialize the paths, targets, and dilated trajectory.
        """
        for i in range(self.curr_state.shape[0]):
            self.dilated_traj.append(Point(self.curr_state[i, 0], self.curr_state[i, 1]).buffer(self.dilation_factor, cap_style=3))
        
    def _carStateStamped_to_array(self, car_list: List[CarStateStamped]) -> np.array:
        car_num = len(car_list)
        car_array = np.empty((car_num, 5))
        for i in range(len(car_list)):
            car_i = car_list[i]
            curr_v = utils.longitudinal_velocity(car_i.vel_x, car_i.vel_y, car_i.turn_angle)
            np_car_i = np.array([car_i.pos_x, car_i.pos_y, car_i.turn_angle, curr_v, car_i.turn_rate])
            car_array[i,:] = np_car_i
        return car_array


    def _calc_dynamic_window(self):
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

    def _predict_trajectory(self, x_init, a, delta):
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

    def _calc_obstacle_cost(self, trajectory, ob):
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
            distance_cost = 1/min_distance
        else:
            distance_cost = 0.0
        
        return distance_cost

    def _calc_to_goal_cost(self, trajectory):
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

        dx = self.goal.x - trajectory[:, 0]
        dy = self.goal.y - trajectory[:, 1]

        cost = min(np.sqrt(dx**2+dy**2))
        return cost

    def _calc_to_goal_heading_cost(self, trajectory):
        """
        Calculate the cost to the goal with angle difference.

        Args:
            trajectory (numpy.ndarray): Trajectory.
            goal (list): Goal position [x(m), y(m)].

        Returns:
            float: Cost to the goal with angle difference.
        """
        dx = self.goal.x - trajectory[-1, 0]
        dy = self.goal.y - trajectory[-1, 1]

        # either using the angle difference or the distance --> if we want to use the angle difference, we need to normalize the angle before taking the difference
        error_angle = utils.normalize_angle(math.atan2(dy, dx))
        cost_angle = error_angle - utils.normalize_angle(trajectory[-1, 2])
        cost_angle = abs(cost_angle)

        return cost_angle
    












    def plot_robot_trajectory(self, x, u, predicted_trajectory, dilated_traj, targets, ax):
        """
        Plots the robot and arrows for visualization.

        Args:
            i (int): Index of the robot.
            x (numpy.ndarray): State vector of shape (4, N), where N is the number of time steps.
            multi_control (numpy.ndarray): Control inputs of shape (2, N).
            targets (list): List of target points.

        """
        plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-", color=color_dict[0])
        plot_polygon(dilated_traj, ax=ax, add_points=False, alpha=0.5, color=color_dict[0])
        # plt.plot(x[0], x[1], "xr")
        plt.plot(targets[0], targets[1], "x", color=color_dict[0], markersize=15)
        self.plot_robot(x[0], x[1], x[2])
        self.plot_arrow(x[0], x[1], x[2], length=1, width=0.5)
        self.plot_arrow(x[0], x[1], x[2] + u[1], length=3, width=0.5)
        self.plot_map()


    def plot_arrow(self, x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
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

    def plot_robot(self, x, y, yaw):  
        """
        Plot the robot.

        Args:
            x (float): X-coordinate of the robot.
            y (float): Y-coordinate of the robot.
            yaw (float): Yaw angle of the robot.
            i (int): Index of the robot.
        """
        outline = np.array([[-self.L / 2, self.L / 2,
                                (self.L / 2), -self.L / 2,
                                -self.L / 2],
                            [self.WB / 2, self.WB / 2,
                                - self.WB / 2, -self.WB / 2,
                                self.WB / 2]])
        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                            [-math.sin(yaw), math.cos(yaw)]])
        outline = (outline.T.dot(Rot1)).T
        outline[0, :] += x
        outline[1, :] += y
        plt.plot(np.array(outline[0, :]).flatten(),
                    np.array(outline[1, :]).flatten(), color_dict[0])

    def plot_map(self):
        """
        Plot the map.
        """
        corner_x = [-10/2.0, 10/2.0, 10/2.0, -10/2.0, -10/2.0]
        corner_y = [10/2.0, 10/2.0, -10/2.0, -10/2.0, 10/2.0]

        plt.plot(corner_x, corner_y)