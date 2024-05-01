import matplotlib.pyplot as plt
import numpy as np
import math
from enum import Enum
# For the parameter file
import pathlib
import json
import yaml
from custom_message.msg import Coordinate
from shapely.geometry import Point, Polygon, LineString
from shapely import intersection, distance
from shapely.plotting import plot_polygon, plot_line

from bumper_cars.classes.CarModel import State, CarModel
from bumper_cars.classes.Controller import Controller
from lar_utils import car_utils as utils
from lar_msgs.msg import CarControlStamped, CarStateStamped
from typing import List
import os


# For animation purposes
fig = plt.figure(1, dpi=90, figsize=(10,10))
ax= fig.add_subplot(111)


np.random.seed(1)

color_dict = {0: 'r', 1: 'b', 2: 'g', 3: 'y', 4: 'm', 5: 'c', 6: 'k', 7: 'tab:orange', 8: 'tab:brown', 9: 'tab:gray', 10: 'tab:olive', 11: 'tab:pink', 12: 'tab:purple', 13: 'tab:red', 14: 'tab:blue', 15: 'tab:green'}

class LBP_algorithm(Controller):
    dir_path = os.path.dirname(os.path.realpath(__file__))

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

        self.v_resolution = yaml_object["LBP"]["v_resolution"] # [m/s]
        self.delta_resolution = math.radians(yaml_object["LBP"]["delta_resolution"])# [rad/s]
        self.to_goal_cost_gain = yaml_object["LBP"]["to_goal_cost_gain"]
        self.speed_cost_gain = yaml_object["LBP"]["speed_cost_gain"]
        self.obstacle_cost_gain = yaml_object["LBP"]["obstacle_cost_gain"]
        self.heading_cost_gain = yaml_object["LBP"]["heading_cost_gain"]
        self.dilation_factor = yaml_object["LBP"]["dilation_factor"]

        self.width_init = yaml_object["Simulation"]["width"]
        self.height_init = yaml_object["Simulation"]["height"]

        self.show_animation = yaml_object["Simulation"]["show_animation"]


    ################# PUBLIC METHODS
    
    def compute_cmd(self, car_list : List[CarStateStamped]) -> CarControlStamped:
        # Init empty command
        car_cmd = CarControlStamped()

        # Update current state of all cars
        self.curr_state = utils.carStateStamped_to_array(car_list)
        if self.dilated_traj == []:
            self.__initialize_paths_targets_dilated_traj()

        # Update expected state of other cars (no input)
        self.__update_others()

        # Compute control   
        u, trajectory, u_history = self.__calc_control_and_trajectory(self.curr_state[self.robot_num])
        print("Traj\n",trajectory)
        # self.dilated_traj[self.robot_num] = LineString(zip(trajectory[:, 0], trajectory[:, 1])).buffer(self.dilation_factor, cap_style=3)

        car_cmd.throttle = u[0]
        car_cmd.steering = u[1]

        # Debug visualization
        if self.show_animation and self.robot_num == 0:
            plt.cla()
            plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
            for i in range(self.curr_state.shape[0]):
                self.plot_robot_trajectory(self.curr_state[i], u, trajectory, self.dilated_traj[i], [self.goal.x, self.goal.y], ax)
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.00001)

        return car_cmd

    def set_goal(self, goal: CarControlStamped) -> None:
        next_state = self.__simulate_input(goal)
        self.goal = next_state

    ################## PRIVATE METHODS

    def __simulate_input(self, car_cmd: CarControlStamped) -> State:
        curr_state = self.car_model.step(car_cmd, self.dt)
        t = self.dt
        while t<self.ph:
            curr_state = self.car_model.step(car_cmd, self.dt, curr_state=curr_state)
            t += self.dt
        return curr_state
    
    def __calc_trajectory(self, curr_state:State, cmd:CarControlStamped):
        """
        Computes the trajectory that is used for each vehicle
        """
        iterations = math.ceil(self.ph/self.dt) + 1
        traj = np.zeros((iterations, 4))
        traj[0,:] = np.array([curr_state.x, curr_state.y, curr_state.yaw, curr_state.v])
        i = 1
        while i < iterations:
            curr_state = self.car_model.step(cmd,self.dt,curr_state)
            x = [curr_state.x, curr_state.y, curr_state.yaw, curr_state.v]
            traj[i,:] = np.array(x)
            i += 1
        return traj



    def __update_others(self):
        emptyControl = CarControlStamped()
        for i in range(len(self.dilated_traj)):
            # if i == self.robot_num:
            #     continue
            car_state = State()
            car_state.x = self.curr_state[i][0]
            car_state.y = self.curr_state[i][1]
            car_state.yaw = self.curr_state[i][2]
            car_state.v = self.curr_state[i][3]
            car_state.omega = self.curr_state[i][4]
            traj_i = self.__calc_trajectory(car_state, emptyControl )
            self.dilated_traj[i] = LineString(zip(traj_i[:, 0], traj_i[:, 1])).buffer(self.dilation_factor, cap_style=3)

    def __initialize_paths_targets_dilated_traj(self):
        """
        Initialize the paths, targets, and dilated trajectory.
        """
        # Read pre-computed trajectories
        with open(self.dir_path + '/../config/LBP.json', 'r') as file:
            self.data = json.load(file)

        for i in range(self.curr_state.shape[0]):
            self.dilated_traj.append(Point(self.curr_state[i, 0], self.curr_state[i, 1]).buffer(self.dilation_factor, cap_style=3))

    def __calc_dynamic_window(self, x):
        """
        Calculates the dynamic window for velocity search based on the current state.

        Args:
            x (list): Current state of the system [x, y, theta, v]

        Returns:
            list: List of possible velocities within the dynamic window
        """
        v_poss = np.arange(self.car_model.min_speed, self.car_model.max_speed+self.v_resolution, self.v_resolution)
        v_achiv = [x[3] + self.car_model.min_acc*self.dt, x[3] + self.car_model.max_acc*self.dt]

        v_search = []

        for v in v_poss:
            if v >= v_achiv[0] and v <= v_achiv[1]:
                v_search.append(v)
        
        return v_search

    def __calc_control_and_trajectory(self, x):
        """
        Calculates the final input with LBP method.

        Args:
            x (list): The current state of the system.
        Returns:
            tuple: A tuple containing the best control input, the best trajectory, and the control input history.
        """

        min_cost = float("inf")
        best_u = [0.0, 0.0]
        best_trajectory = np.array([x])

        ob = [self.dilated_traj[idx] for idx in range(len(self.dilated_traj)) if idx != self.robot_num]
        v_search = self.__calc_dynamic_window(x)

        # Calculate the cost of each possible trajectory and return the minimum
        for v in v_search:
            dict = self.data[str(v)]
            for id, info in dict.items():

                # old_time = time.time()
                geom = np.zeros((len(info['x']),3))
                geom[:,0] = info['x']
                geom[:,1] = info['y']
                geom[:,2] = info['yaw']
                geom[:,0:2] = (geom[:,0:2]) @ utils.rotateMatrix(-x[2]) + [x[0],x[1]]
                
                geom[:,2] = geom[:,2] + x[2] #bringing also the yaw angle in the new frame
                
                # trajectory = predict_trajectory(x_init, a, delta)
                trajectory = geom
                # calc cost

                # TODO: small bug when increasing the factor too much for the to_goal_cost_gain
                to_goal_cost = self.to_goal_cost_gain * self.__calc_to_goal_cost(trajectory)
                
                if v <= 0.0:
                    speed_cost = 30
                else:
                    speed_cost = 0.0
                    
                ob_cost = self.obstacle_cost_gain * self.__calc_obstacle_cost(trajectory, ob)
                heading_cost = self.heading_cost_gain * self.__calc_to_goal_heading_cost(trajectory)
                final_cost = to_goal_cost + ob_cost + heading_cost + speed_cost 
                
                # search minimum trajectory
                if min_cost >= final_cost:
                    min_cost = final_cost

                    # interpolate the control inputs
                    a = (v-x[3])/self.dt

                    # print(f'v: {v}, id: {id}')
                    # print(f"Control seq. {len(info['ctrl'])}")
                    best_u = [a, info['ctrl'][1]]
                    best_trajectory = trajectory
                    # u_history = info['ctrl'].copy()

        # Calculate cost of the previous best trajectory and compare it with that of the new trajectories
        # If the cost of the previous best trajectory is lower, use the previous best trajectory
        
        #TODO: this section has a small bug due to popping elements from the buffer, it gets to a point where there 
        # are no more elements in the buffer to use
        # # # # # if len(u_buf) > 4:
        # # # # #     u_buf.pop(0)
        
        # # # # #     trajectory_buf = trajectory_buf[1:]

        # # # # #     to_goal_cost = to_goal_cost_gain * calc_to_goal_cost(trajectory_buf, goal)
        # # # # #     # speed_cost = speed_cost_gain * np.sign(trajectory[-1, 3]) * trajectory[-1, 3]
        # # # # #     ob_cost = obstacle_cost_gain * calc_obstacle_cost(trajectory_buf, ob)
        # # # # #     heading_cost = heading_cost_gain * calc_to_goal_heading_cost(trajectory_buf, goal)
        # # # # #     final_cost = to_goal_cost + ob_cost + heading_cost #+ speed_cost 

        # # # # #     if min_cost >= final_cost:
        # # # # #         min_cost = final_cost
        # # # # #         best_u = [0, u_buf[1]]
        # # # # #         best_trajectory = trajectory_buf
        # # # # #         u_history = u_buf

        # # # # # elif min_cost == np.inf:
        # # # # #     # emergency stop
        # # # # #     print("Emergency stop")
        # # # # #     if x[3]>0:
        # # # # #         best_u = [min_acc, 0]
        # # # # #     else:
        # # # # #         best_u = [max_acc, 0]
        # # # # #     best_trajectory = np.array([x[0:3], x[0:3]])
        # # # # #     u_history = [min_acc, 0]


        return best_u, best_trajectory, [] #u_history

    def __calc_obstacle_cost(self, trajectory, ob):
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
        
        minxp = min(abs(self.width_init/2-trajectory[:, 0]))
        minxn = min(abs(-self.width_init/2-trajectory[:, 0]))
        minyp = min(abs(self.height_init/2-trajectory[:, 1]))
        minyn = min(abs(-self.height_init/2-trajectory[:, 1]))
        min_distance = min(minxp, minxn, minyp, minyn)
        dilated = line.buffer(self.dilation_factor, cap_style=3)

        x = trajectory[:, 0]
        y = trajectory[:, 1]

        # check if the trajectory is out of bounds
        if any(element < -self.width_init/2+self.car_model.L or element > self.width_init/2-self.car_model.L for element in x):
            return np.inf
        if any(element < -self.height_init/2+self.car_model.L or element > self.height_init/2-self.car_model.L for element in y):
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

    def __calc_to_goal_cost(self, trajectory):
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

        dx = self.goal.x - trajectory[:, 0]
        dy = self.goal.y - trajectory[:, 1]

        cost = min(np.sqrt(dx**2+dy**2))

        return cost

    def __calc_to_goal_heading_cost(self, trajectory):
        """
        Calculate the cost to reach the goal based on the heading angle difference.

        Args:
            trajectory (numpy.ndarray): The trajectory array containing the x, y, and heading values.
            goal (tuple): The goal coordinates (x, y).

        Returns:
            float: The cost to reach the goal based on the heading angle difference.
        """
        dx = self.goal.x - trajectory[-1, 0]
        dy = self.goal.y - trajectory[-1, 1]

        # either using the angle difference or the distance --> if we want to use the angle difference, we need to normalize the angle before taking the difference
        error_angle = utils.normalize_angle(math.atan2(dy, dx))
        cost_angle = error_angle - utils.normalize_angle(trajectory[-1, 2])
        cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

        return cost
    
