import matplotlib.pyplot as plt
import numpy as np
import math

from bumper_cars.classes.State import State
from bumper_cars.classes.Controller import Controller
from bumper_cars.utils import car_utils as utils
from lar_msgs.msg import CarControlStamped, CarStateStamped
from typing import List


# For the parameter file
import yaml
import json

from shapely.geometry import Point, LineString
from shapely import distance
import os


# For animation purposes
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
    
    dir_path = os.path.dirname(os.path.realpath(__file__))

    def __init__(self, controller_path:str, robot_num = 0):
        super().__init__(controller_path)
        self.robot_num = robot_num 
        self.dilated_traj = []
        self.predicted_trajectory = None
        self.u_hist = [[0.0, 0.0]]

        # Opening YAML file
        with open(controller_path, 'r') as openfile:
            # Reading from yaml file
            yaml_object = yaml.safe_load(openfile)

        self.v_resolution = yaml_object["DWA"]["v_resolution"] # [m/s]
        # self.delta_resolution = math.radians(yaml_object["DWA"]["delta_resolution"])# [rad]
        self.delta_resolution = yaml_object["DWA"]["delta_resolution"]# [rad]       #TODO: CHANGE NAME IN YAML FILE
        self.a_resolution = yaml_object["DWA"]["a_resolution"] # [m/ss]

        self.to_goal_cost_gain = yaml_object["DWA"]["to_goal_cost_gain"]
        self.obstacle_cost_gain = yaml_object["DWA"]["obstacle_cost_gain"]
        self.dilation_factor = yaml_object["DWA"]["dilation_factor"]
        self.emergency_brake_distance = yaml_object["DWA"]["emergency_brake_distance"]
        
        np.random.seed(1)

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
        u, trajectory = self.__calc_control_and_trajectory(self.curr_state[self.robot_num])
        self.dilated_traj[self.robot_num] = LineString(zip(trajectory[:, 0], trajectory[:, 1])).buffer(self.dilation_factor, cap_style=3)

        car_cmd.throttle = np.interp(u[0], [self.car_model.min_acc, self.car_model.max_acc], [-1, 1]) * self.car_model.acc_gain
        car_cmd.steering = np.interp(u[1], [-self.car_model.max_steer, self.car_model.max_steer], [-1, 1])

        # Debug visualization
        if self.show_animation and self.robot_num == 0:
            plt.cla()
            plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
            for i in range(self.curr_state.shape[0]):
                self.plot_robot_trajectory(self.curr_state[i], u, trajectory, self.dilated_traj[i], [0,0], ax)
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.00001)

        return car_cmd

    def set_goal(self, goal: CarControlStamped) -> None:
        # next_state = self.__simulate_input(goal)
        self.goal = CarControlStamped()
        self.goal.throttle = np.interp(goal.throttle, [-1, 1], [self.car_model.min_acc, self.car_model.max_acc]) * self.car_model.acc_gain
        self.goal.steering = np.interp(goal.steering, [-1, 1], [-self.car_model.max_steer, self.car_model.max_steer])

    def compute_traj(self):
        """
        Generates trajectories and stores them in a json in the config file
        """
        
        dw = self.__calc_dynamic_window()
        complete_trajectories = {}
        initial_state = State()
        initial_state.x = 0.0
        initial_state.y = 0.0
        initial_state.yaw = np.radians(90.0)

        other_delta = abs(np.random.normal(0.0, 0.1, int(self.delta_resolution)))

        for v in np.arange(self.car_model.min_speed, self.car_model.max_speed+self.v_resolution, self.v_resolution):
            initial_state.v = v
            traj = []
            u_total = []
            cmd = CarControlStamped()
            for a in np.arange(dw[0], dw[1]+self.a_resolution, self.a_resolution):
                cmd.throttle = np.interp(a, [self.car_model.min_acc, self.car_model.max_acc], [-1, 1]) * self.car_model.acc_gain
                init_delta = np.array([0.0, dw[2], dw[3]])
                delta_list = np.hstack((init_delta, other_delta))
                for delta in delta_list:
                    # Calc traj for angle
                    cmd.steering = np.interp(delta, [-self.car_model.max_steer, self.car_model.max_steer], [-1, 1])
                    traj.append(self.__calc_trajectory(initial_state, cmd))
                    u_total.append([a, delta])
                    # Repeat for mirrored angle
                    cmd.steering = np.interp(-delta, [-self.car_model.max_steer, self.car_model.max_steer], [-1, 1])
                    traj.append(self.__calc_trajectory(initial_state, cmd))
                    u_total.append([a, -delta])


            traj = np.array(traj)
            temp2 = {}
            for j in range(len(traj)):
                temp2[u_total[j][0]] = {}
            for i in range(len(traj)):
                temp2[u_total[i][0]][u_total[i][1]] = traj[i, :, :].tolist()
            complete_trajectories[v] = temp2
                
        # saving the complete trajectories to a csv file
        with open(self.dir_path + '/../config/trajectories.json', 'w') as file:
            json.dump(complete_trajectories, file, indent=4)

        print("\nThe JSON data has been written to 'config/trajectories.json'")


    ################## PRIVATE METHODS

    def __update_others(self):
        emptyControl = CarControlStamped()
        for i in range(len(self.dilated_traj)):
            if i == self.robot_num:
                continue
            car_state = State()
            car_state.x = self.curr_state[i][0]
            car_state.y = self.curr_state[i][1]
            car_state.yaw = self.curr_state[i][2]
            car_state.v = self.curr_state[i][3]
            car_state.omega = self.curr_state[i][4]
            traj_i = self.__calc_trajectory(car_state, emptyControl )
            self.dilated_traj[i] = LineString(zip(traj_i[:, 0], traj_i[:, 1])).buffer(self.dilation_factor, cap_style=3)

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

    
    def __initialize_paths_targets_dilated_traj(self):
        """
        Initialize the paths, targets, and dilated trajectory.
        """
        # Read pre-computed trajectories
        with open(self.dir_path + '/../config/trajectories.json', 'r') as file:
            self.trajs = json.load(file)

        for i in range(self.curr_state.shape[0]):
            self.dilated_traj.append(Point(self.curr_state[i, 0], self.curr_state[i, 1]).buffer(self.dilation_factor, cap_style=3))

        self.predicted_trajectory = np.array([self.curr_state[self.robot_num]]*int(self.ph/self.dt))
        


    def __calc_dynamic_window(self):
        """
        Calculate the dynamic window based on the current state.

        Args:
            x (list): Current state [x(m), y(m), yaw(rad), v(m/s), delta(rad)].

        Returns:
            list: Dynamic window [min_throttle, max_throttle, min_steer, max_steer].
        """

        Vs = [self.car_model.min_acc, self.car_model.max_acc,
            -self.car_model.max_steer, self.car_model.max_steer]

        dw = [Vs[0], Vs[1], Vs[2], Vs[3]]

        return dw

    def __calc_obstacle_cost(self, trajectory, ob):
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
        if any(element < -self.width_init/2+self.car_model.width/2 or element > self.width_init/2-self.car_model.width/2 for element in x):
            return np.inf
        if any(element < -self.height_init/2+self.car_model.width/2 or element > self.height_init/2-self.car_model.width/2 for element in y):
            return np.inf

        if ob:
            for obstacle in ob:
                if dilated.intersects(obstacle):
                    return np.inf # collision        
                elif distance(dilated, obstacle) < min_distance:
                    min_distance = distance(dilated, obstacle)
            distance_cost = 1/(min_distance * 10)
        else:
            distance_cost = 0.0
        # print("Robot: "+str(self.robot_num) + " distance: " + str(min_distance))
        return distance_cost

    def __calc_to_goal_cost(self, a = 0.0, delta = 0.0):
        """
        Calculate the cost to the goal.

        Args:
            trajectory (numpy.ndarray): Trajectory.
            goal (list): Goal position [x(m), y(m)].

        Returns:
            float: Cost to the goal.
        """

        dx = self.goal.throttle - a
        dy = self.goal.steering - delta

        cost = np.sqrt(dx**2+dy**2)
        return cost
    

    def __calc_control_and_trajectory(self, x):
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
            
            ob = [self.dilated_traj[idx] for idx in range(len(self.dilated_traj)) if idx != self.robot_num]
            dw = self.__calc_dynamic_window()

            # evaluate all trajectory with sampled input in dynamic window
            nearest = utils.find_nearest(np.arange(self.car_model.min_speed, self.car_model.max_speed+self.v_resolution, self.v_resolution), x[3])

            for a in np.arange(dw[0], dw[1]+self.v_resolution, self.v_resolution):
                for delta in self.trajs[str(nearest)][str(a)]:

                    geom = self.trajs[str(nearest)][str(a)][str(delta)]
                    geom = np.array(geom)
                    geom[:,0:2] = (geom[:,0:2]) @ utils.rotateMatrix(np.radians(90)-x[2]) + [x[0],x[1]]
                    geom[:,2] = geom[:,2] + x[2] - np.pi/2 #bringing also the yaw angle in the new frame
                    trajectory = geom

                    # calc cost
                    to_goal_cost = self.to_goal_cost_gain * self.__calc_to_goal_cost(a, float(delta))
                    ob_cost = self.obstacle_cost_gain * self.__calc_obstacle_cost(trajectory, ob)
                    final_cost = to_goal_cost + ob_cost

                    # search minimum trajectory
                    if min_cost >= final_cost:
                        min_cost = final_cost
                        best_u = [a, float(delta)]
                        best_trajectory = trajectory
                        u_traj = [[a, float(delta)] for _ in range(len(trajectory-1))]

            # Check if previous trajectory was better
            if len(self.u_hist) > self.emergency_brake_distance:  
                self.u_hist.pop(0)
                trajectory_buf = self.predicted_trajectory[1:]

                # Even when following the old trajectory, we need to update it to the position of the robot
                trajectory_buf[:,0:2] -= trajectory_buf[1,0:2]
                trajectory_buf[:,0:2] = (trajectory_buf[:,0:2]) @ utils.rotateMatrix(utils.normalize_angle(-x[2]+trajectory_buf[0,2]))
                trajectory_buf[:,0:2] += x[0:2]
                trajectory_buf[:,2] += utils.normalize_angle(x[2]-trajectory_buf[0,2])

                to_goal_cost = self.to_goal_cost_gain * self.__calc_to_goal_cost(self.u_hist[0][0], self.u_hist[0][1])
                ob_cost = self.obstacle_cost_gain * self.__calc_obstacle_cost(trajectory_buf, ob)
                
                final_cost = to_goal_cost + ob_cost 

                if min_cost >= final_cost:      
                    min_cost = final_cost
                    best_u = self.u_hist[0]
                    best_trajectory = trajectory_buf
                    u_traj = self.u_hist

            elif min_cost == np.inf:
                # emergency stop
                print(f"Emergency stop for vehicle {self.robot_num}")
                best_u = [(0.0 - x[3])/self.dt, 0]
                best_trajectory = np.array([x[0:3], x[0:3]] * int(self.ph/self.dt)) 
                u_traj =  np.array([best_u]*int(self.ph/self.dt))
            self.u_hist = u_traj
            self.predicted_trajectory = best_trajectory
            return best_u, best_trajectory
    
