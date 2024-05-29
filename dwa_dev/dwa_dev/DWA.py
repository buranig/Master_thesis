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

from shapely.geometry import Point, LineString, mapping
from shapely import distance, union
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

        # Empty obstacle set
        self.ob = None
        self.trajectory = None
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
        self.dilated_traj[self.robot_num] = trajectory
        # self.trajectory = trajectory.exterior.coords
        self.trajectory = trajectory.buffer(self.car_model.width * self.dilation_factor, cap_style=3).exterior.coords
        

        car_cmd.throttle = np.interp(u[0], [self.car_model.min_acc, self.car_model.max_acc], [-1, 1]) * self.car_model.acc_gain
        car_cmd.steering = np.interp(u[1], [-self.car_model.max_steer, self.car_model.max_steer], [-1, 1])

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
        initial_state.yaw = 0.0

        mean = 0.0
        std = self.car_model.max_steer/2.0

        other_delta = abs(np.random.normal(mean, std, int(self.delta_resolution)))
        init_delta = np.array([0.0, dw[2], dw[3]])
        delta_list = np.hstack((init_delta, other_delta))

        for v in np.arange(self.car_model.min_speed, self.car_model.max_speed+self.v_resolution, self.v_resolution):
            initial_state.v = v
            traj = []
            u_total = []
            cmd = CarControlStamped()
            for a in np.arange(dw[0], dw[1]+self.a_resolution, self.a_resolution):
                cmd.throttle = np.interp(a, [self.car_model.min_acc, self.car_model.max_acc], [-1, 1]) * self.car_model.acc_gain
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

        ref_x = self.curr_state[self.robot_num][0]
        ref_y = self.curr_state[self.robot_num][1]
        ref_yaw = self.curr_state[self.robot_num][2]
        ref_v = self.curr_state[self.robot_num][3]
        ref_omega = self.curr_state[self.robot_num][4]

        for i in range(len(self.dilated_traj)):
            if i == self.robot_num:
                continue
            car_state = State()
            car_state.x = self.curr_state[i][0]
            car_state.y = self.curr_state[i][1]
            car_state.yaw = self.curr_state[i][2]
            car_state.v = self.curr_state[i][3]
            car_state.omega = self.curr_state[i][4]

            # Compute the relative position
            dx = car_state.x - ref_x
            dy = car_state.y - ref_y
            dtheta = car_state.yaw - ref_yaw

            # Rotate the relative position to the reference frame of the skipped car
            rel_x = np.cos(-ref_yaw) * dx - np.sin(-ref_yaw) * dy
            rel_y = np.sin(-ref_yaw) * dx + np.cos(-ref_yaw) * dy
            rel_yaw = dtheta  # assuming small angles, otherwise normalize angle

            # Compute the relative velocity components
            vx = car_state.v * np.cos(car_state.yaw)
            vy = car_state.v * np.sin(car_state.yaw)
            ref_vx = ref_v * np.cos(ref_yaw)
            ref_vy = ref_v * np.sin(ref_yaw)

            rel_vx = vx - ref_vx
            rel_vy = vy - ref_vy

            # Rotate the relative velocity to the reference frame of the skipped car
            rel_vx_transformed = np.cos(-ref_yaw) * rel_vx - np.sin(-ref_yaw) * rel_vy
            rel_vy_transformed = np.sin(-ref_yaw) * rel_vx + np.cos(-ref_yaw) * rel_vy

            # Calculate the magnitude of the relative velocity
            rel_v = np.sqrt(rel_vx_transformed**2 + rel_vy_transformed**2)
            rel_omega = car_state.omega - ref_omega
            


            ###############################
            #TODO: Check if this is correct
            ###############################

            # Update the car state to the relative state
            car_state.x = rel_x
            car_state.y = rel_y
            car_state.yaw = rel_yaw
            car_state.v = rel_v
            car_state.omega = rel_omega

            traj_i = self.__calc_trajectory(car_state, emptyControl )
            self.dilated_traj[i] = LineString(zip(traj_i[:, 0], traj_i[:, 1]))#.buffer(self.dilation_factor, cap_style=3)

        ########
        # Also move the borders
        x_list = []
        y_list = []
        for i in range(len(self.boundary_points)):
            pos_x = self.boundary_points[i][0]
            pos_y = self.boundary_points[i][1]
            new_x, new_y = utils.transform_point(pos_x,pos_y,ref_x,ref_y,ref_yaw)
            x_list.append(new_x)
            y_list.append(new_y)
            
        x_list.append(x_list[0])
        y_list.append(y_list[0])
        self.borders = LineString(zip(x_list, y_list)) #.buffer(self.dilation_factor, cap_style=3)

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

        # Dilate them for later inference
        for v in np.arange(self.car_model.min_speed, self.car_model.max_speed+self.v_resolution, self.v_resolution):            
            for a in np.arange(self.car_model.min_acc, self.car_model.max_acc+self.a_resolution, self.a_resolution):
                diff_deltas = self.trajs[str(v)][str(a)].keys()
                for delta in diff_deltas:
                    trajectory = np.array(self.trajs[str(v)][str(a)][delta])
                    line = LineString(zip(trajectory[:, 0], trajectory[:, 1]))
                    dilated = line#.buffer(self.dilation_factor, cap_style=3)
                    self.trajs[str(v)][str(a)][str(delta)] = dilated

        # Initialize traj for each vehicle
        for i in range(self.curr_state.shape[0]):
            self.dilated_traj.append(Point(self.curr_state[i, 0], self.curr_state[i, 1]))#.buffer(self.dilation_factor, cap_style=3))

        # Initialize the predicted trajectory
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

    def __calc_obstacle_cost(self, nearest=None, a=None, delta=None, dilated=None):
        """
        Calculate the obstacle cost.

        Args:
            trajectory (numpy.ndarray): Trajectory.
            ob (list): List of obstacles.

        Returns:
            float: Obstacle cost.
        """
        # Retrieve previously dilated trajectory (if not initialized)
        if dilated is None:
            dilated = self.trajs[nearest][a][delta] 

        min_distance = np.inf
        if self.ob:
            dist = distance(dilated, self.ob)
            # if dilated.intersection(self.ob):
            if dist < self.dilation_factor * self.car_model.width:
                return np.inf # collision        
            elif dist < min_distance:
                min_distance = dist
            distance_cost = 1/(min_distance * 10)
        else:
            distance_cost = 0.0
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

            # Variables needed for faster computation
            self.ob = self.borders
            obstacles = [self.dilated_traj[idx] for idx in range(len(self.dilated_traj)) if idx != self.robot_num]
            for ob in obstacles:
                self.ob = ob if self.ob is None else union(self.ob, ob)

            
            dw = self.__calc_dynamic_window()

            # evaluate all trajectory with sampled input in dynamic window
            nearest = utils.find_nearest(np.arange(self.car_model.min_speed, self.car_model.max_speed+self.v_resolution, self.v_resolution), x[3])

            for a in np.arange(dw[0], dw[1]+self.a_resolution, self.a_resolution):
                for delta in self.trajs[str(nearest)][str(a)]:
                
                    # calc cost
                    to_goal_cost = self.to_goal_cost_gain * self.__calc_to_goal_cost(a, float(delta))
                    ob_cost = self.obstacle_cost_gain * self.__calc_obstacle_cost(str(nearest),str(a),str(delta))
                    final_cost = to_goal_cost + ob_cost

                    # search minimum trajectory
                    if min_cost >= final_cost:
                        min_cost = final_cost
                        best_u = [a, float(delta)]

            best_trajectory = self.trajs[str(nearest)][str(best_u[0])][str(best_u[1])]
            u_traj = [[a, float(delta)] for _ in range(int(self.delta_resolution)+3)] #TODO: Check if this 3 has to remain here

            # Check if previous trajectory was better
            if len(self.u_hist) > self.emergency_brake_distance:  
                self.u_hist.pop(0)
                trajectory_buf = self.predicted_trajectory

                to_goal_cost = self.to_goal_cost_gain * self.__calc_to_goal_cost(self.u_hist[0][0], self.u_hist[0][1])
                ob_cost = self.obstacle_cost_gain * self.__calc_obstacle_cost(dilated=trajectory_buf)
                
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
                trajectory = np.array([x[0:3], x[0:3]] * int(self.ph/self.dt)) 
                line = LineString(zip(trajectory[:, 0], trajectory[:, 1]))
                best_trajectory = line#.buffer(self.dilation_factor, cap_style=3)
                u_traj =  np.array([best_u]*int(self.ph/self.dt))

            self.u_hist = u_traj
            self.predicted_trajectory = best_trajectory
            return best_u, best_trajectory
    
