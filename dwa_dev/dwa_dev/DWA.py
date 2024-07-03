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

from shapely.geometry import Point, LineString, MultiLineString
from shapely import distance, union
import os


class DWA_algorithm(Controller):    
    """
    DWA_algorithm class implements the Dynamic Window Approach algorithm for collision avoidance.
    It inherits from the Controller class.

    Args:
    - controller_path (str): The path to the controller YAML file.
    - car_i (int): The index of the car.

    Returns:
    - None
    """

    dir_path = os.path.dirname(os.path.realpath(__file__))

    def __init__(self, controller_path:str, car_i = 0):
        """
        Initializes the DWA_algorithm object.

        Args:
        - controller_path (str): The path to the controller YAML file.
        - car_i (int): The index of the car.

        Returns:
        - None
        """
        super().__init__(controller_path, car_i)

        self.dilated_traj = []
        self.predicted_trajectory = None
        self.u_hist = [[0.0, 0.0]]

        # Opening YAML file
        with open(controller_path, 'r') as openfile:
            # Reading from yaml file
            yaml_object = yaml.safe_load(openfile)

        self.v_resolution = yaml_object["DWA"]["v_resolution"] # [m/s]
        self.a_resolution = yaml_object["DWA"]["a_resolution"] # [m/ss]
        self.delta_samples = yaml_object["DWA"]["delta_samples"]# [rad]

        self.to_goal_cost_gain = yaml_object["DWA"]["to_goal_cost_gain"]
        self.obstacle_cost_gain = yaml_object["DWA"]["obstacle_cost_gain"]
        self.dilation_factor = yaml_object["DWA"]["dilation_factor"]

        # Empty obstacle set
        self.ob = None
        self.trajectory = None
        np.random.seed(1)

    ################# PUBLIC METHODS
    
    def compute_cmd(self, car_list : List[CarStateStamped]) -> CarControlStamped:
        """
        Computes the control command for the car based on the current state of all cars.

        Args:
        - car_list (List[CarStateStamped]): The list of car states.

        Returns:
        - CarControlStamped: The computed safe control command for the car.
        """
        # Init empty command
        car_cmd = CarControlStamped()

        # Update current state of all cars
        self.curr_state = utils.carStateStamped_to_array(car_list)
        if self.dilated_traj == []:
            print("\033[93mLoading trajectories . . . \033[0m")
            self.__initialize_paths_targets_dilated_traj()
            print("\033[93mDone loading!\033[0m")

        # Update expected state of other cars (no input)
        self.__update_others()

        # Compute control   
        u, trajectory = self.__calc_control_and_trajectory(self.curr_state[self.car_i])
        self.dilated_traj[self.car_i] = trajectory
        # self.trajectory = trajectory.exterior.coords
        self.trajectory = trajectory.buffer(self.car_model.width * self.dilation_factor, cap_style=3).exterior.coords
        

        car_cmd.throttle = np.interp(u[0], [self.car_model.min_acc, self.car_model.max_acc], [-1, 1]) * self.car_model.acc_gain
        car_cmd.steering = np.interp(u[1], [-self.car_model.max_steer, self.car_model.max_steer], [-1, 1])

        return car_cmd

    def set_goal(self, goal: CarControlStamped) -> None:
        """
        Sets the goal for the car.

        Args:
        - goal (CarControlStamped): The goal control command for the car.

        Returns:
        - None
        """
        self.goal = CarControlStamped()
        self.goal.throttle = np.interp(goal.throttle, [-1, 1], [self.car_model.min_acc, self.car_model.max_acc]) * self.car_model.acc_gain
        self.goal.steering = np.interp(goal.steering, [-1, 1], [-self.car_model.max_steer, self.car_model.max_steer])

    def compute_traj(self) -> None:
        """
        Computes trajectories for different velocities and saves them to a JSON file.

        This function calculates trajectories for different velocities, accelerations and 
        steering angles and saves them to a JSON file.

        Returns:
            None

        """
        print("\033[93mGenerating trajectories . . . \033[0m")
        
        # Initialize trajectory storing variables
        dw = self.__calc_dynamic_window()
        complete_trajectories = {}
        initial_state = State()
        initial_state.x = 0.0
        initial_state.y = 0.0
        initial_state.yaw = 0.0

        # Compute mean and std for steering normal sampling
        mean = 0.0
        std = self.car_model.max_steer/2.0

        # Precompute steering angles
        other_delta = abs(np.random.normal(mean, std, int(self.delta_samples)))
        init_delta = np.array([0.0, dw[2], dw[3]])
        delta_list = np.hstack((init_delta, other_delta))

        # Iterate for each expected velocity (specified in the YAML file)
        for v in np.arange(self.car_model.min_speed, self.car_model.max_speed+self.v_resolution, self.v_resolution):
            initial_state.v = v
            traj = []
            u_total = []
            cmd = CarControlStamped()
            # Iterate for each expected acceleration
            for a in np.arange(dw[0], dw[1]+self.a_resolution, self.a_resolution):
                cmd.throttle = np.interp(a, [self.car_model.min_acc, self.car_model.max_acc], [-1, 1])
                # Iterate for each sampled steering angle
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
                
        # saving the complete trajectories to a json file
        with open(self.dir_path + '/../config/trajectories.json', 'w') as file:
            json.dump(complete_trajectories, file, indent=4)
        print("\033[93mDone generating!\033[0m")
        print("\033[92mTrajectories were written to 'dwa_dev/config/trajectories.json' \033[00m")


    ################## PRIVATE METHODS
    
    def __update_others(self) -> None:
        """
        Update the states of other cars and the borders.

        This method updates the states of other cars relative to the current car's state.
        It also updates the positions of the borders relative to the current car's state.

        Returns:
            None
        """

        # Initialize variables for change of reference frame
        emptyControl = CarControlStamped()

        ref_x = self.curr_state[self.car_i][0]
        ref_y = self.curr_state[self.car_i][1]
        ref_yaw = self.curr_state[self.car_i][2]
        ref_v = self.curr_state[self.car_i][3]
        ref_omega = self.curr_state[self.car_i][4]

        for i in range(len(self.dilated_traj)):
            if i == self.car_i:
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
            rel_yaw = dtheta

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

            # Update the car state to the relative state
            car_state.x = rel_x
            car_state.y = rel_y
            car_state.yaw = rel_yaw
            car_state.v = rel_v
            car_state.omega = rel_omega

            traj_i = self.__calc_trajectory(car_state, emptyControl)
            self.dilated_traj[i] = LineString(zip(traj_i[:, 0], traj_i[:, 1]))

        # Also move the borders
        x_list = []
        y_list = []
        for i in range(len(self.boundary_points)):
            pos_x = self.boundary_points[i][0]
            pos_y = self.boundary_points[i][1]
            new_x, new_y = utils.transform_point(pos_x, pos_y, ref_x, ref_y, -ref_yaw)
            x_list.append(new_x)
            y_list.append(new_y)

        x_list.append(x_list[0])
        y_list.append(y_list[0])

        self.borders = LineString(zip(x_list, y_list))

    def __calc_trajectory(self, curr_state:State, cmd:CarControlStamped) -> np.ndarray:
        """
        Calculate the trajectory of the car given the current state and control command.

        Args:
            curr_state (State): The current state of the car.
            cmd (CarControlStamped): The control command for the car.

        Returns:
            np.ndarray: The calculated trajectory as a numpy array.

        """
        # Compute the iteration number
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

    
    def __initialize_paths_targets_dilated_traj(self) -> None:
        """
        Initializes the paths, targets, and dilated trajectories for the DWA algorithm.

        Reads pre-computed car trajectories from a JSON file.
        Initializes the trajectory for each vehicle and the predicted trajectory.

        Returns:
            None
        """        
        # Read pre-computed trajectories
        with open(self.dir_path + '/../config/trajectories.json', 'r') as file:
            self.trajs = json.load(file)

        # Store them in dictionary for constant time access
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
        self.predicted_trajectory = np.array([self.curr_state[self.car_i]]*int(self.ph/self.dt))
        


    def __calc_dynamic_window(self) -> List[float]:
        """
        Calculate the dynamic window based on the car model's minimum acceleration, maximum acceleration,
        maximum steering angle, and negative maximum steering angle (assuming steering symmetry).

        Returns:
            dw (List[float]): The dynamic window.
        """
        Vs = [self.car_model.min_acc, self.car_model.max_acc,
              -self.car_model.max_steer, self.car_model.max_steer]

        dw = [Vs[0], Vs[1], Vs[2], Vs[3]]

        return dw

    def __calc_obstacle_cost(self, nearest=None, a=None, delta=None, dilated=None) -> float:
        """
        Calculate the obstacle cost for a given trajectory.

        Parameters:
        - nearest (str): Nearest velocity to car's that is in the dicitonary.
        - a (str): Acceleration to be applied.
        - delta (str): Steering to be applied.
        - dilated (numpy.ndarray): Dilated trajectory.

        Returns:
        - float: The obstacle cost for the trajectory.

        This method calculates the obstacle cost for a given trajectory. It checks if the trajectory intersects with any obstacles
        and calculates the minimum distance to the nearest obstacle. The obstacle cost is inversely proportional to the minimum distance.

        If there are no obstacles, the obstacle cost is 0.0.

        If the trajectory intersects with an obstacle, the obstacle cost is set to infinity.
        """
        # Retrieve previously dilated trajectory (if not initialized)
        if dilated is None:
            dilated = self.trajs[nearest][a][delta] 

        min_distance = np.inf
        if self.ob:
            dist = distance(dilated, self.ob)
            if dist < self.dilation_factor * self.car_model.width:
                return np.inf  # collision        
            elif dist < min_distance:
                min_distance = dist
            distance_cost = 1 / (min_distance * 10)
        else:
            distance_cost = 0.0
        return distance_cost

    def __calc_to_goal_cost(self, a=0.0, delta=0.0) -> float:
        """
        Calculates the cost to reach the goal based on the given throttle and steering values.

        Args:
            a (float): Throttle value.
            delta (float): Steering value.

        Returns:
            float: The cost to reach the goal.

        """
        dx = self.goal.throttle - a
        dy = self.goal.steering - delta

        cost = np.sqrt(dx**2 + dy**2)
        return cost
    

    def __calc_control_and_trajectory(self, x) -> tuple :
        """
        Calculates the best safe control inputs and trajectory for the given state.

        Args:
            x (numpy.ndarray): The current state of the vehicle.

        Returns:
            tuple: A tuple containing the best control inputs and trajectory.
                - best_u (list): The best control inputs [acceleration, steering angle].
                - best_trajectory (numpy.ndarray): The best trajectory for the given control inputs.
        """
        min_cost = np.inf
        best_u = [0.0, 0.0]
        best_trajectory = np.array([x])

        # Variables needed for faster computation
        obstacles = [self.dilated_traj[idx] for idx in range(len(self.dilated_traj)) if idx != self.car_i]
        obstacles.append(self.borders)
        self.ob = MultiLineString(obstacles)

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
                    u_traj = [[a, float(delta)] for _ in range(int(self.delta_samples)+3)]

        best_trajectory = self.trajs[str(nearest)][str(best_u[0])][str(best_u[1])]

        if min_cost == np.inf:
            # emergency stop
            print(f"Emergency stop for vehicle {self.car_i}")
            best_u = [(0.0 - x[3])/self.dt, 0]
            trajectory = np.array([[0.0,0.0,0.0]] * int(self.ph/self.dt)) 
            best_trajectory = LineString(zip(trajectory[:, 0], trajectory[:, 1]))            
            u_traj =  np.array([best_u]*int(self.ph/self.dt))

        best_u = np.clip(best_u, [self.car_model.min_acc, -self.car_model.max_steer], [self.car_model.max_acc, self.car_model.max_steer])
        self.u_hist = u_traj
        self.predicted_trajectory = best_trajectory
        return best_u, best_trajectory
    
