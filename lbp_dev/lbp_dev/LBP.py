import matplotlib.pyplot as plt
import numpy as np
import math
from enum import Enum
# For the parameter file
import pathlib
import json
import yaml
# from custom_message.msg import Coordinate
from shapely.geometry import Point, Polygon, LineString, MultiLineString
from shapely import intersection, distance
from shapely.plotting import plot_polygon, plot_line

from bumper_cars.classes.State import State
from bumper_cars.classes.Controller import Controller
from bumper_cars.utils import car_utils as utils
from lar_msgs.msg import CarControlStamped, CarStateStamped
from typing import List
import os
from scipy.interpolate import interp1d

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

        self.a_resolution = yaml_object["LBP"]["a_resolution"] # [m/ss]
        self.v_resolution = yaml_object["LBP"]["v_resolution"] # [m/s]
        self.delta_samples = yaml_object["LBP"]["delta_samples"] # Number of delta samples
        self.to_goal_cost_gain = yaml_object["LBP"]["to_goal_cost_gain"]
        self.obstacle_cost_gain = yaml_object["LBP"]["obstacle_cost_gain"]
        self.dilation_factor = yaml_object["LBP"]["dilation_factor"]

        self.width_init = yaml_object["Simulation"]["width"]
        self.height_init = yaml_object["Simulation"]["height"]

        self.trajectory = None
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
        self.dilated_traj[self.car_i] = trajectory
        # For plotting purposes
        self.trajectory = trajectory.buffer(self.car_model.width * self.dilation_factor, cap_style=3).exterior.coords

        
        # Update history
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

    def compute_traj(self):
        temp = {}
        initial_state = State()
        initial_state.x = 0.0
        initial_state.y = 0.0
        initial_state.yaw = 0.0 #np.radians(90.0)

        for v in np.arange(self.v_resolution, self.car_model.max_speed+self.v_resolution, self.v_resolution):
            v = round(v,2)
            print(str(v)+'\t/'+str(self.car_model.max_speed+self.v_resolution))  
            temp[v] = {}

            k0 = 0.0
            nxy = 5
            nh = 3
            d = v*self.ph

            angle = self.car_model.max_steer
            a_min = - np.deg2rad(angle)
            a_max = np.deg2rad(angle)
            p_min = - np.deg2rad(angle)
            p_max = np.deg2rad(angle)
            
            states = self.__calc_uniform_polar_states(nxy, nh, d, a_min, a_max, p_min, p_max)
            result = self.__generate_path(states, k0, v)

            i = 0
            for table in result:
                xc, yc, yawc, vc, kp = self.__generate_trajectory(
                    table[3], table[4], table[5], k0, v)
                for id, element in enumerate(kp): kp[id] = np.clip(element, -self.car_model.max_steer, self.car_model.max_steer) # clipping elements withing feasible bounds
                temp[v][i] = {}
                temp[v][i]['ctrl'] = list(kp)
                temp[v][i]['x'] = xc
                temp[v][i]['y'] = yc
                temp[v][i]['yaw'] = yawc
                temp[v][i]['v'] = vc
                i +=1
            
            
            target = [[0.1, 0.5, np.deg2rad(90.0)],
                      [0.1, -0.5, np.deg2rad(-90.0)],
                      [0.3, 0.5, np.deg2rad(90.0)],
                      [0.3, -0.5, np.deg2rad(-90.0)],
                      [0.5, 0.5, np.deg2rad(90.0)],
                      [0.5, -0.5, np.deg2rad(-90.0)],
                      [0.7, 0.5, np.deg2rad(90.0)],
                      [0.7, -0.5, np.deg2rad(-90.0)],
                      [0.9, 0.5, np.deg2rad(90.0)],
                      [0.9, -0.5, np.deg2rad(-90.0)]]
            result = self.__generate_path(target, k0, v)
            i = 0
            for table in result:
                xc, yc, yawc, vc, kp = self.__generate_trajectory(
                    table[3], table[4], table[5], k0, v)
                for id, element in enumerate(kp): kp[id] = np.clip(element, -self.car_model.max_steer, self.car_model.max_steer) # clipping elements withing feasible bounds
                temp[v][i] = {}
                temp[v][i]['ctrl'] = list(kp)
                temp[v][i]['x'] = xc
                temp[v][i]['y'] = yc
                temp[v][i]['yaw'] = yawc
                temp[v][i]['v'] = vc
                i +=1

        # Compute mean and std for steering normal sampling
        mean = 0.0
        std = self.car_model.max_steer/2.0

        # Precompute steering angles
        other_delta = abs(np.random.normal(mean, std, self.delta_samples))
        init_delta = np.array([0.0, -self.car_model.max_steer, self.car_model.max_steer])
        delta_list = np.hstack((init_delta, other_delta))

        for v in np.arange(self.car_model.min_speed, self.v_resolution, self.v_resolution):
            v = round(v,2)
            x_init = [0.0, 0.0, 0.0, v]
            i = 0
            temp[v] = {}
            cmd = CarControlStamped()
            for delta in delta_list:
                initial_state.v = v
                cmd.throttle = 0.0
                cmd.steering = np.interp(delta, [-self.car_model.max_steer, self.car_model.max_steer], [-1, 1])
                
                traj = self.__calc_trajectory(initial_state, cmd)
                # plt.plot(traj[:, 0], traj[:, 1])
                xc = traj[:, 0]
                yc = traj[:, 1]
                yawc = traj[:, 2]
                vc = traj[:, 3]
                kp = [delta]*len(xc)

                temp[v][i] = {}
                temp[v][i]['ctrl'] = list(kp)
                temp[v][i]['x'] = list(xc)
                temp[v][i]['y'] = list(yc)
                temp[v][i]['yaw'] = list(yawc)
                temp[v][i]['v'] = list(vc)
                i +=1
                # print(f'len: {len(xc)}')
                
                # Repeat for other side
        
                initial_state.v = v
                cmd.throttle = 0.0
                cmd.steering = np.interp(-delta, [-self.car_model.max_steer, self.car_model.max_steer], [-1, 1])
                
                traj = self.__calc_trajectory(initial_state, cmd)
                # plt.plot(traj[:, 0], traj[:, 1])
                xc = traj[:, 0]
                yc = traj[:, 1]
                yawc = traj[:, 2]
                vc = traj[:, 3]
                kp = [delta]*len(xc)

                temp[v][i] = {}
                temp[v][i]['ctrl'] = list(kp)
                temp[v][i]['x'] = list(xc)
                temp[v][i]['y'] = list(yc)
                temp[v][i]['yaw'] = list(yawc)
                temp[v][i]['v'] = list(vc)
                i +=1

        # saving the complete trajectories to a csv file
        with open(self.dir_path + '/../config/LBP.json', 'w') as file:
            json.dump(temp, file, indent=4)

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

    def __initialize_paths_targets_dilated_traj(self):
        """
        Initialize the paths, targets, and dilated trajectory.
        """
        # Read pre-computed trajectories
        with open(self.dir_path + '/../config/LBP.json', 'r') as file:
            self.data = json.load(file)

        # Store them in dictionary for constant time access
        self.trajs = {}
        for v, trajectories in self.data.items():
            self.trajs[str(v)] = {}
            # Plot each trajectory
            for a, traj_data in trajectories.items():
                self.trajs[str(v)][str(a)] = {}
                delta = traj_data['ctrl'][1]    #Sample the second input from the control sequence
                traj_x = np.array(traj_data['x'])
                traj_y = np.array(traj_data['y'])
                line = LineString(zip(traj_x, traj_y))
                self.trajs[str(v)][str(a)][str(delta)] = line
        
        # Initialize traj for each vehicle
        for i in range(self.curr_state.shape[0]):
            self.dilated_traj.append(Point(self.curr_state[i, 0], self.curr_state[i, 1]))#.buffer(self.dilation_factor, cap_style=3))

        # Initialize the predicted trajectory
        self.predicted_trajectory = np.array([self.curr_state[self.car_i]]*int(self.ph/self.dt))


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
        
        return v_poss

    def __calc_control_and_trajectory(self, x):
        """
        Calculates the final input with LBP method.

        Args:
            x (list): The current state of the system.
        Returns:
            tuple: A tuple containing the best control input, the best trajectory, and the control input history.
        """

        min_cost = np.inf
        best_u = [0.0, 0.0]
        best_trajectory = np.array([x])

        # Variables needed for faster computation
        obstacles = [self.dilated_traj[idx] for idx in range(len(self.dilated_traj)) if idx != self.car_i]
        obstacles.append(self.borders)
        self.ob = MultiLineString(obstacles)

        # evaluate all trajectory with sampled input in dynamic window
        v_range = self.data.keys()
        v_range = [float(v) for v in v_range]
        nearest = utils.find_nearest(v_range, x[3])
        vsearch = self.__calc_dynamic_window(x)
        
        for v in vsearch:
            v = round(v,2)
            dict = self.trajs[str(v)]
            for i in dict.keys():
                for delta in self.trajs[str(v)][i].keys():
                    a = (v - x[3])/self.dt * self.car_model.acc_gain * self.goal.throttle
                    # calc cost
                    to_goal_cost = self.to_goal_cost_gain * self.__calc_to_goal_cost(float(a), float(delta))
                    ob_cost = self.obstacle_cost_gain * self.__calc_obstacle_cost(str(v),i,delta)
                    final_cost = to_goal_cost + ob_cost
                    # search minimum trajectory
                    
                    if min_cost >= final_cost:
                        min_cost = final_cost
                        best_u = [float(a), float(delta)]
                        u_traj = [[float(a), float(delta)] for _ in range(int(self.delta_samples)+3)]


                        best_trajectory = self.trajs[str(v)][str(i)][delta]

        # # Check if previous trajectory was better
        # if len(self.u_hist) > self.emergency_brake_distance:  
        #     _, self.u_hist = self.u_hist[0], self.u_hist[1:]

        #     trajectory_buf = self.predicted_trajectory

        #     to_goal_cost = self.to_goal_cost_gain * self.__calc_to_goal_cost(self.u_hist[0][0], self.u_hist[0][1])
        #     ob_cost = self.obstacle_cost_gain * self.__calc_obstacle_cost(dilated=trajectory_buf)

        #     final_cost = to_goal_cost + ob_cost 

        #     if min_cost >= final_cost:      
        #         min_cost = final_cost
        #         best_u = self.u_hist[0]
        #         best_trajectory = trajectory_buf
        #         u_traj = self.u_hist
        # el
        if min_cost == np.inf:
            # emergency stop
            print(f"Emergency stop for vehicle {self.car_i}")
            best_u = [(0.0 - x[3])/self.dt, 0]
            trajectory = np.array([x[0:3], x[0:3]] * int(self.ph/self.dt)) 
            line = LineString(zip(trajectory[:, 0], trajectory[:, 1]))
            best_trajectory = line#.buffer(self.dilation_factor, cap_style=3)
            u_traj =  np.array([best_u]*int(self.ph/self.dt))

        best_u = np.clip(best_u, [self.car_model.min_acc, -self.car_model.max_steer], [self.car_model.max_acc, self.car_model.max_steer])
        self.u_hist = u_traj
        self.predicted_trajectory = best_trajectory
        
        return best_u, best_trajectory

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
    
    
    def __calc_uniform_polar_states(self, nxy, nh, d, a_min, a_max, p_min, p_max):
        """
        Calculate the uniform polar states based on the given parameters.

        Parameters
        ----------
        nxy : int
            Number of position sampling.
        nh : int
            Number of heading sampling.
        d : float
            Distance of terminal state.
        a_min : float
            Position sampling min angle.
        a_max : float
            Position sampling max angle.
        p_min : float
            Heading sampling min angle.
        p_max : float
            Heading sampling max angle.

        Returns
        -------
        list
            List of uniform polar states.

        """
        angle_samples = [i / (nxy - 1) for i in range(nxy)]
        states = self.__sample_states(angle_samples, a_min, a_max, d, p_max, p_min, nh)

        return states


    def __sample_states(self, angle_samples, a_min, a_max, d, p_max, p_min, nh):
        """
        Generate a list of states based on the given parameters.

        Args:
            angle_samples (list): List of angle samples.
            a_min (float): Minimum angle value.
            a_max (float): Maximum angle value.
            d (float): Distance value.
            p_max (float): Maximum yaw value.
            p_min (float): Minimum yaw value.
            nh (int): Number of yaw samples.

        Returns:
            list: List of states, each represented as [xf, yf, yawf].
        """
        states = []
        for i in angle_samples:
            a = a_min + (a_max - a_min) * i

            for j in range(nh):
                xf = d * math.cos(a)
                yf = d * math.sin(a)
                if nh == 1:
                    yawf = (p_max - p_min) / 2 + a
                else:
                    yawf = p_min + (p_max - p_min) * j / (nh - 1) + a
                states.append([xf, yf, yawf])

        return states



    def __generate_path(self, target_states, k0, v, k=False):
        """
        Generates a path based on the given target states, initial steering angle, velocity, and a flag indicating whether to use a specific value for k.

        Args:
            target_states (list): List of target states [x, y, yaw].
            k0 (float): Initial steering angle.
            v (float): Velocity.
            k (bool, optional): Flag indicating whether to use a specific value for k. Defaults to False.

        Returns:
            list: List of generated paths [x, y, yaw, p, kp].
        """
        
        result = []

        for state in target_states:
            target = State(x=state[0], y=state[1], yaw=state[2])
            initial_state = State(x=0.0, y=0.0, yaw=0.0, v=0.0)
            throttle, delta = utils.pure_pursuit_steer_control([target.x,target.y], initial_state)
            k0 = delta
            if k:
                km = delta/2
            else:
                km = 0.0
            init_p = np.array(
                [np.hypot(state[0], state[1]), km, 0.0]).reshape(3, 1)

            x, y, yaw, p, kp = self.__optimize_trajectory(target, k0, init_p, v)

            if x is not None:
                # print("find good path")
                result.append(
                    [x[-1], y[-1], yaw[-1], float(p[0, 0]), float(p[1, 0]), float(p[2, 0])])

        # print("finish path generation")
        return result
    
    def __generate_trajectory(self, s, km, kf, k0, v):
        """
        Generate a trajectory based on the given parameters.

        Args:
            s (float): The distance to be covered.
            km (float): The middle curvature.
            kf (float): The final curvature.
            k0 (float): The initial curvature.
            v (float): The velocity.

        Returns:
            tuple: A tuple containing the x-coordinates, y-coordinates, yaw angles, and curvature values of the generated trajectory.
        """

        # n = s / ds
        time = s / v  # [s]
        n = time / self.dt


        if isinstance(time, type(np.array([]))):
            time = time[0]
        if isinstance(km, type(np.array([]))):
            km = km[0]
        if isinstance(kf, type(np.array([]))):
            kf = kf[0]

        tk = np.array([0.0, time / 2.0, time])
        kk = np.array([k0, km, kf])
        t = np.arange(0.0, time, time / n)
        fkp = interp1d(tk, kk, kind="quadratic")
        kp = [float(fkp(ti)) for ti in t]
        # self.dt = abs(float(time / n))

        #  plt.plot(t, kp)
        #  plt.show()

        state = State()
        x, y, yaw, vi = [state.x], [state.y], [state.yaw], [state.v]
        for ikp in kp:
            self.__update(state, v, ikp, self.dt, self.car_model.L)
            x.append(state.x)
            y.append(state.y)
            yaw.append(state.yaw)
            vi.append(state.v)

        return x, y, yaw, vi, kp
    
    def __optimize_trajectory(self, target, k0, p, v):
        """
        Optimize the trajectory to reach the target position.

        Args:
            target (tuple): The target position (x, y, yaw).
            k0 (float): The initial curvature.
            p (numpy.ndarray): The initial trajectory parameters.
            v (float): The velocity.

        Returns:
            tuple: The optimized trajectory (xc, yc, yawc, p, kp).

        Raises:
            LinAlgError: If the path calculation encounters a linear algebra error.
        """
        max_iter = 100
        cost_th = 0.12
        h = np.array([0.3, 0.02, 0.02]).T  # parameter sampling distance

        for i in range(max_iter):
            xc, yc, yawc, _, kp = self.__generate_trajectory(p[0, 0], p[1, 0], p[2, 0], k0, v)
            dc = np.array(self.__calc_diff(target, xc, yc, yawc)).reshape(3, 1)

            cost = np.linalg.norm(dc)
            if cost <= cost_th:
                # print("path is ok cost is:" + str(cost))
                break

            J = self.__calc_j(target, p, h, k0, v)
            try:
                dp = - np.linalg.pinv(J) @ dc
            except np.linalg.linalg.LinAlgError:
                # print("cannot calc path LinAlgError")
                xc, yc, yawc, p = None, None, None, None
                break
            alpha = self.__selection_learning_param(dp, p, k0, target, v)

            p += alpha * np.array(dp)
            # print(p.T)

        else:
            xc, yc, yawc, p = None, None, None, None
            # print("cannot calc path")

        return xc, yc, yawc, p, kp
    
    def __calc_j(self, target, p, h, k0, v):
        """
        Calculate the Jacobian matrix J for a given target and state vector p.

        Args:
            target (list): List of target coordinates [x, y, yaw].
            p (numpy.ndarray): Optimization parameters vector [s, km, kf].
            h (numpy.ndarray): Step sizes for numerical differentiation.
            k0 (float): Curvature of the motion model.
            v (float): Velocity of the motion model.

        Returns:
            numpy.ndarray: Jacobian matrix J.

        """
        xp, yp, yawp = self.__generate_last_state(
            p[0, 0] + h[0], p[1, 0], p[2, 0], k0, v)
        dp = self.__calc_diff(target, [xp], [yp], [yawp])
        xn, yn, yawn = self.__generate_last_state(
            p[0, 0] - h[0], p[1, 0], p[2, 0], k0, v)
        dn = self.__calc_diff(target, [xn], [yn], [yawn])
        d1 = np.array((dp - dn) / (2.0 * h[0])).reshape(3, 1)

        xp, yp, yawp = self.__generate_last_state(
            p[0, 0], p[1, 0] + h[1], p[2, 0], k0, v)
        dp = self.__calc_diff(target, [xp], [yp], [yawp])
        xn, yn, yawn = self.__generate_last_state(
            p[0, 0], p[1, 0] - h[1], p[2, 0], k0, v)
        dn = self.__calc_diff(target, [xn], [yn], [yawn])
        d2 = np.array((dp - dn) / (2.0 * h[1])).reshape(3, 1)

        xp, yp, yawp = self.__generate_last_state(
            p[0, 0], p[1, 0], p[2, 0] + h[2], k0, v)
        dp = self.__calc_diff(target, [xp], [yp], [yawp])
        xn, yn, yawn = self.__generate_last_state(
            p[0, 0], p[1, 0], p[2, 0] - h[2], k0, v)
        dn = self.__calc_diff(target, [xn], [yn], [yawn])
        d3 = np.array((dp - dn) / (2.0 * h[2])).reshape(3, 1)

        J = np.hstack((d1, d2, d3))

        return J
    
    def __calc_diff(self, target, x, y, yaw):
        pi_2_pi = target.yaw - yaw[-1]
        pi_2_pi = (pi_2_pi + math.pi) % (2 * math.pi) - math.pi
        d = np.array([target.x - x[-1],
                    target.y - y[-1],
                    pi_2_pi])

        return d
    
    def __generate_last_state(self, s, km, kf, k0, v):
        """
        Generates the last state of the motion model based on the given parameters.

        Args:
            s (float): The distance traveled.
            km (float): The middle curvature.
            kf (float): The final curvature.
            k0 (float): The initial curvature.
            v (float): The velocity.

        Returns:
            tuple: A tuple containing the x-coordinate, y-coordinate, and yaw angle of the last state.
        """
        ds = 0.1 #TODO: Increment the lookahead distance
        n = s / ds
        time = abs(s / v)  # [s]

        if isinstance(n, type(np.array([]))):
            n = n[0]
        if isinstance(time, type(np.array([]))):
            time = time[0]
        if isinstance(km, type(np.array([]))):
            km = km[0]
        if isinstance(kf, type(np.array([]))):
            kf = kf[0]

        tk = np.array([0.0, time / 2.0, time])
        kk = np.array([k0, km, kf])
        t = np.arange(0.0, time, time / n)
        fkp = interp1d(tk, kk, kind="quadratic")
        kp = [fkp(ti) for ti in t]
        dt = time / n

        # plt.plot(t, kp)
        # plt.show()

        state = State()

        _ = [self.__update(state, v, ikp, dt, self.car_model.L) for ikp in kp]

        return state.x, state.y, state.yaw
    

    def __selection_learning_param(self, dp, p, k0, target, v):
        """
        Selects the learning parameter 'a' that minimizes the cost function.

        Args:
            dp (float): The change in parameter 'p'.
            p (float): The current value of parameter 'p'.
            k0 (float): The value of parameter 'k0'.
            target (float): The target value.
            v (float): The value of parameter 'v'.

        Returns:
            float: The selected value of parameter 'a'.
        """

        mincost = float("inf")
        mina = 1.0
        maxa = 2.0
        da = 0.5

        for a in np.arange(mina, maxa, da):
            tp = p + a * dp
            xc, yc, yawc = self.__generate_last_state(
                tp[0], tp[1], tp[2], k0, v)
            dc = self.__calc_diff(target, [xc], [yc], [yawc])
            cost = np.linalg.norm(dc)

            if cost <= mincost and a != 0.0:
                mina = a
                mincost = cost

        return mina
    
        #TODO: remove this function
    def __update(self, state, v, delta, dt, L):
        state.v = v
        delta = np.clip(delta, -np.radians(45), np.radians(45)) #TODO remove hardcoded limits
        state.x = state.x + state.v * math.cos(state.yaw) * dt
        state.y = state.y + state.v * math.sin(state.yaw) * dt
        state.yaw = state.yaw + state.v / L * math.tan(delta) * dt
        state.yaw = utils.normalize_angle(state.yaw)
        # state.yaw = pi_2_pi(state.yaw)

        return state