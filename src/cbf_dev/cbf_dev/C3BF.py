import numpy as np

from cvxopt import matrix, solvers
from cvxopt import matrix
from piqp import DenseSolver, PIQP_SOLVED
from planner import utils as utils
# import planner.utils as utils

from custom_message.msg import ControlInputs, State, MultiControl, Coordinate

# For the parameter file
import pathlib
import json
import math
import matplotlib.pyplot as plt 
import time
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
path = pathlib.Path(dir_path + '/../../bumper_cars/params.json')

with open(path, 'r') as openfile:
    # Reading from json file
    json_object = json.load(openfile)

L = json_object["Car_model"]["L"]
max_steer = json_object["C3BF"]["max_steer"]  # [rad] max steering angle
max_speed = json_object["Car_model"]["max_speed"] # [m/s]
min_speed = json_object["Car_model"]["min_speed"] # [m/s]
max_acc = json_object["C3BF"]["max_acc"] 
min_acc = json_object["C3BF"]["min_acc"] 
car_max_acc = json_object["Controller"]["max_acc"]
car_min_acc = json_object["Controller"]["min_acc"]
dt = json_object["Controller"]["dt"]
safety_radius = json_object["C3BF"]["safety_radius"]
barrier_gain = json_object["C3BF"]["barrier_gain"]
arena_gain = json_object["C3BF"]["arena_gain"]
Kv = json_object["C3BF"]["Kv"] # interval [0.5-1]
Lr = L / 2.0  # [m]
Lf = L - Lr
WB = json_object["Controller"]["WB"]
robot_num = 5 #json_object["robot_num"]
safety_init = json_object["safety"]
width_init = json_object["width"]
height_init = json_object["height"]
min_dist = json_object["min_dist"]
to_goal_stop_distance = json_object["to_goal_stop_distance"]
boundary_points = np.array([-width_init/2, width_init/2, -height_init/2, height_init/2])
check_collision_bool = False
add_noise = json_object["add_noise"]
noise_scale_param = json_object["noise_scale_param"]
show_animation = json_object["show_animation"]
linear_model = json_object["linear_model"]
np.random.seed(1)

color_dict = {0: 'r', 1: 'b', 2: 'g', 3: 'y', 4: 'm', 5: 'c', 6: 'k', 7: 'tab:orange', 8: 'tab:brown', 9: 'tab:gray', 10: 'tab:olive', 11: 'tab:pink', 12: 'tab:purple', 13: 'tab:red', 14: 'tab:blue', 15: 'tab:green'}
with open(dir_path + '/../../seeds/circular_seed_4.json', 'r') as file:
    data = json.load(file)

def update_paths(paths):
    """
    Updates the given paths.

    Args:
        paths (list): List of paths.

    Returns:
        list: Updated paths.

    """
    updated_paths = []
    for path in paths:
        updated_paths.append(utils.update_path(path))
    return updated_paths

def update_robot_state(i, x, dxu, multi_control, targets):
    """
    Updates the state of all robots.

    Args:
        x (numpy.ndarray): State vector of shape (4, N), where N is the number of time steps.
        dxu (numpy.ndarray): Control inputs of shape (2, N).
        multi_control (MultiControl): Object for storing control inputs for all robots.
        targets (list): List of target positions for each robot.
    
    Returns:
        tuple: Updated state of all robots and updated multi_control object.

    """
    cmd = ControlInputs()

    x1 = utils.array_to_state(x[:, i])
    cmd.throttle, cmd.delta = dxu[0, i], dxu[1, i]
    x1 = utils.linear_model_callback(x1, cmd)
    x1 = utils.state_to_array(x1).reshape(4)
    x[:, i] = x1
    multi_control.multi_control[i] = cmd

    utils.plot_robot_and_arrows(i, x, multi_control, targets)
    
    return x, multi_control

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
        print(f"Vehicle {i} reached goal!")
        return True
    return False

class C3BF_algorithm():
    
    def __init__(self, targets, paths, ax, robot_num=robot_num):
        self.targets = targets
        self.paths = paths
        self.robot_num = robot_num
        self.dxu = np.zeros((2, self.robot_num))
        self.reached_goal = [False]*robot_num
        self.computational_time = []
        self.solver_failure = 0
        self.ax = ax

    def debug_info(self):
        print(f"Solver failure: {self.solver_failure}\n")
        print(f'Paths: {self.paths}\n')
        print(f'Targets: {self.targets}\n')
        print(f'Robot num: {self.robot_num}\n')
        print(f'Computational time: {self.computational_time}\n')
        print(f'Reached goal: {self.reached_goal}\n')
        print(f'dxu: {self.dxu}\n')
        print(f'Solver failure: {self.solver_failure}\n')

    def run_3cbf(self, x, break_flag):
        for i in range(self.robot_num):
            if not self.reached_goal[i]:
                # Step 9: Check if the distance between the current position and the target is less than 5
                if utils.dist(point1=(x[0,i], x[1,i]), point2=self.targets[i]) < 2:
                    # Perform some action when the condition is met
                    self.paths[i].pop(0)
                    if not self.paths[i]:
                        print(f"Path complete for vehicle {i}!")
                        self.dxu[:, i] = 0
                        x[3,i] = 0
                        self.reached_goal[i] = True
                    
                    else: 
                        self.targets[i] = (self.paths[i][0].x, self.paths[i][0].y)
                
                if check_goal_reached(x, self.targets, i):
                    # print(f"Vehicle {i} reached goal!!")
                    self.dxu[:, i] = 0
                    x[3,i] = 0
                    self.reached_goal[i] = True
                else:
                    t_prev = time.time()
                    if add_noise:
                        noise = np.concatenate([np.random.normal(0, 0.21*noise_scale_param, 2).reshape(2, 1), np.random.normal(0, np.radians(5)*noise_scale_param, 1).reshape(1,1), np.random.normal(0, 0.2*noise_scale_param, 1).reshape(1,1), np.random.normal(0, 0.02*noise_scale_param, 1).reshape(1,1)], axis=0)
                        noisy_pos = x + noise
                        self.control_robot(i, noisy_pos)
                        plt.plot(noisy_pos[0,i], noisy_pos[1,i], "x", color=color_dict[i], markersize=10)
                    else:
                        self.control_robot(i, x)

                    self.computational_time.append((time.time() - t_prev))

                    if linear_model:
                        x[:, i] = utils.motion(x[:, i], self.dxu[:, i], dt)
                    else:
                        x[:, i] = utils.nonlinear_model_numpy_stable(x[:, i], self.dxu[:, i])
                    # if x[3,i] < 0.0:
                    #     print(f"Negative speed for robot {i}!")
                x = self.check_collision(x, i)

            utils.plot_robot_and_arrows(i, x, self.dxu, self.targets, self.ax)

        if all(self.reached_goal):
            break_flag = True
        return x, break_flag
    
    def go_to_goal(self, x, break_flag):
        for i in range(self.robot_num):
            # Step 9: Check if the distance between the current position and the target is less than 5
            if not self.reached_goal[i]:
                if check_goal_reached(x, self.targets, i):
                    # print(f"Vehicle {i} reached goal!!")
                    self.dxu[:, i] = 0
                    x[3,i] = 0
                    self.reached_goal[i] = True
                else:
                    t_prev = time.time()
                    if add_noise:
                        noise = np.concatenate([np.random.normal(0, 0.21*noise_scale_param, 2).reshape(2, 1), np.random.normal(0, np.radians(5)*noise_scale_param, 1).reshape(1,1), np.random.normal(0, 0.2*noise_scale_param, 1).reshape(1,1), np.random.normal(0, 0.02*noise_scale_param, 1).reshape(1,1)], axis=0)
                        noisy_pos = x + noise
                        self.control_robot(i, noisy_pos)
                        plt.plot(noisy_pos[0,i], noisy_pos[1,i], "x", color=color_dict[i], markersize=10)
                    else:
                        self.control_robot(i, x)

                    self.computational_time.append((time.time() - t_prev))
            
                    if linear_model:
                        x[:, i] = utils.motion(x[:, i], self.dxu[:, i], dt)
                    else:
                        x[:, i] = utils.nonlinear_model_numpy_stable(x[:, i], self.dxu[:, i])
                    # if x[3,i] < 0.0:
                    #     print(f"Negative speed for robot {i}!")
                x = self.check_collision(x, i) 
            # If we want the robot to disappear when it reaches the goal, indent one more time
            if all(self.reached_goal):
                break_flag = True

            utils.plot_robot_and_arrows(i, x, self.dxu, self.targets, self.ax)
            # print(f"Speed of robot {i}: {x[3, i]}")

        return x, break_flag
    
    def control_robot(self, i, x):
        """
        Controls the movement of a robot based on its current state and target positions.

        Args:
            i (int): Index of the robot.
            x (numpy.ndarray): Array representing the current state of all robots.
            targets (list): List of target positions for each robot.
            robot_num (int): Total number of robots.
            multi_control (MultiControl): Object for storing control inputs for all robots.
            paths (list): List of paths for each robot.

        Returns:
            tuple: Updated state of all robots, updated target positions, and updated multi_control object.
        """
        # dxu = np.zeros((2, self.robot_num))

        cmd = ControlInputs()
        
        # x = self.check_collision(x, i)
        x1 = utils.array_to_state(x[:, i])
        cmd.throttle, cmd.delta = utils.pure_pursuit_steer_control(self.targets[i], x1)
        self.dxu[0, i], self.dxu[1, i] = cmd.throttle, cmd.delta

        self.C3BF(i, x)
    
    def closest_boundary_point(self, i, x):
        """
        Finds the closest boundary point to the robot.

        Args:
            i (int): Index of the robot.
            x (numpy.ndarray): Array representing the current state of all robots.

        Returns:
            numpy.ndarray: Closest boundary point.

        """
        dx = 0
        dy = 0

        if abs(x[0,i] - boundary_points[0]) < abs(x[0,i] - boundary_points[1]):
            dx = boundary_points[0] - x[0,i]
            cpx = boundary_points[0]
        else:
            dx = boundary_points[1] - x[0,i]
            cpx = boundary_points[1]
        
        if abs(x[1,i] - boundary_points[2]) < abs(x[1,i] - boundary_points[3]):
            dy = boundary_points[2] - x[1,i]
            cpy = boundary_points[2]
        else:
            dy = boundary_points[3] - x[1,i]
            cpy = boundary_points[3]

        if abs(dx) < abs(dy):
            closest_point = np.array([cpx, x[1,i], np.pi/2])  
            closest_point2 = np.array([cpx, x[1,i]+safety_radius, np.pi/2])
            closest_point3 = np.array([cpx, x[1,i]-safety_radius, np.pi/2])
        
        # elif abs(dx) == abs(dy):
        #     if cpx == boundary_points[0]:
        #         if cpy == boundary_points[2]:
        #             closest_point = np.array([cpx, cpy, np.pi/4])
        #             closest_point2 = np.array([cpx, cpy+safety_radius, np.pi/2])
        #             closest_point3 = np.array([cpx+safety_radius, cpy, 0.0])
        #         else:
        #             closest_point = np.array([cpx, cpy, -np.pi/4])
        #             closest_point2 = np.array([cpx, cpy-safety_radius, np.pi])
        #             closest_point3 = np.array([cpx+safety_radius, cpy, 0.0])
        #     elif cpx == boundary_points[1]:
        #         if cpy == boundary_points[2]:
        #             closest_point = np.array([cpx, cpy, 3*np.pi/4])
        #             closest_point2 = np.array([cpx, cpy+safety_radius, np.pi])
        #             closest_point3 = np.array([cpx-safety_radius, cpy, 0.0])
        #         else:
        #             closest_point = np.array([cpx, cpy, -3*np.pi/4])
        #             closest_point2 = np.array([cpx, cpy-safety_radius, np.pi/2])
        #             closest_point3 = np.array([cpx-safety_radius, cpy, -np.pi])
              
        else:
            closest_point = np.array([x[0,i], cpy, 0.0])
            closest_point2 = np.array([x[0,i]+safety_radius, cpy, 0.0])
            closest_point3 = np.array([x[0,i]-safety_radius, cpy, 0.0])
        
        return closest_point, closest_point2, closest_point3
  
    def C3BF(self, i, x):
        """
        Computes the control input for the C3BF (Collision Cone Control Barrier Function) algorithm.

        Args:
            x (numpy.ndarray): State vector of shape (4, N), where N is the number of time steps.
            u_ref (numpy.ndarray): Reference control input of shape (2, N).

        Returns:
            numpy.ndarray: Filtered Control input dxu of shape (2, N).

        """
        # start = time.time()

        flag = True
        N = x.shape[1]
        M = self.dxu.shape[0]
        self.dxu[1,i] = utils.delta_to_beta(self.dxu[1,i])

        # if i == 0:
        closest_point, closest_point2, closest_point3 = self.closest_boundary_point(i, x)
        # circle = plt.Circle(closest_point[0:2], safety_radius/1.5, color='r', fill=False)
        # circle2 = plt.Circle(closest_point2[0:2], safety_radius/1.5, color='b', fill=False)
        # circle3 = plt.Circle(closest_point3[0:2], safety_radius/1.5, color='b', fill=False)
        # utils.plot_arrow(closest_point[0], closest_point[1], closest_point[2], length=2, width=1, color=color_dict[i])
        # self.ax.add_patch(circle)
        # self.ax.add_patch(circle2)
        # self.ax.add_patch(circle3)

        count = 0
        G = np.zeros([N-1 + 9,M])
        H = np.zeros([N-1 + 9,1]) 

        f = np.array([x[3,i]*np.cos(x[2,i]),
                            x[3,i]*np.sin(x[2,i]), 
                            0, 
                            0]).reshape(4,1)
        g = np.array([[0, -x[3,i]*np.sin(x[2,i])], 
                        [0, x[3,i]*np.cos(x[2,i])], 
                        [0, x[3,i]/Lr],
                        [1, 0]]).reshape(4,2)
        
        P = np.identity(2)*2
        q = np.array([-2 * self.dxu[0, i], - 2 * self.dxu[1,i]])
        
        for j in range(N):
            arr = np.array([x[0, j] - x[0, i], x[1, j] - x[1,i]])
            dist = np.linalg.norm(arr)

            if j == i or dist > 2 * safety_radius: 
                continue

            v_rel = np.array([x[3,j]*np.cos(x[2,j]) - x[3,i]*np.cos(x[2,i]), 
                                x[3,j]*np.sin(x[2,j]) - x[3,i]*np.sin(x[2,i])])
            p_rel = np.array([x[0,j]-x[0,i],
                                x[1,j]-x[1,i]])
            
            cos_Phi = np.sqrt(abs(np.linalg.norm(p_rel)**2 - safety_radius**2))/np.linalg.norm(p_rel)
            tan_Phi_sq = safety_radius**2 / (np.linalg.norm(p_rel)**2 - safety_radius**2)
            
            h = np.dot(p_rel, v_rel) + np.linalg.norm(v_rel) * np.linalg.norm(p_rel) * cos_Phi
            
            gradH_1 = np.array([- (x[3,j]*np.cos(x[2,j]) - x[3,i]*np.cos(x[2,i])), 
                                - (x[3,j]*np.sin(x[2,j]) - x[3,i]*np.sin(x[2,i])),
                                x[3,i] * (np.sin(x[2,i]) * p_rel[0] - np.cos(x[2,i]) * p_rel[1]),
                                -np.cos(x[2,i]) * p_rel[0] - np.sin(x[2,i]) * p_rel[1]])
            
            gradH_21 = -(1 + tan_Phi_sq) * np.linalg.norm(v_rel)/np.linalg.norm(p_rel) * cos_Phi * p_rel 
            gradH_22 = np.dot(np.array([x[3,i]*np.sin(x[2,i]), -x[3,i]*np.cos(x[2,i])]), v_rel) * np.linalg.norm(p_rel)/(np.linalg.norm(v_rel) + 0.00001) * cos_Phi
            gradH_23 = - np.dot(v_rel, np.array([np.cos(x[2,i]), np.sin(x[2,i])])) * np.linalg.norm(p_rel)/(np.linalg.norm(v_rel) + 0.00001) * cos_Phi

            gradH = gradH_1.reshape(4,1) + np.vstack([gradH_21.reshape(2,1), gradH_22, gradH_23])

            Lf_h = np.dot(gradH.T, f)
            Lg_h = np.dot(gradH.T, g)

            H[count] = np.array([barrier_gain*np.power(h, 3) + Lf_h])
            G[count,:] = -Lg_h
            count+=1

        # Closes point 1
        Lf_h = 2 * x[3, i] * (np.cos(x[2, i]) * (x[0, i] - closest_point[0]) + np.sin(x[2, i]) * (x[1, i] - closest_point[1]))
        Lg_h = 2 * x[3, i] * (np.cos(x[2, i]) * (x[1, i] - closest_point[1]) - np.sin(x[2, i]) * (x[0, i] - closest_point[0]))
        h = (x[0, i] - closest_point[0]) * (x[0, i] - closest_point[0]) + (x[1, i] - closest_point[1]) * (x[1, i] - closest_point[1]) - (
                    safety_radius ** 2 + Kv * abs(x[3, i]))

        H[count] = np.array([barrier_gain * np.power(h, 3) + Lf_h])

        if x[3, i] >= 0:
            # G = np.vstack([G, [Kv, -Lg_h]])
            G[count, :] = np.array([Kv, -Lg_h])
        else:
            # G = np.vstack([G, [-Kv, -Lg_h]])
            G[count, :] = np.array([-Kv, -Lg_h])
        count+=1
        
        # Closest point 2
        Lf_h = 2 * x[3, i] * (np.cos(x[2, i]) * (x[0, i] - closest_point2[0]) + np.sin(x[2, i]) * (x[1, i] - closest_point2[1]))
        Lg_h = 2 * x[3, i] * (np.cos(x[2, i]) * (x[1, i] - closest_point2[1]) - np.sin(x[2, i]) * (x[0, i] - closest_point2[0]))
        h = (x[0, i] - closest_point2[0]) * (x[0, i] - closest_point2[0]) + (x[1, i] - closest_point2[1]) * (x[1, i] - closest_point2[1]) - (
                    safety_radius ** 2 + Kv * abs(x[3, i]))

        H[count] = np.array([barrier_gain * np.power(h, 3) + Lf_h])

        if x[3, i] >= 0:
            # G = np.vstack([G, [Kv, -Lg_h]])
            G[count, :] = np.array([Kv, -Lg_h])
        else:
            # G = np.vstack([G, [-Kv, -Lg_h]])
            G[count, :] = np.array([-Kv, -Lg_h])
        count+=1
        
        # Closest point 3
        Lf_h = 2 * x[3, i] * (np.cos(x[2, i]) * (x[0, i] - closest_point3[0]) + np.sin(x[2, i]) * (x[1, i] - closest_point3[1]))
        Lg_h = 2 * x[3, i] * (np.cos(x[2, i]) * (x[1, i] - closest_point3[1]) - np.sin(x[2, i]) * (x[0, i] - closest_point3[0]))
        h = (x[0, i] - closest_point3[0]) * (x[0, i] - closest_point3[0]) + (x[1, i] - closest_point3[1]) * (x[1, i] - closest_point3[1]) - (
                    safety_radius ** 2 + Kv * abs(x[3, i]))

        H[count] = np.array([barrier_gain * np.power(h, 3) + Lf_h])

        if x[3, i] >= 0:
            # G = np.vstack([G, [Kv, -Lg_h]])
            G[count, :] = np.array([Kv, -Lg_h])
        else:
            # G = np.vstack([G, [-Kv, -Lg_h]])
            G[count, :] = np.array([-Kv, -Lg_h])
        count+=1
        
        # Add the input constraint
        G[count, :] = np.array([0, 1])
        H[count] = np.array([utils.delta_to_beta(max_steer)])
        count += 1

        G[count, :] = np.array([0, -1])
        H[count] = np.array([-utils.delta_to_beta(-max_steer)])
        count += 1

        G[count, :] = np.array([0, x[3,i]/Lr])
        H[count] = np.array([np.deg2rad(50)])
        count += 1

        G[count, :] = np.array([0, x[3,i]/Lr])
        H[count] = np.array([np.deg2rad(50)])
        count += 1

        G[count, :] = np.array([1, 0])
        H[count] = np.array([max_acc])
        count += 1

        G[count, :] = np.array([-1, 0])
        H[count] = np.array([-min_acc])

        start_solver = time.time()
        solver = DenseSolver()
        solver.settings.verbose = False
        solver.setup(P, q, np.empty((0, P.shape[0])), np.empty((0,)), G, H)
        status = solver.solve()
        if status == PIQP_SOLVED:
            self.dxu[:,i] = np.reshape(solver.result.x, (M,))
        else:
            print(f"QP solver failed for robot {i}! Emergency stop.") 
            self.dxu[0,i] = (0 - x[3,i])/dt 
            self.solver_failure += 1
        # end_solver = time.time()
        # end = time.time()
        # print(f"Time: {(end-start)*1e3} ms")
        # print(f"Solver time: {(end_solver-start_solver)*1e3} ms")       
        
        if show_animation:
            circle2 = plt.Circle((x[0,i], x[1,i]), safety_radius, color='b', fill=False)
            self.ax.add_patch(circle2)
    
        self.dxu[1,i] = utils.beta_to_delta(self.dxu[1,i])    

    def check_collision(self, x, i):
        """
        Checks for collision between the robot at index i and other robots.

        Args:
            x (numpy.ndarray): State vector of shape (4, N), where N is the number of time steps.
            i (int): Index of the robot to check collision for.

        Raises:
            Exception: If collision is detected.

        """
        if x[0,i]>=boundary_points[1]-WB/2 or x[0,i]<=boundary_points[0]+WB/2 or x[1,i]>=boundary_points[3]-WB/2 or x[1,i]<=boundary_points[2]+WB/2:
            if check_collision_bool:
                raise Exception('Collision')
            else:
                print("Collision detected")
                self.reached_goal[i] = True
                self.dxu[:, i] = 0
                x[3,i] = 0.0

        for idx in range(self.robot_num):
            if idx == i:
                continue
            if utils.dist([x[0,i], x[1,i]], [x[0, idx], x[1, idx]]) <= WB/2:
                if check_collision_bool:
                    raise Exception('Collision')
                else:
                    print("Collision detected")
                    self.reached_goal[i] = True
                    self.dxu[:, i] = 0.0
                    x[3,i] = 0.0
        return x

def main1(args=None):
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
        - Perform 3CBF control for each robot.
        - Plot the robot trajectory.
        - Update the predicted trajectory.
        - Plot the map and pause for visualization.
    """
    # Step 1: Set the number of iterations
    iterations = 3000
    fig = plt.figure(1, dpi=90, figsize=(20,10))
    ax = fig.add_subplot(111)
    fontsize = 40
    plt.rcParams['font.family'] = ['serif']
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = fontsize
    break_flag = False

    # Step 3: Create an array x with the initial values
    # x = np.array([[5.0, 5.0], [-2.0, 2.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    # u = np.array([[0, 0], [0, 0]])
    # targets = [[10, 2], [10, -2]]
    x = np.array([[5.0, 5.0], [5.0, -5.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    u = np.array([[0, 0], [0, 0]])
    targets = [[20.0, -20.0], [20.0, 20.0]]
    
    # Step 4: Create paths for each robot
    traj = data['trajectories']
    paths = [[Coordinate(x=traj[str(idx)][i][0], y=traj[str(idx)][i][1]) for i in range(len(traj[str(idx)]))] for idx in range(robot_num)]

    # Step 5: Extract the target coordinates from the paths
    # targets = [[path[0].x, path[0].y] for path in paths]

    c3bf = C3BF_algorithm(targets, paths, robot_num=x.shape[1], ax=ax)
    # Step 8: Perform the simulation for the specified number of iterations
    for z in range(iterations):
        plt.cla()
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        
        x, break_flag = c3bf.go_to_goal(x, break_flag) 
        
        utils.plot_map(width=width_init, height=height_init)
        # plt.axis("equal")
        plt.xlabel("x [m]", fontdict={'size': fontsize, 'family': 'serif'})
        plt.ylabel("y [m]", fontdict={'size': fontsize, 'family': 'serif'})
        plt.title('C3BF Corner Case', fontdict={'size': fontsize, 'family': 'serif'})
        
        # plt.xlim(0, 20)
        # plt.ylim(-5, 5)
        plt.axis("equal")
        plt.legend()
        plt.grid(True)
        plt.pause(0.0001)

        if break_flag:
            break

def main_seed(args=None):
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
        - Perform 3CBF control for each robot.
        - Plot the robot trajectory.
        - Update the predicted trajectory.
        - Plot the map and pause for visualization.
    """
    # Step 1: Set the number of iterations
    iterations = 3000
    fig = plt.figure(1, dpi=90, figsize=(10,10))
    ax = fig.add_subplot(111)
    break_flag = False
    
    # Step 2: Sample initial values for x0, y, yaw, v, omega, and model_type
    initial_state = data['initial_position']
    x0 = initial_state['x']
    y = initial_state['y']
    yaw = initial_state['yaw']
    v = initial_state['v']
    omega = [0.0]*len(initial_state['x'])

    # Step 3: Create an array x with the initial values
    x = np.array([x0, y, yaw, v, omega])
    u = np.zeros((2, robot_num))

    trajectory = np.zeros((x.shape[0]+u.shape[0], robot_num, 1))
    trajectory[:, :, 0] = np.concatenate((x,u))
    
    # Step 4: Create paths for each robot
    traj = data['trajectories']
    paths = [[Coordinate(x=traj[str(idx)][i][0], y=traj[str(idx)][i][1]) for i in range(len(traj[str(idx)]))] for idx in range(robot_num)]

    # Step 5: Extract the target coordinates from the paths
    targets = [[path[0].x, path[0].y] for path in paths]

    c3bf = C3BF_algorithm(targets, paths, ax=ax)
    # Step 8: Perform the simulation for the specified number of iterations
    for z in range(iterations):
        plt.cla()
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        
        x, break_flag = c3bf.run_3cbf(x, break_flag) 
        # x, break_flag = c3bf.go_to_goal(x, break_flag) 

        trajectory = np.dstack([trajectory, np.concatenate((x, c3bf.dxu))])
        
        utils.plot_map(width=width_init, height=height_init)
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)

        if break_flag:
            break
    
    print("Done")
    if show_animation:
        for i in range(robot_num):
            plt.plot(trajectory[0, i, :], trajectory[1, i, :], "-", color=color_dict[i])
        plt.pause(0.0001)
        plt.show()
        
if __name__=='__main__':
    main_seed()
    # main1()
        
