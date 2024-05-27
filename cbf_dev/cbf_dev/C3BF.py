import numpy as np

import time
from piqp import DenseSolver, PIQP_SOLVED

from lar_msgs.msg import CarControlStamped, CarStateStamped
from typing import List

from bumper_cars.utils import car_utils as utils
from bumper_cars.classes.State import State
from bumper_cars.classes.Controller import Controller

# For the parameter file
import yaml

np.random.seed(1)

class C3BF_algorithm(Controller):
    def __init__(self, controller_path:str, robot_num = 0):

        ## Init Controller class
        super().__init__(controller_path)

        ## Init public parameters
        self.robot_num = robot_num
        self.dxu = np.zeros((2,), dtype=float)
        self.solver_failure = 0

        # Opening YAML file
        with open(controller_path, 'r') as openfile:
            # Reading from yaml file
            yaml_object = yaml.safe_load(openfile)

        self.safety_radius = yaml_object["C3BF"]["safety_radius"]
        self.barrier_gain = yaml_object["C3BF"]["barrier_gain"]
        self.arena_gain = yaml_object["C3BF"]["arena_gain"]
        self.Kv = yaml_object["C3BF"]["Kv"] # interval [0.5-1]

        self.closest_point = (0.0, 0.0)

    ################# PUBLIC METHODS

    def compute_cmd(self, car_list : List[CarStateStamped]) -> CarControlStamped:
        # Init empty command
        car_cmd = CarControlStamped()

        # Update current state of all cars
        self.curr_state = np.transpose(utils.carStateStamped_to_array(car_list))

        # Compute control   
        u = self.__C3BF(self.robot_num, self.curr_state)

        # Project it to range [-1, 1]
        car_cmd.throttle = np.interp(u[0], [self.car_model.min_acc, self.car_model.max_acc], [-1, 1]) * self.car_model.acc_gain
        car_cmd.steering = np.interp(u[1], [-self.car_model.max_steer, self.car_model.max_steer], [-1, 1])

        return car_cmd

    def set_goal(self, goal: CarControlStamped) -> None:
        self.dxu[0] = np.interp(goal.throttle, [-1, 1], [self.car_model.min_acc, self.car_model.max_acc]) * self.car_model.acc_gain
        self.dxu[1] = np.interp(goal.steering, [-1, 1], [-self.car_model.max_steer, self.car_model.max_steer])
        
    ################# PRIVATE METHODS
    def __delta_to_beta(self, delta):
        """
        Converts the steering angle delta to the slip angle beta.

        Args:
            delta (float): Steering angle in radians.

        Returns:
            float: Slip angle in radians.

        """
        beta = utils.normalize_angle(np.arctan2(self.car_model.Lr*np.tan(delta)/self.car_model.L, 1.0))

        return beta

    def __beta_to_delta(self, beta):
        """
        Converts the slip angle beta to the steering angle delta.

        Args:
            beta (float): Slip angle in radians.

        Returns:
            float: Steering angle in radians.

        """
        delta = utils.normalize_angle(np.arctan2(self.car_model.L*np.tan(beta)/self.car_model.Lr, 1.0))

        return delta  

    def __closest_boundary_point(self, i, x):
        
        dx = 0.0
        dy = 0.0

        if abs(x[0,i]-self.boundary_points[0]) < abs(x[0,i]-self.boundary_points[1]):
            cpx = self.boundary_points[0]
            dx = abs(x[0,i]-self.boundary_points[0])
        else: 
            cpx = self.boundary_points[1]
            dx = abs(x[0,i]-self.boundary_points[1])

        if abs(x[1,i]-self.boundary_points[2]) < abs(x[1,i]-self.boundary_points[3]):
            cpy = self.boundary_points[2]
            dy = abs(x[1,i]-self.boundary_points[2])
        else: 
            cpy = self.boundary_points[3]
            dy = abs(x[1,i]-self.boundary_points[3])

        if dx < dy:
            self.closest_boundary_point = (cpx, x[1,i], 0.0)
        else:
            self.closest_boundary_point = (x[0,i], cpy, np.pi/2)

    def __C3BF(self, i, x):
        """
        Computes the control input for the C3BF (Collision Cone Control Barrier Function) algorithm.

        Args:
            x (numpy.ndarray): State vector of shape (4, N), where N is the number of time steps.
            u_ref (numpy.ndarray): Reference control input of shape (2, N).

        Returns:
            numpy.ndarray: Filtered Control input dxu of shape (2, N).

        """

        N = x.shape[1]
        M = self.dxu.shape[0]
        self.dxu[1] = self.__delta_to_beta(self.dxu[1])

        count = 0
        G = np.zeros([N-1 + 9,M])
        H = np.zeros([N-1 + 9,1]) 

        f = np.array([x[3,i]*np.cos(x[2,i]),
                            x[3,i]*np.sin(x[2,i]), 
                            0, 
                            0]).reshape(4,1)
        g = np.array([[0, -x[3,i]*np.sin(x[2,i])], 
                        [0, x[3,i]*np.cos(x[2,i])], 
                        [0, x[3,i]/self.car_model.Lr],
                        [1, 0]]).reshape(4,2)
        
        P = np.identity(2)*2
        q = np.array([-2 * self.dxu[0], - 2 * self.dxu[1]])
        
        for j in range(N):
            arr = np.array([x[0, j] - x[0, i], x[1, j] - x[1,i]])
            dist = np.linalg.norm(arr)

            if j == i or dist > 2 * self.safety_radius: 
                continue

            v_rel = np.array([x[3,j]*np.cos(x[2,j]) - x[3,i]*np.cos(x[2,i]), 
                                x[3,j]*np.sin(x[2,j]) - x[3,i]*np.sin(x[2,i])])
            p_rel = np.array([x[0,j]-x[0,i],
                                x[1,j]-x[1,i]])
            
            cos_Phi = np.sqrt(abs(np.linalg.norm(p_rel)**2 - self.safety_radius**2))/np.linalg.norm(p_rel)
            tan_Phi_sq = self.safety_radius**2 / (np.linalg.norm(p_rel)**2 - self.safety_radius**2)
            
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

            H[count] = np.array([self.barrier_gain*np.power(h, 3) + Lf_h])
            G[count,:] = -Lg_h
            count+=1

        # Closes point on the boundary
        self.__compute_closest_point(i, x)
        Lf_h = 2 * x[3, i] * (np.cos(x[2, i]) * (x[0, i] - self.closest_point[0]) + np.sin(x[2, i]) * (x[1, i] - self.closest_point[1]))
        Lg_h = 2 * x[3, i] * (np.cos(x[2, i]) * (x[1, i] - self.closest_point[1]) - np.sin(x[2, i]) * (x[0, i] - self.closest_point[0]))
        h = (x[0, i] - self.closest_point[0]) * (x[0, i] - self.closest_point[0]) + (x[1, i] - self.closest_point[1]) * (x[1, i] - self.closest_point[1]) - (
                    self.safety_radius ** 2 + self.Kv * abs(x[3, i])) #TODO: Ask giacomo why he has a 1.5 here

        H[count] = np.array([self.barrier_gain * np.power(h, 3) + Lf_h])

        if x[3, i] >= 0:
            G[count, :] = np.array([self.Kv, -Lg_h])
        else:
            G[count, :] = np.array([-self.Kv, -Lg_h])
        count+=1


        # Add the input constraint
        G[count, :] = np.array([0, 1])
        H[count] = np.array([self.__delta_to_beta(self.car_model.max_steer)])
        count += 1

        G[count, :] = np.array([0, -1])
        H[count] = np.array([-self.__delta_to_beta(-self.car_model.max_steer)])
        count += 1

        # # # # Removed these constraints to allow sharper turns
        # # # G[count, :] = np.array([0, x[3,i]/self.car_model.Lr])
        # # # H[count] = np.array([np.deg2rad(50)])
        # # # count += 1

        # # # G[count, :] = np.array([0, x[3,i]/self.car_model.Lr])
        # # # H[count] = np.array([np.deg2rad(50)])
        # # # count += 1

        G[count, :] = np.array([1, 0])
        H[count] = np.array([self.car_model.max_acc])
        count += 1

        G[count, :] = np.array([-1, 0])
        H[count] = np.array([-self.car_model.min_acc])

        solver = DenseSolver()
        solver.settings.verbose = False
        solver.setup(P, q, np.empty((0, P.shape[0])), np.empty((0,)), G, H)
        status = solver.solve()

        if status == PIQP_SOLVED:
            self.dxu[:] = np.reshape(solver.result.x, (M,))
            self.dxu[1] = self.__beta_to_delta(self.dxu[1])
        else:
            print(f"QP solver failed for robot {i}! Emergency stop.") 
            self.dxu[0] = (0 - x[3,i])/self.dt 
            self.solver_failure += 1
        return self.dxu
        
    def __compute_closest_point(self, i, x):
        dx = 0
        dy = 0

        if abs(x[0,i] - self.boundary_points[0]) < abs(x[0,i] - self.boundary_points[1]):
            dx = self.boundary_points[0] - x[0,i]
            cpx = self.boundary_points[0]
        else:
            dx = self.boundary_points[1] - x[0,i]
            cpx = self.boundary_points[1]
        
        if abs(x[1,i] - self.boundary_points[2]) < abs(x[1,i] - self.boundary_points[3]):
            dy = self.boundary_points[2] - x[1,i]
            cpy = self.boundary_points[2]
        else:
            dy = self.boundary_points[3] - x[1,i]
            cpy = self.boundary_points[3]
            
        if abs(dx) < abs(dy):
            self.closest_point = np.array([cpx, x[1,i]])  
        else:
            self.closest_point = np.array([x[0,i], cpy])
        