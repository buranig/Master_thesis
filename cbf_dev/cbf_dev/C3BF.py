import numpy as np

from cvxopt import matrix, solvers
from cvxopt import matrix

from lar_utils import car_utils as utils
from lar_msgs.msg import CarControlStamped, CarStateStamped
from typing import List

from bumper_cars.classes.CarModel import State, CarModel
from bumper_cars.classes.Controller import Controller

# For the parameter file
import yaml


np.random.seed(1)

class C3BF_algorithm(Controller):
    def __init__(self, controller_path:str, robot_num = 1):

        ## Init Controller class
        super().__init__(controller_path)

        ## Init public parameters

        self.goal = None
        self.robot_num = robot_num - 1
        self.dxu = np.zeros((2, 1), dtype=float)
        self.solver_failure = 0

        # Opening YAML file
        with open(controller_path, 'r') as openfile:
            # Reading from yaml file
            yaml_object = yaml.safe_load(openfile)

        self.safety_radius = yaml_object["C3BF"]["safety_radius"]
        self.barrier_gain = yaml_object["C3BF"]["barrier_gain"]
        self.arena_gain = yaml_object["C3BF"]["arena_gain"]
        self.Kv = yaml_object["C3BF"]["Kv"] # interval [0.5-1]

    ################# PUBLIC METHODS

    def compute_cmd(self, car_list : List[CarStateStamped]) -> CarControlStamped:
        # Init empty command
        car_cmd = CarControlStamped()

        # Update current state of all cars
        self.curr_state = np.transpose(utils.carStateStamped_to_array(car_list))

        # Compute control   
        u = self.__C3BF(self.robot_num, self.curr_state)
        # If solver did not fail
        if u is not None:
            car_cmd.throttle = u[0]
            car_cmd.steering = u[1]

        return car_cmd

    def set_goal(self, goal: CarControlStamped) -> None:
        self.dxu[0] = self.dxu[0]
        self.dxu[1] = self.dxu[1]
        
        # For homogeneity, keep self.goal updated
        self.goal = goal

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
        G = np.zeros([N-1,M])
        H = np.zeros([N-1,1])

        # when the car goes backwards the yaw angle should be flipped --> Why??
        # x[2,i] = (1-np.sign(x[3,i]))*(np.pi/2) + x[2,i]

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
            v = np.array([x[3,i]*np.cos(x[2,i]), x[3,i]*np.sin(x[2,i])])
            scalar_prod = v @ arr

            if j == i or dist > 3 * self.safety_radius or scalar_prod < 0: 
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

        # Adding arena boundary constraints
        # # Pos Y
        # h = (x[1,i] - self.boundary_points[3])**2 - self.safety_radius**2 - self.Kv * abs(x[3,i]*np.sin(x[2,i]))
        # gradH = np.array([2*(x[1,i] - self.boundary_points[3]), 0, 
        #                 -self.Kv*np.abs(x[3,i])*np.sign(np.sin(x[2,i]))*np.cos(x[2,i]), 
        #                 -self.Kv*np.abs(np.sin(x[2,i]))*np.sign(x[3,i])])
        # Lf_h = np.dot(gradH.T, f)
        # Lg_h = np.dot(gradH.T, g)
        # G = np.vstack([G, -Lg_h])
        # H = np.vstack([H, np.array([self.arena_gain*h**3 + Lf_h])])
        
        # # Neg Y
        # h = (x[1,i] - self.boundary_points[2])**2 - self.safety_radius**2 - self.Kv * abs(x[3,i]*np.sin(x[2,i]))
        
        # gradH = np.array([2*(x[1,i] - self.boundary_points[2]), 0, 
        #                 -self.Kv*np.abs(x[3,i])*np.sign(np.sin(x[2,i]))*np.cos(x[2,i]), 
        #                 -self.Kv*np.abs(np.sin(x[2,i]))*np.sign(x[3,i])])
        # Lf_h = np.dot(gradH.T, f)
        # Lg_h = np.dot(gradH.T, g)
        # G = np.vstack([G, -Lg_h])
        # H = np.vstack([H, np.array([self.arena_gain*h**3 + Lf_h])])
        
        # # Pos X
        # h = (x[0,i] - self.boundary_points[1])**2 - self.safety_radius**2 - self.Kv * abs(x[3,i]*np.cos(x[2,i]))
        # gradH = np.array([2*(x[0,i] - self.boundary_points[1]), 0, 
        #                 self.Kv*np.abs(x[3,i])*np.sign(np.cos(x[2,i]))*np.sin(x[2,i]), 
        #                 -self.Kv*np.abs(np.cos(x[2,i]))*np.sign(x[3,i])])
        # Lf_h = np.dot(gradH.T, f)
        # Lg_h = np.dot(gradH.T, g)
        # G = np.vstack([G, -Lg_h])
        # H = np.vstack([H, np.array([self.arena_gain*h**3 + Lf_h])])

        # # Neg X
        # h = (x[0,i] - self.boundary_points[0])**2 - self.safety_radius**2 - self.Kv * abs(x[3,i]*np.cos(x[2,i]))
        # gradH = np.array([2*(x[0,i] - self.boundary_points[0]), 0, 
        #                 self.Kv*np.abs(x[3,i])*np.sign(np.cos(x[2,i]))*np.sin(x[2,i]), 
        #                 -self.Kv*np.abs(np.cos(x[2,i]))*np.sign(x[3,i])])
        # Lf_h = np.dot(gradH.T, f)
        # Lg_h = np.dot(gradH.T, g)
        # G = np.vstack([G, -Lg_h])
        # H = np.vstack([H, np.array([self.arena_gain*h**3 + Lf_h])])

        # Adding arena boundary constraints
        # Pos Y
        h = ((x[1,i] - self.boundary_points[3])**2 - self.safety_radius**2 - self.Kv * abs(x[3,i]))
        if x[3,i] >= 0:
            gradH = np.array([0, 2*(x[1,i] - self.boundary_points[3]), 0, -self.Kv])
        else:
            gradH = np.array([0, 2*(x[1,i] - self.boundary_points[3]), 0, self.Kv])

        Lf_h = np.dot(gradH.T, f)
        Lg_h = np.dot(gradH.T, g)
        G = np.vstack([G, -Lg_h])
        H = np.vstack([H, np.array([self.arena_gain*h**3 + Lf_h])])

        # Neg Y
        h = ((x[1,i] - self.boundary_points[2])**2 - self.safety_radius**2 - self.Kv * abs(x[3,i]))
        if x[3,i] >= 0:
            gradH = np.array([0, 2*(x[1,i] - self.boundary_points[2]), 0, -self.Kv])
        else:
            gradH = np.array([0, 2*(x[1,i] - self.boundary_points[2]), 0, self.Kv])
        Lf_h = np.dot(gradH.T, f)
        Lg_h = np.dot(gradH.T, g)
        G = np.vstack([G, -Lg_h])
        H = np.vstack([H, np.array([self.arena_gain*h**3 + Lf_h])])

        # Pos X
        h = ((x[0,i] - self.boundary_points[1])**2 - self.safety_radius**2 - self.Kv * abs(x[3,i]))
        if x[3,i] >= 0:
            gradH = np.array([2*(x[0,i] - self.boundary_points[1]), 0, 0, -self.Kv])
        else:
            gradH = np.array([2*(x[0,i] - self.boundary_points[1]), 0, 0, self.Kv])
        Lf_h = np.dot(gradH.T, f)
        Lg_h = np.dot(gradH.T, g)
        G = np.vstack([G, -Lg_h])
        H = np.vstack([H, np.array([self.arena_gain*h**3 + Lf_h])])

        # Neg X
        h = ((x[0,i] - self.boundary_points[0])**2 - self.safety_radius**2 - self.Kv * abs(x[3,i]))
        if x[3,i] >= 0:
            gradH = np.array([2*(x[0,i] - self.boundary_points[0]), 0, 0, -self.Kv])
        else:
            gradH = np.array([2*(x[0,i] - self.boundary_points[0]), 0, 0, self.Kv])
        Lf_h = np.dot(gradH.T, f)
        Lg_h = np.dot(gradH.T, g)
        G = np.vstack([G, -Lg_h])
        H = np.vstack([H, np.array([self.arena_gain*h**3 + Lf_h])])
        
        # safety = 1
        # G = np.vstack([G, [[0, -x[3,i]*np.sin(x[2,i])], [0, x[3,i]*np.cos(x[2,i])]]])
        # h1 = ((np.array([self.boundary_points[1]-safety, self.boundary_points[3]-safety])-x[0:2,i])/dt - np.array([x[3,i]*np.cos(x[2,i]), x[3,i]*np.sin(x[2,i])])).reshape(2,1)
        # H = np.vstack([H, h1])
        # G = np.vstack([G, [[0, x[3,i]*np.sin(x[2,i])], [0, -x[3,i]*np.cos(x[2,i])]]])
        # h1 = ((x[0:2,i] - np.array([self.boundary_points[0]+safety, self.boundary_points[2]+safety]))/dt + np.array([x[3,i]*np.cos(x[2,i]), x[3,i]*np.sin(x[2,i])])).reshape(2,1)
        # H = np.vstack([H, h1])
        
        # Input constraints
        G = np.vstack([G, [[0, 1], [0, -1]]])
        H = np.vstack([H, self.__delta_to_beta(self.car_model.max_steer), -self.__delta_to_beta(-self.car_model.max_steer)])
        # TODO: Keeping the following constraints for solves some problem with the circular_seed_10.json --> why??
        G = np.vstack([G, [[0, x[3,i]/self.car_model.Lr], [0, x[3,i]/self.car_model.Lr]]])
        H = np.vstack([H, np.deg2rad(50), np.deg2rad(50)])
        # G = np.vstack([G, [[0, 1], [0, -1]]])
        # H = np.vstack([H, self.dxu[1]+self.__delta_to_beta(5), -self.dxu[1]+self.__delta_to_beta(5)])
        G = np.vstack([G, [[1, 0], [-1, 0]]])
        H = np.vstack([H, self.car_model.max_acc, -self.car_model.min_acc])

        solvers.options['show_progress'] = False
        try:
            sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(H))
            self.dxu = np.reshape(np.array(sol['x']), (M,))
            self.dxu[0] = np.clip(self.dxu[0], self.car_model.min_acc, self.car_model.max_acc)
            self.dxu[1] = self.__beta_to_delta(self.dxu[1])  
            self.dxu[1] = np.clip(self.dxu[1], -self.car_model.max_steer, self.car_model.max_steer)
            return self.dxu

        except:
            print("QP solver failed")   
            self.solver_failure += 1   
            return None      
        
