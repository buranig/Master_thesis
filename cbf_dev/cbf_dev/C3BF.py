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
    """
    This class implements the C3BF (Collision Cone Control Barrier Function) algorithm.

    Attributes:
        dxu (numpy.ndarray): Control input of shape (2,) representing the throttle and steering.
        safety_radius (float): Safety radius for collision avoidance.
        barrier_gain (float): Gain for the barrier function.
        Kv (float): Gain for the velocity term.
        closest_point (tuple): Coordinates of the closest point on the boundary.

    """    
    def __init__(self, controller_path:str, car_i = 0):
        """
        Initializes the C3BF_algorithm class.

        Args:
            controller_path (str): Path to the controller YAML file.
            car_i (int): Index of the car.

        """
        ## Init Controller class
        super().__init__(controller_path, car_i)

        ## Init public parameters
        self.dxu = np.zeros((2,), dtype=float)

        # Opening YAML file
        with open(controller_path, 'r') as openfile:
            # Reading from yaml file
            yaml_object = yaml.safe_load(openfile)

        self.safety_radius = yaml_object["C3BF"]["safety_radius"]
        self.barrier_gain = yaml_object["C3BF"]["barrier_gain"]
        self.Kv = yaml_object["C3BF"]["Kv"] # interval [0.5-1]

        # Instantiate border variables
        self.__compute_track_constants()
        self.closest_point = (0.0, 0.0)

    ################# PUBLIC METHODS

    def compute_cmd(self, car_list: List[CarStateStamped]) -> CarControlStamped:
        """
        Computes the car control command based on the current state of all cars.

        Args:
            car_list (List[CarStateStamped]): A list of CarStateStamped objects representing the current state of all cars.

        Returns:
            CarControlStamped: The computed car control command.

        """
        # Init empty command
        car_cmd = CarControlStamped()

        # Update current state of all cars
        self.curr_state = np.transpose(utils.carStateStamped_to_array(car_list))

        # Compute control   
        u = self.__C3BF(self.car_i, self.curr_state)

        # Project it to range [-1, 1]
        car_cmd.throttle = np.interp(u[0], [self.car_model.min_acc, self.car_model.max_acc], [-1, 1]) * self.car_model.acc_gain
        car_cmd.steering = np.interp(u[1], [-self.car_model.max_steer, self.car_model.max_steer], [-1, 1])

        return car_cmd

    def set_goal(self, goal: CarControlStamped) -> None:
        """
        Sets the goal for the C3BF controller.

        Args:
            goal (CarControlStamped): The goal containing the desired throttle and steering values.

        Returns:
            None
        """
        self.dxu[0] = np.interp(goal.throttle, [-1, 1], [self.car_model.min_acc, self.car_model.max_acc]) * self.car_model.acc_gain
        self.dxu[1] = np.interp(goal.steering, [-1, 1], [-self.car_model.max_steer, self.car_model.max_steer])

    def offset_track(self, off:List[int]) -> None:
        """
        Updates the position of the corners of the map and the corresponding parameters that define the "walls".

        Args:
            off (List[int]): A list of integers representing the offset values for x,y and yaw.

        Returns:
            None
        """
        super().offset_track(off)
        self.__compute_track_constants()
        
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

    def __C3BF(self, i, x) -> np.ndarray:
        """
        Private method that implements the C3BF algorithm for a specific car.

        Args:
            i (int): Index of the car.
            x (numpy.ndarray): Array of shape (4, N) representing the state of all cars.

        Returns:
            numpy.ndarray: Array of shape (2,) representing the control input.

        """

        # Initialize all matrices for faster computation
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
        
        # Add the barrier function constraints for other cars
        for j in range(N):
            arr = np.array([x[0, j] - x[0, i], x[1, j] - x[1,i]])
            dist = np.linalg.norm(arr)

            # Only add if they are close
            if j == i or dist > 5 * self.safety_radius: 
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

        # Closest point on the boundary acts as another car
        self.__compute_closest_point(i, x)
        Lf_h = 2 * x[3, i] * (np.cos(x[2, i]) * (x[0, i] - self.closest_point[0]) + np.sin(x[2, i]) * (x[1, i] - self.closest_point[1]))
        Lg_h = 2 * x[3, i] * (np.cos(x[2, i]) * (x[1, i] - self.closest_point[1]) - np.sin(x[2, i]) * (x[0, i] - self.closest_point[0]))
        h = (x[0, i] - self.closest_point[0]) * (x[0, i] - self.closest_point[0]) + (x[1, i] - self.closest_point[1]) * (x[1, i] - self.closest_point[1]) - (
                    self.safety_radius ** 2 + self.Kv * abs(x[3, i]))

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
            print(f'G: {G}')
            print(f'H: {H}\n')
        else:
            print(f"QP solver failed for robot {i}! Emergency stop.") 
            print(f'G: {G}')
            print(f'H: {H}\n')
            self.dxu[0] = (0 - x[3,i])/self.dt 
        return self.dxu
        
    def __compute_track_constants(self):
        """
        Computes the track constants based on the boundary points.

        This method calculates the constants required for track boundary computation
        using the given boundary points. The constants a, b, and c
        are calculated based on the following equations:

        a = y2 - y1
        b = x1 - x2
        c = y1 * (x2 - x1) - x1 * (y2 - y1)

        They correspond to the constants for the straight lines that join all vertices 
        of the square that defines the bumping arena. These lines follow convention 
        of the form ax + by + c = 0.

        Returns:
            None
        """
        x1, y1 = self.boundary_points[0][0], self.boundary_points[0][1]
        x2, y2 = self.boundary_points[1][0], self.boundary_points[1][1]
        x3, y3 = self.boundary_points[2][0], self.boundary_points[2][1]
        x4, y4 = self.boundary_points[3][0], self.boundary_points[3][1]

        # a = y2-y1
        self.a1 = y2 - y1
        self.a2 = y3 - y2
        self.a3 = y4 - y3
        self.a4 = y1 - y4

        # b = x1-x2
        self.b1 = x1 - x2
        self.b2 = x2 - x3
        self.b3 = x3 - x4
        self.b4 = x4 - x1

        # c = y1*(x2-x1) - x1*(y2-y1)
        self.c1 = y1 * (x2 - x1) - x1 * (y2 - y1)
        self.c2 = y2 * (x3 - x2) - x2 * (y3 - y2)
        self.c3 = y3 * (x4 - x3) - x3 * (y4 - y3)
        self.c4 = y4 * (x1 - x4) - x4 * (y1 - y4)

    def __compute_closest_point(self, i, x) -> None:
        """
        Compute the closest point to the given point (x0, y0) among four lines.

        Args:
            i (int): The index of the point.
            x (numpy.ndarray): The array of points.

        Returns:
            None
        """

        x0 = x[0,i]
        y0 = x[1,i]

        # Calculate distance from current point to each line
        dist_1 = abs(self.a1*x0 + self.b1*y0 + self.c1)/np.sqrt(self.a1**2 + self.b1**2)
        dist_2 = abs(self.a2*x0 + self.b2*y0 + self.c2)/np.sqrt(self.a2**2 + self.b2**2)
        dist_3 = abs(self.a3*x0 + self.b3*y0 + self.c3)/np.sqrt(self.a3**2 + self.b3**2)
        dist_4 = abs(self.a4*x0 + self.b4*y0 + self.c4)/np.sqrt(self.a4**2 + self.b4**2)
        min_dist = min(dist_1, dist_2, dist_3, dist_4)

        # Compute the closest point in the closest line
        if min_dist == dist_1:
            denom = self.a1**2 + self.b1**2
            num_x = self.b1*(self.b1*x0 - self.a1*y0) - self.a1*self.c1
            num_y = self.a1*(-self.b1*x0 + self.a1*y0) - self.b1*self.c1
            self.closest_point = np.array([num_x/denom, num_y/denom])
        elif min_dist == dist_2:
            denom = self.a2**2 + self.b2**2
            num_x = self.b2*(self.b2*x0 - self.a2*y0) - self.a2*self.c2
            num_y = self.a2*(-self.b2*x0 + self.a2*y0) - self.b2*self.c2
            self.closest_point = np.array([num_x/denom, num_y/denom])
        elif min_dist == dist_3:
            denom = self.a3**2 + self.b3**2
            num_x = self.b3*(self.b3*x0 - self.a3*y0) - self.a3*self.c3
            num_y = self.a3*(-self.b3*x0 + self.a3*y0) - self.b3*self.c3
            self.closest_point = np.array([num_x/denom, num_y/denom])
        elif min_dist == dist_4:
            denom = self.a4**2 + self.b4**2
            num_x = self.b4*(self.b4*x0 - self.a4*y0) - self.a4*self.c4
            num_y = self.a4*(-self.b4*x0 + self.a4*y0) - self.b4*self.c4
            self.closest_point = np.array([num_x/denom, num_y/denom])
        else:   
            print("Error in computing closest point")
        