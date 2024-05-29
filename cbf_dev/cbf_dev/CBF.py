import numpy as np

from piqp import DenseSolver, PIQP_SOLVED

from lar_msgs.msg import CarControlStamped, CarStateStamped
from typing import List

from bumper_cars.utils import car_utils as utils
from bumper_cars.classes.State import State
from bumper_cars.classes.Controller import Controller

# For the parameter file
import yaml

np.random.seed(1)

class CBF_algorithm(Controller):
    def __init__(self, controller_path:str, robot_num = 0):

        ## Init Controller class
        super().__init__(controller_path)

        ## Init public parameters
        self.robot_num = robot_num
        self.dxu = np.zeros((2, ), dtype=float)
        self.solver_failure = 0

        # Opening YAML file
        with open(controller_path, 'r') as openfile:
            # Reading from yaml file
            yaml_object = yaml.safe_load(openfile)

        self.safety_radius = yaml_object["CBF"]["safety_radius"]
        self.barrier_gain = yaml_object["CBF"]["barrier_gain"]
        self.arena_gain = yaml_object["CBF"]["arena_gain"]
        self.Kv = yaml_object["CBF"]["Kv"] # interval [0.5-1]

        # Instantiate border variables
        self.__compute_track_constants()
        self.closest_point = (0.0, 0.0)

    ################# PUBLIC METHODS

    def compute_cmd(self, car_list : List[CarStateStamped]) -> CarControlStamped:

        # Init empty command
        car_cmd = CarControlStamped()

        # Update current state of all cars
        self.curr_state = np.transpose(utils.carStateStamped_to_array(car_list))

        # Compute control   
        u = self.__CBF(self.robot_num, self.curr_state)

        # Project it to range [-1, 1]
        car_cmd.throttle = np.interp(u[0], [self.car_model.min_acc, self.car_model.max_acc], [-1, 1]) * self.car_model.acc_gain
        car_cmd.steering = np.interp(u[1], [-self.car_model.max_steer, self.car_model.max_steer], [-1, 1])
            
        return car_cmd

    def set_goal(self, goal: CarControlStamped) -> None:
        self.dxu[0] = np.interp(goal.throttle, [-1, 1], [self.car_model.min_acc, self.car_model.max_acc]) * self.car_model.acc_gain
        self.dxu[1] = np.interp(goal.steering, [-1, 1], [-self.car_model.max_steer, self.car_model.max_steer])

    def offset_track(self, off:List[int]):
        print("Corrected offset!")
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

    def __CBF(self, i:int, x:List[float]) -> List[float]:
        """
        Computes the control input for the CBF (Collision Cone Control Barrier Function) algorithm.

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

        # # f = np.array([x[3,i]*np.cos(x[2,i]),
        # #                     x[3,i]*np.sin(x[2,i]), 
        # #                     0, 
        # #                     0]).reshape(4,1)
        # # g = np.array([[0, -x[3,i]*np.sin(x[2,i])], 
        # #                 [0, x[3,i]*np.cos(x[2,i])], 
        # #                 [0, x[3,i]/self.car_model.Lr],
        # #                 [1, 0]]).reshape(4,2)
        
        P = np.identity(2)*2
        q = np.array([-2 * self.dxu[0], - 2 * self.dxu[1]])

        for j in range(N):
            arr = np.array([x[0, j] - x[0, i], x[1, j] - x[1,i]])
            dist = np.linalg.norm(arr)

            if j == i: 
                continue

            # # # if j != i and dist > 2 * self.safety_radius: 
            # # #     print("\n \n \n ERROR \n \n \n")
            # # #     continue
            
            Lf_h = 2 * x[3, i] * (np.cos(x[2, i]) * (x[0, i] - x[0, j]) + np.sin(x[2, i]) * (x[1, i] - x[1, j]))
            Lg_h = 2 * x[3, i] * (np.cos(x[2, i]) * (x[1, i] - x[1, j]) - np.sin(x[2, i]) * (x[0, i] - x[0, j]))
            h = (x[0, i] - x[0, j]) * (x[0, i] - x[0, j]) + (x[1, i] - x[1, j]) * (x[1, i] - x[1, j]) - (
                        self.safety_radius ** 2 + self.Kv * abs(x[3, i]))

            H[count] = np.array([self.barrier_gain * np.power(h, 3) + Lf_h])

            if x[3, i] >= 0:
                G[count, :] = np.array([self.Kv, -Lg_h])
            else:
                G[count, :] = np.array([-self.Kv, -Lg_h])


            count += 1

        # Closest point on the boundary
        self.__compute_closest_point(i, x)
        Lf_h = 2 * x[3, i] * (np.cos(x[2, i]) * (x[0, i] - self.closest_point[0]) + np.sin(x[2, i]) * (x[1, i] - self.closest_point[1]))
        Lg_h = 2 * x[3, i] * (np.cos(x[2, i]) * (x[1, i] - self.closest_point[1]) - np.sin(x[2, i]) * (x[0, i] - self.closest_point[0]))
        h = (x[0, i] - self.closest_point[0]) * (x[0, i] - self.closest_point[0]) + (x[1, i] - self.closest_point[1]) * (x[1, i] - self.closest_point[1]) - (
                    self.safety_radius ** 2 + self.Kv * abs(x[3, i]))

        H[count] = np.array([self.barrier_gain * np.power(h, 3) + Lf_h])

        if x[3, i] >= 0:
            G[count, :] = np.array([self.Kv, -Lg_h*0.0])
        else:
            G[count, :] = np.array([-self.Kv, -Lg_h*0.0])
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

        self.dxu = np.clip(self.dxu, [self.car_model.min_acc, -self.car_model.max_steer], [self.car_model.max_acc, self.car_model.max_steer])

        return self.dxu      
    
    def __compute_track_constants(self):
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

    def __compute_closest_point(self, i, x):
        x0 = x[0,i]
        y0 = x[1,i]

        dist_1 = abs(self.a1*x0 + self.b1*y0 + self.c1)/np.sqrt(self.a1**2 + self.b1**2)
        dist_2 = abs(self.a2*x0 + self.b2*y0 + self.c2)/np.sqrt(self.a2**2 + self.b2**2)
        dist_3 = abs(self.a3*x0 + self.b3*y0 + self.c3)/np.sqrt(self.a3**2 + self.b3**2)
        dist_4 = abs(self.a4*x0 + self.b4*y0 + self.c4)/np.sqrt(self.a4**2 + self.b4**2)
        min_dist = min(dist_1, dist_2, dist_3, dist_4)

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