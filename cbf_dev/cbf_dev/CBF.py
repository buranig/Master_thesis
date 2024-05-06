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

# For plotting
import matplotlib.pyplot as plt
color_dict = {0: 'r', 1: 'b', 2: 'g', 3: 'y', 4: 'm', 5: 'c', 6: 'k', 7: 'tab:orange', 8: 'tab:brown', 9: 'tab:gray', 10: 'tab:olive', 11: 'tab:pink', 12: 'tab:purple', 13: 'tab:red', 14: 'tab:blue', 15: 'tab:green'}
fig = plt.figure(1, dpi=90, figsize=(10,10))
ax= fig.add_subplot(111)

np.random.seed(1)

class CBF_algorithm(Controller):
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

        self.safety_radius = yaml_object["CBF"]["safety_radius"]
        self.barrier_gain = yaml_object["CBF"]["barrier_gain"]
        self.arena_gain = yaml_object["CBF"]["arena_gain"]
        self.Kv = yaml_object["CBF"]["Kv"] # interval [0.5-1]

    ################# PUBLIC METHODS

    def compute_cmd(self, car_list : List[CarStateStamped]) -> CarControlStamped:
        # Init empty command
        car_cmd = CarControlStamped()

        # Update current state of all cars
        self.curr_state = np.transpose(utils.carStateStamped_to_array(car_list))

        # Compute control   
        u = self.__CBF(self.robot_num, self.curr_state)
        # If solver did not fail
        if u is not None:
            car_cmd.throttle = u[0]
            car_cmd.steering = u[1]
            
        # Debug visualization
        if self.show_animation and self.robot_num == 0:
            plt.cla()
            plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
            for i in range(self.curr_state.shape[1]):
                self.plot_robot(self.curr_state[0, i], self.curr_state[1, i], self.curr_state[2, i])
                self.plot_arrow(self.curr_state[0, i], self.curr_state[1, i], self.curr_state[2, i] + self.dxu[1], length=0.3, width=0.05)
                self.plot_arrow(self.curr_state[0, i], self.curr_state[1, i], self.curr_state[2, i], length=0.1, width=0.05)
            self.plot_map()
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.00001)
            # plt.plot(self.goal[0], self.goal[1], "x", color = color_dict[0])
        return car_cmd

    def set_goal(self, goal: CarControlStamped) -> None:
        self.dxu[0] = goal.throttle
        self.dxu[1] = goal.steering
        
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

    def __CBF(self, i:int, x:List[float]) -> List[float]:
        """
        Computes the control input for the CBF (Collision Cone Control Barrier Function) algorithm.

        Args:
            x (numpy.ndarray): State vector of shape (4, N), where N is the number of time steps.
            u_ref (numpy.ndarray): Reference control input of shape (2, N).

        Returns:
            numpy.ndarray: Filtered Control input dxu of shape (2, N).

        """
        N = x.shape[1]              # Number of robots
        M = self.dxu.shape[0]       # Number of control inputs
        self.dxu[1] = self.__delta_to_beta(self.dxu[1])

        G = np.zeros([N - 1, M])
        H = np.zeros([N - 1, 1])

        # when the car goes backwards the yaw angle should be flipped --> Why??
        # x[2,i] = (1-np.sign(x[3,i]))*(np.pi/2) + x[2,i]
        f = np.array([x[3, i] * np.cos(x[2, i]),
                      x[3, i] * np.sin(x[2, i]),
                      0,
                      0]).reshape(4, 1)
        g = np.array([[0, -x[3, i] * np.sin(x[2, i])],
                      [0, x[3, i] * np.cos(x[2, i])],
                      [0, x[3, i] / self.car_model.Lr],
                      [1, 0]]).reshape(4, 2)

        P = np.identity(2) * 2
        q = np.array([- 2 * self.dxu[0], - 2 * self.dxu[1]])


        # Solves a quadratic program

        # minimize    (1/2)*x'*P*x + q'*x
        # subject to  G*x <= h
        #             A*x = b.
        print("Safety radius:", self.safety_radius)
        print("Kv:", self.Kv)
        print("Barrier gain:", self.barrier_gain)

        count = 0
        for j in range(N):

            if j == i: continue
            
            # Lf_h = 2 * x[3, i] * (np.cos(x[2, i]) * (x[0, i] - x[0, j]) + np.sin(x[2, i]) * (x[1, i] - x[1, j]))
            # Lg_h = 2 * x[3, i] * (np.cos(x[2, i]) * (x[1, i] - x[1, j]) - np.sin(x[2, i]) * (x[0, i] - x[0, j]))
            
            # h = (x[0, i] - x[0, j]) ** 2 + (x[1, i] - x[1, j]) ** 2 - (
            #             self.safety_radius ** 2 + self.Kv * abs(x[3, i]))

            # H[count] = np.array([self.barrier_gain * np.power(h, 3) + Lf_h])
            # G[count, :] = - np.array([- np.sign(x[3, i])*self.Kv, Lg_h])

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

        if self.robot_num ==0:
            print("G\n",G)
            print("H\n",H)

        # Add the input constraint
        G = np.vstack([G, [[0, 1], [0, -1]]])
        H = np.vstack([H, self.__delta_to_beta(self.car_model.max_steer), -self.__delta_to_beta(-self.car_model.max_steer)])
        # TODO: check whether to keep the following constraints
        # G = np.vstack([G, [[0, x[3,i]/self.car_model.Lr], [0, x[3,i]/self.car_model.Lr]]])
        # H = np.vstack([H, np.deg2rad(50), np.deg2rad(50)])
        G = np.vstack([G, [[1, 0], [-1, 0]]])
        H = np.vstack([H, self.car_model.max_acc, self.car_model.min_acc])

        # Adding arena boundary constraints
        # Pos Y
        h = ((x[1, i] - self.boundary_points[3]) ** 2 - self.safety_radius ** 2 - self.Kv * abs(x[3, i]))
        if x[3, i] >= 0:
            gradH = np.array([0, 2 * (x[1, i] - self.boundary_points[3]), 0, -self.Kv])
        else:
            gradH = np.array([0, 2 * (x[1, i] - self.boundary_points[3]), 0, self.Kv])

        Lf_h = np.dot(gradH.T, f)
        Lg_h = np.dot(gradH.T, g)
        G = np.vstack([G, -Lg_h])
        H = np.vstack([H, np.array([self.arena_gain * h ** 3 + Lf_h])])

        # Neg Y
        h = ((x[1, i] - self.boundary_points[2]) ** 2 - self.safety_radius ** 2 - self.Kv * abs(x[3, i]))
        if x[3, i] >= 0:
            gradH = np.array([0, 2 * (x[1, i] - self.boundary_points[2]), 0, -self.Kv])
        else:
            gradH = np.array([0, 2 * (x[1, i] - self.boundary_points[2]), 0, self.Kv])
        Lf_h = np.dot(gradH.T, f)
        Lg_h = np.dot(gradH.T, g)
        G = np.vstack([G, -Lg_h])
        H = np.vstack([H, np.array([self.arena_gain * h ** 3 + Lf_h])])

        # Pos X
        h = ((x[0, i] - self.boundary_points[1]) ** 2 - self.safety_radius ** 2 - self.Kv * abs(x[3, i]))
        if x[3, i] >= 0:
            gradH = np.array([2 * (x[0, i] - self.boundary_points[1]), 0, 0, -self.Kv])
        else:
            gradH = np.array([2 * (x[0, i] - self.boundary_points[1]), 0, 0, self.Kv])
        Lf_h = np.dot(gradH.T, f)
        Lg_h = np.dot(gradH.T, g)
        G = np.vstack([G, -Lg_h])
        H = np.vstack([H, np.array([self.arena_gain * h ** 3 + Lf_h])])

        # Neg X
        h = ((x[0, i] - self.boundary_points[0]) ** 2 - self.safety_radius ** 2 - self.Kv * abs(x[3, i]))
        if x[3, i] >= 0:
            gradH = np.array([2 * (x[0, i] - self.boundary_points[0]), 0, 0, -self.Kv])
        else:
            gradH = np.array([2 * (x[0, i] - self.boundary_points[0]), 0, 0, self.Kv])
        Lf_h = np.dot(gradH.T, f)
        Lg_h = np.dot(gradH.T, g)
        G = np.vstack([G, -Lg_h])
        H = np.vstack([H, np.array([self.arena_gain * h ** 3 + Lf_h])])

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

