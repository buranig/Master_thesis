import numpy as np

# To perform mpc
from casadi import *
import do_mpc

import time

from lar_msgs.msg import CarControlStamped, CarStateStamped
from typing import List

from bumper_cars.utils import car_utils as utils
from bumper_cars.classes.State import State
from bumper_cars.classes.Controller import Controller

# For the parameter file
import yaml



np.random.seed(1)

class MPC_algorithm(Controller):
    """
    This class implements the MPC (Model Predictive Control) algorithm.

    Attributes:
        dxu (numpy.ndarray): Control input of shape (2,) representing the throttle and steering.
        safety_radius (float): Safety radius for collision avoidance.
        barrier_gain (float): Gain for the barrier function.
        Kv (float): Gain for the velocity term.
        closest_point (tuple): Coordinates of the closest point on the boundary.

    """    
    def __init__(self, controller_path:str, car_i = 0):
        """
        Initializes the MPC_algorithm class.

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

        self.safety_radius = yaml_object["MPC"]["safety_radius"]
        self.safety_radius = yaml_object["MPC"]["safety_radius"]

        # Instantiate border variables
        self.__compute_track_constants()
        self.closest_point = (0.0, 0.0)

        self.goal_input = CarControlStamped()
        self.goal = State()
        self.mpc = None
        self.__prepare_mpc()

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
        u = self.__MPC(self.car_i, self.curr_state)

        # Project it to range [-1, 1]
        car_cmd.throttle = np.interp(u[0][0], [self.car_model.min_acc, self.car_model.max_acc], [-1, 1]) * self.car_model.acc_gain
        car_cmd.steering = np.interp(u[1][0], [-self.car_model.max_steer, self.car_model.max_steer], [-1, 1])

        x_traj = self.mpc.data.prediction(('_x', 'x')).flatten()   
        y_traj = self.mpc.data.prediction(('_x', 'y')).flatten()
        self.trajectory = zip(x_traj, y_traj)


        return car_cmd

    def set_goal(self, goal: CarControlStamped) -> None:
        """
        Sets the goal for the C3BF controller.

        Args:
            goal (CarControlStamped): The goal containing the desired throttle and steering values.

        Returns:
        """
        self.goal_input = CarControlStamped()
        self.goal_input.throttle = np.interp(goal.throttle, [-1, 1], [self.car_model.min_acc, self.car_model.max_acc]) * self.car_model.acc_gain
        self.goal_input.steering = np.interp(goal.steering, [-1, 1], [-self.car_model.max_steer, self.car_model.max_steer])

        # # # next_state = self.__simulate_input(self.goal_input)
        # # # self.goal = next_state

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
    
    def __simulate_input(self, car_cmd: CarControlStamped) -> State:
        curr_state = self.car_model.step(car_cmd, self.dt)
        t = self.dt
        while t<self.ph:
            curr_state = self.car_model.step(car_cmd, self.dt, curr_state=curr_state)
            t += self.dt
        return curr_state
        
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

        # Precompute denominators for distance calculation
        self.denom1 = np.sqrt(self.a1**2 + self.b1**2)
        self.denom2 = np.sqrt(self.a2**2 + self.b2**2)
        self.denom3 = np.sqrt(self.a3**2 + self.b3**2)
        self.denom4 = np.sqrt(self.a4**2 + self.b4**2)


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
        self.dist_1 = abs(self.a1*x0 + self.b1*y0 + self.c1)/np.sqrt(self.a1**2 + self.b1**2)
        self.dist_2 = abs(self.a2*x0 + self.b2*y0 + self.c2)/np.sqrt(self.a2**2 + self.b2**2)
        self.dist_3 = abs(self.a3*x0 + self.b3*y0 + self.c3)/np.sqrt(self.a3**2 + self.b3**2)
        self.dist_4 = abs(self.a4*x0 + self.b4*y0 + self.c4)/np.sqrt(self.a4**2 + self.b4**2)
        

    def __MPC(self, i, x) -> np.ndarray:

        # Precompute distances
        self.__compute_closest_point(i, x)

        # Initial state
        x0 = np.array([
                x[0,i],         # x
                x[1,i],         # y
                x[2,i],         # yaw
                x[3,i],         # vel
                self.dist_1,    
                self.dist_2,
                self.dist_3,
                self.dist_4])
        
        self.mpc.x0 = x0

        # Use initial state to set the initial guess
        self.mpc.set_initial_guess()

        # MPC step
        # pre = time.time()
        u0 = self.mpc.make_step(x0)
        # post = time.time()
        # print(f"Iteration time: {post - pre:.4f}s")
        # print(f"Control action: {u0}")
        return u0


    # Define the function to update the time-varying parameters
    def __tvp_fun(self, t_now):
        self.goal_param['_tvp', :] = np.array([self.goal_input.throttle,
                                               self.goal_input.steering,
                                               self.goal.x,
                                               self.goal.y])
        
        return self.goal_param


    def __prepare_mpc(self):
        # Initialize model
        model_type = 'continuous'
        model = do_mpc.model.Model(model_type)

        # States are described by x,y,yaw,vel,omega
        x = model.set_variable('_x',  'x', shape=(1,1))
        y = model.set_variable('_x',  'y', shape=(1,1))
        yaw = model.set_variable('_x',  'yaw', shape=(1,1))
        v = model.set_variable('_x',  'v', shape=(1,1))
        dist_1 = model.set_variable('_x',  'dist_1', shape=(1,1))
        dist_2 = model.set_variable('_x',  'dist_2', shape=(1,1))
        dist_3 = model.set_variable('_x',  'dist_3', shape=(1,1))
        dist_4 = model.set_variable('_x',  'dist_4', shape=(1,1))
        # omega = model.set_variable('_x',  'omega')
        
        # Input is defined by acceleration (a) and steering (delta)
        a = model.set_variable('_u',  'a', shape=(1,1))
        delta = model.set_variable('_u',  'delta', shape=(1,1))

        # Parameters (e.g., desired values or any time-varying parameter)
        p_a = model.set_variable('_tvp', 'p_a')
        p_delta = model.set_variable('_tvp', 'p_delta')
        p_x = model.set_variable('_tvp', 'p_x')
        p_y = model.set_variable('_tvp', 'p_y')

        # Start by using linear model, maybe if it works I upgrade
        x_dot = model.set_variable('_z',  'dx', shape=(1,1))
        y_dot = model.set_variable('_z',  'dy', shape=(1,1))
        yaw_dot = model.set_variable('_z',  'dyaw', shape=(1,1))
        v_dot = model.set_variable('_z',  'dv', shape=(1,1))

        ddist_1 = model.set_variable('_z',  'ddist_1', shape=(1,1))
        ddist_2 = model.set_variable('_z',  'ddist_2', shape=(1,1))
        ddist_3 = model.set_variable('_z',  'ddist_3', shape=(1,1))
        ddist_4 = model.set_variable('_z',  'ddist_4', shape=(1,1))



        # Assign the differential equations to the model
        model.set_rhs('x', x_dot)
        model.set_rhs('y', y_dot)
        model.set_rhs('yaw', yaw_dot)
        model.set_rhs('v', v_dot)
        model.set_rhs('dist_1', ddist_1)
        model.set_rhs('dist_2', ddist_2)
        model.set_rhs('dist_3', ddist_3)
        model.set_rhs('dist_4', ddist_4)

        # Define the model equations
        euler_lagrange = vertcat(
            x_dot - v * np.cos(yaw),
            y_dot - v * np.sin(yaw),
            yaw_dot - v / self.car_model.L * tan(delta),
            v_dot - a,
            ddist_1 - sign(self.a1*x + self.b1*y + self.c1)/self.denom1*(self.a1*x_dot + self.b1*y_dot),
            ddist_2 - sign(self.a2*x + self.b2*y + self.c2)/self.denom2*(self.a2*x_dot + self.b2*y_dot),
            ddist_3 - sign(self.a3*x + self.b3*y + self.c3)/self.denom3*(self.a3*x_dot + self.b3*y_dot),
            ddist_4 - sign(self.a4*x + self.b4*y + self.c4)/self.denom4*(self.a4*x_dot + self.b4*y_dot)
            )
        
        model.set_alg('euler_lagrange', euler_lagrange)

        # Finalize the model setup
        model.setup()
        
        
        self.mpc = do_mpc.controller.MPC(model)

        setup_mpc = {
            'n_horizon': int(self.ph/self.dt),
            'n_robust': 0,
            'open_loop': True,
            't_step': self.dt,
            'state_discretization': 'collocation',
            'collocation_type': 'radau',
            'collocation_deg': 2,
            'collocation_ni': 2,
            'store_full_solution': True,
            # Use MA27 linear solver in ipopt for faster calculations:
            'nlpsol_opts': {
                # 'ipopt.print_level': 0,  # Suppress solver output
                # 'ipopt.linear_solver': 'MA27'
            }
        }

        self.mpc.set_param(**setup_mpc)
        self.mpc.settings.supress_ipopt_output()
        
        # # Safety distance
        self.mpc.bounds['lower', '_x', 'dist_1'] = 0.1
        self.mpc.bounds['lower', '_x', 'dist_2'] = 0.1
        self.mpc.bounds['lower', '_x', 'dist_3'] = 0.1
        self.mpc.bounds['lower', '_x', 'dist_4'] = 0.1


        # upper and lower bounds of the control input
        self.mpc.bounds['lower','_u','a'] = self.car_model.min_acc
        self.mpc.bounds['lower','_u','delta'] = -self.car_model.max_steer
        self.mpc.bounds['upper','_u','a'] = self.car_model.max_acc
        self.mpc.bounds['upper','_u','delta'] = self.car_model.max_steer

        mterm = -(x - p_x)**2 *0.0 - (y - p_y)**2*0.0           # terminal cost
        lterm = (a - p_a)**2 + (delta - p_delta)**2             # stage cost
        
        # Set the time-varying parameters
        self.goal_param = self.mpc.get_tvp_template()
        self.goal_param['_tvp', :] = np.empty([4,1])
        self.mpc.set_tvp_fun(self.__tvp_fun)
        
        self.mpc.set_objective(mterm=mterm, lterm=lterm)
        self.mpc.set_rterm(a=0.0, delta=0.0) # penalty on input changes
        self.mpc.setup()


        # Initial state
        e = np.ones([model.n_x,1])
        x0 = np.random.uniform(-1*e, 1*e)  # Values between -1 and 1 for all states
        self.mpc.x0 = x0

        # Use initial state to set the initial guess
        self.mpc.set_initial_guess()

        # MPC step
        pre = time.time()
        u0 = self.mpc.make_step(x0)
        post = time.time()
        print(f"First iteration time: {post - pre:.4f}s")
        print(f"Control action: {u0}")
