"""

Lookup Table generation for model predictive trajectory generator

author: Atsushi Sakai

"""
import sys
import pathlib
path_planning_dir = pathlib.Path(__file__).parent.parent
sys.path.append(str(path_planning_dir))

from matplotlib import pyplot as plt
import numpy as np
import math
from lar_utils import car_utils as utils
from bumper_cars.classes.CarModel import State
from lbp_dev.LBP import LBP_algorithm as LBP
from lar_msgs.msg import CarControlStamped
import json

from scipy.interpolate import interp1d
import os
TABLE_PATH = os.path.dirname(os.path.abspath(__file__)) + "/lookup_table.csv"

from ament_index_python.packages import get_package_share_directory


car_yaml = os.path.join(
    get_package_share_directory('bumper_cars'),
    'config',
    'controller.yaml'
)

class TrajectoryGenerator(LBP):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    def __init__(self) -> None:
        super().__init__(car_yaml)
        pass


    def calc_trajectory(self, curr_state:State, cmd:CarControlStamped):
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


    def generate_lookup_table(self):
        temp = {}
        initial_state = State()
        initial_state.x = 0.0
        initial_state.y = 0.0
        initial_state.yaw = np.radians(90.0)

        for v in np.arange(0.5, 2.0+0.5, 0.5): # TODO: Remove this hard-coded velocity range  
            print(v)  
            temp[v] = {}
            k0 = 0.0
            nxy = 5
            nh = 3
            d = v*self.ph
            # print(f'distance: {d}')

            if v == 0.5:    # TODO: Remove this hard-coded velocity limit
                angle = 45
                a_min = - np.deg2rad(angle)
                a_max = np.deg2rad(angle)
                p_min = - np.deg2rad(angle)
                p_max = np.deg2rad(angle)
            else:
                angle = 60
                a_min = - np.deg2rad(angle)
                a_max = np.deg2rad(angle)
                p_min = - np.deg2rad(angle)
                p_max = np.deg2rad(angle)
            states = self.calc_uniform_polar_states(nxy, nh, d, a_min, a_max, p_min, p_max)
            result = self.generate_path(states, k0, v)

            i = 0
            for table in result:
                xc, yc, yawc, kp = self.generate_trajectory(
                    table[3], table[4], table[5], k0, v)
                for id, element in enumerate(kp): kp[id] = np.clip(element, -self.car_model.max_steer, self.car_model.max_steer) # clipping elements withing feasible bounds
                temp[v][i] = {}
                temp[v][i]['ctrl'] = list(kp)
                temp[v][i]['x'] = xc
                temp[v][i]['y'] = yc
                temp[v][i]['yaw'] = yawc
                i +=1
            
            if v==1.0:
                target = [[1.0, 3.0, np.deg2rad(90.0)],
                        [1.0, -3.0, np.deg2rad(-90.0)],
                        [1.5, 3.0, np.deg2rad(90.0)],
                        [1.5, -3.0, np.deg2rad(-90.0)]]
                result = self.generate_path(target, k0, v, k=True)
                i = 0
                for table in result:
                    xc, yc, yawc, kp = self.generate_trajectory(
                        table[3], table[4], table[5], k0, v)
                    for id, element in enumerate(kp): kp[id] = np.clip(element, -self.car_model.max_steer, self.car_model.max_steer) # clipping elements withing feasible bounds
                    temp[v][i] = {}
                    temp[v][i]['ctrl'] = list(kp)
                    temp[v][i]['x'] = xc
                    temp[v][i]['y'] = yc
                    temp[v][i]['yaw'] = yawc
                    i +=1

        for v in np.arange(-1.0, 0.5, 0.5):
            x_init = [0.0, 0.0, 0.0, v]
            i = 0
            temp[v] = {}
            for delta in np.arange(-self.car_model.max_steer, self.car_model.max_steer+self.delta_resolution, self.delta_resolution):
                u = [0.0, delta]
                initial_state.v = v
                cmd = CarControlStamped()
                cmd.throttle = 0.0
                cmd.steering = np.interp(delta, [-self.car_model.max_steer, self.car_model.max_steer], [-1, 1])
                
                traj = self.calc_trajectory(initial_state, cmd)
                # plt.plot(traj[:, 0], traj[:, 1])
                xc = traj[:, 0]
                yc = traj[:, 1]
                yawc = traj[:, 2]
                kp = [delta]*len(xc)

                temp[v][i] = {}
                temp[v][i]['ctrl'] = list(kp)
                temp[v][i]['x'] = list(xc)
                temp[v][i]['y'] = list(yc)
                temp[v][i]['yaw'] = list(yawc)
                i +=1
                # print(f'len: {len(xc)}')

        # saving the complete trajectories to a csv file
        with open(self.dir_path + '/../config/LBP.json', 'w') as file:
            json.dump(temp, file, indent=4)
        

    def generate_trajectory(self, s, km, kf, k0, v):
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
        x, y, yaw = [state.x], [state.y], [state.yaw]

        for ikp in kp:
            self.update(state, v, ikp, self.dt, self.car_model.L)
            x.append(state.x)
            y.append(state.y)
            yaw.append(state.yaw)

        return x, y, yaw, kp


    def calc_uniform_polar_states(self, nxy, nh, d, a_min, a_max, p_min, p_max):
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
        states = self.sample_states(angle_samples, a_min, a_max, d, p_max, p_min, nh)

        return states


    def sample_states(self, angle_samples, a_min, a_max, d, p_max, p_min, nh):
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



    def generate_path(self, target_states, k0, v, k=False):
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

            x, y, yaw, p, kp = self.optimize_trajectory(target, k0, init_p, v)

            if x is not None:
                # print("find good path")
                result.append(
                    [x[-1], y[-1], yaw[-1], float(p[0, 0]), float(p[1, 0]), float(p[2, 0])])

        # print("finish path generation")
        return result
    
    def calc_diff(self, target, x, y, yaw):
        pi_2_pi = target.yaw - yaw[-1]
        pi_2_pi = (pi_2_pi + math.pi) % (2 * math.pi) - math.pi
        d = np.array([target.x - x[-1],
                    target.y - y[-1],
                    pi_2_pi])

        return d

    def optimize_trajectory(self, target, k0, p, v):
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
            xc, yc, yawc, kp = self.generate_trajectory(p[0, 0], p[1, 0], p[2, 0], k0, v)
            dc = np.array(self.calc_diff(target, xc, yc, yawc)).reshape(3, 1)

            cost = np.linalg.norm(dc)
            if cost <= cost_th:
                # print("path is ok cost is:" + str(cost))
                break

            J = self.calc_j(target, p, h, k0, v)
            try:
                dp = - np.linalg.pinv(J) @ dc
            except np.linalg.linalg.LinAlgError:
                # print("cannot calc path LinAlgError")
                xc, yc, yawc, p = None, None, None, None
                break
            alpha = self.selection_learning_param(dp, p, k0, target, v)

            p += alpha * np.array(dp)
            # print(p.T)

        else:
            xc, yc, yawc, p = None, None, None, None
            # print("cannot calc path")

        return xc, yc, yawc, p, kp


    def calc_j(self, target, p, h, k0, v):
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
        xp, yp, yawp = self.generate_last_state(
            p[0, 0] + h[0], p[1, 0], p[2, 0], k0, v)
        dp = self.calc_diff(target, [xp], [yp], [yawp])
        xn, yn, yawn = self.generate_last_state(
            p[0, 0] - h[0], p[1, 0], p[2, 0], k0, v)
        dn = self.calc_diff(target, [xn], [yn], [yawn])
        d1 = np.array((dp - dn) / (2.0 * h[0])).reshape(3, 1)

        xp, yp, yawp = self.generate_last_state(
            p[0, 0], p[1, 0] + h[1], p[2, 0], k0, v)
        dp = self.calc_diff(target, [xp], [yp], [yawp])
        xn, yn, yawn = self.generate_last_state(
            p[0, 0], p[1, 0] - h[1], p[2, 0], k0, v)
        dn = self.calc_diff(target, [xn], [yn], [yawn])
        d2 = np.array((dp - dn) / (2.0 * h[1])).reshape(3, 1)

        xp, yp, yawp = self.generate_last_state(
            p[0, 0], p[1, 0], p[2, 0] + h[2], k0, v)
        dp = self.calc_diff(target, [xp], [yp], [yawp])
        xn, yn, yawn = self.generate_last_state(
            p[0, 0], p[1, 0], p[2, 0] - h[2], k0, v)
        dn = self.calc_diff(target, [xn], [yn], [yawn])
        d3 = np.array((dp - dn) / (2.0 * h[2])).reshape(3, 1)

        J = np.hstack((d1, d2, d3))

        return J
    


    def generate_last_state(self, s, km, kf, k0, v):
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

        _ = [self.update(state, v, ikp, dt, self.car_model.L) for ikp in kp]

        return state.x, state.y, state.yaw
    

    def selection_learning_param(self, dp, p, k0, target, v):
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
            xc, yc, yawc = self.generate_last_state(
                tp[0], tp[1], tp[2], k0, v)
            dc = self.calc_diff(target, [xc], [yc], [yawc])
            cost = np.linalg.norm(dc)

            if cost <= mincost and a != 0.0:
                mina = a
                mincost = cost

        #  print(mincost, mina)
        #  input()

        return mina
    
    # TODO: Remove this function
    def update(self, state, v, delta, dt, L):
        state.v = v
        delta = np.clip(delta, -np.radians(45), np.radians(45)) #TODO remove hardcoded limits
        state.x = state.x + state.v * math.cos(state.yaw) * dt
        state.y = state.y + state.v * math.sin(state.yaw) * dt
        state.yaw = state.yaw + state.v / L * math.tan(delta) * dt
        state.yaw = utils.normalize_angle(state.yaw)
        # state.yaw = pi_2_pi(state.yaw)

        return state

    
def main():
    asdf = TrajectoryGenerator()
    asdf.generate_lookup_table()

if __name__ == '__main__':
    main()
