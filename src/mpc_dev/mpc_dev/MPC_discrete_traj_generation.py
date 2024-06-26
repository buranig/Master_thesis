import abc
import time
import casadi
import matplotlib.pyplot as plt
import planner.utils as utils
import numpy as np
import math
import lbp_dev.lattice as lt
from MPC import get_switch_back_course

import json
import pathlib

path = pathlib.Path('/home/giacomo/thesis_ws/src/bumper_cars/params.json')
# Opening JSON file
with open(path, 'r') as openfile:
    # Reading from json file
    json_object = json.load(openfile)

L = json_object["Car_model"]["L"]
max_steer = json_object["CBF_simple"]["max_steer"]  # [rad] max steering angle
max_speed = json_object["Car_model"]["max_speed"] # [m/s]
min_speed = json_object["Car_model"]["min_speed"] # [m/s]
max_acc = json_object["CBF_simple"]["max_acc"] 
min_acc = json_object["CBF_simple"]["min_acc"] 
car_max_acc = json_object["Controller"]["max_acc"]
car_min_acc = json_object["Controller"]["min_acc"]
dt = json_object["Controller"]["dt"]
safety_radius = json_object["CBF_simple"]["safety_radius"]
barrier_gain = json_object["CBF_simple"]["barrier_gain"]
arena_gain = json_object["CBF_simple"]["arena_gain"]
Kv = json_object["CBF_simple"]["Kv"] # interval [0.5-1]
Lr = L / 2.0  # [m]
Lf = L - Lr
Cf = json_object["Car_model"]["Cf"]  # N/rad
Cr = json_object["Car_model"]["Cr"] # N/rad
Iz = json_object["Car_model"]["Iz"]  # kg/m2
m = json_object["Car_model"]["m"]  # kg
WB = json_object["Controller"]["WB"]

robot_num = 12 #json_object["robot_num"]
safety_init = json_object["safety"]
min_dist = json_object["min_dist"]
width_init = json_object["width"]
height_init = json_object["height"]
to_goal_stop_distance = json_object["to_goal_stop_distance"]
boundary_points = np.array([-width_init/2, width_init/2, -height_init/2, height_init/2])
check_collision_bool = False
add_noise = json_object["add_noise"]
noise_scale_param = json_object["noise_scale_param"]
show_animation = json_object["show_animation"]
predict_time = json_object["LBP"]["predict_time"] # [s]
dilation_factor = json_object["LBP"]["dilation_factor"]
linear_model = True #json_object["linear_model"]

debug = True

color_dict = {0: "r", 1: "b", 2: "g", 3: "y", 4: "m", 5: "c", 6: "k", 7: "tab:orange", 8: "tab:brown", 9: "tab:gray", 10: "tab:olive", 11: "tab:pink", 12: "tab:purple", 13: "tab:red", 14: "tab:blue", 15: "tab:green"}

def uniform_sampling(d, a_max, a_min, nxy):
    """
    Uniform sampling of states

    Args:
        d (float): distance
        a_max (float): maximum angle
        a_min (float): minimum angle
        nxy (int): number of states

    Returns:
        states (list): list of states
    """
    states = []
    for i in range(nxy+1):
        angle = a_min + (a_max - a_min) * i / (nxy - 1)
        states.append([d*np.cos(angle), d*np.sin(angle), utils.normalize_angle(angle+np.pi)])
    return states

class Controller:
    """Abstract Base Class for control implementation."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def solve(self):
        """Returns a dictionary sol_dict with control input to apply,
        as well as other useful information (e.g. MPC solution).

        In particular, sol_dict must have a key "u_control" such that
                    sol_dict["u_control"][0] = acceleration input to apply
                    sol_dict["u_control"][1] = steering input to apply
        """
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, update_dict):
        """Updates the state of the controller with feedback info contained in update_dict."""
        raise NotImplementedError

# XY Nonlinear Kinematic MPC Module.

class KinMPCPathFollower(Controller):

    def __init__(
        self,
        obsx,
        obsy,
        N=20,  # timesteps in MPC Horizon
        DT=0.1,  # discretization time between timesteps (s)
        L_F=Lf,  # distance from CoG to front axle (m)
        L_R=Lf,  # distance from CoG to rear axle (m)
        V_MIN=min_speed,  # min/max velocity constraint (m/s)
        V_MAX=max_speed,
        A_MIN=min_acc,  # min/max acceleration constraint (m/s^2)
        A_MAX=max_acc,
        DF_MIN=-max_steer,  # min/max front steer angle constraint (rad)
        DF_MAX=max_steer,
        A_DOT_MIN=-1.5,  # min/max jerk constraint (m/s^3)
        A_DOT_MAX=1.5,
        DF_DOT_MIN=-0.5,  # min/max front steer angle rate constraint (rad/s)
        DF_DOT_MAX=0.5,
        Q=[4.0, 4.0, 10.0, 4.0, 0.0],  # weights on x, y, psi, and v.
        R=[0.0, 0.0],
    ):  # weights on jerk and slew rate (steering angle derivative)
        for key in list(locals()):
            if key == "self":
                pass
            elif key == "Q":
                self.Q = casadi.diag(Q)
            elif key == "R":
                self.R = casadi.diag(R)
            else:
                setattr(self, "%s" % key, locals()[key])
        self.opti = casadi.Opti()

        """ 
		(1) Parameters
		"""
        self.obsx = obsx
        self.obsy = obsy
        self.u_prev = self.opti.parameter(
            2
        )  # previous input: [u_{acc, -1}, u_{df, -1}]
        
        if linear_model:
            self.z_curr = self.opti.parameter(4)  # current state:  [x_0, y_0, psi_0, v_0]
        else:
            self.z_curr = self.opti.parameter(5)

        # Reference trajectory we would like to follow.
        # First index corresponds to our desired state at timestep k+1:
        #   i.e. z_ref[0,:] = z_{desired, 1}.
        # Second index selects the state element from [x_k, y_k, psi_k, v_k].
        if linear_model:
            self.z_ref = self.opti.parameter(self.N, 4)
        else:
            self.z_ref = self.opti.parameter(self.N, 5)

        self.x_ref = self.z_ref[:, 0]
        self.y_ref = self.z_ref[:, 1]
        self.psi_ref = self.z_ref[:, 2]
        self.v_ref = self.z_ref[:, 3]
        if not linear_model:
            self.omega_ref = self.z_ref[:, 4]

        """
		(2) Decision Variables
		"""
        # Actual trajectory we will follow given the optimal solution.
        # First index is the timestep k, i.e. self.z_dv[0,:] is z_0.
        # It has self.N+1 timesteps since we go from z_0, ..., z_self.N.
        # Second index is the state element, as detailed below.
        if linear_model:
            self.z_dv = self.opti.variable(self.N + 1, 4)
        else:
            self.z_dv = self.opti.variable(self.N + 1, 5)

        # self.x_dv = self.z_dv[:, 0]
        # self.y_dv = self.z_dv[:, 1]
        # self.psi_dv = self.z_dv[:, 2]
        # self.v_dv = self.z_dv[:, 3]

        # Control inputs used to achieve self.z_dv according to dynamics.
        # First index is the timestep k, i.e. self.u_dv[0,:] is u_0.
        # Second index is the input element as detailed below.
        self.u_dv = self.opti.variable(self.N, 2)

        self.acc_dv = self.u_dv[:, 0]
        self.df_dv = self.u_dv[:, 1]

        # Slack variables used to relax input rate constraints.
        # Matches self.u_dv in structure but timesteps range from -1, ..., N-1.
        self.sl_dv = self.opti.variable(self.N, 2)

        self.sl_acc_dv = self.sl_dv[:, 0]
        self.sl_df_dv = self.sl_dv[:, 1]

        """
		(3) Problem Setup: Constraints, Cost, Initial Solve
		"""
        self._add_constraints()

        self._add_cost()

        self._update_initial_condition(0.0, 0.0, 0.0, 0.0)

        self._update_reference(
            self.N * [5.0], self.N * [1.0], self.N * [0.0], self.N * [0.0]
        )

        self._update_previous_input(0.0, 0.0)

        # Ipopt with custom options: https://web.casadi.org/docs/ -> see sec 9.1 on Opti stack.
        p_opts = {"expand": True}
        s_opts = {"max_cpu_time": 1, "print_level": 1}
        self.opti.solver("ipopt", p_opts, s_opts)

        sol = self.solve()

    def _add_constraints(self):
        # State Bound Constraints
        self.opti.subject_to(self.opti.bounded(self.V_MIN, self.z_dv[:,3], self.V_MAX))

        # Initial State Constraint
        self.opti.subject_to(self.z_dv[0,0] == self.z_curr[0])
        self.opti.subject_to(self.z_dv[0,1] == self.z_curr[1])
        self.opti.subject_to(self.z_dv[0,2] == self.z_curr[2])
        self.opti.subject_to(self.z_dv[0,3] == self.z_curr[3])
        if not linear_model:
            self.opti.subject_to(self.z_dv[0,4] == self.z_curr[4])        

        # State Dynamics Constraints
        # Kinematic model:
        if linear_model:
            for i in range(self.N):
                beta = casadi.atan(
                    self.L_R / (self.L_F + self.L_R) * casadi.tan(self.df_dv[i])
                )

                self.opti.subject_to(
                    self.z_dv[i + 1, 0]
                    == self.z_dv[i, 0]
                    + self.DT * (self.z_dv[i, 3] * casadi.cos(self.z_dv[i, 2] + beta))
                )

                self.opti.subject_to(
                    self.z_dv[i + 1, 1]
                    == self.z_dv[i, 1]
                    + self.DT * (self.z_dv[i, 3] * casadi.sin(self.z_dv[i, 2] + beta))
                )

                self.opti.subject_to(
                    self.z_dv[i + 1, 2]
                    == self.z_dv[i, 2]
                    + self.DT * (self.z_dv[i, 3] / self.L_R * casadi.sin(beta))
                )

                self.opti.subject_to(
                    self.z_dv[i + 1, 3] == self.z_dv[i, 3] + self.DT * (self.acc_dv[i])
                )
        # Dynamic Model
        else:
            for i in range(self.N):
                beta = casadi.atan(
                    self.L_R / (self.L_F + self.L_R) * casadi.tan(self.df_dv[i])
                )
                kf = -Cf
                kr = -Cr

                self.opti.subject_to(
                    self.z_dv[i+1, 0] 
                    ==  self.z_dv[i, 0] 
                    + self.z_dv[i, 3] * casadi.cos(beta) * casadi.cos(self.z_dv[i, 2]) * self.DT - self.z_dv[i, 3] * casadi.sin(beta) * casadi.sin(self.z_dv[i, 2]) * self.DT
                    )
                
                self.opti.subject_to(
                    self.z_dv[i+1, 1]
                    ==  self.z_dv[i, 1]
                    + self.z_dv[i, 3] * casadi.cos(beta) * casadi.sin(self.z_dv[i, 2]) * self.DT + self.z_dv[i, 3] * casadi.sin(beta) * casadi.cos(self.z_dv[i, 2]) * self.DT
                    )

                self.opti.subject_to(
                    self.z_dv[i+1, 2]
                    ==  self.z_dv[i, 2]
                    + self.z_dv[i, 4] * self.DT
                    )
                
                self.opti.subject_to(
                    self.z_dv[i+1, 3]
                    == self.z_dv[i, 3]
                    + casadi.sqrt(
                        (self.acc_dv[i] * self.DT)**2 
                        + ((m * self.z_dv[i, 3]*casadi.cos(beta)*self.z_dv[i, 3]*casadi.sin(beta) + 
                            (self.L_F*kf-self.L_R*kr)*self.z_dv[i, 4] 
                            - self.DT*kf*self.df_dv[i]*self.z_dv[i, 3]*casadi.cos(beta) 
                            - self.DT*m*self.z_dv[i, 3]*casadi.cos(beta)**2*self.z_dv[i, 4])
                            /(m*self.z_dv[i, 3]*casadi.cos(beta) - self.DT*(kf+kr)))**2
                    )
                    )
                
                self.opti.subject_to(
                    self.z_dv[i+1, 4]
                    ==  (Iz*self.z_dv[i, 4]*self.z_dv[i, 3]*casadi.cos(beta) + self.DT*(self.L_F*kf-self.L_R*kr)*self.z_dv[i, 4] - self.DT*self.L_F*kf*self.df_dv[i]*self.z_dv[i, 3]*casadi.cos(beta))/(Iz*self.z_dv[i, 3]*casadi.cos(beta) - self.DT*(self.L_F**2*kf+self.L_R**2*kr))
                    )

        # Input Bound Constraints
        self.opti.subject_to(self.opti.bounded(self.A_MIN, self.acc_dv, self.A_MAX))
        self.opti.subject_to(self.opti.bounded(self.DF_MIN, self.df_dv, self.DF_MAX))

        # Input Rate Bound Constraints
        self.opti.subject_to(
            self.opti.bounded(
                self.A_DOT_MIN * self.DT - self.sl_acc_dv[0],
                self.acc_dv[0] - self.u_prev[0],
                self.A_DOT_MAX * self.DT + self.sl_acc_dv[0],
            )
        )

        self.opti.subject_to(
            self.opti.bounded(
                self.DF_DOT_MIN * self.DT - self.sl_df_dv[0],
                self.df_dv[0] - self.u_prev[1],
                self.DF_DOT_MAX * self.DT + self.sl_df_dv[0],
            )
        )

        for i in range(self.N - 1):
            self.opti.subject_to(
                self.opti.bounded(
                    self.A_DOT_MIN * self.DT - self.sl_acc_dv[i + 1],
                    self.acc_dv[i + 1] - self.acc_dv[i],
                    self.A_DOT_MAX * self.DT + self.sl_acc_dv[i + 1],
                )
            )
            self.opti.subject_to(
                self.opti.bounded(
                    self.DF_DOT_MIN * self.DT - self.sl_df_dv[i + 1],
                    self.df_dv[i + 1] - self.df_dv[i],
                    self.DF_DOT_MAX * self.DT + self.sl_df_dv[i + 1],
                )
            )
        # Other Constraints
        self.opti.subject_to(0 <= self.sl_df_dv)
        self.opti.subject_to(0 <= self.sl_acc_dv)

        # e.g. things like collision avoidance or lateral acceleration bounds could go here.
        # for obs in range(len(self.obsx)):
        #     self.opti.subject_to(
		# 		casadi.dot(self.z_dv[0] - self.obsx[obs], self.z_dv[0] - self.obsx[obs])
		# 		+ casadi.dot(self.z_dv[1] - self.obsy[obs], self.z_dv[1] - self.obsy[obs])
		# 		> self.L_R**2
		# 	)

    def _add_cost(self):
        def _quad_form(z, Q):
            return casadi.mtimes(z, casadi.mtimes(Q, z.T))

        cost = 0
        cost += _quad_form(self.z_dv[0, :] - self.z_curr.T, 100*self.Q)  # initial state cost
        for i in range(self.N):
            cost += _quad_form(
                self.z_dv[i + 1, :] - self.z_ref[i, :], self.Q
            )  # tracking cost

        for i in range(self.N - 1):
            cost += _quad_form(
                self.u_dv[i + 1, :] - self.u_dv[i, :], self.R
            )  # input derivative cost

        cost += casadi.sum1(self.sl_df_dv) + casadi.sum1(self.sl_acc_dv)  # slack cost

        self.opti.minimize(cost)

    def solve(self):
        st = time.time()
        try:
            sol = self.opti.solve()
            # Optimal solution.
            u_mpc = sol.value(self.u_dv)
            z_mpc = sol.value(self.z_dv)
            sl_mpc = sol.value(self.sl_dv)
            z_ref = sol.value(self.z_ref)
            is_opt = True
        except:
            # Suboptimal solution (e.g. timed out).
            u_mpc = self.opti.debug.value(self.u_dv)
            z_mpc = self.opti.debug.value(self.z_dv)
            sl_mpc = self.opti.debug.value(self.sl_dv)
            z_ref = self.opti.debug.value(self.z_ref)
            is_opt = False

        solve_time = time.time() - st

        sol_dict = {}
        sol_dict["u_control"] = u_mpc[0, :]  # control input to apply based on solution
        sol_dict["optimal"] = is_opt  # whether the solution is optimal or not
        sol_dict["solve_time"] = solve_time  # how long the solver took in seconds
        sol_dict["u_mpc"] = u_mpc  # solution inputs (N by 2, see self.u_dv above)
        sol_dict["z_mpc"] = z_mpc  # solution states (N+1 by 4, see self.z_dv above)
        sol_dict["sl_mpc"] = (
            sl_mpc  # solution slack vars (N by 2, see self.sl_dv above)
        )
        sol_dict["z_ref"] = z_ref  # state reference (N by 4, see self.z_ref above)

        return sol_dict

    def update(self, update_dict):
        self._update_initial_condition(
            *[update_dict[key] for key in ["x0", "y0", "psi0", "v0"]]
        )
        self._update_reference(
            *[update_dict[key] for key in ["x_ref", "y_ref", "psi_ref", "v_ref"]]
        )
        self._update_previous_input(
            *[update_dict[key] for key in ["acc_prev", "df_prev"]]
        )

        if "warm_start" in update_dict.keys():
            # Warm Start used if provided.  Else I believe the problem is solved from scratch with initial values of 0.
            self.opti.set_initial(self.z_dv, update_dict["warm_start"]["z_ws"])
            self.opti.set_initial(self.u_dv, update_dict["warm_start"]["u_ws"])
            self.opti.set_initial(self.sl_dv, update_dict["warm_start"]["sl_ws"])

    def _update_initial_condition(self, x0, y0, psi0, vel0, omega0=0.0):
        if linear_model:
            self.opti.set_value(self.z_curr, [x0, y0, psi0, vel0])
        else:
            self.opti.set_value(self.z_curr, [x0, y0, psi0, vel0, omega0])

    def _update_reference(self, x_ref, y_ref, psi_ref, v_ref, omega_ref=0.0):
        self.opti.set_value(self.x_ref, x_ref)
        self.opti.set_value(self.y_ref, y_ref)
        self.opti.set_value(self.psi_ref, psi_ref)
        self.opti.set_value(self.v_ref, v_ref)
        if not linear_model:
            self.opti.set_value(self.omega_ref, omega_ref)

    def _update_previous_input(self, acc_prev, df_prev):
        self.opti.set_value(self.u_prev, [acc_prev, df_prev])

def main2():
    kmpc = KinMPCPathFollower(None, None)

    traj = {}
    temp = {}
    for v0 in np.arange(min_speed, max_speed+0.5, 0.5):
        traj[v0] = {}
        for v in np.arange(min_speed, max_speed+0.5, 0.5):
            if v==0.0 and v0==0.0:
                temp[v][i] = {}
                temp[v][i]['x'] = [0.0]*kmpc.N
                temp[v][i]['y'] = [0.0]*kmpc.N
                temp[v][i]['yaw'] = [0.0]*kmpc.N
                temp[v][i]['v'] = [0.0]*kmpc.N
                temp[v][i]['throttle'] = [0.0]*kmpc.N
                temp[v][i]['steering'] = [0.0]*kmpc.N
                traj[v0][v] = temp[v]
                continue
            print(f'velocity: {v}')
            temp[v] = {}
            
            k0 = 0.0
            nxy = 8
            nh = 3
            d = v*3
            # print(f'distance: {d}')

            if v<0.0:
                angle = 80
                a_min = - np.deg2rad(180 - angle)
                a_max = np.deg2rad(180 + angle)
                states = uniform_sampling(-d, a_max, a_min, nxy)
            else:
                angle = 80
                a_min = - np.deg2rad(angle)
                a_max = np.deg2rad(angle)
                p_min = - np.deg2rad(angle)
                p_max = np.deg2rad(angle)
                states = lt.calc_uniform_polar_states(nxy, nh, d, a_min, a_max, p_min, p_max)
            # print(f'states: {states}')

            plt.cla()
            plt.gcf().canvas.mpl_connect(
                "key_release_event",
                lambda event: [exit(0) if event.key == "escape" else None],
            )
            for i, state in enumerate(states):
                kmpc._update_initial_condition(0.0, 0.0, 0.0, v0, 0.0)
                kmpc._update_reference(
                    [state[0]] * kmpc.N, [state[1]] * kmpc.N, [state[2]] * kmpc.N, [v] * kmpc.N, [0.0]*kmpc.N)
                sol_dict = kmpc.solve()

                # Saving the trajectory in a dictionary
                x_mpc = sol_dict['z_mpc'][:,0]
                y_mpc = sol_dict['z_mpc'][:,1]
                psi_mpc = sol_dict['z_mpc'][:,2]
                v_mpc = sol_dict['z_mpc'][:,3]

                temp[v][i] = {}
                temp[v][i]['x'] = list(x_mpc)
                temp[v][i]['y'] = list(y_mpc)
                temp[v][i]['yaw'] = list(psi_mpc)
                temp[v][i]['v'] = list(v_mpc)
                temp[v][i]['throttle'] = list(sol_dict['u_mpc'][:,0])
                temp[v][i]['steering'] = list(sol_dict['u_mpc'][:,1])

                if debug:
                    plt.plot(sol_dict['z_mpc'][:,0], sol_dict['z_mpc'][:,1], 'b--')
                    # utils.plot_arrow(sol_dict['z_mpc'][z,0], sol_dict['z_mpc'][z,1], sol_dict['z_mpc'][z,2] + sol_dict['u_mpc'][z,1] , length=3, width=0.5)
                    utils.plot_arrow(sol_dict['z_mpc'][-1,0], sol_dict['z_mpc'][-1,1], sol_dict['z_mpc'][-1,2], length=0.1, width=0.1)
                    # plt.scatter(sol_dict['z_ref'][-1,0], sol_dict['z_ref'][-1,1], marker="x", color=color_dict[0], s=200)
                    print('\n')
            if debug:
                utils.plot_robot(0.0, 0.0, 0.0, 0)
                plt.show()

            traj[v0][v] = temp[v]

    # saving the complete trajectories to a csv file
    with open('src/mpc_dev/mpc_dev/MPC_casadi.json', 'w') as file:
        json.dump(traj, file, indent=4)

    print("\nThe JSON data has been written to 'src/mpc_dev/mpc_dev/MPC_casadi.json'")

if __name__ == "__main__":
    main2()