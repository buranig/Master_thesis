import abc
import time
import casadi
import matplotlib.pyplot as plt
import planner.utils as utils
import numpy as np
import math
import lbp_dev.lattice as lt
from MPC import get_switch_back_course

color_dict = {
    0: "r",
    1: "b",
    2: "g",
    3: "y",
    4: "m",
    5: "c",
    6: "k",
    7: "tab:orange",
    8: "tab:brown",
    9: "tab:gray",
    10: "tab:olive",
    11: "tab:pink",
    12: "tab:purple",
    13: "tab:red",
    14: "tab:blue",
    15: "tab:green",
}


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
        N=80,  # timesteps in MPC Horizon
        DT=0.15,  # discretization time between timesteps (s)
        L_F=1.5213,  # distance from CoG to front axle (m)
        L_R=1.4987,  # distance from CoG to rear axle (m)
        V_MIN=0.0,  # min/max velocity constraint (m/s)
        V_MAX=20.0,
        A_MIN=-3.0,  # min/max acceleration constraint (m/s^2)
        A_MAX=2.0,
        DF_MIN=-0.78,  # min/max front steer angle constraint (rad)
        DF_MAX=0.78,
        A_DOT_MIN=-1.5,  # min/max jerk constraint (m/s^3)
        A_DOT_MAX=1.5,
        DF_DOT_MIN=-0.5,  # min/max front steer angle rate constraint (rad/s)
        DF_DOT_MAX=0.5,
        Q=[10.0, 10.0, 4.0, 0.1],  # weights on x, y, psi, and v.
        R=[10.0, 10.0],
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
        self.z_curr = self.opti.parameter(4)  # current state:  [x_0, y_0, psi_0, v_0]

        # Reference trajectory we would like to follow.
        # First index corresponds to our desired state at timestep k+1:
        #   i.e. z_ref[0,:] = z_{desired, 1}.
        # Second index selects the state element from [x_k, y_k, psi_k, v_k].
        self.z_ref = self.opti.parameter(self.N, 4)

        self.x_ref = self.z_ref[:, 0]
        self.y_ref = self.z_ref[:, 1]
        self.psi_ref = self.z_ref[:, 2]
        self.v_ref = self.z_ref[:, 3]

        """
		(2) Decision Variables
		"""
        # Actual trajectory we will follow given the optimal solution.
        # First index is the timestep k, i.e. self.z_dv[0,:] is z_0.
        # It has self.N+1 timesteps since we go from z_0, ..., z_self.N.
        # Second index is the state element, as detailed below.
        self.z_dv = self.opti.variable(self.N + 1, 4)

        self.x_dv = self.z_dv[:, 0]
        self.y_dv = self.z_dv[:, 1]
        self.psi_dv = self.z_dv[:, 2]
        self.v_dv = self.z_dv[:, 3]

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
        s_opts = {"max_cpu_time": 0.1, "print_level": 0}
        self.opti.solver("ipopt", p_opts, s_opts)

        sol = self.solve()

    def _add_constraints(self):
        # State Bound Constraints
        self.opti.subject_to(self.opti.bounded(self.V_MIN, self.v_dv, self.V_MAX))

        # Initial State Constraint
        self.opti.subject_to(self.x_dv[0] == self.z_curr[0])
        self.opti.subject_to(self.y_dv[0] == self.z_curr[1])
        self.opti.subject_to(self.psi_dv[0] == self.z_curr[2])
        self.opti.subject_to(self.v_dv[0] == self.z_curr[3])

        # State Dynamics Constraints
        for i in range(self.N):
            beta = casadi.atan(
                self.L_R / (self.L_F + self.L_R) * casadi.tan(self.df_dv[i])
            )
            self.opti.subject_to(
                self.x_dv[i + 1]
                == self.x_dv[i]
                + self.DT * (self.v_dv[i] * casadi.cos(self.psi_dv[i] + beta))
            )
            self.opti.subject_to(
                self.y_dv[i + 1]
                == self.y_dv[i]
                + self.DT * (self.v_dv[i] * casadi.sin(self.psi_dv[i] + beta))
            )
            self.opti.subject_to(
                self.psi_dv[i + 1]
                == self.psi_dv[i]
                + self.DT * (self.v_dv[i] / self.L_R * casadi.sin(beta))
            )
            self.opti.subject_to(
                self.v_dv[i + 1] == self.v_dv[i] + self.DT * (self.acc_dv[i])
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

    def _update_initial_condition(self, x0, y0, psi0, vel0):
        self.opti.set_value(self.z_curr, [x0, y0, psi0, vel0])

    def _update_reference(self, x_ref, y_ref, psi_ref, v_ref):
        self.opti.set_value(self.x_ref, x_ref)
        self.opti.set_value(self.y_ref, y_ref)
        self.opti.set_value(self.psi_ref, psi_ref)
        self.opti.set_value(self.v_ref, v_ref)

    def _update_previous_input(self, acc_prev, df_prev):
        self.opti.set_value(self.u_prev, [acc_prev, df_prev])


def motion(kmpc, x, u, dt):
    beta = casadi.atan(kmpc.L_R / (kmpc.L_F + kmpc.L_R) * casadi.tan(u[1]))
    x[0] = x[0] + kmpc.DT * (x[3] * casadi.cos(x[2] + beta))
    x[1] = x[1] + kmpc.DT * (x[3] * casadi.sin(x[2] + beta))
    x[2] = x[2] + kmpc.DT * (x[3] / kmpc.L_R * casadi.sin(beta))
    x[3] = x[3] + u[0] * kmpc.DT

    return x


def plot_robot(kmpc, x, y, yaw, i):
    """
    Plot the robot.

    Args:
        x (float): X-coordinate of the robot.
        y (float): Y-coordinate of the robot.
        yaw (float): Yaw angle of the robot.
        i (int): Index of the robot.
    """
    L = kmpc.L_F + kmpc.L_R
    WB = L / 2
    outline = np.array(
        [
            [-L / 2, L / 2, (L / 2), -L / 2, -L / 2],
            [WB / 2, WB / 2, -WB / 2, -WB / 2, WB / 2],
        ]
    )
    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])
    outline = (outline.T.dot(Rot1)).T
    outline[0, :] += x
    outline[1, :] += y
    plt.plot(
        np.array(outline[0, :]).flatten(),
        np.array(outline[1, :]).flatten(),
        color_dict[i],
        label="Robot " + str(i),
    )

def main():
    cx, cy, cyaw, ck = get_switch_back_course(3)
    # plt.plot(cx, cy, 'r--')
    # plt.show()
    idx = 1
    u_idx = 0
    acc_prev = 0.0
    df_prev = 0.0
    x = np.array([0.0, 0.0, 0.0, 0.0])

    kmpc = KinMPCPathFollower(None, None)
    # kmpc._update_initial_condition(x[0], x[1], x[2], x[3])
    # # kmpc._update_initial_condition(0., 0., 0., 0.)
    # kmpc._update_reference(
    #     [cx[idx]] * kmpc.N, [cy[idx]] * kmpc.N, [cyaw[idx]] * kmpc.N, [0] * kmpc.N
    # )
    # kmpc._update_previous_input(acc_prev, df_prev)
    # kmpc._add_constraints()
    sol_dict = kmpc.solve()
    print("x: ", x)
    print(f"goal: {sol_dict['z_mpc'][0, :]}")

    fig, ax = plt.subplots()

    for z in range(1000):
        plt.cla()
        plt.gcf().canvas.mpl_connect(
            "key_release_event",
            lambda event: [exit(0) if event.key == "escape" else None],
        )

        if u_idx == kmpc.N - 1 or np.linalg.norm([x[0] - cx[idx], x[1] - cy[idx]]) < 2:
            print("Switching to next reference point.")
            idx += 1
            print("idx: ", idx)

            # kmpc = KinMPCPathFollower(None, None)
            kmpc._update_initial_condition(x[0], x[1], x[2], x[3])
            kmpc._update_reference(
                [cx[idx]] * kmpc.N,
                [cy[idx]] * kmpc.N,
                [cyaw[idx]] * kmpc.N,
                [0] * kmpc.N,
            )
            kmpc._update_previous_input(acc_prev, df_prev)

            sol_dict = kmpc.solve()
            u_idx = 0
            # for key in sol_dict:
            #     print(key, sol_dict[key])
            # print("\n")
            print("x: ", x)
            print(f"goal: {sol_dict['z_mpc'][-1, :]}")
            print(f"ref: {sol_dict['z_ref'][-1, :]}")
            print('\n')
        else:
            u = sol_dict["u_mpc"][u_idx, :]
            x = motion(kmpc, x, u, kmpc.DT).reshape(-1)
            u_idx += 1
            acc_prev = u[0]
            df_prev = u[1]

        plot_robot(kmpc, x[0], x[1], x[2], 0)
        plt.plot(sol_dict["z_mpc"][:, 0], sol_dict["z_mpc"][:, 1], "b--")
        utils.plot_arrow(x[0], x[1], x[2] + u[1], length=3, width=0.5)
        utils.plot_arrow(x[0], x[1], x[2], length=1, width=0.5)
        plt.scatter(
            sol_dict["z_ref"][-1, 0],
            sol_dict["z_ref"][-1, 1],
            marker="x",
            color=color_dict[0],
            s=200,
        )
        plt.plot(cx, cy, "r--")
        if kmpc.obsx is not None:
            for obs in range(len(kmpc.obsx)):
                ax.add_patch(plt.Circle((kmpc.obsx[obs], kmpc.obsy[obs]), kmpc.L_R, color='r', alpha=0.5))
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.1)

    # sol_dict = kmpc.solve()

    # for key in sol_dict:
    # 	print(key, sol_dict[key])

    # fig, ax = plt.subplots()

    # # Plot simulation
    # for z in range(kmpc.N):
    # 	plt.cla()
    # 	utils.plot_robot(sol_dict['z_mpc'][z,0], sol_dict['z_mpc'][z,1], sol_dict['z_mpc'][z,2], 0)
    # 	plt.plot(sol_dict['z_mpc'][:,0], sol_dict['z_mpc'][:,1], 'b--')
    # 	utils.plot_arrow(sol_dict['z_mpc'][z,0], sol_dict['z_mpc'][z,1], sol_dict['z_mpc'][z,2] + sol_dict['u_mpc'][z,1] , length=3, width=0.5)
    # 	utils.plot_arrow(sol_dict['z_mpc'][z,0], sol_dict['z_mpc'][z,1], sol_dict['z_mpc'][z,2], length=1, width=0.5)
    # 	plt.scatter(sol_dict['z_ref'][-1,0], sol_dict['z_ref'][-1,1], marker="x", color=color_dict[0], s=200)
    # 	ax.add_patch(plt.Circle((kmpc.obsx, kmpc.obsy), 0.5, color='r', alpha=0.5))
    # 	plt.axis("equal")
    # 	plt.grid(True)
    # 	plt.pause(0.2)
    # plt.show()

def main2():
    kmpc = KinMPCPathFollower(None, None)

    temp = {}
    for v in np.arange(0.5, 2.0+0.5, 0.5):
        temp[v] = {}
        k0 = 0.0
        nxy = 5
        nh = 3
        d = v*3
        print(f'distance: {d}')

        if v == 0.5:
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
        states = lt.calc_uniform_polar_states(nxy, nh, d, a_min, a_max, p_min, p_max)
        print(f'states: {states}')

        for state in states:
            kmpc._update_reference(
                [state[0]] * kmpc.N, [state[1]] * kmpc.N, [state[2]] * kmpc.N, [v] * kmpc.N)
            sol_dict = kmpc.solve()

            plt.plot(sol_dict['z_mpc'][:,0], sol_dict['z_mpc'][:,1], 'b--')
            # utils.plot_arrow(sol_dict['z_mpc'][z,0], sol_dict['z_mpc'][z,1], sol_dict['z_mpc'][z,2] + sol_dict['u_mpc'][z,1] , length=3, width=0.5)
            utils.plot_arrow(sol_dict['z_mpc'][-1,0], sol_dict['z_mpc'][-1,1], sol_dict['z_mpc'][-1,2], length=0.1, width=0.1)
            # plt.scatter(sol_dict['z_ref'][-1,0], sol_dict['z_ref'][-1,1], marker="x", color=color_dict[0], s=200)
            print('\n')
        plt.show()

if __name__ == "__main__":
    main()