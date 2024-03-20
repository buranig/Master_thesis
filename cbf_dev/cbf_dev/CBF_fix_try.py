import itertools
import numpy as np
from scipy.special import comb
import time
from matplotlib.animation import FuncAnimation

from cvxopt import matrix, solvers
from cvxopt.blas import dot
from cvxopt.solvers import qp, options
from cvxopt import matrix, sparse
from planner import utils as utils
from planner.predict_traj import predict_trajectory
import planner.utils as utils

from custom_message.msg import ControlInputs, State, Path, Coordinate, MultiplePaths, MultiState, MultiControl

# For the parameter file
import pathlib
import json
import math
import matplotlib.pyplot as plt

# TODO: import all this parameters from a config file so that we can easily change them in one place
path = pathlib.Path('/home/giacomo/thesis_ws/src/bumper_cars/params.json')
# Opening JSON file
with open(path, 'r') as openfile:
    # Reading from json file
    json_object = json.load(openfile)

L = json_object["Car_model"]["L"]
max_steer = 0.785398 #json_object["C3BF"]["max_steer"]  # [rad] max steering angle
max_speed = 2 #json_object["Car_model"]["max_speed"] # [m/s]
min_speed = 0.05 #json_object["Car_model"]["min_speed"] # [m/s]
max_acc = 100 #json_object["C3BF"]["max_acc"] 
min_acc = -100 #json_object["C3BF"]["min_acc"] 
dt = 0.1 #json_object["C3BF"]["dt"]
safety_radius = 1.5 #json_object["C3BF"]["safety_radius"]
barrier_gain = 1 #json_object["C3BF"]["barrier_gain"]
arena_gain = 0.01 #json_object["C3BF"]["arena_gain"]
Kv = 0.1 #json_object["C3BF"]["Kv"] # interval [0.5-1]
Lr = L / 2.0  # [m]
Lf = L - Lr
WB = json_object["Controller"]["WB"]
robot_num = json_object["robot_num"]
safety_init = json_object["safety"]
width_init = json_object["width"]
height_init = json_object["height"]
min_dist = json_object["min_dist"]
boundary_points = np.array([-width_init/2, width_init/2, -height_init/2, height_init/2])

def motion(x, u, dt):
    """
    motion model
    initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    """
    delta = u[1]
    delta = np.clip(delta, -max_steer, max_steer)
    throttle = u[0]

    x[0] = x[0] + x[3] * math.cos(x[2]) * dt
    x[1] = x[1] + x[3] * math.sin(x[2]) * dt
    x[2] = x[2] + x[3] / L * math.tan(delta) * dt
    x[2] = utils.normalize_angle(x[2])
    x[3] = x[3] + throttle * dt
    x[3] = np.clip(x[3], min_speed, max_speed)

    return x

def C3BF(x, u_ref):
    """
    Computes the control input for the C3BF (Collision Cone Control Barrier Function) algorithm.

    Args:
        x (numpy.ndarray): State vector of shape (4, N), where N is the number of time steps.
        u_ref (numpy.ndarray): Reference control input of shape (2, N).

    Returns:
        numpy.ndarray: Filtered Control input dxu of shape (2, N).

    """
    N = x.shape[1]
    M = u_ref.shape[0]
    dxu = np.zeros([u_ref.shape[0], u_ref.shape[1]])
    count_dxu = 0

    u_ref[1,:] = delta_to_beta_array(u_ref[1,:])

    for i in range(N):
        count = 0
        G = np.zeros([N-1,M])
        H = np.zeros([N-1,1])

        # when the car goes backwards the yaw angle should be flipped --> Why??
        x[2,i] = (1-np.sign(x[3,i]))*(np.pi/2) + x[2,i]

        f = np.array([x[3,i]*np.cos(x[2,i]),
                          x[3,i]*np.sin(x[2,i]), 
                          0, 
                          0]).reshape(4,1)
        g = np.array([[0, -x[3,i]*np.sin(x[2,i])], 
                        [0, x[3,i]*np.cos(x[2,i])], 
                        [0, x[3,i]/Lr],
                        [1, 0]]).reshape(4,2)
        
        P = np.identity(2)*2
        q = np.array([-2 * u_ref[0, i], - 2 * u_ref[1,i]])
        
        for j in range(N):
            arr = np.array([x[0, j] - x[0, i], x[1, j] - x[1,i]])
            dist = np.linalg.norm(arr)
            v = np.array([x[3,i]*np.cos(x[2,i]), x[3,i]*np.sin(x[2,i])])
            scalar_prod = v @ arr

            if j == i or dist > 3 * safety_radius or scalar_prod < 0: 
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

            H[count] = np.array([barrier_gain*np.power(h, 1) + Lf_h])
            G[count,:] = -Lg_h
            count+=1

        # Adding arena boundary constraints TODO check the calculation/propagation of Kv in the lie derivatives
        # Pos Y
        # h = 0.1*(boundary_points[3] - safety_radius - x[1,i] - Kv * x[3,i])**3
        # gradH = np.array([0,-1, 0, -Kv])
        # Lf_h = np.dot(gradH.T, f)
        # Lg_h = np.dot(gradH.T, g)
        # G = np.vstack([G, -Lg_h])
        # H = np.vstack([H, np.array([h + Lf_h])])

        # # Neg Y
        # h = 0.1*(-boundary_points[2] - safety_radius + x[1,i] - Kv * x[3,i])**3
        # gradH = np.array([0,1, x[3,i]*np.cos(x[2,i]), np.sin(x[2,i])])
        # Lf_h = np.dot(gradH.T, f)
        # Lg_h = np.dot(gradH.T, g)
        # G = np.vstack([G, -Lg_h])
        # H = np.vstack([H, np.array([h + Lf_h])])

        # # Pos X
        # h = 0.1*(boundary_points[1] - safety_radius - x[0,i] - Kv * x[3,i])**3
        # gradH = np.array([-1,0, x[3,i]*np.sin(x[2,i]), -np.cos(x[2,i])])
        # Lf_h = np.dot(gradH.T, f)
        # Lg_h = np.dot(gradH.T, g)
        # G = np.vstack([G, -Lg_h])
        # H = np.vstack([H, np.array([h + Lf_h])])

        # # Neg X
        # h = 0.1*(-boundary_points[0] - safety_radius + x[0,i] - Kv * x[3,i])**3
        # gradH = np.array([1,0, -x[3,i]*np.sin(x[2,i]), np.cos(x[2,i])])
        # Lf_h = np.dot(gradH.T, f)
        # Lg_h = np.dot(gradH.T, g)
        # G = np.vstack([G, -Lg_h])
        # H = np.vstack([H, np.array([h + Lf_h])])

        # Adding arena boundary constraints
        # Pos Y
        h = ((x[1,i] - boundary_points[3])**2 - safety_radius**2 - Kv * abs(x[3,i]))
        if x[3,i] >= 0:
            gradH = np.array([0, 2*(x[1,i] - boundary_points[3]), 0, -Kv])
        else:
            gradH = np.array([0, 2*(x[1,i] - boundary_points[3]), 0, Kv])
        Lf_h = np.dot(gradH.T, f)
        Lg_h = np.dot(gradH.T, g)
        G = np.vstack([G, -Lg_h])
        H = np.vstack([H, np.array([arena_gain*h**3 + Lf_h])])
        
        # Neg Y
        h = ((x[1,i] - boundary_points[2])**2 - safety_radius**2 - Kv * abs(x[3,i]))
        if x[3,i] >= 0:
            gradH = np.array([0, 2*(x[1,i] - boundary_points[2]), 0, -Kv])
        else:
            gradH = np.array([0, 2*(x[1,i] - boundary_points[2]), 0, Kv])
        Lf_h = np.dot(gradH.T, f)
        Lg_h = np.dot(gradH.T, g)
        G = np.vstack([G, -Lg_h])
        H = np.vstack([H, np.array([arena_gain*h**3 + Lf_h])])
        
        # Pos X
        h = ((x[0,i] - boundary_points[1])**2 - safety_radius**2 - Kv * abs(x[3,i]))
        if x[3,i] >= 0:
            gradH = np.array([2*(x[0,i] - boundary_points[1]), 0, 0, -Kv])
        else:
            gradH = np.array([2*(x[0,i] - boundary_points[1]), 0, 0, Kv])
        Lf_h = np.dot(gradH.T, f)
        Lg_h = np.dot(gradH.T, g)
        G = np.vstack([G, -Lg_h])
        H = np.vstack([H, np.array([arena_gain*h**3 + Lf_h])])

        # Neg X
        h = ((x[0,i] - boundary_points[0])**2 - safety_radius**2 - Kv * abs(x[3,i]))
        if x[3,i] >= 0:
            gradH = np.array([2*(x[0,i] - boundary_points[0]), 0, 0, -Kv])
        else:
            gradH = np.array([2*(x[0,i] - boundary_points[0]), 0, 0, Kv])
        Lf_h = np.dot(gradH.T, f)
        Lg_h = np.dot(gradH.T, g)
        G = np.vstack([G, -Lg_h])
        H = np.vstack([H, np.array([arena_gain*h**3 + Lf_h])])
        
        # Input constraints
        G = np.vstack([G, [[0, 1], [0, -1]]])
        H = np.vstack([H, delta_to_beta(max_steer), -delta_to_beta(-max_steer)])
        # G = np.vstack([G, [[1, 0], [-1, 0]]])
        # H = np.vstack([H, max_acc, -min_acc])

        solvers.options['show_progress'] = False
        sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(H))
        dxu[:,count_dxu] = np.reshape(np.array(sol['x']), (M,))
        count_dxu += 1
    
    dxu[1,:] = beta_to_delta(dxu[1,:])    
    return dxu
            
def delta_to_beta(delta):
    beta = utils.normalize_angle(np.arctan2(Lr*np.tan(delta)/L, 1.0))

    return beta

def delta_to_beta_array(delta):
    beta = utils.normalize_angle_array(np.arctan2(Lr*np.tan(delta)/L, 1.0))

    return beta

def beta_to_delta(beta):
    delta = utils.normalize_angle_array(np.arctan2(L*np.tan(beta)/Lr, 1.0))

    return delta           

def plot_rect(x, y, yaw, r):  # pragma: no cover
        outline = np.array([[-r / 2, r / 2,
                                (r / 2), -r / 2,
                                -r / 2],
                            [r / 2, r/ 2,
                                - r / 2, -r / 2,
                                r / 2]])
        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                            [-math.sin(yaw), math.cos(yaw)]])
        outline = (outline.T.dot(Rot1)).T
        outline[0, :] += x
        outline[1, :] += y
        plt.plot(np.array(outline[0, :]).flatten(),
                    np.array(outline[1, :]).flatten(), "-k")

def plot_robot(x, y, yaw):  # pragma: no cover
    outline = np.array([[-L / 2, L / 2,
                            (L / 2), -L / 2,
                            -L / 2],
                        [WB / 2, WB / 2,
                            - WB / 2, -WB / 2,
                            WB / 2]])
    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                        [-math.sin(yaw), math.cos(yaw)]])
    outline = (outline.T.dot(Rot1)).T
    outline[0, :] += x
    outline[1, :] += y
    plt.plot(np.array(outline[0, :]).flatten(),
                np.array(outline[1, :]).flatten(), "-k")
   

def main(args=None):
    # Instantiate Robotarium object
    # The robots will never reach their goal points so set iteration number
    iterations = 3000
    # Define goal points outside of the arena
    x0, y, yaw, v, omega, model_type = utils.samplegrid(width_init, height_init, min_dist, robot_num, safety_init)
    x = np.array([x0, y, yaw, v])

    paths = []
    targets = []
    # multi_traj = MultiplePaths()
    multi_control = MultiControl()
    dxu = np.zeros((2,robot_num))
    for i in range(robot_num):
        paths.append(utils.create_path())
        targets.append([paths[i][0].x, paths[i][0].y])
        initial_state = State(x=x0[i], y=y[i], yaw=yaw[i], v=v[i], omega=omega[i])
        # multi_traj.multiple_path.append(predict_trajectory(initial_state, targets[i]))
        multi_control.multi_control.append(ControlInputs(delta=0.0, throttle=0.0))
        
    # While the number of robots at the required poses is less
    # than N...
    for z in range(iterations):
        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        
        # Create single-integrator control inputs
        for i in range(robot_num):
            x1 = utils.array_to_state(x[:,i])
            if utils.dist(point1=(x[0,i], x[1,i]), point2=targets[i]) < 5:
                paths[i] = utils.update_path(paths[i])
                targets[i] = (paths[i][0].x, paths[i][0].y)
                # multi_traj.multiple_path[i] = predict_trajectory(x1, targets[i])

            cmd = ControlInputs()
            cmd.throttle, cmd.delta= utils.pure_pursuit_steer_control(targets[i], x1)

            dxu[0,i], dxu[1,i] = cmd.throttle, cmd.delta            
            dxu = C3BF(x, dxu)
            cmd.throttle, cmd.delta = dxu[0,i], dxu[1,i]
            x1 = utils.linear_model_callback(x1, cmd)
            x1 = utils.state_to_array(x1).reshape(4)
            x[:, i] = x1
            multi_control.multi_control[i] = cmd
    
            # plt.plot(x[0,i], x[1,i], "xr")
            # plt.plot(goal[0,i], goal[1,i], "xb")utils.plot_arrow(x1.x, x1.y, x1.yaw)
            plot_robot(x[0,i], x[1,i], x[2,i])
            # plot_rect(x[0,i], x[1,i], x[2,i], safety_radius)
            utils.plot_arrow(x[0,i], x[1,i], x[2,i] + multi_control.multi_control[i].delta, length=2, width=1)
            utils.plot_arrow(x[0,i], x[1,i], x[2,i], length=2, width=1)
            plt.plot(targets[i][0], targets[i][1], "xg")
            # plot_path(multi_traj.multiple_path[i])

        utils.plot_map(width=width_init, height=height_init)
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)
    
if __name__=='__main__':
    main()
        