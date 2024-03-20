import itertools
import numpy as np
from scipy.special import comb
import time
from matplotlib.animation import FuncAnimation
from custom_message.msg import ControlInputs

from cvxopt import matrix, solvers
from cvxopt.blas import dot
from cvxopt.solvers import qp, options
from cvxopt import matrix, sparse
from planner import utils as utils
from planner.predict_traj import predict_trajectory

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
max_steer = json_object["CBF_simple"]["max_steer"]  # [rad] max steering angle
max_speed = json_object["CBF_simple"]["max_speed"] # [m/s]
min_speed = json_object["CBF_simple"]["min_speed"] # [m/s]
magnitude_limit= json_object["CBF_simple"]["max_speed"] 
max_acc = 40 #json_object["CBF_simple"]["max_acc"] 
min_acc = -40 #json_object["CBF_simple"]["min_acc"] 
dt = json_object["CBF_simple"]["dt"]
safety_radius = 3#json_object["CBF_simple"]["safety_radius"]
barrier_gain = 1# json_object["CBF_simple"]["barrier_gain"]
Kv = json_object["CBF_simple"]["Kv"] # interval [0.5-1]
Lr = L / 2.0  # [m]
Lf = L - Lr
boundary_points = np.array([-50, 50, -50, 50])

# define x initially --> state: [x, y, yaw, v]
x = np.array([[0, 20], [0, 0], [0, np.pi], [0, 0]])
x1 = x[:,0]
x2 = x[:, 1]
goal1 = np.array([20, 0])
goal2 = np.array([0, 0])
cmd1 = ControlInputs()
cmd2 = ControlInputs()
trajectory, tx, ty = predict_trajectory(utils.array_to_state(x[:,0]), goal1)
trajectory2, tx2, ty2 = predict_trajectory(utils.array_to_state(x[:,1]), goal2)
# Instantiate Robotarium object
N = 2
debug_time = time.time()

def C3BF(x, u_ref):
    N = x.shape[1]
    M = u_ref.shape[0]
    dxu = np.zeros([u_ref.shape[0], u_ref.shape[1]])
    count_dxu = 0

    u_ref[1,:] = delta_to_beta_array(u_ref[1,:])

    for i in range(N):
        count = 0
        G = np.zeros([N-1,M])
        H = np.zeros([N-1,1])

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

            if j == i: continue

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

        # # Adding arena boundary constraints
        # # Pos Y
        # h = 0.01*(boundary_points[3] - safety_radius - x[1,i])**3
        # gradH = np.array([0,-1, -x[3,i]*np.cos(x[2,i]), -np.sin(x[2,i])])
        # Lf_h = np.dot(gradH.T, f)
        # Lg_h = np.dot(gradH.T, g)
        # G = np.vstack([G, -Lg_h])
        # H = np.vstack([H, np.array([h + Lf_h])])

        # # Neg Y
        # h = 0.01*(-boundary_points[2] - safety_radius + x[1,i])**3
        # gradH = np.array([0,1, x[3,i]*np.cos(x[2,i]), np.sin(x[2,i])])
        # Lf_h = np.dot(gradH.T, f)
        # Lg_h = np.dot(gradH.T, g)
        # G = np.vstack([G, -Lg_h])
        # H = np.vstack([H, np.array([h + Lf_h])])

        # # Pos X
        # h = 0.01*(boundary_points[1] - safety_radius - x[0,i])**3
        # gradH = np.array([-1,0, x[3,i]*np.sin(x[2,i]), -np.cos(x[2,i])])
        # Lf_h = np.dot(gradH.T, f)
        # Lg_h = np.dot(gradH.T, g)
        # G = np.vstack([G, -Lg_h])
        # H = np.vstack([H, np.array([h + Lf_h])])

        # # Neg X
        # h = 0.01*(-boundary_points[0] - safety_radius + x[0,i])**3
        # gradH = np.array([1,0, -x[3,i]*np.sin(x[2,i]), np.cos(x[2,i])])
        # Lf_h = np.dot(gradH.T, f)
        # Lg_h = np.dot(gradH.T, g)
        # G = np.vstack([G, -Lg_h])
        # H = np.vstack([H, np.array([h + Lf_h])])

        # # Adding arena boundary constraints
        # # Pos Y
        # h = ((x[1,i] - boundary_points[3])**2 - 6**2)
        # gradH = np.array([0, 2*(x[1,i] - boundary_points[3]), 0, -Kv])
        # Lf_h = np.dot(gradH.T, f)
        # Lg_h = np.dot(gradH.T, g)
        # G = np.vstack([G, -Lg_h])
        # H = np.vstack([H, np.array([0.1*h**3 + Lf_h])])

        # # Neg Y
        # h = ((x[1,i] - boundary_points[2])**2 - 6**2)
        # gradH = np.array([0, 2*(x[1,i] - boundary_points[2]), 0, -Kv])
        # Lf_h = np.dot(gradH.T, f)
        # Lg_h = np.dot(gradH.T, g)
        # G = np.vstack([G, -Lg_h])
        # H = np.vstack([H, np.array([0.1*h**3 + Lf_h])])

        # # Pos X
        # h = ((x[0,i] - boundary_points[1])**2 - 6**2)
        # gradH = np.array([2*(x[0,i] - boundary_points[1]), 0, 0, -Kv])
        # Lf_h = np.dot(gradH.T, f)
        # Lg_h = np.dot(gradH.T, g)
        # G = np.vstack([G, -Lg_h])
        # H = np.vstack([H, np.array([0.1*h**3 + Lf_h])])

        # # Neg X
        # h = ((x[0,i] - boundary_points[0])**2 - 6**2)
        # gradH = np.array([2*(x[0,i] - boundary_points[0]), 0, 0, -Kv])
        # Lf_h = np.dot(gradH.T, f)
        # Lg_h = np.dot(gradH.T, g)
        # G = np.vstack([G, -Lg_h])
        # H = np.vstack([H, np.array([0.1*h**3 + Lf_h])])
        
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

def update_state(x):
        # Create single-integrator control inputs
        x1 = x[:,0]
        x2 = x[:, 1]
        x1 = utils.array_to_state(x1)
        x2 = utils.array_to_state(x2)
        
        dxu = np.zeros((2,N))
        dxu[0,0], dxu[1,0] = utils.pure_pursuit_steer_control(goal1, x1)
        dxu[0,1], dxu[1,1] = utils.pure_pursuit_steer_control(goal2, x2)

        # Create safe control inputs (i.e., no collisions)
        # print(dxu)
        dxu = C3BF(x, dxu)
        # print(dxu)
        # print('\n')

        cmd1.throttle, cmd1.delta = dxu[0,0], dxu[1,0]
        cmd2.throttle, cmd2.delta = dxu[0,1], dxu[1,1]

        # Applying command and current state to the model
        x1 = utils.linear_model_callback(x1, cmd1)
        x2 = utils.linear_model_callback(x2, cmd2)

        return x1, x2

def main(args=None):
    # Initialize the figure and axis
    fig, ax = plt.subplots()

    def update(frame):
        global x, debug_time
        x1, x2 = update_state(x)

        ax.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        plt.plot(x1.x, x1.y, 'xk')
        plt.plot(x2.x, x2.y, 'xb')
        utils.plot_arrow(x1.x, x1.y, x1.yaw)
        utils.plot_arrow(x1.x, x1.y, x1.yaw + cmd1.delta)
        plot_rect(x1.x, x1.y, x1.yaw, safety_radius)

        utils.plot_arrow(x2.x, x2.y, x2.yaw)
        utils.plot_arrow(x2.x, x2.y, x2.yaw + cmd2.delta)
        plot_rect(x2.x, x2.y, x2.yaw, safety_radius)

        utils.plot_path(trajectory)
        utils.plot_path(trajectory2)
        ax.plot(goal1[0], goal1[1], '.k')
        ax.plot(goal2[0], goal2[1], '.b')

        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_aspect('equal')
        ax.grid(True)

        x1 = utils.state_to_array(x1)
        x2 = utils.state_to_array(x2)
        x = np.concatenate((x1, x2), axis=1)

        print(time.time()-debug_time)
        debug_time = time.time()

    # Create the animation with a faster frame rate (50 milliseconds per frame)
    animation = FuncAnimation(fig, update, frames=range(1000), repeat=False, interval=30)
    # Show the animation
    plt.show()


    
if __name__=='__main__':
    main()
        
