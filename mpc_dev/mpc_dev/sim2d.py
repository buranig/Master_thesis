import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.optimize import minimize
import time
import sys
sys.path.append('/home/giacomo/thesis_ws/src/bumper_cars/src')
print(sys.path)
from planner.utils import *


MAX_STEER = np.deg2rad(45.0)  # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(30.0)  # maximum steering speed [rad/s]
MAX_SPEED = 3.0 # maximum speed [m/s]
MIN_SPEED = -3.0  # minimum speed [m/s]
MAX_ACCEL = 5.0  # maximum accel [m/ss]

def dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def sim_run(options, MPC, initial_state, cx, cy, cyaw, ck):
    start = time.process_time()
    # Simulator Options
    FIG_SIZE = options['FIG_SIZE'] # [Width, Height]
    OBSTACLES = options['OBSTACLES']

    mpc = MPC(obs_x=cx[5:-1:12], obs_y=cy[5:-1:12])

    num_inputs = 2
    u = np.zeros(mpc.horizon*num_inputs)
    bounds = []
    constraints = []

    # Set bounds for inputs bounded optimization.
    for i in range(mpc.horizon):
        bounds += [[-MAX_ACCEL, MAX_ACCEL]]
        bounds += [[-MAX_STEER, MAX_STEER]]
        # print(i, 2*i, 2*i+1)
        # constraints += [{'type': 'ineq', 'fun': lambda u: u[i*2] - MAX_ACCEL},
        #                 {'type': 'ineq', 'fun': lambda u: -u[i*2] - MAX_ACCEL},
        #                 {'type': 'ineq', 'fun': lambda u: u[i*2+1] - MAX_STEER},
        #                 {'type': 'ineq', 'fun': lambda u: -u[i*2+1] - MAX_STEER}]

    ref_1 = mpc.reference1
    ref_2 = mpc.reference2
    # ref = ref_1
    target_ind = 1
    ref = [cx[target_ind], cy[target_ind], cyaw[target_ind]]

    state_i = np.array([initial_state])
    u_i = np.array([[0,0]])
    sim_total = 1000
    predict_info = [state_i]

    # Total Figure
    fig = plt.figure(figsize=(FIG_SIZE[0], FIG_SIZE[1]))
    gs = gridspec.GridSpec(8,8)

    # Elevator plot settings.
    ax = fig.add_subplot(gs[:8, :8])

    plt.xlim(-3, 17)
    ax.set_ylim([-3, 17])
    plt.xticks(np.arange(0,11, step=2))
    plt.yticks(np.arange(0,11, step=2))
    plt.title('MPC 2D')

    for i in range(1,sim_total+1):
        u = np.delete(u,0)
        u = np.delete(u,0)
        u = np.append(u, u[-2])
        u = np.append(u, u[-2])
        start_time = time.time()

        # explore possibility of iterative MPC: for z in range(3):
        # Non-linear optimization.
        u_solution = minimize(mpc.cost_function, u, (state_i[-1], ref),
                                method='SLSQP',
                                bounds=bounds,
                                tol = 1e-5)
        # print('Step ' + str(i) + ' of ' + str(sim_total) + '   Time ' + str(round(time.time() - start_time,5)))
        u = u_solution.x
        y = mpc.plant_model(state_i[-1], mpc.dt, u[0], u[1])
        if (target_ind < len(cx)-1):
            if dist([y[0], y[1]], [cx[target_ind], cy[target_ind]]) < 4:
                target_ind+=1
                ref[0] = cx[target_ind]
                ref[1] = cy[target_ind]
                ref[2] = cyaw[target_ind]

        predicted_state = np.array([y])
        for j in range(1, mpc.horizon):
            if u[2*j]>MAX_ACCEL or u[2*j]<-MAX_ACCEL:
                print('Acceleration out of bounds')
                break
            elif u[2*j+1]>MAX_STEER or u[2*j+1]<-MAX_STEER:
                print('Steering out of bounds')
                break
            predicted = mpc.plant_model(predicted_state[-1], mpc.dt, u[2*j], u[2*j+1])
            predicted_state = np.append(predicted_state, np.array([predicted]), axis=0)
        predict_info += [predicted_state]
        state_i = np.append(state_i, np.array([y]), axis=0)
        # print(f'yaw angle: {y[2]}')
        # print(f'speed: {y[3]}')
        # print(f'ref yaw angle: {ref[2]}')
        u_i = np.append(u_i, np.array([(u[0], u[1])]), axis=0)

        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
        if OBSTACLES:
            for zz in range(len(mpc.x_obs)):
                patch_obs = mpatches.Circle((mpc.x_obs[zz], mpc.y_obs[zz]),0.5)
                ax.add_patch(patch_obs)
        plot_robot(state_i[i,0], state_i[i,1], state_i[i,2])
        plot_robot(ref[0],ref[1],ref[2])
        plt.plot(cx, cy, "-r", label="course")
        plt.plot(predicted_state[:,0], predicted_state[:,1])
        plt.xlim(-10, 40)
        plt.ylim(-10, 40)
        plt.title('MPC 2D')
        plt.grid(True)
        plt.pause(0.0001)
