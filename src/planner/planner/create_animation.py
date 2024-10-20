import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import pickle
import planner.utils as utils
from shapely.geometry import Point, Polygon, LineString
from shapely.plotting import plot_polygon
import pathlib
import json
import os
import matplotlib
matplotlib.use("Agg") # Uncomment to save but not visualize

dir_path = os.path.dirname(os.path.realpath(__file__))
path = pathlib.Path(dir_path + '/../../bumper_cars/params.json')

# Opening JSON file
with open(path, 'r') as openfile:
    # Reading from json file
    json_object = json.load(openfile)

# robot_num = json_object["robot_num"]
width_init = json_object["width"]
height_init = json_object["height"]
show_animation = json_object["show_animation"]
add_noise = json_object["add_noise"]
noise_scale_param = json_object["noise_scale_param"]
linear_model = json_object["linear_model"]
go_to_goal_bool = True
iterations = 700

color_dict = {0: 'r', 1: 'b', 2: 'g', 3: 'y', 4: 'm', 5: 'c', 6: 'k', 7: 'tab:orange', 8: 'tab:brown', 9: 'tab:gray', 10: 'tab:olive', 11: 'tab:pink', 12: 'tab:purple', 13: 'tab:red', 14: 'tab:blue', 15: 'tab:green'}

# method = "CBF-LBP"
method = 'MPC'

if method == "LBP":
    # fpath = '/home/giacomo/thesis_ws/src/lbp_dev/lbp_dev/LBP_trajectories.pkl'
    fpath = pathlib.Path(dir_path + '/../../lbp_dev/lbp_dev/LBP_trajectories_harem.pkl')

elif method == "DWA":
    fpath = pathlib.Path(dir_path + '/../../dwa_dev/dwa_dev/DWA_trajectories_harem.pkl')
    # fpath = '/home/giacomo/thesis_ws/src/dwa_dev/dwa_dev/DWA_trajectories.pkl'

elif method == "MPC":
    fpath = pathlib.Path(dir_path + '/../../mpc_dev/mpc_dev/MPC_trajectories_harem.pkl')
    # fpath = '/home/giacomo/thesis_ws/src/mpc_dev/mpc_dev/MPC_trajectories.pkl'

elif method == 'CBF':
    fpath = pathlib.Path(dir_path + '/../../cbf_dev/cbf_dev/CBF_trajectories.pkl')

elif method == 'CBF-LBP':
    fpath = pathlib.Path(dir_path + '/../../cbf_dev/cbf_dev/CBF_LBP_trajectories.pkl')

elif method == 'CBF-MPC':
    fpath = pathlib.Path(dir_path + '/../../mpc_dev/mpc_dev/CBF_MPC_trajectories.pkl')

f = open(fpath, 'rb')
obj = pickle.load(f)
f.close()

trajectory = obj[0]
targets = obj[1]

if method == "LBP" or method == "DWA" or method == "CBF-MPC" or method == "CBF-LBP" or method=="MPC":
    
    if method == "LBP": 
        # fpath = '/home/giacomo/thesis_ws/src/lbp_dev/lbp_dev/LBP_dilated_traj.pkl'
        fpath = pathlib.Path(dir_path + '/../../lbp_dev/lbp_dev/LBP_dilated_traj_harem.pkl')
    
    if method == "DWA":
        fpath = pathlib.Path(dir_path + '/../../dwa_dev/dwa_dev/DWA_dilated_traj_harem.pkl')

    if method == "MPC":
        fpath = pathlib.Path(dir_path + '/../../mpc_dev/mpc_dev/MPC_dilated_traj_harem.pkl')

    if method == "CBF-LBP":
        fpath = pathlib.Path(dir_path + '/../../cbf_dev/cbf_dev/CBF_LBP_dilated_traj.pkl')

    if method == "CBF-MPC":
        fpath = pathlib.Path(dir_path + '/../../mpc_dev/mpc_dev/CBF_MPC_dilated_traj.pkl')

    f = open(fpath, 'rb')
    obj = pickle.load(f)
    f.close()
    dilated_traj = obj[0]

robot_num = len(trajectory[0, :, 0])

fig, ax = plt.subplots(figsize=(10,10), dpi=90)

def update(frame):
    z = frame
    ax.cla()
    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])
    for i in range(robot_num):
        utils.plot_robot(trajectory[0, i, z], trajectory[1, i, z], trajectory[2, i, z], i)
        utils.plot_arrow(trajectory[0, i, z], trajectory[1, i, z], trajectory[2, i, z], length=1, width=0.5)
        if linear_model:
            utils.plot_arrow(trajectory[0, i, z], trajectory[1, i, z], trajectory[2, i, z] + trajectory[5, i, z], length=3, width=0.5)
        else:
            utils.plot_arrow(trajectory[0, i, z], trajectory[1, i, z], trajectory[2, i, z] + trajectory[6, i, z], length=3, width=0.5)


        # plt.scatter(targets[i][0], targets[i][1], marker="x", color=color_dict[i], s=200)
        plt.scatter(targets[z][i][0], targets[z][i][1], marker="x", color=color_dict[i], s=200)

        if method == "LBP" or method == "DWA" or method == 'CBF-MPC' or method == "CBF-LBP" or method=="MPC":
            try:
                predicted_trajectory1 = dilated_traj[z][i]
                predicted_trajectory1 = LineString(zip(predicted_trajectory1[:, 0], predicted_trajectory1[:, 1])).buffer(0.35, cap_style=3)
                plot_polygon(Polygon(predicted_trajectory1), ax=ax, add_points=False, alpha=0.5, color=color_dict[i])
            except: 
                print(f'Robot {i}, finished')
    utils.plot_map(width=width_init, height=height_init)
    plt.xlabel("x [m]", fontdict={'size': 17, 'family': 'serif'})
    plt.ylabel("y [m]", fontdict={'size': 17, 'family': 'serif'})
    plt.title(method + ' simulation', fontdict={'size': 25, 'family': 'serif'})
    # plt.axis("equal")
    plt.xlim(-17,17)
    plt.ylim(-17,17)
    plt.grid(True)


# Create the animation with a faster frame rate (50 milliseconds per frame)
if method == "LBP" or method == "DWA":
    frame_num = min(len(dilated_traj), len(trajectory[0, 0, :]))
else:
    frame_num = len(trajectory[0, 0, :])
anim = FuncAnimation(fig, update, frames=frame_num-1, repeat=False, interval=30)

plt.show()

writergif = animation.PillowWriter(fps=10) 
# writergif.setup(fig, "2D_Schrodinger_Equation.gif", dpi = 300) 
# anim.save('/home/giacomo/Video/Video/' + method + '.gif', writer=writergif, dpi="figure")
anim.save('/home/la-user/Videos/gif/' + method + '_harem.gif', writer=writergif, dpi="figure")
# /home/la-user/Videos/gif