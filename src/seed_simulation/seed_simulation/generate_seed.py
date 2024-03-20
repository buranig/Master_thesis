import numpy as np
import matplotlib.pyplot as plt
import random
import os
from cvxopt import matrix, solvers
from cvxopt import matrix
from planner import utils as utils
# For the parameter file
import pathlib
import json

# TODO: import all this parameters from a config file so that we can easily change them in one place
path = pathlib.Path('/home/giacomo/thesis_ws/src/bumper_cars/params.json')
# Opening JSON file
with open(path, 'r') as openfile:
    # Reading from json file
    json_object = json.load(openfile)

# robot_num = json_object["robot_num"]
safety_init = json_object["safety"]
width_init = json_object["width"]
height_init = json_object["height"]
min_dist = json_object["min_dist"]
color_dict = {0: 'r', 1: 'b', 2: 'g', 3: 'y', 4: 'm', 5: 'c', 6: 'k', 7: 'tab:orange', 8: 'tab:brown', 9: 'tab:gray', 10: 'tab:olive', 11: 'tab:pink', 12: 'tab:purple', 13: 'tab:red', 14: 'tab:blue', 15: 'tab:green'}

# write a main function that generates a path for robot_num robots using the function create_path and saves the the generated trajectories to a dictionary.
# The dictionary is then saved to a file using the function save_dict_to_file
def save_dict_to_file(dict, filename='src/seeds/seed_'):
    # save the dictionary to a file
    i = 0
    while os.path.exists(f"{filename}{i}.json"):
        i += 1
    print(f"Saving seed to {filename}{i}.json")
    with open(f'{filename}{i}.json', 'w') as fp:
        json.dump(dict, fp, indent=3)

def random_seed(robot_num):
    # generate a path for robot_num robots
    # save the generated trajectories to a dictionary
    # save the dictionary to a file
    # create a dictionary to save the generated trajectories
    initial_position = {}
    x0, y, yaw, v, omega, model_type = utils.samplegrid(width_init, height_init, min_dist, robot_num, safety_init)
    
    initial_position['x'] = x0
    initial_position['y'] = y
    initial_position['yaw'] = yaw
    initial_position['v'] = v

    trajectories = {}
    for i in range(robot_num):
        # generate a path for each robot
        trajectories[i] = utils.create_seed(len_path=10)
    # save the dictionary to a file
        
    seed = {}
    seed['initial_position'] = initial_position
    seed['trajectories'] = trajectories
    seed['robot_num'] = robot_num
    save_dict_to_file(seed, filename='src/seeds/seed_')

def circular_seed(robot_num, R=10.0):
    # generate a path for robot_num robots
    # save the generated trajectories to a dictionary
    # save the dictionary to a file
    # create a dictionary to save the generated trajectories
    initial_position = {}
    x0, y, yaw, v, omega, model_type = utils.circular_samples(width_init, height_init, R, robot_num, safety_init)
    
    # plt.plot(x0, y, 'ro')
    for i in range(len(x0)):
        utils.plot_arrow(x0[i], y[i], yaw[i], length=2.5, width=1.0, color=color_dict[i])
        utils.plot_map(width_init, height_init)

    initial_position['x'] = x0
    initial_position['y'] = y
    initial_position['yaw'] = yaw
    initial_position['v'] = v

    trajectories = {}
    possibles_indexes = np.arange(0, len(x0)).tolist()
    for i in range(robot_num):
        path = []
        # generate a path for each robot
        idx = random.choice([ele for ele in possibles_indexes if ele != i])
        path.append([x0[idx], y[idx]])
        plt.plot(x0[idx], y[idx], 'x', color=color_dict[i])
        trajectories[i] = path
        possibles_indexes.remove(idx)
    # save the dictionary to a file
    plt.xlabel("x [m]", fontdict={'size': 15, 'family': 'serif'})
    plt.ylabel("y [m]", fontdict={'size': 15, 'family': 'serif'})
    plt.title('Seed Simulation Setup', fontdict={'size':20, 'family': 'serif'})
    plt.show() 
    seed = {}
    seed['initial_position'] = initial_position
    seed['trajectories'] = trajectories
    seed['robot_num'] = robot_num
    save_dict_to_file(seed, filename='src/seeds/circular_seed_')

if __name__ == "__main__":

    circular_seed(R=11.5, robot_num=14)
    # random_seed(robot_num=9)
    # random_seed()