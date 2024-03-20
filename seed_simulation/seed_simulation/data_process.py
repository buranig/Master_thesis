# class to process the data from the simulation

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import json
import pathlib

# TODO: import all this parameters from a config file so that we can easily change them in one place
path = pathlib.Path('/home/giacomo/thesis_ws/src/bumper_cars/params.json')
# Opening JSON file
with open(path, 'r') as openfile:
    # Reading from json file
    json_object = json.load(openfile)

L = json_object["Car_model"]["L"] # [m] Wheel base of vehicle
WB = json_object["Controller"]["WB"] # Wheel base
safety_init = json_object["safety"]
width_init = json_object["width"]
height_init = json_object["height"]
min_dist = json_object["min_dist"]
to_goal_stop_distance = json_object["to_goal_stop_distance"]
add_noise = json_object["add_noise"]
noise_scale_param = json_object["noise_scale_param"]

class DataProcessor:
    def __init__(self, robot_num, file_name, seed):
        self.robot_num = robot_num
        self.file_name = file_name
        self.seed = seed
    
    def calculate_goal_reached_index(self, trajectory, i):
        last_idx = 0
        for idx in range(1, len(trajectory[4,i,:])):
            idx_prev = idx-1
            if trajectory[4, i, idx] == 0.0 and trajectory[5, i, idx] == 0.0 and idx > last_idx:
                if trajectory[4, i, idx_prev] != 0.0 or trajectory[5, i, idx_prev] != 0.0:
                    last_idx = idx
        
        return last_idx
    
    def get_initial_length(self, trajectory, i):
        initial_state = self.seed['initial_position']
        x0 = initial_state['x']
        y = initial_state['y']
        yaw = initial_state['yaw']
        v = initial_state['v']

        # Step 3: Create an array x with the initial values
        x = np.array([x0, y, yaw, v])

        traj = self.seed['trajectories']
        paths = [[(traj[str(idx)][i][0], traj[str(idx)][i][1]) for i in range(len(traj[str(idx)]))] for idx in range(self.robot_num)]

        # Step 5: Extract the target coordinates from the paths
        targets = [[path[0][0], path[0][1]] for path in paths]

        # Step 6: Calculate the distance between the initial position and the target position
        initial_length = np.sqrt((x[0, i] - targets[i][0]) ** 2 + (x[1,i] - targets[i][1]) ** 2)

        return initial_length - to_goal_stop_distance

    def calculate_path_length(self, trajectory, i):
        """
        Calculate the percentage increase of the path length of the trajectory or robot i with respect to
        initial distance to the goal.
        :param trajectory: the trajectory
        :return: the path length
        """
        initial_length = self.get_initial_length(trajectory, i)
        x_list = trajectory[0, i, :]
        y_list = trajectory[1, i, :]

        path_length = 0.0

        initial_x = x_list[0]
        initial_y = y_list[0]

        assert len(x_list) == len(y_list)

        for idx in range(len(x_list)):

            x = x_list[idx]
            y = y_list[idx]

            dx = x - initial_x
            dy = y - initial_y

            initial_x = x
            initial_y = y

            path_length += np.sqrt(dx ** 2 + dy ** 2)
        
        return (path_length/initial_length-1)*100
    
    def calculate_avg_path_length(self, trajectory):
        """
        Calculate the average percentage increase of the path length of the trajectory or robot i with respect to
        initial distance to the goal.
        :param trajectory: the trajectory
        :return: the average path length
        """
        print("Calculating Path Length...")
        path_length = 0
        for i in range(self.robot_num):
            path_length += self.calculate_path_length(trajectory, i)
        return path_length / self.robot_num
    
    def calculate_acceleration_usage(self, trajectory, i):
        """
        Calculate the acceleration usage of the trajectory or robot i
        :param trajectory: the trajectory
        :return: the acceleration usage
        """
        idx = self.calculate_goal_reached_index(trajectory, i)
        a = trajectory[4, i, :idx+1]
        return np.mean(np.abs(a))
    
    def calculate_avg_acceleration_usage(self, trajectory):
        """
        Calculate the average acceleration usage of the trajectory over all the robots
        :param trajectory: the trajectory
        :return: the average acceleration usage
        """
        print("Calculating Acceleration Usage...")
        acceleration_usage = 0
        for i in range(self.robot_num):
            acceleration_usage += self.calculate_acceleration_usage(trajectory, i)
        return acceleration_usage / self.robot_num
    
    def calculate_steering_usage(self, trajectory, i):
        """
        Calculate the steering usage of the trajectory or robot i
        :param trajectory: the trajectory
        :return: the steering usage
        """
        idx = self.calculate_goal_reached_index(trajectory, i)
        steering = trajectory[5, i, :idx+1]
        return np.mean(np.abs(steering))
    
    def calculate_avg_steering_usage(self, trajectory):
        """
        Calculate the average steering usage of the trajectory over all the robots
        :param trajectory: the trajectory
        :return: the average steering usage
        """
        print("Calculating Steering Usage...")
        steering_usage = 0
        for i in range(self.robot_num):
            steering_usage += self.calculate_steering_usage(trajectory, i)
        return steering_usage / self.robot_num
    
    def calculate_speed_avg(self, trajectory, i):
        """
        Calculate the average speed of the trajectory or robot i
        :param trajectory: the trajectory
        :return: the average speed
        """
        idx = self.calculate_goal_reached_index(trajectory, i)
        v = trajectory[3, i, :idx+1]
        return np.mean(v)
    
    def calculate_avg_speed(self, trajectory):
        """
        Calculate the average speed of the trajectory over all the robots
        :param trajectory: the trajectory
        :return: the average speed
        """
        print("Calculating Average Speed...")
        speed = 0
        for i in range(self.robot_num):
            speed += self.calculate_speed_avg(trajectory, i)
        return speed / self.robot_num
    
    def calculate_avg_computational_time(self, computational_time):
        """
        Calculate the computational time of the trajectory
        :param computational_time: the computational time
        :return: the computational time
        """
        print("Calculating Average Computational Time...")
        return sum(computational_time) / len(computational_time)
    
    def calculate_initial_dist(self, trajectory, i):
        """
        Calculate the initial distance of the trajectory or robot i
        :param trajectory: the trajectory
        :return: the initial distance
        """
        x = trajectory[0, i, :]
        y = trajectory[1, i, :]
        return np.sqrt(x[0] ** 2 + y[0] ** 2)
    
    def calculate_avg_initial_dist(self, trajectory):
        """
        Calculate the average initial distance of the trajectory over all the robots
        :param trajectory: the trajectory
        :return: the average initial distance
        """
        print("Calculating Initial Distance...")
        initial_dist = 0
        for i in range(self.robot_num):
            initial_dist += self.calculate_initial_dist(trajectory, i)
        return initial_dist / self.robot_num
    
    def count_collision(self, trajectory):
        """
        Count the number of collisions in the trajectory
        :param trajectory: the trajectory
        :return: the number of collisions
        """
        print("Counting Collisions...")
        collision = 0
        for i in range(self.robot_num):
            x = trajectory[0, i, :]
            y = trajectory[1, i, :]
            for j in range(i + 1, self.robot_num):
                x2 = trajectory[0, j, :]
                y2 = trajectory[1, j, :]
                for k in range(len(x)):
                    if np.sqrt((x[k] - x2[k]) ** 2 + (y[k] - y2[k]) ** 2) <= WB/2:
                        collision += 1
                        break
            
            if (x>=width_init/2-WB/2).any() or (x<=-width_init/2+WB/2).any() or (y>=height_init/2-WB/2).any() or (y<=-height_init/2+WB/2).any():
                collision += 1
        return collision

    def post_process_simultation(self, trajectory, computational_time, method, solver_failure=0):
        """
        Post process the simulation
        :param trajectory: the trajectory
        :param computational_time: the computational time
        :return: the post processed data
        """
        avg_path_length = self.calculate_avg_path_length(trajectory)
        acceleration_usage = self.calculate_avg_acceleration_usage(trajectory)
        steering_usage = self.calculate_avg_steering_usage(trajectory)
        avg_speed = self.calculate_avg_speed(trajectory)
        avg_computational_time = self.calculate_avg_computational_time(computational_time)
        avg_initial_dist = self.calculate_avg_initial_dist(trajectory)
        collision_number = self.count_collision(trajectory)

        if add_noise:
            noise_scaling = noise_scale_param
        else:
            noise_scaling = 0.0

        data = {
            "Path Length": avg_path_length,
            "Acceleration Usage": acceleration_usage,
            "Steering Usage": steering_usage,
            "Average Speed": avg_speed,
            "Avg Computational Time": avg_computational_time,
            "Initial Distance": avg_initial_dist,
            "Solver Failure": solver_failure,
            "Robot Number": self.robot_num,
            "File Name": self.file_name,
            "Method": method,
            "Collision Number": collision_number,
            "Noise Scaling": noise_scaling
        }

        print("Data Processed Successfully!\n")
        return data
    
    def remove_df_duplicates(self, df):
        """
        Remove the duplicates from the dataframe
        :param df: the dataframe
        :return: the dataframe without duplicates
        """
        return df.drop_duplicates(subset=["Robot Number", "File Name", "Method", "Noise Scaling"], keep="last")