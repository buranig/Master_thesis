import math
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits import mplot3d
from custom_message.msg import ControlInputs

import pathlib
import json

from planner import utils as utils

# TODO: import all this parameters from a config file so that we can easily change them in one place
path = pathlib.Path('/home/giacomo/thesis_ws/src/bumper_cars/params.json')
# path = pathlib.Path('/home/giacomo/thesis_ws/src/bumper_cars/params_small.json')

# Opening JSON file
with open(path, 'r') as openfile:
    # Reading from json file
    json_object = json.load(openfile)

max_steer = json_object["CBF_simple"]["max_steer"]   # [rad] max steering angle
max_speed = json_object["Car_model"]["max_speed"] # [m/s]
min_speed = json_object["Car_model"]["min_speed"]  # [m/s]
car_max_acc = json_object["Controller"]["max_acc"]
car_min_acc = json_object["Controller"]["min_acc"]
dt = json_object["Controller"]["dt"] 
safety_init = json_object["safety"]
width_init = json_object["width"]
height_init = json_object["height"]
min_dist = json_object["min_dist"]
L = json_object["Car_model"]["L"] # [m] Wheel base of vehicle
Lr = L / 2.0  # [m]
Lf = L - Lr
Cf = json_object["Car_model"]["Cf"]  # N/rad
Cr = json_object["Car_model"]["Cr"] # N/rad
Iz = json_object["Car_model"]["Iz"]  # kg/m2
m = json_object["Car_model"]["m"]  # kg
# Aerodynamic and friction coefficients
c_a = json_object["Car_model"]["c_a"]
c_r1 = json_object["Car_model"]["c_r1"]
WB = json_object["Controller"]["WB"]
predict_time = json_object["LBP"]["predict_time"] # [s]
show_animation = json_object["show_animation"]

class PotentialFieldController:
    def __init__(self, goal, obstacles):
        self.goal = goal
        self.obstacles = obstacles

        self.k_rep = 100 # Repulsive force gain
        self.k_att = 7 # Attractive force gain
        self.k_arena = 150

        self.n = 2

        self.sigmax = 2
        self.sigmay = 2
        self.sigma_arena = 1.5
        self.definition = 64
        self.width = 15
        self.height = 15
        self.m = m

    def attractive_potential(self, position):
        # Calculate attractive potential towards the goal
        attractive_potential = 0.5 * self.k_att*np.linalg.norm([position[0]- self.goal[0], position[1]- self.goal[1]])**2
        return attractive_potential

    def attractive_force(self, position):
        # Calculate attractive force towards the goal
        attractive_force = [-self.k_att*(position[0]- self.goal[0]), -self.k_att*(position[1]- self.goal[1])]
        return attractive_force
    
    def repulsive_potential(self, position):
        print("position", position)
        print("goal", self.goal)
        # Calculate repulsive potential from obstacles
        repulsive_potential = 0.0
        goal_distance = math.sqrt((self.goal[0] - position[0])**2 + (self.goal[1] - position[1])**2)
        for obstacle in self.obstacles:
            distance = math.sqrt((obstacle[0] - position[0])**2 + (obstacle[1] - position[1])**2)
            if distance < obstacle[2]:
                if distance == 0.0:
                    distance += 0.0001
                repulsive_potential += 0.5*self.k_rep*(1/distance - 1/obstacle[2]) * (goal_distance**self.n/((1 + goal_distance**self.n)))
        return repulsive_potential
    
    def repulsive_gaussian_potential(self, position):
        # Calculate repulsive potential from obstacles
        repulsive_potential = 0.0
        goal_distance = self.euclidean_distance(position, self.goal)

        for obstacle in self.obstacles:
            repulsive_potential += self.k_rep*np.exp(-(position[0] - obstacle[0])**2/(self.sigmax**2) - (position[1] - obstacle[1])**2/(self.sigmay**2))
        return repulsive_potential
    
    def repulsive_gaussian_potential_imp(self, position):
        # Calculate repulsive potential from obstacles
        repulsive_potential = 0.0
        goal_distance = self.euclidean_distance(position, self.goal)

        for obstacle in self.obstacles:
            repulsive_potential += self.k_rep*np.exp(-(position[0] - obstacle[0])**2/(self.sigmax**2) - (position[1] - obstacle[1])**2/(self.sigmay**2))*goal_distance
        return repulsive_potential
    
    def repulsive_gaussian_force(self, position):
        # Calculate repulsive force from obstacles
        repulsive_force = [0.0, 0.0]
        for obstacle in self.obstacles:
            repulsive_force[0] += 2*(position[0] - obstacle[0])/(self.sigmax**2)*self.k_rep*np.exp(-(position[0] - obstacle[0])**2/(self.sigmax**2) - (position[1] - obstacle[1])**2/(self.sigmay**2)) 
            repulsive_force[1] += 2*(position[1] - obstacle[1])/(self.sigmay**2)*self.k_rep*np.exp(-(position[0] - obstacle[0])**2/(self.sigmax**2) - (position[1] - obstacle[1])**2/(self.sigmay**2))
        return repulsive_force

    def repulsive_gaussian_force_imp(self, position):
        # Calculate repulsive force from obstacles
        goal_distance = self.euclidean_distance(position, self.goal)
        if goal_distance == 0.0:
            goal_distance += 0.0001
        repulsive_force = [0.0, 0.0]
        for obstacle in self.obstacles:
            potential = self.k_rep*np.exp(-(position[0] - obstacle[0])**2/(self.sigmax**2) - (position[1] - obstacle[1])**2/(self.sigmay**2))
            repulsive_force[0] += -potential*goal_distance*(-2*(position[0] - obstacle[0])/(self.sigmax**2))-potential*(position[0]-self.goal[0])/goal_distance
            repulsive_force[1] += -potential*goal_distance*(-2*(position[1] - obstacle[1])/(self.sigmay**2))-potential*(position[1]-self.goal[1])/goal_distance
        return repulsive_force
    
    def repulsive_arena_potential(self, position):
        # Calculate repulsive potential from obstacles
        repulsive_potential = 0.0
        repulsive_potential += self.k_arena*(np.exp((position[0]-self.width)/(2*self.sigma_arena**2)))
        repulsive_potential += self.k_arena*(np.exp(-(position[0]+self.width)/(2*self.sigma_arena**2)))
        repulsive_potential += self.k_arena*(np.exp((position[1]-self.height)/(2*self.sigma_arena**2)))
        repulsive_potential += self.k_arena*(np.exp(-(position[1]+self.height)/(2*self.sigma_arena**2)))
        return repulsive_potential

    def repulsive_arena_force(self, position):
        # Calculate repulsive force from obstacles
        repulsive_force = [0.0, 0.0]
        repulsive_force[0] -= self.k_arena*(np.exp((position[0]-self.width)/(2*self.sigma_arena**2)))*(1/(self.sigma_arena**2))
        repulsive_force[1] -= self.k_arena*(np.exp((position[1]-self.height)/(2*self.sigma_arena**2)))*(1/(self.sigma_arena**2))
        repulsive_force[0] -= self.k_arena*(np.exp(-(position[0]+self.width)/(2*self.sigma_arena**2)))*(-1/(self.sigma_arena**2))
        repulsive_force[1] -= self.k_arena*(np.exp(-(position[1]+self.height)/(2*self.sigma_arena**2)))*(-1/(self.sigma_arena**2))
        return repulsive_force
    
    def euclidean_distance(self, position, obstacle):
        return math.sqrt((obstacle[0] - position[0])**2 + (obstacle[1] - position[1])**2)
    
    def grad_eucledian_distance(self, position, obstacle):
        distance = self.euclidean_distance(position, obstacle)
        if distance == 0.0:
            distance += 0.0001
        grad_x = (position[0] - obstacle[0])/distance
        grad_y = (position[1] - obstacle[1])/distance
        return grad_x, grad_y

    def repulsive_force(self, position):
        # Calculate repulsive force from obstacles
        repulsive_force = [0, 0]
        if position[0] == 8.0 and position[1] == 8.0:
            print('Obstacle')
        goal_distance = self.euclidean_distance(position, self.goal)
        grad_goal_x, grad_goal_y = self.grad_eucledian_distance(position, self.goal)

        for obstacle in self.obstacles:
            distance = self.euclidean_distance(position, obstacle)
            grad_obstacle_x, grad_obstacle_y = self.grad_eucledian_distance(position, obstacle)
            if distance < obstacle[2]:
                if distance == 0.0:
                    distance += 0.001
                
                f_rep1_x = self.k_rep*(1/distance - 1/obstacle[2])*grad_obstacle_x * (goal_distance**self.n/((1 + goal_distance**self.n)*distance**2))
                f_rep1_y = self.k_rep*(1/distance - 1/obstacle[2])*grad_obstacle_y * (goal_distance**self.n/((1 + goal_distance**self.n)*distance**2))

                f_rep2_x = self.k_rep*self.n/2*(1/distance - 1/obstacle[2])**2*grad_goal_x * (goal_distance**(self.n-1)/(1 + goal_distance**self.n)**2)
                f_rep2_y = self.k_rep*self.n/2*(1/distance - 1/obstacle[2])**2*grad_goal_y * (goal_distance**(self.n-1)/(1 + goal_distance**self.n)**2)

                repulsive_force[0] += abs(f_rep1_x + f_rep2_x) #TODO: Check if adding the abs is correct
                repulsive_force[1] += abs(f_rep1_y + f_rep2_y)

        return repulsive_force

    def calculate_total_force(self, position):
        # Calculate the total force acting on the robot
        attractive_force = self.attractive_force(position)
        repulsive_force = self.repulsive_force(position)
        total_force = [attractive_force[0] + repulsive_force[0], attractive_force[1] + repulsive_force[1]]
        # total_force = [repulsive_force[0], repulsive_force[1]]
        # total_force = [attractive_force[0], attractive_force[1]]

        return total_force
    
    def calculate_total_potential(self, position):
        # Calculate the total potential acting on the robot
        attractive_potential = self.attractive_potential(position)
        repulsive_potential = self.repulsive_potential(position)
        total_potential = attractive_potential + repulsive_potential
        return total_potential
    
    def calculate_total_gaussian_potential(self, position):
        # # Calculate the total potential acting on the robot
        attractive_potential = self.attractive_potential(position)
        repulsive_potential = self.repulsive_gaussian_potential_imp(position)
        arena_potential = self.repulsive_arena_potential(position)
        total_potential = arena_potential + repulsive_potential + attractive_potential
        return total_potential
    
    def calculate_total_gaussian_force(self, position):
        # # Calculate the total force acting on the robot
        total_force = [0.0, 0.0]
        attractive_force = self.attractive_force(position)
        repulsive_force = self.repulsive_gaussian_force_imp(position)
        arena_force = self.repulsive_arena_force(position)
        total_force[0] = repulsive_force[0] + attractive_force[0] + arena_force[0]
        total_force[1] = repulsive_force[1] + attractive_force[1] + arena_force[1]
        return total_force
    
    def force_to_input(self, position, force, force_pred):
        # Convert the force to robot's acceleration and steering angle
        ax = (force[0]*np.cos(position[2]) + force[1]*np.sin(position[2]))/self.m
        ay = (-force_pred[0]*np.sin(position[2]) + force_pred[1]*np.cos(position[2]))/self.m 
        steering = 3*ay*(L**2*2*Cf*Cr + self.m*position[3]**2*(Lf*Cf-Lr*Cr))/(2*Cf*Cr*L*(position[3]+0.001)**2)

        # ax = np.linalg.norm(force)
        # steering = np.arctan2(force[1], force[0]) - position[2]
        steering = np.clip(steering, -max_steer, max_steer)
        
        # print(f'Yaw angle: {np.rad2deg(position[2])}, Force field angle: {np.rad2deg(np.arctan2(force[1], force[0]))}')
        # print(f'Steering angle: {np.rad2deg(steering)}')

        return ax, steering

    def move_robot(self, position):
        # Plot the potential field
        fig = plt.figure(1, dpi=90, figsize=(10,10))
        ax = fig.add_subplot(111)

        x = np.linspace(-self.width, self.width, self.definition)
        y = np.linspace(-self.height, self.height, self.definition)
        X, Y = np.meshgrid(x, y)
        total_force_angle = np.zeros((x.shape[0], y.shape[0]))
        Z = np.zeros_like(X)
        P = np.zeros_like(X)
        for i in range(len(x)):
            for j in range(len(y)):
                position_ = [X[i, j], Y[i, j]]
                # print(position_)
                # total_force = self.calculate_total_force(position_)
                # total_force = self.calculate_total_potential(position_)
                total_potential = self.calculate_total_gaussian_potential(position_)
                total_force = self.calculate_total_gaussian_force(position_)
                # total_force_buf[i, j] = np.linalg.norm(total_force)
                total_force_angle[i, j] = np.arctan2(total_force[1], total_force[0])

                Z[i, j] = np.linalg.norm(total_force)
                # Z[i, j] = min(Z[i, j], 80)
                Z[i, j] = max(Z[i, j], 0.0)

                P[i, j] = np.linalg.norm(total_potential)

        dy, dx = np.gradient(P)
        dy, dx = -dy, -dx
       
        # Move the robot based on the calculated force
        total_force = self.calculate_total_gaussian_force(position)
        x_pred = [position[0] + dt*np.cos(position[2])*position[3], position[1] + dt*np.sin(position[2])*position[3]]
        total_force_pred = self.calculate_total_gaussian_force(x_pred)
        throttle, steering = self.force_to_input(position, total_force, total_force_pred)

        position_buf = [[position[0], position[1]]]

        while self.euclidean_distance(position, self.goal) > 2:
            plt.cla()
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            
            # Car Model
            # position[0] = position[0] + dt*total_force[0]
            # position[1] = position[1] + dt*total_force[1]

            position = utils.nonlinear_model_numpy_stable(position, [throttle, steering])

            position_buf.append([position[0], position[1]])

            # Plot the robot's position
            plt.contourf(X, Y, P, levels=20, cmap='jet')
            plt.quiver(X, Y, dx, dy, pivot='mid', scale=2000)
            utils.plot_robot(position[0], position[1], position[2], 0)
            utils.plot_arrow(position[0], position[1], position[2], length=1, width=0.5)
            utils.plot_arrow(position[0], position[1], position[2]+steering, length=3, width=0.5)

            for obstacle in self.obstacles:
                circle = plt.Circle(obstacle[0:2], obstacle[2], color='r', fill=False)
                ax.add_patch(circle)
            # plt.plot(position_buf[-1][0], position_buf[-1][1], 'ro', label='Robot')
            # plt.legend()
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Potential Field')
            plt.pause(0.01)

            # Break if the robot reaches the goal
            if self.euclidean_distance(position, self.goal) < 2:
                print('Goal Reached')
                break

            total_force = self.calculate_total_gaussian_force(position)
            # x_pred = utils.nonlinear_model_numpy_stable(position, [0, 0])
            # total_force_pred = self.calculate_total_force(x_pred[0:2])
            throttle, steering = self.force_to_input(position, total_force, total_force)

            print('Throttle:', throttle, 'Steering:', steering)

        for elem in position_buf:
            plt.plot(elem[0], elem[1], 'ro', label='Robot')
        plt.pause(0.0001)
        plt.show()
        # print(position_buf)

    def plot_moving_field(self, position):
        # Plot the potential field
        fig = plt.figure(1, dpi=90, figsize=(10,10))
        ax = fig.add_subplot(111)

        self.obstacles = [[position[0], position[1], 2]]

        x = np.linspace(-self.width, self.width, self.definition)
        y = np.linspace(-self.height, self.height, self.definition)
        X, Y = np.meshgrid(x, y)
        total_force_angle = np.zeros((x.shape[0], y.shape[0]))
        Z = np.zeros_like(X)
        P = np.zeros_like(X)
        for i in range(len(x)):
            for j in range(len(y)):
                position_ = [X[i, j], Y[i, j]]
                # print(position_)
                # total_force = self.calculate_total_force(position_)
                # total_force = self.calculate_total_potential(position_)
                total_potential = self.calculate_total_gaussian_potential(position_)
                total_force = self.calculate_total_gaussian_force(position_)
                # total_force_buf[i, j] = np.linalg.norm(total_force)
                total_force_angle[i, j] = np.arctan2(total_force[1], total_force[0])

                Z[i, j] = np.linalg.norm(total_force)
                # Z[i, j] = min(Z[i, j], 80)
                Z[i, j] = max(Z[i, j], 0.0)

                P[i, j] = np.linalg.norm(total_potential)

        dy, dx = np.gradient(P)
        dy, dx = -dy, -dx

        plt.contourf(X, Y, P, levels=20, cmap='jet')
        plt.quiver(X, Y, dx, dy, pivot='mid', scale=2000)
       
        # Move the robot based on the calculated force
        total_force = self.calculate_total_gaussian_force(position)
        x_pred = [position[0] + dt*np.cos(position[2])*position[3], position[1] + dt*np.sin(position[2])*position[3]]
        total_force_pred = self.calculate_total_gaussian_force(x_pred)
        throttle, steering = self.force_to_input(position, total_force, total_force_pred)

        position_buf = [[position[0], position[1]]]

        while self.euclidean_distance(position, self.goal) > 2:
            plt.cla()
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            
            # Car Model
            x1 = utils.array_to_state(position)
            throttle, steering = utils.pure_pursuit_steer_control(self.goal, x1)

            position = utils.nonlinear_model_numpy_stable(position, [throttle, steering])
            self.obstacles = [[position[0], position[1], 2]]
            position_buf.append([position[0], position[1]])

            # Plot the robot's position
            for i in range(len(x)):
                for j in range(len(y)):
                    position_ = [X[i, j], Y[i, j]]
                    total_potential = self.calculate_total_gaussian_potential(position_)

                    P[i, j] = np.linalg.norm(total_potential)

            plt.contourf(X, Y, P, levels=20, cmap='jet')

            utils.plot_robot(position[0], position[1], position[2], 0)
            utils.plot_arrow(position[0], position[1], position[2], length=1, width=0.5)
            utils.plot_arrow(position[0], position[1], position[2]+steering, length=3, width=0.5)

            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Potential Field')
            plt.pause(0.01)

            # Break if the robot reaches the goal
            if self.euclidean_distance(position, self.goal) < 2:
                print('Goal Reached')
                break

        for elem in position_buf:
            plt.plot(elem[0], elem[1], 'ro', label='Robot')
        plt.pause(0.0001)
        plt.show()
        # print(position_buf)

    def remap(self, x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def plot_field(self):
        # Plot the potential field
        x = np.linspace(-self.width, self.width, self.definition)
        y = np.linspace(-self.height, self.height, self.definition)
        X, Y = np.meshgrid(x, y)
        total_force_angle = np.zeros((x.shape[0], y.shape[0]))
        Z = np.zeros_like(X)
        P = np.zeros_like(X)

        for i in range(len(x)):
            for j in range(len(y)):
                position = [X[i, j], Y[i, j]]
                # print(position)
                # total_force = self.calculate_total_force(position)
                # total_force = self.calculate_total_potential(position)
                total_potential = self.calculate_total_gaussian_potential(position)
                total_force = self.calculate_total_gaussian_force(position)
                # total_force_buf[i, j] = np.linalg.norm(total_force)
                total_force_angle[i, j] = np.arctan2(total_force[1], total_force[0])

                Z[i, j] = np.linalg.norm(total_force)
                Z[i, j] = min(Z[i, j], 400)
                Z[i, j] = max(Z[i, j], 0.0)

                P[i, j] = np.linalg.norm(total_potential)

        dy, dx = np.gradient(P)
        dy, dx = -dy, -dx

        # plt.figure(1)
        # plt.contourf(X, Y, P, levels=20, cmap='jet')
        # plt.quiver(X, Y, dx, dy, pivot='mid', scale=500)
        # plt.scatter(self.goal[0], self.goal[1], color='green', marker='o', label='Goal')
        # plt.legend()
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.title('Potential Field, np.gradient()')


        # plt.figure(2)
        # plt.contourf(X, Y, P, levels=20, cmap='jet')
        # for i in range(len(x)):
        #     for j in range(len(y)):
        #         plt.arrow(X[i, j], Y[i, j], self.remap(Z[i, j], np.min(Z), np.max(Z), 0.0, 0.7)*np.cos(total_force_angle[i, j]), self.remap(Z[i, j], np.min(Z), np.max(Z), 0.1, 1)*np.sin(total_force_angle[i, j]), head_width=0.1, head_length=0.1, fc='k', ec='k')

        # plt.scatter(self.goal[0], self.goal[1], color='green', marker='o', label='Goal')
        # plt.legend()
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.title('Potential Field, manual calculations')

        # plt.show()


        fig = plt.figure(figsize =(14, 9))
        ax = plt.axes(projection='3d')
        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False)
        ax.contour3D(X, Y, Z, 50)
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

# Example usage
goal = [10, 10]
obstacles = [[5, 5, 2], [8, 8, 2], [0, -10, 2], [-10, 0, 2], [10, -5, 2], [10, 0, 2]]  # [x, y, radius]
controller = PotentialFieldController(goal, obstacles)
robot_position = [-15.0, -10.0, 0.0, 0.0, 0.0]
# controller.plot_field()
# controller.move_robot(robot_position)
controller.plot_moving_field(robot_position)