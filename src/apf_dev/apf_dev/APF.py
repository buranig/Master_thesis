import math
import numpy as np
import matplotlib.pyplot as plt

from custom_message.msg import Coordinate

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
np.random.seed(1)

color_dict = {0: 'r', 1: 'b', 2: 'g', 3: 'y', 4: 'm', 5: 'c', 6: 'k', 7: 'tab:orange', 8: 'tab:brown', 9: 'tab:gray', 10: 'tab:olive', 11: 'tab:pink', 12: 'tab:purple', 13: 'tab:red', 14: 'tab:blue', 15: 'tab:green'}
with open('/home/giacomo/thesis_ws/src/seeds/seed_8.json', 'r') as file:
    data = json.load(file)

class PotentialFieldController:
    def __init__(self, goal, robot_num, paths):
        self.goal = goal
        self.obstacles = []
        self.dxu = np.zeros((2, robot_num))
        self.robot_num = robot_num
        self.paths = paths

        self.k_rep = 3000 # Repulsive force gain
        self.k_att = 100 # Attractive force gain
        self.k_arena = 0 #7000

        self.n = 2

        self.sigmax = 2
        self.sigmay = 2
        self.sigma_arena = 1.5
        self.definition = 64
        self.width = 15
        self.height = 15
        self.m = m

        self.reached_goal = [False]*robot_num

    def attractive_potential(self, position, i):
        # Calculate attractive potential towards the goal
        attractive_potential = 0.5 * self.k_att*np.linalg.norm([position[0]- self.goal[i][0], position[1]- self.goal[i][1]])**2
        return attractive_potential

    def attractive_force(self, position, i):
        # Calculate attractive force towards the goal
        attractive_force = [-self.k_att*(position[0]- self.goal[i][0]), -self.k_att*(position[1]- self.goal[i][1])]
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
    
    def repulsive_gaussian_potential_imp(self, position, i):
        # Calculate repulsive potential from obstacles
        repulsive_potential = 0.0
        goal_distance = self.euclidean_distance(position, self.goal[i])

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

    def repulsive_gaussian_force_imp(self, position, i):
        # Calculate repulsive force from obstacles
        goal_distance = self.euclidean_distance(position, self.goal[i])
        if goal_distance == 0.0:
            goal_distance += 0.0001
        repulsive_force = [0.0, 0.0]
        for obstacle in self.obstacles:
            potential = self.k_rep*np.exp(-(position[0] - obstacle[0])**2/(self.sigmax**2) - (position[1] - obstacle[1])**2/(self.sigmay**2))
            repulsive_force[0] += -potential*goal_distance*(-2*(position[0] - obstacle[0])/(self.sigmax**2))-potential*(position[0]-self.goal[i][0])/goal_distance
            repulsive_force[1] += -potential*goal_distance*(-2*(position[1] - obstacle[1])/(self.sigmay**2))-potential*(position[1]-self.goal[i][1])/goal_distance
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
    
    def calculate_total_gaussian_potential(self, position, i):
        # # Calculate the total potential acting on the robot
        attractive_potential = self.attractive_potential(position, i)
        repulsive_potential = self.repulsive_gaussian_potential_imp(position, i)
        arena_potential = self.repulsive_arena_potential(position)
        total_potential = arena_potential + repulsive_potential + attractive_potential
        return total_potential
    
    def calculate_total_gaussian_force(self, position, i):
        # # Calculate the total force acting on the robot
        total_force = [0.0, 0.0]
        attractive_force = self.attractive_force(position, i)
        repulsive_force = self.repulsive_gaussian_force_imp(position, i)
        arena_force = self.repulsive_arena_force(position)
        total_force[0] = repulsive_force[0] + attractive_force[0] + arena_force[0]
        total_force[1] = repulsive_force[1] + attractive_force[1] + arena_force[1]
        return total_force
    
    def force_to_input(self, position, force, force_pred):
        # Convert the force to robot's acceleration and steering angle
        ax = (force[0]*np.cos(position[2]) + force[1]*np.sin(position[2]))/self.m
        ay = (-force_pred[0]*np.sin(position[2]) + force_pred[1]*np.cos(position[2]))/self.m 
        steering = ay*(L**2*2*Cf*Cr + self.m*position[3]**2*(Lf*Cf-Lr*Cr))/(2*Cf*Cr*L*(position[3]+0.001)**2)

        # ax = np.linalg.norm(force)
        # steering = np.arctan2(force_pred[1], force_pred[0]) - position[2]
        steering = np.clip(steering, -max_steer, max_steer)

        return ax, steering
    
    def calculate_complete_field(self):
        x = np.linspace(-self.width, self.width, self.definition)
        y = np.linspace(-self.height, self.height, self.definition)
        X, Y = np.meshgrid(x, y)
        total_force_angle = np.zeros((x.shape[0], y.shape[0]))
        Z = np.zeros_like(X)
        P = np.zeros_like(X)
        for i in range(len(x)):
            for j in range(len(y)):
                position_ = [X[i, j], Y[i, j]]
                total_potential = self.calculate_total_gaussian_potential(position_, 0)
                total_force = self.calculate_total_gaussian_force(position_, 0)

                total_force_angle[i, j] = np.arctan2(total_force[1], total_force[0])

                Z[i, j] = np.linalg.norm(total_force)
                Z[i, j] = max(Z[i, j], 0.0)

                P[i, j] = np.linalg.norm(total_potential)

        dy, dx = np.gradient(P)
        dy, dx = -dy, -dx

        return P, Z, dx, dy, X, Y
    
    def calculate_map_potential(self):
        x = np.linspace(-self.width, self.width, self.definition)
        y = np.linspace(-self.height, self.height, self.definition)
        X, Y = np.meshgrid(x, y)
        P = np.zeros_like(X)
        for i in range(len(x)):
            for j in range(len(y)):
                position_ = [X[i, j], Y[i, j]]
                total_potential = self.calculate_total_gaussian_potential(position_, 0)
                
                P[i, j] = np.linalg.norm(total_potential)

        dy, dx = np.gradient(P)
        dy, dx = -dy, -dx

        return P, dx, dy, X, Y

    def move_robot(self, position):
        # Plot the potential field
        fig = plt.figure(1, dpi=90, figsize=(10,10))
        ax = fig.add_subplot(111)

        self.obstacles = [[0, 0, 2]]

        P, Z, dx, dy, X, Y = self.calculate_complete_field()
       
        # Move the robot based on the calculated force
        total_force = self.calculate_total_gaussian_force(position, 0)
        x_pred = [position[0] + dt*np.cos(position[2])*position[3], position[1] + dt*np.sin(position[2])*position[3]]
        total_force_pred = self.calculate_total_gaussian_force(x_pred, 0)
        throttle, steering = self.force_to_input(position, total_force, total_force_pred)

        position_buf = [[position[0], position[1]]]

        while self.euclidean_distance(position, self.goal[0]) > 2:
            plt.cla()
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

            position = utils.nonlinear_model_numpy_stable(position, [throttle, steering])

            position_buf.append([position[0], position[1]])

            # Plot the robot's position
            plt.contourf(X, Y, P, levels=20, cmap='jet')
            plt.quiver(X, Y, dx, dy, pivot='mid', scale=100000)
            utils.plot_robot(position[0], position[1], position[2], 0)
            utils.plot_arrow(position[0], position[1], position[2], length=1, width=0.5)
            utils.plot_arrow(position[0], position[1], position[2]+steering, length=3, width=0.5)

            plt.plot(x_pred[0], x_pred[1], 'ro', label='Predicted Robot')

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
            if self.euclidean_distance(position, self.goal[0]) < 2:
                print('Goal Reached')
                break

            total_force = self.calculate_total_gaussian_force(position, 0)
            position_copy = position.copy()
            x_pred = utils.nonlinear_model_numpy_stable(position_copy, [throttle, steering], dt=0.2)
            # x_pred = utils.nonlinear_model_numpy_stable(position, [0, 0])
            total_force_pred = self.calculate_total_gaussian_force(x_pred[0:2], 0)
            throttle, steering = self.force_to_input(position, total_force, total_force_pred)

            # print('Throttle:', throttle, 'Steering:', steering)

        for elem in position_buf:
            plt.plot(elem[0], elem[1], 'ro', label='Robot')
        plt.pause(0.0001)
        plt.show()
        # print(position_buf)

    def step(self, x, i):
        self.obstacles = [[x[0, z], x[1, z], 2] for z in range(self.robot_num) if z != i]

        x_copy = x[:, i].copy()
        x_pred = utils.nonlinear_model_numpy_stable(x_copy, self.dxu[:, i], dt=0.5)
        plt.plot(x_pred[0], x_pred[1], 'ko', label='Predicted Robot')
        total_force = self.calculate_total_gaussian_force(x[:, i], i)
        total_force_pred = self.calculate_total_gaussian_force(x_pred, i)
        self.dxu[0, i], self.dxu[1, i] = self.force_to_input(x[:, i], total_force, total_force_pred)
        x[:, i] = utils.nonlinear_model_numpy_stable(x[:, i], self.dxu[:, i])

        return x

    def run_apf(self, x, break_flag):
        for i in range(self.robot_num):
            if not self.reached_goal[i]:
                # Step 9: Check if the distance between the current position and the target is less than 5
                if utils.dist(point1=(x[0,i], x[1,i]), point2=self.goal[i]) < 2:
                    print(f'Updating target for vehicle {i}!')
                    # Perform some action when the condition is met
                    self.paths[i].pop(0)
                    if not self.paths[i]:
                        print(f"Path complete for vehicle {i}!")
                        self.dxu[:, i] = np.zeros(2)
                        x[3, i] = 0
                        self.reached_goal[i] = True
                    else: 
                        self.goal[i] = (self.paths[i][0].x, self.paths[i][0].y)
                else:
                    x = self.step(x, i) 

            if i == 0:
                self.obstacles = []
                for z in range(self.robot_num):
                    if z == i:
                        continue
                    else:
                        self.obstacles.append([x[0, z], x[1, z], 2])
                P, dx, dy, X, Y = self.calculate_map_potential()

                plt.contourf(X, Y, P, levels=20, cmap='jet')
                # force_angle = np.arctan2(total_force[1], total_force[0])    
                # utils.plot_arrow(x[0, i], x[1, i], force_angle, length=5, width=0.5)
            
            utils.plot_robot(x[0, i], x[1, i], x[2, i], i)
            utils.plot_arrow(x[0, i], x[1, i], x[2, i], length=1, width=0.5)
            utils.plot_arrow(x[0, i], x[1, i], x[2, i]+ self.dxu[1, i], length=3, width=0.5)
            
            plt.plot(self.goal[i][0], self.goal[i][1], 'rx') 

        if all(self.reached_goal):
            break_flag = True
        return x, break_flag


    def remap(self, x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def plot_field(self):
        # Plot the potential field
        
        P, Z, dx, dy, X, Y = self.calculate_complete_field()

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

def main_seed(args=None):
    """
    Main function for controlling multiple robots using Model Predictive Control (MPC).

    Steps:
    1. Initialize the necessary variables and parameters.
    2. Create an instance of the ModelPredictiveControl class.
    3. Set the initial state and control inputs.
    4. Generate the reference trajectory for each robot.
    5. Plot the initial positions and reference trajectory.
    6. Set the bounds and constraints for the MPC.
    7. Initialize the predicted trajectory for each robot.
    8. Enter the main control loop:
        - Check if the distance between the current position and the target is less than 5.
            - If yes, update the path and target.
        - Perform 3CBF control for each robot.
        - Plot the robot trajectory.
        - Update the predicted trajectory.
        - Plot the map and pause for visualization.
    """
    # Step 1: Set the number of iterations
    iterations = 3000
    fig = plt.figure(1, dpi=90, figsize=(10,10))
    ax = fig.add_subplot(111)
    break_flag = False
    
    # Step 2: Sample initial values for x0, y, yaw, v, omega, and model_type
    initial_state = data['initial_position']
    x0 = initial_state['x']
    y = initial_state['y']
    yaw = initial_state['yaw']
    v = initial_state['v']
    omega = [0.0]*len(initial_state['x'])

    robot_num = data['robot_num']

    # Step 3: Create an array x with the initial values
    x = np.array([x0, y, yaw, v, omega])
    u = np.zeros((2, robot_num))

    trajectory = np.zeros((x.shape[0]+u.shape[0], robot_num, 1))
    trajectory[:, :, 0] = np.concatenate((x,u))
    
    # Step 4: Create paths for each robot
    traj = data['trajectories']
    paths = [[Coordinate(x=traj[str(idx)][i][0], y=traj[str(idx)][i][1]) for i in range(len(traj[str(idx)]))] for idx in range(robot_num)]

    # Step 5: Extract the target coordinates from the paths
    targets = [[path[0].x, path[0].y] for path in paths]

    apf = PotentialFieldController(goal=targets, robot_num=robot_num, paths=paths)
    # Step 8: Perform the simulation for the specified number of iterations
    for z in range(iterations):
        plt.cla()
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        
        x, break_flag = apf.run_apf(x, break_flag) 

        trajectory = np.dstack([trajectory, np.concatenate((x, apf.dxu))])
        
        utils.plot_map(width=width_init, height=height_init)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Potential Field')
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)

        if break_flag:
            break
    
    print("Done")
    if show_animation:
        for i in range(robot_num):
            plt.plot(trajectory[0, i, :], trajectory[1, i, :], "-", color=color_dict[i])
        plt.pause(0.0001)
        plt.show()
    
    apf.plot_field()

if __name__ == '__main__':
    main_seed()
    # # Example usage
    # goal = [[10, 10], [-15.0, -10.0], [-15.0, 0.0]]
    # obstacles = []  # [x, y, radius]

    # x = np.array([[-15.0, -10.0, 0.0, 0.0, 0.0], 
    #               [10.0, 10.0, -3*np.pi/4, 0.0, 0.0], 
    #               [15.0, 0.0, -np.pi, 0.0, 0.0]])
    # obstacles = [[x[1, 0], x[1,1], 2], [x[2, 0], x[2, 1], 2]]
    # controller = PotentialFieldController(goal, robot_num=3)
    # # controller.plot_field()
    # # controller.move_robot(x[0,:])
    # controller.run_apf(x)