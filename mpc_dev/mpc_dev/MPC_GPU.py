import numpy as np

import os
import time

from lar_msgs.msg import CarControlStamped, CarStateStamped
from typing import List

from bumper_cars.utils import car_utils as utils
from bumper_cars.classes.State import State
from bumper_cars.classes.Controller import Controller

# For the parameter file
import yaml

# For the GPU
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import copy
import tqdm

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

np.random.seed(1)

class MPC_GPU_algorithm(Controller):
    """
    This class implements the MPC (Model Predictive Control) algorithm.

    Attributes:
        dxu (numpy.ndarray): Control input of shape (2,) representing the throttle and steering.
        safety_radius (float): Safety radius for collision avoidance.
        barrier_gain (float): Gain for the barrier function.
        Kv (float): Gain for the velocity term.
        closest_point (tuple): Coordinates of the closest point on the boundary.

    """    
    def __init__(self, controller_path:str, car_i = 0):
        """
        Initializes the MPC_algorithm class.

        Args:
            controller_path (str): Path to the controller YAML file.
            car_i (int): Index of the car.

        """
        ## Init Controller class
        super().__init__(controller_path, car_i)

        # Opening YAML file
        with open(controller_path, 'r') as openfile:
            # Reading from yaml file
            yaml_object = yaml.safe_load(openfile)

        self.safety_radius = yaml_object["MPPI"]["safety_radius"]

        self.input_size = 5  # Number of input features
        self.hidden_size = 128  # Number of hidden units
        self.output_size = 4  # Number of output features

        self.learning_rate = yaml_object["MPPI"]["learning_rate"]
        self.epochs = yaml_object["MPPI"]["epochs"]
        self.batch_size = yaml_object["MPPI"]["batch_size"]
        self.sample_size = yaml_object["MPPI"]["sample_size"]
        

        self.nn_path = os.path.dirname(os.path.realpath(__file__)) + "/../models/statePred.pt"

        self.best_i = 0
        self.prev_a = 0.0
        self.prev_delta = 0.0
        self.std_a = yaml_object["MPPI"]["std_a"]
        self.std_delta = yaml_object["MPPI"]["std_delta"]
        self.sampleNum = yaml_object["MPPI"]["num_samples"]
        self.pred_hor = int(self.ph/self.dt)

        try:
            self.device = "cpu"
            self.__load_model()
        except:
            print("Trained model not found. Run this with option gen_traj:=True")


    ################# PUBLIC METHODS

    def compute_cmd(self, car_list: List[CarStateStamped]) -> CarControlStamped:
        """
        Computes the car control command based on the current state of all cars.

        Args:
            car_list (List[CarStateStamped]): A list of CarStateStamped objects representing the current state of all cars.

        Returns:
            CarControlStamped: The computed car control command.

        """
        # Init empty command
        car_cmd = CarControlStamped()


        # Update current state of all cars
        self.curr_state = np.transpose(utils.carStateStamped_to_array(car_list))

        # Compute control
        u, traj = self.__MPC_GPU(self.car_i, self.curr_state)
        
        # Project it to range [-1, 1]
        car_cmd.throttle = np.interp(u[0].cpu(), [self.car_model.min_acc, self.car_model.max_acc], [-1, 1]) * self.car_model.acc_gain
        car_cmd.steering = np.interp(u[1].cpu(), [-self.car_model.max_steer, self.car_model.max_steer], [-1, 1])
        
        self.trajectory = torch.zeros_like(traj, device='cpu')
        self.trajectory.copy_(traj,non_blocking=True)
    
        # x_traj = self.mpc.data.prediction(('_x', 'x')).flatten()   
        # y_traj = self.mpc.data.prediction(('_x', 'y')).flatten()
        # self.trajectory = zip(x_traj, y_traj)


        return car_cmd

    def set_goal(self, goal: CarControlStamped) -> None:
        """
        Sets the goal for the C3BF controller.

        Args:
            goal (CarControlStamped): The goal containing the desired throttle and steering values.

        Returns:
        """
        self.goal = CarControlStamped()
        self.goal.throttle = np.interp(goal.throttle, [-1, 1], [self.car_model.min_acc, self.car_model.max_acc]) * self.car_model.acc_gain
        self.goal.steering = np.interp(goal.steering, [-1, 1], [-self.car_model.max_steer, self.car_model.max_steer])

        self.goal_tensor = torch.tensor([self.goal.throttle, self.goal.steering], dtype=torch.float32, device=self.device)

        # self.prev_a = self.goal.throttle
        # self.prev_delta = self.goal.steering

    def offset_track(self, off:List[int]) -> None:
        """
        Updates the position of the corners of the map and the corresponding parameters that define the "walls".

        Args:
            off (List[int]): A list of integers representing the offset values for x,y and yaw.

        Returns:
            None
        """
        super().offset_track(off)
        self.__compute_track_constants()

    def compute_traj(self):
        # Generate data for NN
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__gen_model()
        self.device = "cpu"
        self.__load_model()
        # Load model and evaluate with new data
        # self.__test_model()
        
    ################# PRIVATE METHODS
    
    def __load_model(self):
        self.model = SimpleNN(self.input_size, self.hidden_size, self.output_size).to(self.device)
        self.model.load_state_dict(torch.load(self.nn_path))
        self.model.eval()

    def __simulate_input(self, car_cmd: CarControlStamped) -> State:
        curr_state = self.car_model.step(car_cmd, self.dt)
        t = self.dt
        while t<self.ph:
            curr_state = self.car_model.step(car_cmd, self.dt, curr_state=curr_state)
            t += self.dt
        return curr_state
        
    def __compute_track_constants(self):
        """
        Computes the track constants based on the boundary points.

        This method calculates the constants required for track boundary computation
        using the given boundary points. The constants a, b, and c
        are calculated based on the following equations:

        a = y2 - y1
        b = x1 - x2
        c = y1 * (x2 - x1) - x1 * (y2 - y1)

        They correspond to the constants for the straight lines that join all vertices 
        of the square that defines the bumping arena. These lines follow convention 
        of the form ax + by + c = 0.

        Returns:
            None
        """
        x1, y1 = self.boundary_points[0][0], self.boundary_points[0][1]
        x2, y2 = self.boundary_points[1][0], self.boundary_points[1][1]
        x3, y3 = self.boundary_points[2][0], self.boundary_points[2][1]
        x4, y4 = self.boundary_points[3][0], self.boundary_points[3][1]

        # a = y2-y1
        self.a1 = y2 - y1
        self.a2 = y3 - y2
        self.a3 = y4 - y3
        self.a4 = y1 - y4

        self.a = torch.tensor([self.a1, self.a2, self.a3, self.a4], dtype=torch.float32, device=self.device).unsqueeze_(1).unsqueeze_(1)

        # b = x1-x2
        self.b1 = x1 - x2
        self.b2 = x2 - x3
        self.b3 = x3 - x4
        self.b4 = x4 - x1

        self.b = torch.tensor([self.b1, self.b2, self.b3, self.b4], dtype=torch.float32, device=self.device).unsqueeze_(1).unsqueeze_(1)

        # c = y1*(x2-x1) - x1*(y2-y1)
        self.c1 = y1 * (x2 - x1) - x1 * (y2 - y1)
        self.c2 = y2 * (x3 - x2) - x2 * (y3 - y2)
        self.c3 = y3 * (x4 - x3) - x3 * (y4 - y3)
        self.c4 = y4 * (x1 - x4) - x4 * (y1 - y4)

        self.c = torch.tensor([self.c1, self.c2, self.c3, self.c4], dtype=torch.float32, device=self.device).unsqueeze_(1).unsqueeze_(1)

        # Precompute denominators for distance calculation
        self.denom1 = np.sqrt(self.a1**2 + self.b1**2)
        self.denom2 = np.sqrt(self.a2**2 + self.b2**2)
        self.denom3 = np.sqrt(self.a3**2 + self.b3**2)
        self.denom4 = np.sqrt(self.a4**2 + self.b4**2)

        self.denom = torch.tensor([self.denom1, self.denom2, self.denom3, self.denom4], dtype=torch.float32, device=self.device).unsqueeze_(1).unsqueeze_(1)


    def __eval_input(self, input: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the input tensor by calculating the minimum difference to the goal tensor and returning the index.

        Parameters:
        - input (torch.Tensor): The input tensor with shape (batch_size, num_samples, 2).

        Returns:
        - cost (torch.Tensor): The calculated cost for each batch.
        - min_indices (torch.Tensor): The indices of the minimum differences for each batch.
        """
        # Compute the difference between each element of input and goal_tensor
        diff = self.goal_tensor - input[:, 0:2]  
        # Compute the Euclidean norm of the differences
        norm_diff = torch.norm(diff, dim=1)  # Compute the norm along the feature dimension

        return norm_diff
    
    def __eval_distances(self, states: torch.Tensor) -> torch.Tensor:
            
        x = states[:, :, 0]
        y = states[:, :, 1]

        # Compute the distances
        distance = torch.abs(self.a * x + self.b * y + self.c)/self.denom
        cost = torch.where(distance < 2.0*self.safety_radius, 1/distance, torch.zeros_like(distance))
        # cost = torch.where(distance < self.safety_radius, np.inf, torch.zeros_like(distance))
        cost = torch.sum(cost.squeeze(), dim=2)
        cost = torch.sum(cost.squeeze(), dim=0)
        return cost
        
    def __eval_other_cars(self, states: torch.Tensor, others: torch.Tensor) -> torch.Tensor:

        my = states[:, :, 0:2].unsqueeze_(2)
        other = others[:, :, :, 0:2]

        # Calculate the difference between the tensors
        difference = my - other

        # Square the differences
        squared_difference = difference ** 2

        # Sum the squared differences along the last dimension
        sum_squared_difference = torch.sum(squared_difference, dim=-1)
        
        # Take the square root to get the Euclidean distance
        distance = torch.sqrt(sum_squared_difference)

        cost = torch.where(distance < self.safety_radius, 1/distance, torch.zeros_like(distance))
        # cost = torch.where(distance < self.safety_radius, np.inf, torch.zeros_like(distance))
        # cost = torch.sum(cost, dim=2)
        cost = torch.sum(cost.squeeze(), dim=1)
        return cost


    def __MPC_GPU(self, car_i, x) -> np.ndarray:
        with torch.no_grad():

            # Initialize costs
            cost_input = torch.zeros((self.sampleNum), device=self.device)
            cost_dist = torch.zeros((self.sampleNum), device=self.device)
            cost_others = torch.zeros((self.sampleNum), device=self.device)
            
            # Obstacle cars
            num_other_cars = x.shape[1]-1

            # Make tensors in device
            gen_a = torch.normal(self.prev_a, self.std_a, size=(self.sampleNum, self.pred_hor)).to(self.device)
            gen_delta = torch.normal(self.prev_delta, self.std_delta, size=(self.sampleNum, self.pred_hor)).to(self.device)

            gen_a = torch.clamp(gen_a, self.car_model.min_acc, self.car_model.max_acc)
            gen_delta = torch.clamp(gen_delta, -self.car_model.max_steer, self.car_model.max_steer)


            input = torch.zeros((self.sampleNum,5),device=self.device)
            input[:,0] = gen_a[:,0]
            input[:,1] = gen_delta[:,0]
            input[:,2] = np.cos(x[2,car_i])    # Input curr yaw
            input[:,3] = np.sin(x[2,car_i])    # Input curr yaw
            input[:,4] = x[3,car_i]    # Input curr vel

            states = torch.zeros((self.sampleNum, self.pred_hor, self.output_size), device=self.device)
            states[:,0,0] = x[0,car_i]
            states[:,0,1] = x[1,car_i]
            states[:,0,2] = x[2,car_i]
            states[:,0,3] = x[3,car_i]
            
            for i in range(self.pred_hor - 1):
                states[:,i+1,:] = states[:,i,:] + self.model(input)
                input[:,0] = gen_a[:,i+1]
                input[:,1] = gen_delta[:,i+1]
                input[:,2] = torch.cos(states[:,i+1,2])
                input[:,3] = torch.sin(states[:,i+1,2])
                input[:,4] = states[:,i+1,3]
                


            
            if num_other_cars > 0:
                other_states = torch.zeros((self.sampleNum, self.pred_hor, num_other_cars, self.output_size), device=self.device)

                other_i = 0
                for i in range(0, num_other_cars+1):
                    if i == car_i:
                        continue
                    other_states[:,0,other_i,0] = x[0,i]
                    other_states[:,0,other_i,1] = x[1,i]
                    other_states[:,0,other_i,2] = x[2,i]
                    other_states[:,0,other_i,3] = x[3,i]

                    other_input = torch.zeros((self.sampleNum,5),device=self.device)

                    other_input[:,0] = 0.0
                    other_input[:,1] = 0.0
                    other_input[:,2] = np.cos(x[2,i])       # Input curr yaw
                    other_input[:,3] = np.sin(x[2,i])       # Input curr yaw
                    other_input[:,4] = x[3,i]               # Input curr vel

                    
                    for j in range(self.pred_hor - 1):
                        other_states[:,j+1,other_i,:] = other_states[:,j,other_i,:] + self.model(other_input)
                        other_input[:,2] = torch.cos(other_states[:,j+1,other_i,2])
                        other_input[:,3] = torch.sin(other_states[:,j+1,other_i,2])
                        other_input[:,4] = other_states[:,j+1,other_i,3]
                    other_i += 1
                
                cost_others = self.__eval_other_cars(states, other_states)
                cost_others2 = torch.log((cost_others - torch.min(cost_others))/torch.max(cost_others) + 1)
                cost_others2[torch.isnan(cost_others2)] = 0.0

            cost_input = self.__eval_input(input)
            cost_dist = self.__eval_distances(states)
            
            cost_input2 = torch.log((cost_input - torch.min(cost_input))/torch.max(cost_input) + 1)
            cost_dist2 = torch.log((cost_dist - torch.min(cost_dist))/torch.max(cost_dist) + 1)
            
            cost_input2[torch.isnan(cost_input2)] = 0.0
            cost_dist2[torch.isnan(cost_dist2)] = 0.0

            # if self.car_i == 0:
            #     print(cost_input2, cost_dist2, cost_others2)


            cost = 100*cost_dist2 + 100*cost_others2 + 2*cost_input2 
            
            min_cost, min_index = torch.min(cost, 0)
            self.best_i = min_index
            control = input[min_index,:2]        
            self.prev_a = control[0].item()
            self.prev_delta = control[1].item()

            if min_cost.item() == np.inf:
                print("Emergency stop!")
                control[0] =  (0.0 - x[3,car_i])/self.dt
                self.prev_a = 0.0
                
            return control.detach().clone(), states.detach().clone()
            

    def __gen_model(self):

        N = self.sample_size  # Sample size

        gen_u = torch.FloatTensor(N, 1).uniform_(self.car_model.min_acc, self.car_model.max_acc).to(self.device)
        gen_delta = torch.FloatTensor(N, 1).uniform_(-self.car_model.max_steer, self.car_model.max_steer).to(self.device)

        # Move tensors to device
        gen_yaw = torch.FloatTensor(N, 1).uniform_(self.car_model.min_omega, self.car_model.max_omega).to(self.device)
        gen_v = torch.FloatTensor(N, 1).uniform_(self.car_model.min_speed, self.car_model.max_speed).to(self.device)

        # Add this to solve 0 = 2*pi issue
        cos_yaw = torch.cos(gen_yaw)
        sin_yaw = torch.sin(gen_yaw)

        # Calculate new state using the vehicle model equations
        x_dot = gen_v * torch.cos(gen_yaw) * self.dt
        y_dot = gen_v * torch.sin(gen_yaw) * self.dt
        yaw_dot = gen_v / self.car_model.L * torch.tan(gen_delta) * self.dt
        v_dot = gen_u * self.dt

        # Define input and outputs
        X = torch.cat((gen_u, gen_delta, cos_yaw, sin_yaw, gen_v), dim=1)
        Y = torch.cat((x_dot, y_dot, yaw_dot, v_dot), dim=1)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, shuffle=True)
        # X_train = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        # y_train = torch.tensor(y_train, dtype=torch.float32, device=self.device).reshape(-1, 1)
        # X_test = torch.tensor(X_test, dtype=torch.float32, device=self.device)
        # y_test = torch.tensor(y_test, dtype=torch.float32, device=self.device).reshape(-1, 1)

        # Define the neural network
        model = SimpleNN(self.input_size, self.hidden_size, self.output_size).to(self.device)

        # training parameters
        n_epochs = self.epochs   # number of epochs to run
        batch_size = self.batch_size  # size of each batch
        batch_start = torch.arange(0, len(X_train), batch_size)

        # Hold the best model
        best_mse = np.inf   # init to infinity
        best_weights = None
        history = []
        
        # loss function and optimizer
        loss_fn = nn.MSELoss()  # mean square error
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        # training loop
        for epoch in range(n_epochs):
            print("Epoch: ", epoch, " of ", n_epochs)
            model.train()
            with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
                bar.set_description(f"Epoch {epoch}")
                for start in bar:
                    # take a batch
                    X_batch = X_train[start:start+batch_size]
                    y_batch = y_train[start:start+batch_size]
                    # forward pass
                    y_pred = model(X_batch)
                    loss = loss_fn(y_pred, y_batch)
                    # backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    # update weights
                    optimizer.step()
                    # print progress
                    bar.set_postfix(mse=float(loss))
            # evaluate accuracy at end of each epoch
            model.eval()
            y_pred = model(X_test)
            mse = loss_fn(y_pred, y_test)
            mse = float(mse)
            print("Loss function value: " + str(mse))
            history.append(mse)
            if mse < best_mse:
                best_mse = mse
                best_weights = copy.deepcopy(model.state_dict())

        # restore model and return best accuracy
        model.load_state_dict(best_weights)
        
        torch.save(model.state_dict(), self.nn_path)



    def __test_model(self):

        N = self.sample_size

        self.model = SimpleNN(self.input_size, self.hidden_size, self.output_size).to(self.device)
        self.model.load_state_dict(torch.load(self.nn_path))
        self.model.eval()

        gen_u = torch.FloatTensor(N, 1).uniform_(self.car_model.min_acc, self.car_model.max_acc).to(self.device)
        gen_delta = torch.FloatTensor(N, 1).uniform_(-self.car_model.max_steer, self.car_model.max_steer).to(self.device)

        # Move tensors to device
        gen_yaw = torch.FloatTensor(N, 1).uniform_(self.car_model.min_omega, self.car_model.max_omega).to(self.device)
        gen_v = torch.FloatTensor(N, 1).uniform_(self.car_model.min_speed, self.car_model.max_speed).to(self.device)

        # Add this to solve 0 = 2*pi issue
        cos_yaw = torch.cos(gen_yaw)
        sin_yaw = torch.sin(gen_yaw)

        # Calculate new state using the vehicle model equations
        x_dot = gen_v * torch.cos(gen_yaw) * self.dt
        y_dot = gen_v * torch.sin(gen_yaw) * self.dt
        yaw_dot = gen_v / self.car_model.L * torch.tan(gen_delta) * self.dt
        v_dot = gen_u * self.dt

        # Define input and outputs
        X = torch.cat((gen_u, gen_delta, cos_yaw, sin_yaw, gen_v), dim=1)
        Y = torch.cat((x_dot, y_dot, yaw_dot, v_dot), dim=1)

        loss_fn = nn.MSELoss()  # mean square error
        y_pred = self.model(X)
        mse = loss_fn(y_pred, Y)
        mse = float(mse)
        print("------------------------------------------")
        print("Eval loss function value: " + str(mse))
