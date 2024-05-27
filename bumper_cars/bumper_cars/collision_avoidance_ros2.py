#!/usr/bin/env python3
    
import rclpy
import yaml
from rclpy.node import Node
import numpy as np
from bumper_cars.utils import car_utils as utils

# Plot goal in Rviz
from visualization_msgs.msg import Marker

# import control models
from dwa_dev.DWA import DWA_algorithm as DWA
from cbf_dev.CBF import CBF_algorithm as CBF
from cbf_dev.C3BF import C3BF_algorithm as C3BF
from lbp_dev.LBP import LBP_algorithm as LBP
from mpc_dev.MPC import MPC_algorithm as MPC

controller_map = {
    "dwa": DWA,
    "c3bf": C3BF,
    "cbf": CBF,
    "lbp": LBP,
    "mpc": MPC
}


from lar_msgs.msg import CarControlStamped
from bumper_msgs.srv import EnvState, CarCommand, JoySafety


class CollisionAvoidance(Node):
    def __init__(self):
        super().__init__('ca')
        self.declare_parameter('car_i', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('gen_traj', rclpy.Parameter.Type.BOOL)
        self.declare_parameter('source', rclpy.Parameter.Type.STRING)
        self.declare_parameter('car_yaml', rclpy.Parameter.Type.STRING)
        self.declare_parameter('alg', rclpy.Parameter.Type.STRING)
        
        ### Global variables

        # Init variables
        self.car_i = int(self.get_parameter('car_i').value)
        self.car_yaml = self.get_parameter('car_yaml').value
        self.car_alg = self.get_parameter('alg').value
        self.gen_traj = self.get_parameter('gen_traj').value
        self.source = self.get_parameter('source').value

        with open(self.car_yaml, 'r') as openfile:
            yaml_object = yaml.safe_load(openfile)

        # Simulation params
        self.dt = yaml_object["Controller"]["dt"]

        self.car_str = '' if self.car_i == 0 else str(self.car_i + 1)

        # Init controller
        self.algorithm = controller_map[self.car_alg.lower()](self.car_yaml,self.car_i)

        # Generate trajectory (if needed)
        if self.gen_traj:
            self.algorithm.compute_traj()

        # Service to query state
        self.state_cli = self.create_client(EnvState, 'env_state' + self.car_str)
        self.cmd_cli = self.create_client(CarCommand, 'car_cmd')
        self.joy_cli = self.create_client(JoySafety, 'joy_safety')
        while not self.state_cli.wait_for_service(timeout_sec=1.0) or not self.cmd_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        
        # Initialize service messages
        self.state_req = EnvState.Request()
        self.cmd_req = CarCommand.Request()
        self.cmd_req.car = self.car_i
        self.joy_req = JoySafety.Request()

        if self.source == 'sim':
            self.publisher_ = self.create_publisher(CarControlStamped, '/sim/car'+self.car_str+'/set/control', 10)
        else:
            self.publisher_ = self.create_publisher(CarControlStamped, '/car'+self.car_str+'/set/control', 10)

        self.goal_publisher_ = self.create_publisher(Marker, '/vis/trace', 10)

        # Update timer
        self.timer = self.create_timer(self.dt, self.timer_callback)
        self.update_time = False

    def timer_callback(self):
        self.update_time = True
        

    def state_request(self):
        self.future = self.state_cli.call_async(self.state_req)
        rclpy.spin_until_future_complete(self, self.future)
        if not self.future.done():
            print("Timeout")
        return self.future.result()
        
    def cmd_request(self):
        self.cmd_future = self.cmd_cli.call_async(self.cmd_req)
        rclpy.spin_until_future_complete(self, self.cmd_future)
        return self.cmd_future.result()

    def joy_request(self):
        self.joy_future = self.joy_cli.call_async(self.joy_req)
        rclpy.spin_until_future_complete(self, self.joy_future)
        return self.joy_future.result()

    def run(self):
        while rclpy.ok():
            # Update the current state of the car (and do rcply spin to update timer)
            try:
                curr_state = self.state_request() # Might return empty list
                curr_car = curr_state.env_state[self.car_i] # Select desired car
                updated_state = utils.carStateStamped_to_State(curr_car)
                self.algorithm.set_state(updated_state)
            except:
                continue

            # Skip rest if no update
            if self.update_time == False:
                continue

            # Check desired action
            des_action = self.cmd_request()

            # Check if CA is activated
            ca_active = self.joy_request()

            if ca_active.ca_activated:
                # Set desired action as a goal for the CA algorithm
                self.algorithm.set_goal(des_action.cmd)

                # Compute safe control input
                cmd_out = self.algorithm.compute_cmd(curr_state.env_state)
            else:
                cmd_out = des_action.cmd

            # Send command to car
            self.publisher_.publish(cmd_out)

            # Wait for next period
            self.update_time = False



def main(args=None):
    rclpy.init(args=args)
    node = CollisionAvoidance()

    # Run control alg
    node.run()

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()