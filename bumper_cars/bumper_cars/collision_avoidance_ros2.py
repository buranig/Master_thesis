
#!/usr/bin/env python3
    
import rclpy
import yaml
from rclpy.node import Node
import numpy as np
from bumper_cars.classes.CarModel import CarModel, State
from lar_utils import car_utils as utils

# Plot goal in Rviz
from visualization_msgs.msg import Marker

# import control models
# from planner import utils
from dwa_dev.DWA import DWA_algorithm as DWA
from cbf_dev.CBF import CBF_algorithm as CBF
from cbf_dev.C3BF import C3BF_algorithm as C3BF
# from lbp_dev import LBP
# from mpc_dev import MPC

controller_map = {
    "dwa": DWA,
    "c3bf": C3BF,
    "cbf": CBF
    # "lbp": LBP,
    # "LBP": LBP,
    # "mpc": MPC,
    # "MPC": MPC
}


from lar_msgs.msg import CarControlStamped, CarStateStamped
from lar_msgs.srv import EnvState, CarCommand


class CollisionAvoidance(Node):
    def __init__(self):
        super().__init__('ca')
        self.declare_parameter('car_i', 1)
        self.declare_parameter('car_yaml', rclpy.Parameter.Type.STRING)
        self.declare_parameter('alg', rclpy.Parameter.Type.STRING)

        # Init variables
        self.car_i = self.get_parameter('car_i').value
        self.car_yaml = self.get_parameter('car_yaml').value
        self.car_alg = self.get_parameter('alg').value
        print("############################")
        print("############################")
        print(self.car_alg)
        print("############################")
        print("############################")

        self.car_str = '' if self.car_i == 1 else str(self.car_i)

        # Init controller
        self.algorithm = controller_map[self.car_alg.lower()](self.car_yaml,self.car_i)

        # Service to query state
        self.state_cli = self.create_client(EnvState, 'env_state')
        self.cmd_cli = self.create_client(CarCommand, 'car_cmd')
        while not self.state_cli.wait_for_service(timeout_sec=1.0) or not self.cmd_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        
        # Initialize service messages
        self.state_req = EnvState.Request()
        self.cmd_req = CarCommand.Request()
        self.cmd_req.car = self.car_i - 1

        self.publisher_ = self.create_publisher(CarControlStamped, '/sim/car'+self.car_str+'/set/control', 10)
        self.goal_publisher_ = self.create_publisher(Marker, '/vis/trace', 10)

    def state_request(self):
        self.future = self.state_cli.call_async(self.state_req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
        
    def cmd_request(self):
        self.cmd_future = self.cmd_cli.call_async(self.cmd_req)
        rclpy.spin_until_future_complete(self, self.cmd_future)
        return self.cmd_future.result()


    def run(self):
        while rclpy.ok():
            # Update the current state of the car
            curr_state = self.state_request()
            curr_car = curr_state.env_state[self.car_i - 1] # Select desired car
            updated_state = carStateStamped_to_State(curr_car)
            self.algorithm.set_state(updated_state)

            # Compute desired action
            des_action = self.cmd_request()

            # Set desired action as a goal for the CA algorithm
            self.algorithm.set_goal(des_action.cmd)

            # Compute safe control input
            safe_cmd = self.algorithm.compute_cmd(curr_state.env_state)

            # Send command to car
            self.publisher_.publish(safe_cmd)

            # Draw in Rviz desired goal
            if type(self.algorithm.goal) is State:
                self.goal_marker(self.algorithm.goal)

    def goal_marker(self, next_state):
        marker = Marker()
        marker.header.frame_id = 'world'  # Set the frame of reference
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "goal" + self.car_str
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = next_state.x  # Set the position
        marker.pose.position.y = next_state.y
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2  # Set the scale
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0  # Set the color and transparency
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        # Publish the marker
        self.goal_publisher_.publish(marker)
        
        
def carStateStamped_to_State(curr_car: CarStateStamped) -> State:
    car_state = State()
    car_state.x = curr_car.pos_x
    car_state.y = curr_car.pos_y
    car_state.yaw = utils.normalize_angle(curr_car.turn_angle) #turn angle is with respect to the car, not wheel
    car_state.omega = curr_car.turn_rate
    car_state.v = utils.longitudinal_velocity(curr_car.vel_x, curr_car.vel_y, car_state.omega)
    return car_state


def main(args=None):
    rclpy.init(args=args)
    node = CollisionAvoidance()

    # Run control alg
    node.run()
    
    node.destroy_node()
    rclpy.shutdown()