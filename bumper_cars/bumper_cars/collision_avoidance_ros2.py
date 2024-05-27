#!/usr/bin/env python3
    
from typing import List
import rclpy
import yaml
from rclpy.node import Node
import numpy as np
from bumper_cars.utils import car_utils as utils

# Plot goal in Rviz
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import Header, ColorRGBA
import colorsys

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
        self.declare_parameter('debug_rviz', rclpy.Parameter.Type.BOOL)
        
        ### Global variables

        # Init variables
        self.car_i = int(self.get_parameter('car_i').value)
        self.car_yaml = self.get_parameter('car_yaml').value
        self.car_alg = self.get_parameter('alg').value
        self.gen_traj = self.get_parameter('gen_traj').value
        self.source = self.get_parameter('source').value
        self.debug_rviz = self.get_parameter('debug_rviz').value

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

        #Plot map in Rviz
        hue = (self.car_i * 0.618033988749895) % 1  # Use golden ratio conjugate to distribute colors
        self.rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)  # Convert HSV to RGB
        
        if self.debug_rviz:   
            self.debug_publisher = self.create_publisher(Marker, '/debug_markers', 10)

            alg = self.car_alg.lower() 
            if alg == "cbf" or alg == "c3bf":
                self.car_radius_publish(self.algorithm.safety_radius)

            if self.car_i == 0:
                self.publish_map(self.algorithm.boundary_points)
            




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
        return self.joy_future.result().ca_activated

    def run(self):
        ca_active = True
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
            if self.car_i == 0: # Only move car 0
                ca_active = self.joy_request()

            if ca_active:
                # Set desired action as a goal for the CA algorithm
                self.algorithm.set_goal(des_action.cmd)

                # Compute safe control input
                cmd_out = self.algorithm.compute_cmd(curr_state.env_state)
            else:
                cmd_out = des_action.cmd

            # Send command to car
            self.publisher_.publish(cmd_out)

            # Draw debug info in Rviz
            if self.debug_rviz:
                self.barrier_publisher(self.algorithm.closest_point)


            # Wait for next period
            self.update_time = False


    def publish_map(self, boundary_points: List[float]):
        marker = Marker()
        marker.header = Header()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = ""
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        # Define the points for the line strip
        points = [
            Point(x=boundary_points[0], y=boundary_points[2], z=0.0),
            Point(x=boundary_points[0], y=boundary_points[3], z=0.0),
            Point(x=boundary_points[1], y=boundary_points[3], z=0.0),
            Point(x=boundary_points[1], y=boundary_points[2], z=0.0),
            Point(x=boundary_points[0], y=boundary_points[2], z=0.0),
        ]

        marker.points = points

        # Define the color and scale of the line strip
        marker.scale.x = 0.01  # Line width

        color = ColorRGBA()
        color.r = 1.0
        color.g = 0.0
        color.b = 0.0
        color.a = 1.0
        marker.color = color

        self.debug_publisher.publish(marker)

    def car_radius_publish(self, radius: float):
        marker = Marker()
        marker.header = Header()
        marker.header.frame_id = "body" + self.car_str
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "car_radius"
        marker.id = 1 + self.car_i
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Define the position for the sphere
        marker.pose.position = Point(x=0.0, y=0.0, z=0.0)
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # Define the scale (diameter) of the sphere
        marker.scale.x = 1.0 * radius
        marker.scale.y = 1.0 * radius
        marker.scale.z = 1.0 * radius

        # Define the color and transparency of the sphere
        color = ColorRGBA()
        color.r = self.rgb[0]
        color.g = self.rgb[1]
        color.b = self.rgb[2]
        color.a = 0.5  
        marker.color = color
        marker.frame_locked = True

        self.debug_publisher.publish(marker)
    
    def barrier_publisher(self, point: List[float]):
        marker = Marker()
        marker.header = Header()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "barrier"
        marker.id = 101 + self.car_i
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Define the position for the sphere
        marker.pose.position = Point(x=point[0], y=point[1], z=0.0)
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        
        # Define the scale (diameter) of the sphere
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        # Define the color and transparency of the sphere
        color = ColorRGBA()
        color.r = self.rgb[0]
        color.g = self.rgb[1]
        color.b = self.rgb[2]
        color.a = 0.5  
        marker.color = color

        self.debug_publisher.publish(marker)
    
    

def main(args=None):
    rclpy.init(args=args)
    node = CollisionAvoidance()

    # Run control alg
    node.run()

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()