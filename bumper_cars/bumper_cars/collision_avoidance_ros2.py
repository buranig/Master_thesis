#!/usr/bin/env python3
    
from typing import List
import rclpy
import yaml
from rclpy.node import Node
from bumper_cars.utils import car_utils as utils

# Plot goal in Rviz
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Pose2D
from std_msgs.msg import Header, ColorRGBA
import colorsys

import numpy as np

# Data storing
import time
import os

# import control models
from dwa_dev.DWA import DWA_algorithm as DWA
from cbf_dev.CBF import CBF_algorithm as CBF
from cbf_dev.C3BF import C3BF_algorithm as C3BF
from lbp_dev.LBP import LBP_algorithm as LBP
from mpc_dev.MPC import MPC_algorithm as MPC
from mpc_dev.MPC_GPU import MPC_GPU_algorithm as MPC_GPU

controller_map = {
    "dwa": DWA,
    "c3bf": C3BF,
    "cbf": CBF,
    "lbp": LBP,
    "mpc": MPC,
    "mpc_gpu": MPC_GPU
}

from lar_msgs.msg import CarControlStamped
from bumper_msgs.srv import EnvState, CarCommand, JoySafety, TrackState, WheelPosition, MainControl
from ros_g29_force_feedback.msg import ForceFeedback

class CollisionAvoidance(Node):
    """
    This class represents the Collision Avoidance node in the bumper cars system.
    It handles the collision avoidance algorithm and communication with other nodes.

    Args:
        None

    Attributes:
        car_i (int): The index of the car.
        car_yaml (str): The path to the car's YAML file.
        car_alg (str): The algorithm to be used for collision avoidance.
        gen_traj (bool): Flag indicating whether to generate a trajectory.
        source (str): The source of the car's control commands, can be 'sim' or 'real'.
        debug_rviz (bool): Flag indicating whether to enable debug visualization in RViz.
        dt (float): The time step for the simulation.
        car_str (str): The string representation of the car's index.
        algorithm (Controller): The collision avoidance algorithm instance.
        state_cli (Client): The client for querying the environment state.
        cmd_cli (Client): The client for querying the car's command.
        joy_cli (Client): The client for querying the joystick safety status.
        track_cli (Client): The client for querying the track state.
        publisher_ (Publisher): The publisher for sending control commands to the car.
        timer (Timer): The timer for updating the car's state.
        update_time (bool): Flag indicating whether to update the car's state.
        timer_debug (Timer): The timer for updating debug visualization in RViz.
        debug_publisher (Publisher): The publisher for sending debug markers to RViz.
    """

    def __init__(self):
        """
        Initialize the necessary configuration for CollisionAvoidance node.
        Args:
            None

        Returns:
            None
        """
        super().__init__('ca')
        self.declare_parameter('car_i', rclpy.Parameter.Type.INTEGER)
        self.declare_parameter('gen_traj', rclpy.Parameter.Type.BOOL)
        self.declare_parameter('source', rclpy.Parameter.Type.STRING)
        self.declare_parameter('car_yaml', rclpy.Parameter.Type.STRING)
        self.declare_parameter('alg', rclpy.Parameter.Type.STRING)
        self.declare_parameter('debug_rviz', rclpy.Parameter.Type.BOOL)
        self.declare_parameter('write_csv', rclpy.Parameter.Type.BOOL)
        
        ### Global variables

        # Init variables
        self.car_i = int(self.get_parameter('car_i').value)
        self.car_yaml = self.get_parameter('car_yaml').value
        self.car_alg = self.get_parameter('alg').value
        self.gen_traj = self.get_parameter('gen_traj').value
        self.source = self.get_parameter('source').value
        self.debug_rviz = self.get_parameter('debug_rviz').value
        self.write_data = self.get_parameter('write_csv').value

        with open(self.car_yaml, 'r') as openfile:
            yaml_object = yaml.safe_load(openfile)

        # Simulation params
        self.dt = yaml_object["Controller"]["dt"]

        self.car_str = '' if self.car_i == 0 else str(self.car_i + 1)

        # Save car's info
        csv_file = os.path.dirname(os.path.realpath(__file__))
        csv_file += "/../csv/data_"+str(self.car_i)+".csv" 
        if self.write_data:
            self.data_file = open(csv_file, mode='w', newline='')

        # Init controller
        self.alg = self.car_alg.lower() 
        self.algorithm = controller_map[self.alg](self.car_yaml,self.car_i)

        # Generate trajectory (if needed)
        if self.gen_traj:
            self.algorithm.compute_traj()

        # Service to query state
        self.state_cli = self.create_client(EnvState, 'env_state' + self.car_str)
        self.cmd_cli = self.create_client(CarCommand, 'car_cmd')
        self.joy_cli = self.create_client(JoySafety, 'joy_safety')
        self.track_cli = self.create_client(TrackState, 'track_pose')
        self.wheel_cli = self.create_client(WheelPosition, 'wheel_buffer')
        self.main_control_cli = self.create_client(MainControl, 'main_control')

        while not self.state_cli.wait_for_service(timeout_sec=1.0) or not self.cmd_cli.wait_for_service(timeout_sec=1.0) or not self.wheel_cli.wait_for_service(timeout_sec=1.0) or not self.main_control_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        
        # Initialize service messages
        self.state_req = EnvState.Request()
        self.cmd_req = CarCommand.Request()
        self.cmd_req.car = self.car_i
        self.joy_req = JoySafety.Request()
        self.track_req = TrackState.Request()
        self.wheel_pos_req = WheelPosition.Request()
        self.main_control_req = MainControl.Request()

        if self.source == 'sim':
            self.publisher_ = self.create_publisher(CarControlStamped, '/sim/car'+self.car_str+'/set/control', 10)
        else:
            self.publisher_ = self.create_publisher(CarControlStamped, '/car'+self.car_str+'/set/control', 10)
        
        if self.car_i == 0:
            self.force_pub = self.create_publisher(ForceFeedback, '/ff_target', 10)

        # Update timer
        self.timer = self.create_timer(self.dt, self.timer_callback)
        self.update_time = False

        # Get track position
        track_msg = self.track_request()
        while track_msg.updated == False:
            print("Getting track position")
            track_msg = self.track_request()
        track_pos = track_msg.track_state

        # Set track boundaries offset
        track_offset = [track_pos.x, track_pos.y, track_pos.theta]
        self.algorithm.boundary_points[1][0] = self.algorithm.boundary_points[1][0]*2.0/5.0
        self.algorithm.boundary_points[2][0] = self.algorithm.boundary_points[2][0]*2.0/5.0
        self.algorithm.offset_track(track_offset)
        
        self.send_once = False
        # Draw in Rviz
        if self.debug_rviz:   
            # Colormap for debug
            hue = (self.car_i * 0.618033988749895) % 1
            self.rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)

            self.timer_debug = self.create_timer(1.0, self.debug_callback)
            self.debug_publisher = self.create_publisher(MarkerArray, '/debug_markers'+self.car_str, 10)

            
    def timer_callback(self) -> None:
        """
        Timer callback function.
        Updates the flag to indicate that it's time to send another command to the car.
        Args:
            None
        Returns:
            None
        """
        self.update_time = True

    def debug_callback(self) -> None:
        """
        Debug callback function.
        Publishes debug markers to RViz for visualization.
        Args:
            None
        Returns:
            None
        """
        self.markers=MarkerArray()
        if self.car_i == 0:
            self.publish_map(self.algorithm.boundary_points)

        if self.alg == "cbf" or self.alg == "c3bf":
            self.car_radius_publish(self.algorithm.safety_radius)
        self.debug_publisher.publish(self.markers)

    def main_control_request(self):
        self.main_control_future = self.main_control_cli.call_async(self.main_control_req)
        rclpy.spin_until_future_complete(self, self.main_control_future)
        if not self.main_control_future.done():
            print("Timeout")
        return self.main_control_future.result().main_control

    def wheel_pos_request(self):
        self.wheel_pos_future = self.wheel_cli.call_async(self.wheel_pos_req)
        rclpy.spin_until_future_complete(self, self.wheel_pos_future)
        if not self.wheel_pos_future.done():
            print("Timeout")
        return self.wheel_pos_future.result().wheel_position

    def state_request(self) -> EnvState.Response:
        """
        Sends a request to get the current environment state.
        Args:
            None
        Returns:
            EnvState.Response: The response from the environment state service.
        """
        self.future = self.state_cli.call_async(self.state_req)
        rclpy.spin_until_future_complete(self, self.future)
        if not self.future.done():
            print("Timeout")
        return self.future.result()
        
    def cmd_request(self) -> CarCommand.Response:
        """
        Sends a request to get the desired command for a car.
        Args:
            None
        Returns:
            CarCommand.Response: The response from the car command service.
        """
        self.cmd_future = self.cmd_cli.call_async(self.cmd_req)
        rclpy.spin_until_future_complete(self, self.cmd_future)
        return self.cmd_future.result()

    def joy_request(self) -> bool:
        """
        Sends a request to the joystick safety service to see if Collision Avoidance should be enabled.
        Args:
            None
        Returns:
            bool: Whether CA should be active (True) or not (False).
        """
        self.joy_future = self.joy_cli.call_async(self.joy_req)
        rclpy.spin_until_future_complete(self, self.joy_future)
        return self.joy_future.result().ca_activated

    def track_request(self) -> TrackState.Response:
        """
        Sends a request to get the position of the track's center.
        Args:
            None
        Returns:
            TrackState.Response: The response from the track state service.
        """
        self.track_future = self.track_cli.call_async(self.track_req)
        rclpy.spin_until_future_complete(self, self.track_future)
        return self.track_future.result()

    def run(self) -> None:
        """
        Main function to run the CollisionAvoidance node.
        Args:
            None
        Returns:
            None
        """
        ca_active = True
        main_control = False
        self.start_time = time.time()
        self.lap_time = time.time()

        # Lap counting
        self.prev_lap = -1
        self.lap_number = 0
        self.just_changed = True

        while rclpy.ok():
            try:
                # Update the current state of the car (and do rcply spin to update timer)
                curr_state = self.state_request()
                if curr_state.updated == False:
                    continue

                curr_car = curr_state.env_state[self.car_i] # Select desired car
                updated_state = utils.carStateStamped_to_State(curr_car)
                self.algorithm.set_state(updated_state)

                self.__update_lap(curr_car)
                
                # Skip rest if no update
                if self.update_time == False:
                    continue

                # Check desired action
                des_action = self.cmd_request()

                # Check if CA is activated
                if self.car_i == 0: # Only move car 0
                    ca_active = self.joy_request()
                    main_control = self.main_control_request()

                if ca_active:
                    # Set desired action as a goal for the CA algorithm
                    self.algorithm.set_goal(des_action.cmd)

                    # Compute safe control input
                    pre = time.time()
                    cmd_out = self.algorithm.compute_cmd(curr_state.env_state)
                    it_time = time.time() - pre
                else:
                    cmd_out = des_action.cmd
                    it_time = 0.0

                # Send command to car
                v_long = curr_state.env_state[self.car_i].vel_x*np.cos(curr_state.env_state[self.car_i].turn_angle) + curr_state.env_state[self.car_i].vel_y*np.sin(curr_state.env_state[self.car_i].turn_angle)
                
                if cmd_out.throttle<=-0.01 and v_long>0.001:
                    cmd_out.throttle = 0.0
                    self.publisher_.publish(cmd_out)
                    self.send_once = False
                elif cmd_out.throttle<=-0.01 and v_long<0.001 and v_long>-0.001:
                    # self.publisher_.publish(cmd_out)
                    if not self.send_once:
                        throttle_buf = cmd_out.throttle
                        cmd_out.throttle = 0.0
                        self.publisher_.publish(cmd_out)
                        self.send_once = True
                    cmd_out.throttle = throttle_buf
                    self.publisher_.publish(cmd_out)
                else:
                    self.publisher_.publish(cmd_out)
                    self.send_once = False

                # Draw debug info in Rviz
                if self.debug_rviz:
                    self.markers=MarkerArray()
                    if self.alg == "dwa" or self.alg == "lbp" or self.alg == "mpc":
                        self.publish_trajectory(self.algorithm.trajectory)
                    elif self.alg == "cbf" or self.alg == "c3bf":
                        self.barrier_publisher(self.algorithm.closest_point)
                    elif self.alg=="mpc_gpu":
                        self.publish_trajectories(self.algorithm.trajectory, self.algorithm.best_i)
                    self.debug_publisher.publish(self.markers)

                # Add info to csv
                if self.write_data:
                    self.write_csv(des_action.cmd, cmd_out, it_time, curr_car)
                
                if self.car_i == 0 and (main_control or ca_active):
                    self.publish_ff(cmd_out.steering, 0.2)
                elif self.car_i == 0 and (main_control==False or ca_active==False):
                    self.publish_ff(0.0, 0.0)


                # Wait for next period
                self.update_time = False
            except KeyboardInterrupt:
                if self.write_data:
                    self.data_file.close()
                    print("\033[92m  Successfully closed csv file \033[00m")
                break

    def publish_ff(self, steering, torque):
        msg = ForceFeedback()
        msg.header.stamp.sec = 0
        msg.header.stamp.nanosec = 0
        msg.header.frame_id = ''
        msg.position = -steering
        msg.torque = torque
        self.force_pub.publish(msg)

    def publish_map(self, boundary_points: List[float]) -> None:
        """
        Publishes the boundary points of the track as a line strip marker in RViz.
        Args:
            boundary_points (List[float]): The boundary points of the track.
        Returns:
            None
        """
        marker = Marker()
        marker.header = Header()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "world_map"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        # Define the points for the line strip
        points = [
            Point(x=boundary_points[0][0], y=boundary_points[0][1], z=0.0),
            Point(x=boundary_points[1][0], y=boundary_points[1][1], z=0.0),
            Point(x=boundary_points[2][0], y=boundary_points[2][1], z=0.0),
            Point(x=boundary_points[3][0], y=boundary_points[3][1], z=0.0),
            Point(x=boundary_points[0][0], y=boundary_points[0][1], z=0.0)
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

        self.markers.markers.append(marker)
        
    def car_radius_publish(self, radius: float):
        """
        Publishes the safety radius of the car as a sphere marker in RViz.
        Args:
            radius (float): The safety radius of the car.
        Returns:
            None
        """
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

        self.markers.markers.append(marker)
    
    def barrier_publisher(self, point: List[float]) -> None:
        """
        Publishes a marker on boundary of arena to visualize the active barrier function in RViz.

        Args:
            point (List[float]): The position of the active barrier in the world frame.

        Returns:
            None
        """
        marker = Marker()
        marker.header = Header()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "barrier"
        marker.id = 51 + self.car_i
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

        self.markers.markers.append(marker)
    
    def publish_trajectory(self, trajectory: List[List[float]]) -> None:
        """
        Publishes a trajectory as a line strip marker.

        Args:
            trajectory (List[List[float]]): The trajectory to be published. Each element in the list represents a point in the trajectory, specified as [x, y].

        Returns:
            None
        """
        marker = Marker()
        marker.header = Header()
        frame = "world" if self.alg == "mpc" else "body" + self.car_str
        marker.header.frame_id = frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "predicted_trajectory"
        marker.id = 100 + self.car_i
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # Define the points for the line strip
        points = []
        for point in trajectory:
            p = Point(x=point[0], y=point[1], z=0.0)
            points.append(p)
        marker.points = points
        # Define the color and scale of the line strip
        marker.scale.x = 0.01  # Line width
        color = ColorRGBA()
        color.r = self.rgb[0]
        color.g = self.rgb[1]
        color.b = self.rgb[2]
        color.a = 1.0
        marker.color = color
        
        self.markers.markers.append(marker)

    def publish_trajectories(self, trajectory: List[List[List[float]]], best: int) -> None:
        """
        Publishes all trajectoreis as a line strip marker.

        Args:
            trajectory (List[List[float]]): The trajectory to be published. Each element in the list represents a point in the trajectory, specified as [x, y].
            best (int): The index of the best trajectory.

        Returns:
            None
        """

        for i in range(len(trajectory)):
            marker = Marker()
            marker.header = Header()
            frame = "world"
            marker.header.frame_id = frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "predicted_trajectory"
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
        
            marker.id = 100 + i
            # Define the points for the line strip
            points = []
            
            for point in trajectory[i, :, :]:
                p = Point(x=point[0].item(), y=point[1].item(), z=0.0)
                points.append(p)
            marker.points = points
            # Define the color and scale of the line strip
            marker.scale.x = 0.005  # Line width
            color = ColorRGBA()
            color.r = 1.0
            color.g = 0.0
            color.b = 0.0
            color.a = 1.0
            if i == best:
                marker.scale.x = 0.01  # Line width
                color.r = 0.0
                color.g = 1.0
                color.b = 0.0
            marker.color = color
            self.markers.markers.append(marker)

    def __update_lap(self, car_state):
        # if car_state.pos_x>-0.05 and car_state.pos_x<0.05 and car_state.pos_y > -0.8 and car_state.pos_y < 0.0:
        if car_state.pos_x>0.2 and car_state.pos_x<0.4 and car_state.pos_y > -1.75 and car_state.pos_y < -0.85:
            if not self.just_changed:
                self.just_changed = True
                self.lap_number +=1
        else:
            self.just_changed = False

        # if car_state.turn_angle % (2*3.14) > 0.5:
        #     self.just_changed = False
        # elif car_state.turn_angle % (2*3.14) <= 0.5 and not self.just_changed:
        #     self.lap_number +=1
        #     self.just_changed = True
    

    def write_csv(self, des_action, cmd_out, it_time, curr_car):
        """
        Writes data to a CSV file.

        Args:
            des_action (DesiredAction): The desired action.
            cmd_out (CommandOutput): The command output.
            it_time (float): The iteration time.

        Returns:
            None
        """
        if self.prev_lap != self.lap_number:
            self.prev_lap = self.lap_number
            self.lap_time = time.time()

        # Skip first lap
        if self.lap_number == 0:
            return

        x = curr_car.pos_x
        y = curr_car.pos_y

        cum_time = time.time() - self.start_time
        lap_time = time.time() - self.lap_time
        isd = (des_action.steering - cmd_out.steering)**2 + (des_action.throttle - cmd_out.throttle)**2

        self.data_file.write(str(self.car_i) + "," +str(cum_time) + "," + str(lap_time) + "," + str(self.lap_number) + "," + str(it_time) + "," + str(isd) + "," + str(x) + "," + str(y) +  "\n")

def main(args=None):
    rclpy.init(args=args)
    node = CollisionAvoidance()

    # Run control alg
    node.run()

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()