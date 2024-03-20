#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from custom_message.msg import MultiState, FullState, State, MultiplePaths
import message_filters
import time
from geometry_msgs.msg import Pose, PoseArray, Point32
from sensor_msgs.msg import PointCloud
from nav_msgs import msg as nav_msgs_msg
from visualization_msgs.msg import Marker, MarkerArray
import math
import numpy as np

# For the parameter file
import pathlib
import json
from geometry_msgs.msg import Pose

path = pathlib.Path('/home/giacomo/thesis_ws/src/bumper_cars/params.json')
# Opening JSON file
with open(path, 'r') as openfile:
    # Reading from json file
    json_object = json.load(openfile)

plot_traj = json_object["plot_traj"]
robot_num = json_object["robot_num"]
timer_freq = json_object["timer_freq"]
controller_type = json_object["Controller"]["controller_type"]
L = json_object["Car_model"]["L"]

class Converter(Node):
    """
    This class represents a converter node that converts robot states to poses, steering poses, and markers.
    It subscribes to the "/multi_fullstate" topic to receive robot states and publishes the converted poses and markers.
    """

    def __init__(self):
        super().__init__("converter")

        self.declare_parameters(
            namespace='',
            parameters=[
                ('model_type', rclpy.Parameter.Type.STRING_ARRAY),
                ('x0', rclpy.Parameter.Type.DOUBLE_ARRAY),
                ('y0', rclpy.Parameter.Type.DOUBLE_ARRAY),
                ('yaw', rclpy.Parameter.Type.DOUBLE_ARRAY),
                ('v', rclpy.Parameter.Type.DOUBLE_ARRAY),
                ('omega', rclpy.Parameter.Type.DOUBLE_ARRAY)
            ]
        )

        x0 = self.get_parameter('x0').get_parameter_value().double_array_value
        y0 = self.get_parameter('y0').get_parameter_value().double_array_value
        yaw = self.get_parameter('yaw').get_parameter_value().double_array_value
        v = self.get_parameter('v').get_parameter_value().double_array_value
        omega = self.get_parameter('omega').get_parameter_value().double_array_value

        # self.pose_array_state = PoseArray()
        # self.pose_array_state.header.frame_id = "map"
        self.pose_array_steering = PoseArray()
        self.pose_array_steering.header.frame_id = "map"
        if controller_type == "DWA" or controller_type == "LBP" or controller_type == "MPC":
            self.marker_array = MarkerArray()
        if plot_traj:
            self.point_cloud = PointCloud()

        # Initializing the robots TODO add in the launch file parameters the initial control--> here it's hardcoded
        for i in range(robot_num):
            # pose = self.convert_to_pose(FullState(x=x0[i], y=y0[i], yaw=yaw[i], v=v[i],
            #                                                  omega=omega[i], delta=0.0, throttle=0.0))
            steering = self.convert_to_steering_pose(FullState(x=x0[i], y=y0[i], yaw=yaw[i], v=v[i], omega=omega[i],
                                                             delta=0.0, throttle=0.0))
            if controller_type == "DWA" or controller_type == "LBP" or controller_type == "MPC":
                marker = self.convert_to_marker(FullState(x=x0[i], y=y0[i], yaw=yaw[i], v=v[i], omega=omega[i],
                                                             delta=0.0, throttle=0.0), i)
                self.marker_array.markers.append(marker)
            # self.pose_array_state.poses.append(pose)
            self.pose_array_steering.poses.append(steering)
        
        # self.pose_array_pub = self.create_publisher(PoseArray, "/pose_array", 2)
        self.pose_array_steering_pub = self.create_publisher(PoseArray, "/pose_array_steering", 2)
        if controller_type == "DWA" or controller_type == "LBP" or controller_type == "MPC":
            self.marker_array_pub = self.create_publisher(MarkerArray, "/marker_array", 2)
        self.point_cloud_pub = self.create_publisher(PointCloud, "/point_cloud", 2)
        self.timer = self.create_timer(timer_freq, self.timer_callback)

        multi_state_subscriber = message_filters.Subscriber(self, MultiState, "/multi_fullstate")
        multi_trajectory_subscriber = message_filters.Subscriber(self, MultiplePaths, "/robot_multi_traj")
        if plot_traj:
            ts = message_filters.ApproximateTimeSynchronizer([multi_state_subscriber, multi_trajectory_subscriber], 6, 1, allow_headerless=True)
        else:
            ts = message_filters.ApproximateTimeSynchronizer([multi_state_subscriber], 6, 1, allow_headerless=True)
        ts.registerCallback(self.state_converter_callback)

        self.get_logger().info("Converter has been started")

    def get_quaternion_from_euler(self, roll, pitch, yaw):
        """
        Convert an Euler angle to a quaternion.
        
        Input
            :param roll: The roll (rotation around x-axis) angle in radians.
            :param pitch: The pitch (rotation around y-axis) angle in radians.
            :param yaw: The yaw (rotation around z-axis) angle in radians.
        
        Output
            :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
        """
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        
        return [qx, qy, qz, qw]
        
    def convert_to_pose(self, state: FullState):
        """
        Convert a robot state to a pose.

        Input
            :param state: The robot state.

        Output
            :return pose: The converted pose.
        """
        pose = Pose()
        pose.position.x = state.x
        pose.position.y = state.y
        pose.position.z = 0.0

        q = self.get_quaternion_from_euler(0, 0, state.yaw)
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]
        return pose
    
    def convert_to_steering_pose(self, state: FullState):
        """
        Convert a robot state to a steering pose.

        Input
            :param state: The robot state.

        Output
            :return pose: The converted steering pose.
        """
        pose = Pose()        
        pose.position.x = state.x
        pose.position.y = state.y
        pose.position.z = 0.0

        q = self.get_quaternion_from_euler(0, 0, state.yaw+state.delta)
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]
        return pose
    
    def convert_to_marker(self, state: FullState, idx):
        """
        Convert a robot state to a marker.

        Input
            :param state: The robot state.
            :param idx: The index of the marker.

        Output
            :return marker: The converted marker.
        """
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = idx
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose = self.convert_to_pose(state)
        # so that the marker is on the ground
        marker.pose.position.z = 0.5
        marker.scale.x = L
        marker.scale.y = L/2.0
        marker.scale.z = 1.0
        if state.v > 0.0:
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
        else:
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        marker.color.a = 1.0
        return marker

    def state_converter_callback(self, multi_state_in: MultiState, multi_trajectory_in: MultiplePaths = None):
        """
        Callback function for processing robot states.

        Input
            :param multi_state_in: The received multi-state message.
        """
        # self.pose_array_state = PoseArray()
        # self.pose_array_state.header.frame_id = "map"
        # self.pose_array_state.header.stamp = self.get_clock().now().to_msg()

        self.pose_array_steering = PoseArray()
        self.pose_array_steering.header.frame_id = "map"
        self.pose_array_steering.header.stamp = self.get_clock().now().to_msg()

        if plot_traj:
            self.point_cloud = PointCloud()
            self.point_cloud.header.frame_id = "map"
            self.point_cloud.header.stamp = self.get_clock().now().to_msg()

        self.marker_array = MarkerArray()

        for idx, state in enumerate(multi_state_in.multiple_state):
            # pose = self.convert_to_pose(state)
            steering = self.convert_to_steering_pose(state)
            if controller_type == "DWA" or controller_type == "LBP" or controller_type == "MPC":
                marker = self.convert_to_marker(state, idx)
                self.marker_array.markers.append(marker)
            # self.pose_array_state.poses.append(pose)
            self.pose_array_steering.poses.append(steering)
            

            if plot_traj:
                self.path_to_pointcloud(multi_trajectory_in.multiple_path[idx].path)

        
    def path_to_pointcloud(self, path):
        """
        Convert a path to a point cloud.

        Input
            :param path: The path.

        Output
            :return point_cloud: The converted point cloud.
        """
        for point in path:
            p = Point32()
            p.x = point.x
            p.y = point.y
            p.z = 0.0
            self.point_cloud.points.append(p)
        
    def timer_callback(self):
        """
        Timer callback function for publishing the converted poses and markers.
        """
        # self.pose_array_pub.publish(self.pose_array_state)
        self.pose_array_steering_pub.publish(self.pose_array_steering)
        if controller_type == "DWA"or controller_type == "LBP" or controller_type == "MPC":
            self.marker_array_pub.publish(self.marker_array)
        if plot_traj:
            self.point_cloud_pub.publish(self.point_cloud)


def main(args=None):
    rclpy.init(args=args)

    node = Converter()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()