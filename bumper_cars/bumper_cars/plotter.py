#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
from matplotlib import pyplot as plt
from custom_message.msg import FullState, Path, MultiplePaths, MultiState, Coordinate
# import the string type for ROS2 python to publish
from std_msgs.msg import String
import message_filters
import numpy as np
import time

# For the parameter file
import pathlib
import json

path = pathlib.Path('/home/giacomo/thesis_ws/src/bumper_cars/params.json')
# Opening JSON file
with open(path, 'r') as openfile:
    # Reading from json file
    json_object = json.load(openfile)


robot_num = json_object["robot_num"]
controller_type = json_object["Controller"]["controller_type"]
safety = json_object["safety"]
width = json_object["width"]
height = json_object["height"]
timer_freq = json_object["timer_freq"]
debug = False
plot_traj = False

class Config:
    """
    simulation parameter class
    """

    def __init__(self):
        self.robot_width = 1.45  # [m] for collision check
        self.robot_length = 2.9  # [m] for collision check
        # obstacles [x(m) y(m), ....]

        self.trail_length = 5

config = Config()

class Plotter(Node):

    def __init__(self):
        super().__init__("plotter")

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
        model_type = self.get_parameter('model_type').get_parameter_value().string_array_value

        self.multi_state = MultiState()
        self.multi_traj = MultiplePaths()

        # Initializing the robots TODO add in the launch file parameters the initial control--> here it's hardcoded
        for i in range(robot_num):
            self.multi_state.multiple_state.append(FullState(x=x0[i], y=y0[i], yaw=yaw[i], v=v[i],
                                                             omega=omega[i], delta=0.0, throttle=0.0))
            self.multi_traj.multiple_path.append(Path(path=[Coordinate(x=0.0, y=0.0)]))

        multi_state_sub = message_filters.Subscriber(self, MultiState, "/multi_fullstate")
        self.multi_path_sub = message_filters.Subscriber(self, MultiplePaths, "/robot_multi_traj")

        # Creating a publisher to publish a string message on the topic "/plotter" only if the debug flag is on
        if debug:
            self.publisher_ = self.create_publisher(String, "/plotter", 10)
            # define a class string message with the message type String with the value "plotter"
            self.msg = String(data="plotter")

        self.states_x_buf = np.zeros((robot_num, 1))
        self.states_y_buf = np.zeros((robot_num, 1))
        
        self.controller_type = controller_type

        self.width = width
        self.heigth = height

        if self.controller_type == "random_walk":
            ts = message_filters.ApproximateTimeSynchronizer([multi_state_sub], 10, 1, allow_headerless=True)
            ts.registerCallback(self.general_callback)

            # timer
            self.timer = self.create_timer(timer_freq, self.plotter_callback)

        elif self.controller_type == "random_harem":
            ts = message_filters.ApproximateTimeSynchronizer([multi_state_sub], 10, 1, allow_headerless=True)
            ts.registerCallback(self.general_callback)

            # timer
            self.timer = self.create_timer(timer_freq, self.plotter_callback)
            
        else:
            ts = message_filters.ApproximateTimeSynchronizer([multi_state_sub, self.multi_path_sub], 10, 1, allow_headerless=True)
            ts.registerCallback(self.complete_callback)

            # timer
            self.timer = self.create_timer(timer_freq, self.complete_plotter_callback)

        self.get_logger().info("Plotter has been started")

    def general_callback(self, multi_state: MultiState):
        self.multi_state = multi_state

    def complete_callback(self, multi_state: MultiState, multi_traj: MultiplePaths):
        self.multi_state = multi_state
        self.multi_traj = multi_traj

    def plotter_callback(self):        
        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])

        for i in range(robot_num):
            self.plot_robot(self.multi_state.multiple_state[i])
            if debug:
                self.get_logger().info("robot" + str(i) + ", x: " + str(self.multi_state.multiple_state[i].x) + ", " +
                                    "y: " + str(self.multi_state.multiple_state[i].y) + ", " +
                                    "yaw: " + str(self.multi_state.multiple_state[i].yaw) + ", " +
                                    "linear velocity: " + str(self.multi_state.multiple_state[i].v))

        self.plot_map()
        plt.axis("equal")
        plt.pause(0.000001)

        #publish the self.msg string message on the topic "/plotter"
        if debug:
            self.publisher_.publish(self.msg)


    def complete_plotter_callback(self):

        buf = np.zeros((robot_num, 2))
        for i in range(robot_num):
            buf[i, 0] = self.multi_state.multiple_state[i].x
            buf[i, 1] = self.multi_state.multiple_state[i].y

        self.states_x_buf = np.concatenate((self.states_x_buf, buf[:,0].reshape(robot_num,1)), axis=1)
        self.states_y_buf = np.concatenate((self.states_y_buf, buf[:,1].reshape(robot_num,1)), axis=1)

        if len(self.states_x_buf[0, :]) > config.trail_length or len(self.states_y_buf[0, :]) > config.trail_length:
            self.states_x_buf = np.delete(self.states_x_buf, 0, 1)
            self.states_y_buf = np.delete(self.states_y_buf, 0, 1)
        
        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        
        for i in range(robot_num):

            self.plot_situation(self.multi_state.multiple_state[i], self.multi_traj.multiple_path[i], 
                                self.states_x_buf[i, :], self.states_y_buf[i, :])
            if debug:
                self.get_logger().info("robot" + str(i) + ", x: " + str(self.multi_state.multiple_state[i].x) + ", " +
                                    "y: " + str(self.multi_state.multiple_state[i].y) + ", " +
                                    "yaw: " + str(self.multi_state.multiple_state[i].yaw) + ", " +
                                    "linear velocity: " + str(self.multi_state.multiple_state[i].v))

        self.plot_map()
        plt.axis("equal")
        plt.pause(0.000001)
            
        #publish the self.msg string message on the topic "/plotter"
        if debug:
            self.publisher_.publish(self.msg)

    def plot_situation(self, state: FullState, trajectory, state_buf_x, state_buf_y):
        if plot_traj:
            self.plot_path(trajectory)
            plt.scatter(state_buf_x[:], state_buf_y[:], marker='.', s=4)
            plt.plot(state.x, state.y, 'b.')
        self.plot_robot(state)


    def plot_robot(self, fullstate: FullState):
        self.plot_rect(fullstate.x, fullstate.y, fullstate.yaw, config)
        self.plot_cs_robot(fullstate.x, fullstate.y, fullstate.yaw, length=2)
        self.plot_arrow(fullstate.x, fullstate.y, fullstate.yaw + fullstate.delta, length=4)

    def plot_map(self):
        corner_x = [-self.width/2.0, self.width/2.0, self.width/2.0, -self.width/2.0, -self.width/2.0]
        corner_y = [self.heigth/2.0, self.heigth/2.0, -self.heigth/2.0, -self.heigth/2.0, self.heigth/2.0]

        plt.plot(corner_x, corner_y)

    def plot_path(self, path: Path):
        x = []
        y = []
        for coord in path.path:
            x.append(coord.x)
            y.append(coord.y)
        plt.scatter(x, y, marker='.', s=10)
        plt.scatter(x[0], y[0], marker='x', s=20)


    def plot_arrow(self, x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                head_length=width, head_width=width)
    
    def plot_cs_robot(self, x, y, yaw, length=1, width=0.1):  # pragma: no cover
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                head_length=width, head_width=width)
        plt.arrow(x, y, length * -math.sin(yaw), length * math.cos(yaw),
                head_length=width, head_width=width)
        
    def plot_rect(self, x, y, yaw, config):  # pragma: no cover
        outline = np.array([[-config.robot_length / 2, config.robot_length / 2,
                                (config.robot_length / 2), -config.robot_length / 2,
                                -config.robot_length / 2],
                            [config.robot_width / 2, config.robot_width / 2,
                                - config.robot_width / 2, -config.robot_width / 2,
                                config.robot_width / 2]])
        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                            [-math.sin(yaw), math.cos(yaw)]])
        outline = (outline.T.dot(Rot1)).T
        outline[0, :] += x
        outline[1, :] += y
        plt.plot(np.array(outline[0, :]).flatten(),
                    np.array(outline[1, :]).flatten(), "-k")
        
def main(args=None):
    rclpy.init(args=args)

    node = Plotter()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

# colcon build --symlink-install to be able to run the node without building it
# or
# colcon build in the source file and the source ~/.bashrc !!!