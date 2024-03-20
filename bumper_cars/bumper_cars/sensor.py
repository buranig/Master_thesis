#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from custom_message.msg import MultiState, FullState, State
import message_filters
import time

# For the parameter file
import pathlib
import json

path = pathlib.Path('/home/giacomo/thesis_ws/src/bumper_cars/params.json')

# Opening JSON file
with open(path, 'r') as openfile:
    # Reading from json file
    json_object = json.load(openfile)

debug = False
robot_num = json_object["robot_num"]
timer_freq = json_object["timer_freq"]

class SensorMeasurement(Node):
    """
    Class representing a sensor measurement node.

    This class is responsible for initializing the sensor node, receiving sensor measurements,
    applying noise to the measurements, and publishing the multi-state information.

    Args:
        Node: The base class for creating a ROS node.

    Attributes:
        multi_state (MultiState): The multi-state information.
        multi_state_pub (Publisher): The publisher for multi-state information.
        timer (Timer): The timer for publishing multi-state information.
    """

    def __init__(self):
        """
        Initializes the Sensor class.

        This method sets up the necessary parameters and initializes the multi_state object.
        It also creates a publisher for multi-state messages and a timer for the timer_callback method.
        Finally, it sets up a subscriber for multi-state messages and registers the sensor_callback method.

        Args:
            None

        Returns:
            None
        """
        super().__init__("sensor")

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

        # Initializing the robots TODO add in the launch file parameters the initial control--> here it's hardcoded
        for i in range(robot_num):
            self.multi_state.multiple_state.append(FullState(x=x0[i], y=y0[i], yaw=yaw[i], v=v[i],
                                                             omega=omega[i], delta=0.0, throttle=0.0))

        self.multi_state_pub = self.create_publisher(MultiState, "/robot_multi_state", 2)
        self.timer = self.create_timer(timer_freq, self.timer_callback)

        multi_state_subscriber = message_filters.Subscriber(self, MultiState, "/multi_fullstate")

        ts = message_filters.ApproximateTimeSynchronizer([multi_state_subscriber], 6, 1, allow_headerless=True)
        ts.registerCallback(self.sensor_callback)

        self.get_logger().info("Sensor has been started")
        
    def sensor_callback(self, multi_state_in: MultiState):
        """
        Callback function for receiving sensor measurements.

        Args:
            multi_state_in (MultiState): The received multi-state information.
        """

        for i in range(robot_num):
            self.multi_state.multiple_state[i] = self.apply_noise(multi_state_in.multiple_state[i])
            if debug:
                self.get_logger().info("Publishing robot" + str(i) + " new state, x: " + str(multi_state_in.multiple_state[i].x) + ", " +
                                    "y: " + str(multi_state_in.multiple_state[i].y) + ", " +
                                    "theta: " + str(multi_state_in.multiple_state[i].yaw) + ", " +
                                    "linear velocity: " + str(multi_state_in.multiple_state[i].v))

    def timer_callback(self):
        """
        Callback function for the timer to publish multi-state information.
        """
        self.multi_state_pub.publish(self.multi_state)
    
    def apply_noise(self, fullstate: FullState):
        """
        Apply noise to the given full state.

        Args:
            fullstate (FullState): The full state to apply noise to.

        Returns:
            FullState: The full state with applied noise.
        """
        return fullstate
        
def main(args=None):
    rclpy.init(args=args)

    node = SensorMeasurement()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()