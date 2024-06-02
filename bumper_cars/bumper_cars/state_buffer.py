#!/usr/bin/env python3
    
import rclpy
from rclpy.node import Node
from lar_msgs.msg import CarStateStamped, CarControlStamped
from bumper_msgs.srv import EnvState, CarCommand, TrackState
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Pose2D

class StateBufferNode(Node):
    """
    Node class for managing the state buffer for the bumper cars system.

    This class provides methods for retrieving the environment
    state, car commands, and track position.
    """

    def __init__(self):
        """
        Initialize the StateBufferNode class.

        This method initializes the necessary parameters, services, and subscriptions
        for managing the state, car commands, and track position buffers.

        Args:
            None

        Returns:
            None
        """
        super().__init__('state_buffer')

        # Initialise car number
        self.declare_parameter('carNumber', 0)
        self.car_amount = self.get_parameter('carNumber').value

        # Initialize if we are on sim or real
        self.declare_parameter('source', 'sim')
        self.source = self.get_parameter('source').value
        topic_str = "/sim/car" if self.source == 'sim' else "/car"
        track_str = "/sim/track/pose2d" if self.source == 'sim' else "/mocap/track/pose2d"

        # Establish publishing service
        self.srv = self.create_service(EnvState, 'env_state', self._get_env_state)
        self.cmd_srv = self.create_service(CarCommand, 'car_cmd', self._get_cmd_i)
        self.track_srv = self.create_service(TrackState, 'track_pose', self._get_track_pos)

        self.cmd = [CarControlStamped()] * self.car_amount

        # Subscribe to each car's state and commands
        self.sub_state = []
        self.sub_cmd = []

        self.sub_track = self.create_subscription(Pose2D, track_str,\
                                    self._received_track, qos_profile_sensor_data)


        for i in range(int(self.car_amount)):
            car_str = '' if i==0 else str(i+1)
            self.sub_state.append(self.create_subscription(CarStateStamped, topic_str + car_str + "/state",\
                                             lambda msg, car_i=i: self._received_state(msg, car_i), qos_profile_sensor_data))
            self.sub_cmd.append(self.create_subscription(CarControlStamped, topic_str + car_str + "/desired_control",\
                                             lambda msg, car_i=i: self._received_cmd(msg, car_i), qos_profile_sensor_data))
        
        # Initialize states (at default) and update bit
        self.env_state = [CarStateStamped()] * self.car_amount
        self.updated = [False] * self.car_amount
        self.track_pos = Pose2D()
        self.track_updated = False

    def _get_env_state(self, request, response) -> EnvState.Response:
        response.env_state = self.env_state
        response.updated = all(self.updated)
        return response
    
    def _get_cmd_i(self, request, response) -> CarCommand.Response:
        response.cmd = self.cmd[request.car]
        return response
    
    def _get_track_pos(self, request, response) -> TrackState.Response:
        response.track_state.x = self.track_pos.x
        response.track_state.y = self.track_pos.y
        response.track_state.theta = self.track_pos.theta
        response.updated = self.track_updated
        return response

    def _received_state(self, msg, car_i: int) -> None:
        self.updated[car_i] = True
        self.env_state[car_i] = msg 
    
    def _received_cmd(self, msg, car_i: int) -> None:
        self.cmd[car_i] = msg

    def _received_track(self, msg) -> None:
        self.track_updated = True
        self.track_pos.x = msg.x
        self.track_pos.y = msg.y
        self.track_pos.theta = msg.theta



def main(args=None):
    rclpy.init(args=args)
    node = StateBufferNode()

    rclpy.spin(node)
    node.destroy_node()
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()