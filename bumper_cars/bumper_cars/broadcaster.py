import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from custom_message.msg import MultiState
from tf2_ros import TransformBroadcaster
# For the parameter file
import pathlib
import json
import numpy as np

path = pathlib.Path('/home/giacomo/thesis_ws/src/bumper_cars/params.json')
# Opening JSON file
with open(path, 'r') as openfile:
    # Reading from json file
    json_object = json.load(openfile)

plot_traj = json_object["plot_traj"]
robot_num = json_object["robot_num"]
timer_freq = json_object["timer_freq"]

class FramePublisher(Node):

    def __init__(self):
        """
        A class that publishes transformation frames for multiple turtle robots.

        This class subscribes to a topic containing the pose information of multiple turtle robots,
        converts the pose information into transformation frames, and broadcasts the frames using tf2.

        Attributes:
            tf_broadcaster (TransformBroadcaster): The transform broadcaster object.
            subscription (Subscription): The subscription object for receiving turtle pose messages.

        """
        super().__init__('tf2_frame_publisher')

        # Initialize the transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscribe to a turtle{1}{2}/pose topic and call handle_turtle_pose
        # callback function on each message
        self.subscription = self.create_subscription(
            MultiState,
            f'/multi_fullstate',
            self.handle_turtle_pose,
            1)
        self.subscription  # prevent unused variable warning
    
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

    def handle_turtle_pose(self, msg: MultiState):

        for i in range(robot_num):
            t = TransformStamped()

            # Read message content and assign it to
            # corresponding tf variables
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'map'
            t.child_frame_id = f'robot_frame_{i}'

            # Turtle only exists in 2D, thus we get x and y translation
            # coordinates from the message and set the z coordinate to 0
            t.transform.translation.x = msg.multiple_state[i].x
            t.transform.translation.y = msg.multiple_state[i].y
            t.transform.translation.z = 0.0

            # For the same reason, turtle can only rotate around one axis
            # and this why we set rotation in x and y to 0 and obtain
            # rotation in z axis from the message
            q = self.get_quaternion_from_euler(0, 0, msg.multiple_state[i].yaw)
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]

            # Send the transformation
            self.tf_broadcaster.sendTransform(t)


def main():
    rclpy.init()
    node = FramePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()

if __name__ == '__main__':
    main()
