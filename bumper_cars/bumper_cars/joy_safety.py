import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from bumper_msgs.srv import JoySafety

class JoySafetyNode(Node):
    def __init__(self):
        super().__init__('joy_safety_node')

        # Initialise car number
        self.declare_parameter('car_str', '')
        self.car_str = self.get_parameter('car_str').value

        self.subscription = self.create_subscription(
            Joy,
            '/joy'+car_str,
            self.joy_callback,
            10
        )
        # Establish publishing service
        self.srv = self.create_service(JoySafety, 'joy_safety', self._get_joy_state)

        # Establish safety variables
        self.button_pressed = False
        self.ca_activated = True

    def joy_callback(self, msg):
        if msg.buttons[5] == 1 and self.button_pressed == False:
            self.button_pressed = True
            self.ca_activated = not self.ca_activated
            if self.ca_activated:
                self.get_logger().info('[ CA MODE CHANGE ]: collision avoidance for car'+self.car_str+' ACTIVATED')
            else:
                self.get_logger().info('[ CA MODE CHANGE ]: collision avoidance for car'+self.car_str+' DISABLED')
        elif msg.buttons[5] == 0:
            self.button_pressed = False

    def _get_joy_state(self, _, response):
        response.ca_activated = self.ca_activated
        return response


def main(args=None):
    rclpy.init(args=args)
    joy_safety_node = JoySafetyNode()
    rclpy.spin(joy_safety_node)
    joy_safety_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()