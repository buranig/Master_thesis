import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from bumper_msgs.srv import JoySafety

class JoySafetyNode(Node):
    """
    Node responsible for handling joystick input and toggles collision avoidance for a car.
    """

    def __init__(self):
        """
        Initializes the JoySafetyNode class.
        """
        super().__init__('joy_safety_node')

        # Initialise car number
        self.declare_parameter('car_str', '')
        self.car_str = self.get_parameter('car_str').value

        self.subscription = self.create_subscription(
            Joy,
            '/joy'+self.car_str,
            self.joy_callback,
            10
        )
        # Establish publishing service
        self.srv = self.create_service(JoySafety, 'joy_safety'+self.car_str, self._get_joy_state)

        # Establish safety variables
        self.button_pressed = False
        self.ca_activated = True

    def joy_callback(self, msg) -> None:
        """
        Callback function for processing joystick input. 
        
        If R1 is pressed, toggles collision avoidance for the car.

        Args:
            msg (sensor_msgs.msg.Joy): The joystick message.

        Returns:
            None
        """
        if msg.buttons[5] == 1 and self.button_pressed == False:
            self.button_pressed = True
            self.ca_activated = not self.ca_activated
            if self.ca_activated:
                self.get_logger().info('[ CA MODE CHANGE ]: collision avoidance for car'+self.car_str+' ACTIVATED')
            else:
                self.get_logger().info('[ CA MODE CHANGE ]: collision avoidance for car'+self.car_str+' DISABLED')
        elif msg.buttons[5] == 0:
            self.button_pressed = False

    def _get_joy_state(self, _, response) -> JoySafety.Response:
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