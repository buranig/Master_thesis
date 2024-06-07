import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Header
from bumper_msgs.srv import JoySafety
from bumper_msgs.srv import EnvState, CarCommand, JoySafety, TrackState, WheelPosition, MainControl

class WheelJoystickRemap(Node):
    def __init__(self):
        super().__init__('wheel_joystick_remap')
        
        self.subscription = self.create_subscription(
            Joy,
            '/joy',
            self.joy_callback,
            10
        )
        self.declare_parameter('car_i', rclpy.Parameter.Type.INTEGER)
        self.car_i = int(self.get_parameter('car_i').value)

        self.publisher_ = self.create_publisher(Joy, '/joy_remap', 10)

        if self.car_i == 0:
            self.wheel_service = self.create_service(WheelPosition, 'wheel_buffer', self._get_env_state)
            self.main_control_service = self.create_service(MainControl, 'main_control', self._get_main_control)
        self.wheel_position = 0.0

        self.button_pressed = False
        self.main_control = False

    def _get_env_state(self, _, response):
        response.wheel_position = self.wheel_position
        return response
    
    def _get_main_control(self, _, response):
        response.main_control = self.main_control
        return response
    
    def joy_callback(self, msg: Joy):
        if len(msg.axes)>6:
            self.publisher_.publish(msg)

            if msg.buttons[4]==1 and self.button_pressed==False:
                self.button_pressed = True
                self.main_control = not self.main_control
            elif msg.buttons[4] == 0:
                self.button_pressed = False

        else:
            if msg.buttons[5]==1 and self.button_pressed==False:
                self.button_pressed = True
                self.main_control = not self.main_control
            elif msg.buttons[5] == 0:
                self.button_pressed = False

            self.wheel_position = msg.axes[0]
            new_msg = self.wheel_to_joystick(msg)
            self.publisher_.publish(new_msg)
        
    def remap(self, old_val, old_min, old_max, new_min, new_max):
        return (new_max - new_min)*(old_val - old_min) / (old_max - old_min) + new_min
    
    def wheel_to_joystick(self, msg: Joy):

        new_msg = Joy()
        new_msg.header = msg.header
        new_msg.axes = [0.0, 
                        self.remap(msg.axes[2]-msg.axes[3], -2.0, 2.0, -1.0, 1.0),
                        0.0,
                        msg.axes[0],
                        0.0,
                        0.0,
                        msg.axes[4],
                        msg.axes[5]]
        new_msg.buttons = [int(msg.buttons[0]),
                           int(msg.buttons[2]),
                           int(msg.buttons[3]),
                           int(msg.buttons[1]),
                           int(msg.buttons[5]),
                           int(msg.buttons[4]),
                           int(msg.buttons[7]),
                           int(msg.buttons[6]),
                           int(msg.buttons[1]),
                           int(msg.buttons[8]),
                           int(msg.buttons[9]),
                           int(msg.buttons[11]),
                           int(msg.buttons[10]),
                           0]
        return new_msg


def main(args=None):
    rclpy.init(args=args)
    wheel_joystick_remap = WheelJoystickRemap()
    rclpy.spin(wheel_joystick_remap)
    wheel_joystick_remap.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()