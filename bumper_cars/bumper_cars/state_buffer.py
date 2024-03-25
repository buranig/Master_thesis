
#!/usr/bin/env python3
    
import rclpy
from rclpy.node import Node
from lar_msgs.msg import CarStateStamped as State
from lar_msgs.srv import EnvState

class StateBuffer(Node):
    def __init__(self):
        super().__init__('state_buffer')

        # Initialise car number
        self.declare_parameter('carNumber', 1)
        self.car_amount = self.get_parameter('carNumber').value

        # Establish publishing service
        self.srv = self.create_service(EnvState, 'env_state', self._get_env_state)

        # Subscribe to each car's state
        self.sub = []
        
        for i in range(int(self.car_amount)):
            car_str = '' if i==0 else str(i+1)
            self.sub.append(self.create_subscription(State, "/sim/car" + car_str + "/state",\
                                             lambda msg, car_i=i: self._received_state(msg, car_i), 10))
        
        # Initialize states (at default) and update bit
        response = State()
        self.env_state = [response] * self.car_amount
        self.updated = [False] * self.car_amount

    def _get_env_state(self, request, response):
        if all(self.updated):
            response.env_state = self.env_state
        return response
    

    def _received_state(self, msg, car_i: int):
        self.updated[car_i] = True
        self.env_state[car_i] = msg 


def main(args=None):
    rclpy.init(args=args)
    node = StateBuffer()

    rclpy.spin(node)
    node.destroy_node()
    
    rclpy.shutdown()