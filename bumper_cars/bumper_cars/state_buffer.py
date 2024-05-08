#!/usr/bin/env python3
    
import rclpy
from rclpy.node import Node
from lar_msgs.msg import CarStateStamped as State, CarControlStamped
from lar_msgs.srv import EnvState, CarCommand

class StateBuffer(Node):
    def __init__(self):
        super().__init__('state_buffer')

        # Initialise car number
        self.declare_parameter('carNumber', 0)
        self.car_amount = self.get_parameter('carNumber').value

        # Establish publishing service
        self.srv = self.create_service(EnvState, 'env_state', self._get_env_state)
        self.cmd_srv = self.create_service(CarCommand, 'car_cmd', self._get_cmd_i)

        self.cmd = [CarControlStamped()] * self.car_amount

        # Subscribe to each car's state and commands
        self.sub_state = []
        self.sub_cmd = []
        
        for i in range(int(self.car_amount)):
            car_str = '' if i==0 else str(i+1)
            self.sub_state.append(self.create_subscription(State, "/sim/car" + car_str + "/state",\
                                             lambda msg, car_i=i: self._received_state(msg, car_i), 10))
            self.sub_cmd.append(self.create_subscription(CarControlStamped, "/sim/car" + car_str + "/desired_control",\
                                             lambda msg, car_i=i: self._received_cmd(msg, car_i), 10))
        
        # Initialize states (at default) and update bit
        self.env_state = [State()] * self.car_amount
        self.updated = [False] * self.car_amount

    def _get_env_state(self, request, response):
        if all(self.updated):
            response.env_state = self.env_state
        return response
    
    def _get_cmd_i(self, request, response):
        response.cmd = self.cmd[request.car]
        return response
    

    def _received_state(self, msg, car_i: int):
        self.updated[car_i] = True
        self.env_state[car_i] = msg 
    
    def _received_cmd(self, msg, car_i: int):
        self.cmd[car_i] = msg
        

def main(args=None):
    rclpy.init(args=args)
    node = StateBuffer()

    rclpy.spin(node)
    node.destroy_node()
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()