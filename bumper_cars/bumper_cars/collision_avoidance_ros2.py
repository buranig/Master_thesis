
#!/usr/bin/env python3
    
import rclpy
import yaml
from rclpy.node import Node

# import control models
# from planner import utils
from dwa_dev.DWA import DWA_algorithm as DWA
# from cbf_dev. import CBF_robotarium
# from cbf_dev import CBF_simple
# from cbf_dev import C3BF
# from lbp_dev import LBP
# from mpc_dev import MPC

controller_map = {
    "dwa": DWA,
    "DWA": DWA
    # "c3bf": C3BF,
    # "C3BF": C3BF,
    # "lbp": LBP,
    # "LBP": LBP,
    # "mpc": MPC,
    # "MPC": MPC
}


from lar_msgs.msg import CarControlStamped
from lar_msgs.srv import EnvState, CarCommand


class CollisionAvoidance(Node):
    def __init__(self):
        super().__init__('ca')
        self.declare_parameter('car_i', 1)
        self.declare_parameter('car_yaml', rclpy.Parameter.Type.STRING)
        self.declare_parameter('alg', rclpy.Parameter.Type.STRING)

        # Init variables
        self.car_i = self.get_parameter('car_i').value
        self.car_yaml = self.get_parameter('car_yaml').value
        self.car_alg = self.get_parameter('alg').value

        self.car_str = '' if self.car_i == 1 else str(self.car_i)

        # Init controller
        self.algorithm = controller_map[self.car_alg](self.car_yaml,self.car_i)

        # Service to query state
        self.state_cli = self.create_client(EnvState, 'env_state')
        self.cmd_cli = self.create_client(CarCommand, 'car_cmd')
        while not self.state_cli.wait_for_service(timeout_sec=1.0) or not self.cmd_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        
        # Initialize service messages
        self.state_req = EnvState.Request()
        self.cmd_req = CarCommand.Request()
        self.cmd_req.car = self.car_i - 1

        self.publisher_ = self.create_publisher(CarControlStamped, '/sim/car'+self.car_str+'/set/control', 10)

        

    def state_request(self):
        self.future = self.state_cli.call_async(self.state_req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
        
    def cmd_request(self):
        self.cmd_future = self.cmd_cli.call_async(self.cmd_req)
        rclpy.spin_until_future_complete(self, self.cmd_future)
        return self.cmd_future.result()
        

    def run(self):        
        while rclpy.ok():
            # set_comm = CarControlStamped()

            des_action = self.cmd_request()
            curr_state = self.state_request()
            
            
            self.publisher_.publish(des_action.cmd)
        

def main(args=None):
    rclpy.init(args=args)
    node = CollisionAvoidance()

    # Run control alg
    node.run()
    
    node.destroy_node()
    rclpy.shutdown()