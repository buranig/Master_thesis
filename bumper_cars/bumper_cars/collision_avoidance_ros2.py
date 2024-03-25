
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


class CollisionAvoidance(Node):
    def __init__(self):
        super().__init__('ca')
        self.declare_parameter('car_i', 1)
        self.declare_parameter('car_yaml', rclpy.Parameter.Type.STRING)
        self.declare_parameter('alg', rclpy.Parameter.Type.STRING)

        self.car_i = self.get_parameter('car_i').value
        self.car_yaml = self.get_parameter('car_yaml').value
        self.car_alg = self.get_parameter('alg').value

        self.algorithm = controller_map[self.car_alg]




def main(args=None):
    rclpy.init(args=args)
    node = CollisionAvoidance()

    rclpy.spin(node)
    node.destroy_node()
    
    rclpy.shutdown()