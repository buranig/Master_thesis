from launch import LaunchDescription
from launch_ros.actions import Node
from custom_message.msg import ControlInputs, State, FullState
# For the parameter file
import pathlib
import json

path = pathlib.Path('/home/giacomo/thesis_ws/src/bumper_cars/params.json')
init_path = pathlib.Path('/home/giacomo/thesis_ws/src/bumper_cars/init.json')
# Opening JSON file
with open(init_path, 'r') as openfile:
    # Reading from json file
    json_object = json.load(openfile)

model_type = json_object["model_type"]
x = json_object["x0"]
y = json_object["y0"]
yaw = json_object["yaw"]
v = json_object["v"]
omega = json_object["omega"]

def generate_launch_description():
    ld = LaunchDescription()

    robot_node = Node(
        package="bumper_cars",
        executable="model",
        name="robot_models",
        remappings=[
            ("/robot_control", "/multi_control"),
            ("/robot_fullstate", "/multi_fullstate")
        ],
        parameters=[
            {'model_type': model_type},
            {'x0': x},
            {'y0': y},
            {'yaw': yaw},
            {'v': v},
            {'omega': omega}
            ]
        )
    
    ld.add_action(robot_node)  

    return ld