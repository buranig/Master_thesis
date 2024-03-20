from launch import LaunchDescription
from launch_ros.actions import Node
import random
import numpy as np

# For the parameter file
import pathlib
import json

path = pathlib.Path('/home/giacomo/thesis_ws/src/bumper_cars/params.json')
init_path = pathlib.Path('/home/giacomo/thesis_ws/src/bumper_cars/init.json')
# Opening JSON file
with open(path, 'r') as openfile:
    # Reading from json file
    json_object = json.load(openfile)

robot_num = json_object["robot_num"]
safety_init = json_object["safety"]
width_init = json_object["width"]
height_init = json_object["height"]
min_dist = json_object["min_dist"]

def samplegrid():
    # defining the boundaries
    safety = safety_init # safety border around the map boundaries
    width = width_init - safety
    height = height_init - safety
    # min_dis
    N = int(width/min_dist)
    M = int(height/min_dist)
    x_mesh = np.linspace(-width/2, width/2, N)
    y_mesh = np.linspace(-height/2, height/2, M)

    list = [divmod(i, M) for i in random.sample(range(N * M), robot_num)]
    list = np.array(list)
    x = x_mesh[list[:, 0]]
    y = y_mesh[list[:, 1]]
    yaw = []
    while len(yaw)<robot_num:
        yaw.append(np.radians(random.randint(-180, 180)))

    v = robot_num * [0.0]
    omega = robot_num * [0.0]
    model_type = robot_num * ['linear']
    return x.tolist(), y.tolist(), yaw, v, omega, model_type

x, y, yaw, v, omega, model_type = samplegrid()
dictionary = {
    'model_type': model_type,
    'x0': x,
    'y0': y,
    'yaw': yaw,
    'v': v,
    'omega': omega
}
with open(init_path, 'w') as outfile:
    json.dump(dictionary, outfile)

def generate_launch_description():
    ld = LaunchDescription()

    controller_node = Node(
        package="bumper_cars",
        executable="controller",
        parameters=[
            {'model_type': model_type},
            {'x0': x},
            {'y0': y},
            {'yaw': yaw},
            {'v': v},
            {'omega': omega}
            ]
    )

    sensor_node = Node(
        package="bumper_cars",
        executable="sensor",
        parameters=[
            {'model_type': model_type},
            {'x0': x},
            {'y0': y},
            {'yaw': yaw},
            {'v': v},
            {'omega': omega}
            ]
    )

    # plotter_node = Node(
    #     package="bumper_cars",
    #     executable="plotter",
    #     parameters=[
    #         {'model_type': model_type},
    #         {'x0': x},
    #         {'y0': y},
    #         {'yaw': yaw},
    #         {'v': v},
    #         {'omega': omega}
    #         ]
    # )

    converter_node = Node(
        package="bumper_cars",
        executable="converter",
        parameters=[
            {'model_type': model_type},
            {'x0': x},
            {'y0': y},
            {'yaw': yaw},
            {'v': v},
            {'omega': omega}
            ]
    )
    # launch node to start rviz2 
    rviz2_node = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", "/home/giacomo/thesis_ws/src/bumper_cars/rviz2_config.rviz"]
    )

    # launch node for static broadcaster following this example ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 1 map base_link
    static_tf_pub = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=["0", "0", "0", "0", "0", "0", "1", "map", "base_link"]
    )

    broadcaster_node = Node(
        package="bumper_cars",
        executable="broadcaster",
        parameters=[
            {'model_type': model_type},
            {'x0': x},
            {'y0': y},
            {'yaw': yaw},
            {'v': v},
            {'omega': omega}
            ]
    )

    # ld.add_action(plotter_node)
    ld.add_action(sensor_node)
    ld.add_action(controller_node)
    ld.add_action(converter_node)
    # ld.add_action(static_tf_pub)  
    ld.add_action(rviz2_node) 
    ld.add_action(broadcaster_node)

    return ld