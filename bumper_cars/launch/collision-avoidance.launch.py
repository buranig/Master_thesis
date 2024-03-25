import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def add_car(context, ld):
    param_value = LaunchConfiguration('carNumber').perform(context)


    car_yaml = os.path.join(
        get_package_share_directory('bumper_cars'),
        'config',
        'controller.yaml'
    )

    for i in range(int(param_value)):
        node = Node(
            package='bumper_cars',
            executable='collision_avoidance_ros2',
            name='ca_node' + str(i + 1),
            parameters=[
                    {'car_yaml': car_yaml},
                    {'alg': 'dwa'},
                    {'car_i': i + 1}
            ],
            emulate_tty=True,
            output='both'
        )
        ld.add_action(node)


def generate_launch_description():
    ld = LaunchDescription()

    carNumber_arg = DeclareLaunchArgument(
        'carNumber',
        description='number of cars to be controlled, can be sim or real',
        default_value='1'
    )

    node = Node(
        package='bumper_cars',
        executable='state_buffer',
        name='state_buffer_node',
        parameters=[
                {'carNumber': LaunchConfiguration('carNumber')}
        ],
        emulate_tty=True,
        output='both'
    )

    ld.add_action(carNumber_arg)
    ld.add_action(node)
    ld.add_action(OpaqueFunction(function=add_car, args=[ld]))

    return ld
