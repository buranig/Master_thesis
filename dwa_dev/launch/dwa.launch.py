import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    generate_traj_condition = DeclareLaunchArgument(
        'gen_traj',
        description='whether trajectories are generated (True) for the current DWA configuration  or not (False)',
        default_value='False'
    )
    seed_param = DeclareLaunchArgument(
        'seed',
        description='whether trajectories follow those established by Seed 1 (True) or not (False)',
        default_value='True'
    )


    config_json = os.path.join(
        get_package_share_directory('dwa_dev'),
        'config',
        'dwa_config.json'
    )

    traj_json = os.path.join(
        get_package_share_directory('dwa_dev'),
        'config',
        'trajectories.json'
    )

    seed_json = os.path.join(
        get_package_share_directory('dwa_dev'),
        'config',
        'seed_1.json'
    )

    dwa_node = Node(
        package='dwa_dev',
        executable='dwa',
        name='dwa_simulation',
        parameters=[
            {'car_json': config_json},
            {'traj_json': traj_json},
            {'seed': LaunchConfiguration('seed')},
            {'seed_json': seed_json},
        ],
        condition=UnlessCondition(LaunchConfiguration('gen_traj')),
        output="screen"
    )
    
    traj_node = Node(
        package='dwa_dev',
        executable='generate_trajectories',
        name='traj_generator',
        parameters=[
            {'car_json': config_json},
        ],
        condition=IfCondition(LaunchConfiguration('gen_traj')),
        output="screen"
    )


    return LaunchDescription([
        generate_traj_condition,
        seed_param,
        dwa_node,
        traj_node
    ])
