import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    generate_traj_condition = DeclareLaunchArgument(
        'gen_traj',
        description='whether trajectories are generated (True) for the current DWA configuration  or not (False)',
        default_value='False'
    )


    config_yaml = os.path.join(
        get_package_share_directory('dwa_dev'),
        'config',
        'dwa_config.yaml'
    )

    traj_yaml = os.path.join(
        get_package_share_directory('dwa_dev'),
        'config',
        'trajectories.yaml'
    )

    dwa_node = Node(
        package='dwa_dev',
        executable='dwa',
        name='dwa_simulation',
        parameters=[
            {'car_yaml': config_yaml},
            {'traj_yaml': traj_yaml}
        ],
    )
    
    traj_node = Node(
        package='dwa_dev',
        executable='generate_trajectories',
        name='traj_generator',
        parameters=[
            {'car_yaml': config_yaml},
        ],
        condition=IfCondition(LaunchConfiguration('gen_traj'))
    )


    return LaunchDescription([
        generate_traj_condition,
        dwa_node,
        traj_node
    ])
