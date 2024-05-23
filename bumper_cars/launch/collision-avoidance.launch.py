import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource


def add_car(context, ld):
    """
    Function that adds the Nodes required to control any car
    """
    carAmount_value = LaunchConfiguration('carNumber').perform(context)
    offset_value = LaunchConfiguration('carNumOffset').perform(context)

    car_yaml = os.path.join(
        get_package_share_directory('bumper_cars'),
        'config',
        'controller.yaml'
    )

    track_yaml = os.path.join(
        get_package_share_directory('lar_utils'),
        'config',
        'track',
        'la_track.yaml'
    )

    # One *state buffer* / *controller* / *safety controller* per car
    for car_num in range(int(carAmount_value)):
        if car_num == 1:
            continue
        # Assign names and numbers to each car
        car_i = car_num + int(offset_value)
        car_str = '' if car_i == 0 else str(car_i + 1)
        
        # State buffer
        stateBuffer_node = Node(
            package='bumper_cars',
            executable='state_buffer',
            name='state_buffer_node' + car_str,
            parameters=[
                    {'carNumber': int(carAmount_value)},
                    {'source': LaunchConfiguration('source_target')}
            ],
            remappings=[('/env_state', '/env_state' + car_str),
                        ('/car_cmd', '/car_cmd' + car_str)],
            emulate_tty=True,
            output='both'
        )
        ld.add_action(stateBuffer_node)

        # Collision avoidance controller
        collisionAvoidance_node = Node(
            package='bumper_cars',
            executable='collision_avoidance_ros2',
            name='ca_node' + car_str,
            parameters=[
                    {'car_yaml': car_yaml},
                    {'alg': LaunchConfiguration('alg')},
                    {'car_i': car_i},
                    {'gen_traj': LaunchConfiguration('gen_traj')},
                    {'source': LaunchConfiguration('source_target')}
            ],
            emulate_tty=True,
            output='both'
        )
        ld.add_action(collisionAvoidance_node)

        # Control node
        node = Node(
            package='main_control_racing_ros2',
            executable='amzmini_control_node2',
            name='main_control_node' + car_str,
            parameters=[
                    {'track_yaml': track_yaml},
                    # {'static_throttle': 0.5},
                    {'control_mode': 'pursuit'},    
                    {'state_source': LaunchConfiguration('source_target')},
                    {'control_target': LaunchConfiguration('source_target')},
                    {'arm_mpc': False},
                    {'carNumber': car_i}
            ],
                remappings=[('/sim/car'+car_str+'/set/control', '/sim/car'+car_str+'/desired_control'),
                            ('/car'+car_str+'/set/control', '/car'+car_str+'/desired_control') ],
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

    carNumOffset_arg = DeclareLaunchArgument(
        'carNumOffset',
        description='offset to be added to the numbers of the cars to be controlled',
        default_value='0'
    )

    car_alg = DeclareLaunchArgument(
        'alg',
        description='algorithm to be used to control the cars (dwa/cbf/c3bf/lbc/mpc)',
        default_value='dwa'
    )

    source_target_arg = DeclareLaunchArgument(
        'source_target',
        description='source of the information, can be sim or real',
        default_value='sim'
    )

    gen_traj = DeclareLaunchArgument(
        'gen_traj',
        description='boolean to select whether trajectories must be re-computed',
        default_value='False'
    )

    ld.add_action(carNumber_arg)
    ld.add_action(carNumOffset_arg)
    ld.add_action(source_target_arg)
    ld.add_action(car_alg)
    ld.add_action(gen_traj)
    ld.add_action(OpaqueFunction(function=add_car, args=[ld]))

    return ld
