import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource


def add_car(context, ld):
    param_value = LaunchConfiguration('carNumber').perform(context)

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


    for i in range(int(param_value)):
        #
        node = Node(
            package='bumper_cars',
            executable='collision_avoidance_ros2.py',
            name='ca_node' + str(i + 1),
            parameters=[
                    {'car_yaml': car_yaml},
                    {'alg': LaunchConfiguration('alg')},
                    {'car_i': i + 1},
                    {'gen_traj': LaunchConfiguration('gen_traj')},
                    {'source': LaunchConfiguration('source_target')}
            ],
            emulate_tty=True,
            output='both'
        )
        ld.add_action(node)

        # Control node
        car_i = '' if i==0 else str(i+1)
        node = Node(
            package='main_control_racing_ros2',
            executable='amzmini_control_node2',
            name='main_control_node' + str(i + 1),
            parameters=[
                    {'track_yaml': track_yaml},
                    # {'static_throttle': 0.5},
                    {'control_mode': 'pursuit'},
                    {'state_source': LaunchConfiguration('source_target')},
                    {'control_target': LaunchConfiguration('source_target')},
                    {'arm_mpc': False},
                    {'carNumber': i + 1}
            ],
                remappings=[('/sim/car'+car_i+'/set/control', '/sim/car'+car_i+'/desired_control'),
                            ('/car'+car_i+'/set/control', '/car'+car_i+'/desired_control') ],
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

    sim = DeclareLaunchArgument(
        'sim',
        description='boolean to select whether we are working in simulation or real world',
        default_value='False'
    )


    node = Node(
        package='bumper_cars',
        executable='state_buffer.py',
        name='state_buffer_node',
        parameters=[
                {'carNumber': LaunchConfiguration('carNumber')}
        ],
        emulate_tty=True,
        output='both'
    )

    ld.add_action(carNumber_arg)
    ld.add_action(source_target_arg)
    ld.add_action(car_alg)
    ld.add_action(gen_traj)
    ld.add_action(sim)
    ld.add_action(node)
    ld.add_action(OpaqueFunction(function=add_car, args=[ld]))

    return ld
