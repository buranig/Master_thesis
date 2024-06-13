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
                    {'source': LaunchConfiguration('source')}
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
                    {'source': LaunchConfiguration('source')},
                    {'debug_rviz': LaunchConfiguration('debug_rviz')}
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
                    {'static_throttle': LaunchConfiguration('static_throttle')},
                    {'control_mode': LaunchConfiguration('control_mode')},    
                    {'state_source': LaunchConfiguration('source')},
                    {'control_target': LaunchConfiguration('source')},
                    {'arm_mpc': False},
                    {'carNumber': car_i}
            ],
                remappings=[('/sim/car'+car_str+'/set/control', '/sim/car'+car_str+'/desired_control'),
                            ('/car'+car_str+'/set/control', '/car'+car_str+'/desired_control'),
                            ('/joy', '/joy'+car_str) ],
            emulate_tty=True,
            output='both'
        )
        ld.add_action(node)


        # Joy Safety Node
        joy1Safety_node = Node(
            package='bumper_cars',
            executable='joy_safety',
            name='joy'+car_str+'_safety_node',
            parameters=[
                    {'car_str': car_str},
            ],
            emulate_tty=True,
            output='both'
        )

        # Joy Node
        joy1 = Node(
            package='joy',
            executable='joy_node',
            name='joy'+car_str+'_node',
            parameters=[
                    {'device_id': car_num},
            ],
            remappings=[('/joy', '/joy'+str(car_i)) ],
            emulate_tty=True,
            output='both'
        )

        joy_remap = Node(
            package='bumper_cars',
            executable='wheel_remap',
            name='joy'+car_str+'_remap',
            parameters=[
                    {'car_i': car_i}       
            ],
            remappings=[('/joy', '/joy'+str(car_i)),
                        ('/joy_remap', '/joy'+car_str) ],
            emulate_tty=True,
            output='both'
        )

        ld.add_action(joy1Safety_node)
        ld.add_action(joy1)
        ld.add_action(joy_remap)
    
    g29_ff = Node(
        package="ros_g29_force_feedback",
        executable="g29_force_feedback",
        name="g29_force_feedback",
        output="screen",
        parameters=[
            {'device_name': "/dev/input/event12"}
        ],
    )
    ld.add_action(g29_ff)



def generate_launch_description():
    ld = LaunchDescription()

    carNumber_arg = DeclareLaunchArgument(
        'carNumber',
        description='number of cars to be controlled, can be sim or real',
        default_value='1'
    )
    staticThrottle_arg = DeclareLaunchArgument(
        'static_throttle',
        description='number of cars to be controlled, can be sim or real',
        default_value='0.1'
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

    source_arg = DeclareLaunchArgument(
        'source',
        description='source of the information, can be sim or real',
        default_value='sim'
    )

    gen_traj = DeclareLaunchArgument(
        'gen_traj',
        description='boolean to select whether trajectories must be re-computed',
        default_value='False'
    )

    debug_rviz = DeclareLaunchArgument(
        'debug_rviz',
        description='boolean to select whether to draw the debug information in rviz',
        default_value='True'
    )

    control_mode = DeclareLaunchArgument(
        'control_mode',
        description='Main control algorithm',
        default_value='pursuit'
    )

    

    ld.add_action(carNumber_arg)
    ld.add_action(staticThrottle_arg)
    ld.add_action(carNumOffset_arg)
    ld.add_action(source_arg)
    ld.add_action(car_alg)
    ld.add_action(gen_traj)
    ld.add_action(debug_rviz)
    ld.add_action(control_mode)
    ld.add_action(OpaqueFunction(function=add_car, args=[ld]))

    return ld
