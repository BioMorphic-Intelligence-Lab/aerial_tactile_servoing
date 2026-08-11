from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition

import os

"""
Launch simulation with one arm.

The package can be launched with 'ros2 launch ats_bringup gz_sim_one_arm.launch.py'
"""

def generate_launch_description():
    ld = LaunchDescription()

    arg_logging = DeclareLaunchArgument(
        'logging',
        default_value='false',
        description='Enable logging'
    )
    arg_tactip_enable = DeclareLaunchArgument(
        'tactip_enable',
        default_value='false',
        description='Enable tactip driver node'
    )
    arg_major_frequency = DeclareLaunchArgument(
        'major_frequency',
        default_value='25.0',
        description='Main frequency of all nodes'
    )
    ld.add_action(arg_logging)
    ld.add_action(arg_tactip_enable)
    ld.add_action(arg_major_frequency)
    print(LaunchConfiguration(''))


    mission_director = Node(
        package='mission_director_py',
        executable='flight_mission_director_contact',
        name='mission_director',
        output='screen',
        parameters=[
            {'frequency': LaunchConfiguration('major_frequency')},
            {'takeoff_altitude': -1.5},
            {'landing_velocity': -0.5},
            {'search_velocity': 0.05},
            {'ssim_contact_threshold': 0.7},
            {'hover_time': 3.0},
            {'position_clip': 3.0}
        ],
        arguments=['--ros-args', '--log-level', 'info']
    )
    ld.add_action(mission_director)

    sim_remapper = Node(
        package='sim_remapper',
        executable='sim_remapper',
        name='sim_remapper',
        output='screen',
        arguments=['--ros-args', '--log-level', 'info']
    )
    ld.add_action(sim_remapper)

    controller = Node(
        package='pose_based_ats',
        executable='pose_based_ats',
        name='ats_controller',
        output='screen',
        parameters=[
            {'frequency': LaunchConfiguration('major_frequency')},
            {'reference_pose': [0., 0., -0.002]},
            {'Kp': 1.5},
            {'Ki': 0.1},
            {'windup_clip': 1.},
            {'regularization_weight': 0.001},
            {'ssim_contact_threshold': 0.7}
        ],
        arguments=['--ros-args', '--log-level', 'info']
    )
    ld.add_action(controller)

    # tactip_driver = Node(
    #     package='tactip_ros2_driver',
    #     executable='tactip_ros2_driver',
    #     name='tactip_driver',
    #     output='screen',
    #     parameters=[
    #         {'source': 4},
    #         {'frequency': LaunchConfiguration('major_frequency')},
    #         {'verbose': False},
    #         {'test_model_time': False},
    #         {'save_debug_image': False},
    #         {'save_directory': os.path.join('/home','martijn','aerial_tactile_servoing','data','tactip_images')}
    #     ],
    #     arguments=['--ros-args', '--log-level', 'info'],
    #     condition=IfCondition(LaunchConfiguration('tactip_enable'))
    # )
    # ld.add_action(tactip_driver)    

    return ld