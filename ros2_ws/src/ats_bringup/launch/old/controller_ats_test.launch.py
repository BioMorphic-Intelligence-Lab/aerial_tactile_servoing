import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
"""
Launch file for testing ATS controller with only manipulator

The package can be launched with 'ros2 launch ats_bringup ats_test_launch.launch.py'
"""

def generate_launch_description():
    ld = LaunchDescription()

    ats_planner = Node(
        package='ats_planner',
        executable='ats_planner',
        name='ats_planner',
        output='screen',
        parameters=[
            {'frequency': 15.}
        ],
        arguments=['--ros-args', '--log-level', 'info']
    )
    ats_velocity_controller = Node(
        package='ats_velocity_controller',
        executable='ats_velocity_controller',
        name='ats_velocity_controller',
        output='screen',
        parameters=[
            {'frequency': 15.},
            {'kp': 1.0},
            {'ki': 0.0},
            {'kd': 0.0},
            {'max_integral': 1.0},
            {'ewma_alpha': 0.3}
        ],
        arguments=['--ros-args', '--log-level', 'info']
    )

    inverse_dif_kinematics = Node(
        package='inverse_differential_kinematics',
        executable='inverse_differential_kinematics',
        name='inverse_differential_kinematics',
        output='screen',
        parameters=[
            {'frequency': 15.},
            {'mode': 'dry'}, # Set to 'flight to enable flight testing'
            {'start_active': True}
        ],
        arguments=['--ros-args', '--log-level', 'info']
    )

    if False:
        tactip_driver = Node(
            package='tactip_ros2_driver',
            executable='tactip_ros2_driver',
            name='tactip_driver',
            output='screen',
            parameters=[
                {'source': 0},
                {'frequency': 10.},
                {'verbose': True},
                {'test_model_time': False}
            ],
            arguments=['--ros-args', '--log-level', 'info']
        )
        ld.add_action(tactip_driver)

    ld.add_action(ats_planner)
    ld.add_action(ats_velocity_controller)
    ld.add_action(inverse_dif_kinematics)
    
    return ld