from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition

from ament_index_python.packages import get_package_share_directory
import os
import datetime

logging = True
md_name = 'flight_mission_director_waypoints'

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
    arg_major_frequency = DeclareLaunchArgument(
        'major_frequency',
        default_value='25.0',
        description='Main frequency of all nodes'
    )
    ld.add_action(arg_logging)
    ld.add_action(arg_major_frequency)

    if logging:
        rosbag_name = datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S')+'tactip_ros2bag'
        ros2bag = ExecuteProcess(
            cmd=['ros2', 'bag', 'record', '-o', '/ros2_ws/aerial_tactile_servoing/rosbags/'+rosbag_name, '-a'], 
            output='screen', 
            log_cmd=True,
        )
        ld.add_action(ros2bag)

    mission_director = Node(
        package='mission_director_py',
        executable=md_name,
        name='mission_director',
        output='screen',
        parameters=[
            {'frequency': LaunchConfiguration('major_frequency')},
            {'takeoff_altitude': -1.5},
            {'landing_velocity': -0.5},
            {'period_time': 10.0},
            {'position_clip': 3.0}
        ],
        arguments=['--ros-args', '--log-level', 'info']
    )
    ld.add_action(mission_director)

    return ld