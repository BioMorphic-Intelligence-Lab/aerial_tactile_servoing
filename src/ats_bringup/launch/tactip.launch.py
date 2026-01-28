
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
import os
import datetime

def generate_launch_description():
    ld = LaunchDescription()


    tactip_driver = Node(
        package='tactip_ros2_driver',
        executable='tactip_ros2_driver',
        name='tactip_driver',
        output='screen',
        parameters=[
            {'source': 0},
            {'frequency': 15.},
            {'dimension': 5},
            {'verbose': True},
            {'test_model_time': False},
            {'save_debug_image': True},
            {'save_interval': 10.0},
            {'ssim_contact_threshold': 0.65},
            {'save_directory': os.path.join('/ros2_ws','aerial_tactile_servoing','data','tactip_images')},
            {'zero_when_no_contact': True},
            {'fake_data': False}
        ],
        arguments=['--ros-args', '--log-level', 'info']
    )
    ld.add_action(tactip_driver)
    return ld