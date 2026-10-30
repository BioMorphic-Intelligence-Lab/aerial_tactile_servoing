from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

import datetime

logging = False
"""
Launch tactip driver ros2 node.

The package can be launched with 'ros2 launch tactip_ros2_driver tactip_bringup.launch.py'
"""

def generate_launch_description():
    ld = LaunchDescription()

    if logging:
        rosbag_name = 'ros2bag_tactip_'+datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S')
        ros2bag = ExecuteProcess(
            cmd=['ros2', 'bag', 'record', '-o', '/home/martijn/aerial_tactile_servoing/rosbags/'+rosbag_name, '-a'], 
            output='screen', 
            log_cmd=True,
        )
        ld.add_action(ros2bag)

    tactip_driver = Node(
        package='tactip_ros2_driver',
        executable='tactip_ros2_driver',
        name='tactip_driver',
        output='screen',
        parameters=[
            {'sources': [2]},
            {'dimensions': [8]},
            {'ssim_contact_thresholds': [0.8]},
            {'model_names': ['simple_cnn_B2']},
            {'frequency': 10.},
            {'fake_data': False},
            {'verbose': True},
            {'test_model_time': False},
            {'zero_when_no_contact': True},
        ],
        arguments=['--ros-args', '--log-level', 'info']
    )
    ld.add_action(tactip_driver)

    return ld