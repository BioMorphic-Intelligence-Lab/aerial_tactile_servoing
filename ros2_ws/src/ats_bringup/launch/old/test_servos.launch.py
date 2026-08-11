from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
import os
import datetime

import launch

"""
Launch file for testing the ROS2 Dynamixel driver (for Feetech servos). 

The package can be launched with 'ros2 launch ats_bringup test_servos.launch.py'
"""

logging = False

def generate_launch_description():
    ld = LaunchDescription()
    

    param_file = os.path.join(get_package_share_directory('ats_bringup'), 'config', 'feetech_ros2.yaml')
    servo_driver = Node(
        package="feetech_ros2",
        executable="feetech_ros2_interface",
        name="feetech_ros2_interface",
        output="screen",
        parameters=[param_file],
        arguments=["--ros-args", "--log-level", "info"]
    )

    if logging:
        rosbag_name = 'ros2bag_servo_'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        ros2bag = ExecuteProcess(
            cmd=['ros2', 'bag', 'record', '-o', '/ros2_ws/aerial_tactile_servoing/rosbags/'+rosbag_name, '-a'], 
            output='screen', 
            log_cmd=True,
        )
        ld.add_action(ros2bag)

    ld.add_action(servo_driver)
    
    return ld
