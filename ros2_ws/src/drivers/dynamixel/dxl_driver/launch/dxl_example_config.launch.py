from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
import os
import datetime

"""
Launch file for testing the ROS2 Dynamixel driver. 

The package can be launched with 'ros2 launch dxl_driver dxl_example_config.launch.py'
"""

logging = False
log_path = '/ros2_ws/aerial_tactile_servoing/rosbags/'

def generate_launch_description():
    ld = LaunchDescription()
    

    param_file = os.path.join(get_package_share_directory('dxl_driver'), 'config', 'dxl_ros2_ats.yaml')
    servo_driver = Node(
        package="dxl_driver",
        executable="dxl_driver_node",
        name="dxl_driver",
        output="screen",
        parameters=[param_file],
        arguments=["--ros-args", "--log-level", "info"]
    )

    if logging:
        rosbag_name = 'ros2bag_servo_'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        ros2bag = ExecuteProcess(
            cmd=['ros2', 'bag', 'record', '-o', log_path+rosbag_name, '-a'], 
            output='screen', 
            log_cmd=True,
        )
        ld.add_action(ros2bag)

    ld.add_action(servo_driver)
    
    return ld
