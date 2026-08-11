from launch import LaunchDescription
from launch_ros.actions import Node

"""
Example launch file. You can create launch files for subtests in this package in the same way as this file. 
This file launches the cv_aruco_detector node as an example.
To add nodes to be launched, copy the code here and add the package containing the node as a dependency in the package.xml file.

The package can be launched with 'ros2 launch tucan_bringup example.launch.py'
"""

def generate_launch_description():
    ld = LaunchDescription()
    
    return ld