import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

"""
door (real) -- door-opening tactile servoing on hardware.

Thin wrapper over the shared stack. All tuning lives in config/missions/door.yaml.
There is no sim variant: the mission depends on a mocap door pose (/mocap/door/pose)
that the simulation does not provide.

  ros2 launch ats_launch door.launch.py
  ros2 launch ats_launch door.launch.py logging:=true
  ros2 launch ats_launch door.launch.py fcu_on:=false   # dry test, no flight controller
"""


def generate_launch_description():
    pkg = get_package_share_directory("ats_launch")
    stack = os.path.join(pkg, "launch", "ats_stack.launch.py")

    return LaunchDescription(
        [
            DeclareLaunchArgument("logging", default_value="false"),
            DeclareLaunchArgument("fcu_on", default_value="true"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(stack),
                launch_arguments={
                    "md_executable": "door",
                    "mission_params": os.path.join(pkg, "config", "missions", "door.yaml"),
                    "rosbag_prefix": "dooropening",
                    "morphology": "single_arm",
                    "sim": "false",
                    "fcu_on": LaunchConfiguration("fcu_on"),
                    "logging": LaunchConfiguration("logging"),
                }.items(),
            ),
        ]
    )
