import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

"""
hover (real) -- baseline hover mission.

Thin wrapper: selects this mission's parameter file, servo config and
mission_director executable, then includes the shared stack. All tuning lives
in config/missions/hover.yaml, not here. See vbats_sim.launch.py for simulation.

  ros2 launch ats_launch hover.launch.py
  ros2 launch ats_launch hover.launch.py logging:=true
  ros2 launch ats_launch hover.launch.py fcu_on:=false   # dry test, no flight controller
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
                    "md_executable": "hover",
                    "mission_params": os.path.join(pkg, "config", "missions", "hover.yaml"),
                    "rosbag_prefix": "hover",
                    "morphology": "single_arm",
                    "sim": "false",
                    "fcu_on": LaunchConfiguration("fcu_on"),
                    "logging": LaunchConfiguration("logging"),
                }.items(),
            ),
        ]
    )
