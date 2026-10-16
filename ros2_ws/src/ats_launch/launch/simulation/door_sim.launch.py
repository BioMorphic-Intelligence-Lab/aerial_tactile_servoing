import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

"""
door (sim) -- door closing demo with tactile servoing in Gazebo.

Same mission, same parameter file (config/missions/door.yaml), same
mission_director executable as the real launch. The only difference is sim:=true,
which makes the shared stack run the Gazebo graph + sim_remapper instead of the
hardware drivers and override the sim-only params (sm.sim, fcu_on, fake_data).
The mission director itself is unchanged and unaware of sim vs real.

  ros2 launch ats_launch door_sim.launch.py
  ros2 launch ats_launch door_sim.launch.py logging:=true
"""


def generate_launch_description():
    pkg = get_package_share_directory("ats_launch")
    stack = os.path.join(pkg, "launch", "ats_stack.launch.py")

    return LaunchDescription(
        [
            DeclareLaunchArgument("logging", default_value="false"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(stack),
                launch_arguments={
                    "md_executable": "door",
                    "mission_params": os.path.join(pkg, "config", "missions", "door.yaml"),
                    "rosbag_prefix": "door_sim",
                    "sim": "true",
                    "fcu_on": "false",  # no flight controller in sim (also forced by the base class)
                    "logging": LaunchConfiguration("logging"),
                }.items(),
            ),
        ]
    )
