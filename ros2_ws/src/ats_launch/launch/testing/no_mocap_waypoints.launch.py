import os
import datetime

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

"""
waypoints -- bare flight test: take off, fly a list of waypoints, land.

Unlike the mission wrappers in launch/missions/, this does NOT include the full
ats_stack. It launches *only* the mission_director node (running the waypoints
mission) plus an optional rosbag. No drivers, no manipulator, no perception --
useful for checking take-off / navigation / land against a flight controller
(real PX4 over DDS, or a separately started SITL) in isolation.

  ros2 launch ats_launch waypoints.launch.py
  ros2 launch ats_launch waypoints.launch.py logging:=true
  ros2 launch ats_launch waypoints.launch.py fcu_on:=false   # dry run, no flight controller
  ros2 launch ats_launch waypoints.launch.py sim:=true       # FCU forced off by the base class

Waypoints are tuned in config/missions/waypoints.yaml, not here.
"""

_LOG_ARGS = ["--ros-args", "--log-level", "info"]


def _build(context, *args, **kwargs):
    pkg = get_package_share_directory("ats_launch")
    mission_params = os.path.join(pkg, "config", "testing", "waypoints.yaml")

    sim = LaunchConfiguration("sim").perform(context).lower() == "true"
    fcu_on = LaunchConfiguration("fcu_on").perform(context).lower() == "true"
    morphology = LaunchConfiguration("morphology").perform(context)

    # sm.sim / sm.fcu_on describe the run environment, not the mission, so the
    # launch sets them explicitly (matching ats_stack.launch.py).
    md_params = [mission_params, {"sm.sim": sim, "sm.fcu_on": fcu_on}]

    actions = [
        Node(
            package="mission_director",
            executable="no_mocap_waypoints",
            name="mission_director",
            output="screen",
            parameters=md_params,
            arguments=_LOG_ARGS,
        )
    ]

    if sim:
        sim_launch = os.path.join(
            get_package_share_directory("px4_uam_sim"),
            "launch",
            "gz_martijn_"+morphology+".launch.py",
        )
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(sim_launch),
                launch_arguments={
                    "logging": "false",
                    "tactip_enable": "false",
                    "major_frequency": "25.0",
                }.items(),
            )
        )

    if LaunchConfiguration("logging").perform(context).lower() == "true":
        prefix = LaunchConfiguration("rosbag_prefix").perform(context)
        log_path = LaunchConfiguration("log_path").perform(context)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dest = os.path.join(log_path, f"{prefix}_ros2bag_{stamp}")
        actions.append(
            ExecuteProcess(
                cmd=["ros2", "bag", "record", "-o", dest, "-a"],
                output="screen",
                log_cmd=True,
            )
        )

    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "sim",
                default_value="false",
                description="Run environment is simulation (forces the flight controller off in the base class).",
            ),
            DeclareLaunchArgument(
                "fcu_on",
                default_value="true",
                description="Flight controller in the loop (set false for a dry test; forced off in sim).",
            ),
            DeclareLaunchArgument(
                "logging",
                default_value="false",
                description="Record a rosbag (ros2 bag record -a) for the run.",
            ),
            DeclareLaunchArgument(
                "morphology",
                default_value="single_arm",
                description="Aerial manipulator morphology",
            ),
            DeclareLaunchArgument(
                "log_path",
                default_value="../data/rosbags",
                description="Directory the recorded rosbag is written to.",
            ),
            DeclareLaunchArgument(
                "rosbag_prefix",
                default_value="waypoints",
                description="Filename prefix for the recorded rosbag.",
            ),
            OpaqueFunction(function=_build),
        ]
    )
