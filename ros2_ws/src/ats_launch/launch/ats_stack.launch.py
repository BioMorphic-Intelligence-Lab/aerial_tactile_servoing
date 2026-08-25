import os
import datetime

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

"""
Shared node graph for a velocity-based aerial tactile servoing mission.

The full stack is declared ONCE here. Mission wrappers in launch/missions/
include this file and vary it through launch arguments:

  * mission_params : YAML with the per-node parameters for the mission
  * md_executable  : which mission_director executable to run
  * servo_config   : which dxl_driver parameter file to load (real only)
  * sim            : swap the driver layer for the Gazebo graph and override the
                     handful of sim-only params -- the mission_director node
                     itself is launched identically in sim and real.

No node parameters are hard-coded here; to add or retune a mission, edit a
config/missions/*.yaml. The sim/real difference is a launch overlay, not a
property of the mission or the mission director.

Node names are fixed (dxl_driver, mission_director, tactip_driver, controller,
planner) so the sections in the mission YAML match by name.
"""

_LOG_ARGS = ["--ros-args", "--log-level", "info"]


def _build(context, *args, **kwargs):
    sim = LaunchConfiguration("sim").perform(context).lower() == "true"
    fcu_on = LaunchConfiguration("fcu_on").perform(context).lower() == "true"
    mission_params = LaunchConfiguration("mission_params").perform(context)
    servo_config = LaunchConfiguration("servo_config").perform(context)
    md_executable = LaunchConfiguration("md_executable").perform(context)

    # The mission director is sim/real agnostic. sm.sim and sm.fcu_on describe
    # the run environment, not the mission, so they are NOT in the mission YAML:
    # the launch owns them and sets them explicitly here. (The base class also
    # forces fcu_on off whenever sm.sim is true, so in sim its value is moot.)
    # tactip runs on fake data in sim.
    md_params = [mission_params, {"sm.sim": sim, "sm.fcu_on": fcu_on}]
    tactip_params = [mission_params, {"fake_data": True}] if sim else [mission_params]

    actions = []

    # --- Driver layer: real hardware vs Gazebo -------------------------------
    if sim:
        sim_launch = os.path.join(
            get_package_share_directory("px4_uam_sim"),
            "launch",
            "gz_martijn_one_arm.launch.py",
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
        actions.append(
            Node(
                package="sim_remapper",
                executable="sim_remapper",
                name="sim_remapper",
                output="screen",
                parameters=[
                    {"frequency": 100.0},
                    {"verbose": False},
                    {"operating_mode": "velocity"},
                ],
                arguments=_LOG_ARGS,
            )
        )
    else:
        actions.append(
            Node(
                package="dxl_driver",
                executable="dxl_driver_node",
                name="dxl_driver",
                output="screen",
                parameters=[servo_config],
                arguments=_LOG_ARGS,
            )
        )

    # --- Mission stack (identical in sim and real) ---------------------------
    actions.append(
        Node(
            package="mission_director",
            executable=md_executable,
            name="mission_director",
            output="screen",
            parameters=md_params,
            arguments=_LOG_ARGS,
        )
    )
    actions.append(
        Node(
            package="tactip_ros2_driver",
            executable="tactip_ros2_driver",
            name="tactip_driver",
            output="screen",
            parameters=tactip_params,
            arguments=_LOG_ARGS,
        )
    )
    actions.append(
        Node(
            package="pose_based_ats",
            executable="velocity_based_ats",
            name="controller",
            output="screen",
            parameters=[mission_params],
            arguments=_LOG_ARGS,
        )
    )
    actions.append(
        Node(
            package="ats_planner",
            executable="ats_planner",
            name="planner",
            output="screen",
            parameters=[mission_params],
            arguments=_LOG_ARGS,
        )
    )

    # --- Optional rosbag recording -------------------------------------------
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
                "mission_params",
                description="Path to the per-node parameter YAML for this mission.",
            ),
            DeclareLaunchArgument(
                "md_executable",
                description="mission_director executable to run for this mission.",
            ),
            DeclareLaunchArgument(
                "servo_config",
                default_value="",
                description="Path to the dxl_driver parameter YAML (real only).",
            ),
            DeclareLaunchArgument(
                "sim",
                default_value="false",
                description="Run the Gazebo graph instead of the hardware drivers.",
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
                "log_path",
                default_value="/home/martijn/aerial_tactile_servoing/data/rosbags",
                description="Directory the recorded rosbag is written to.",
            ),
            DeclareLaunchArgument(
                "rosbag_prefix",
                default_value="ats",
                description="Filename prefix for the recorded rosbag.",
            ),
            OpaqueFunction(function=_build),
        ]
    )
