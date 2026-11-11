import os
import datetime

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

"""
Launch the tactip driver for both TacTips (dual-arm), standalone.

  ros2 launch tactip_ros2_driver tactip_dual_bringup.launch.py
  ros2 launch tactip_ros2_driver tactip_dual_bringup.launch.py logging:=true
  ros2 launch tactip_ros2_driver tactip_dual_bringup.launch.py backend:=torch

backend:=rknn (default) runs the models on the RK3588 NPU; backend:=torch runs them with PyTorch
on the CPU (also the automatic fallback if the NPU model cannot load).

Rosbag recording follows the ats_launch convention (see ats_stack.launch.py):
logging:=true records all topics to <log_path>/<rosbag_prefix>_ros2bag_<stamp>,
where an empty log_path (the default) means the repo's data/rosbags.
"""

_LOG_ARGS = ["--ros-args", "--log-level", "info"]


def _find_repo_root(start):
    """Walk up from `start` until a directory containing .git is found.

    Anchors the default rosbag path to the repo regardless of the caller's
    working directory or the colcon install layout (isolated/merge/symlink).
    """
    d = os.path.abspath(start)
    while d != "/":
        if os.path.isdir(os.path.join(d, ".git")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("repo root (.git) not found above %s" % start)


def _build(context, *args, **kwargs):
    share_dir = get_package_share_directory("tactip_ros2_driver")

    actions = [
        Node(
            package="tactip_ros2_driver",
            executable="tactip_ros2_driver",
            name="tactip_driver",
            output="screen",
            emulate_tty=True,
            parameters=[
                {"sources": [0, 2]},
                {"dimensions": [8, 8]},
                {"ssim_contact_thresholds": [0.8, 0.85]},
                {"model_names": ["simple_cnn_B1", "simple_cnn_B2"]},
                {"frequency": 0.},  # max publish rate [Hz]; 0 = every camera frame
                {"fake_data": False},
                {"verbose": False},
                {"test_model_time": True},
                {"backend": 'rknn'}, # Set to 'torch' on pc
                {"zero_when_no_contact": True},
                {"ssim_dims": [320, 240]},
            ],
            arguments=_LOG_ARGS,
        )
    ]

    # --- Optional rosbag recording -------------------------------------------
    if LaunchConfiguration("logging").perform(context).lower() == "true":
        prefix = LaunchConfiguration("rosbag_prefix").perform(context)
        # Empty log_path (the default) means "repo's data/rosbags", resolved
        # from the .git marker so it is independent of cwd and install layout.
        log_path = LaunchConfiguration("log_path").perform(context)
        if not log_path:
            log_path = os.path.join(_find_repo_root(share_dir), "data", "rosbags")
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
                "backend",
                default_value="rknn",
                description="Inference backend: 'rknn' (NPU) or 'torch' (PyTorch, CPU).",
            ),
            DeclareLaunchArgument(
                "logging",
                default_value="false",
                description="Record a rosbag (ros2 bag record -a) for the run.",
            ),
            DeclareLaunchArgument(
                "log_path",
                default_value="",
                description="Directory the recorded rosbag is written to. "
                "Empty = the repo's data/rosbags (found via the .git marker).",
            ),
            DeclareLaunchArgument(
                "rosbag_prefix",
                default_value="tactip_dual",
                description="Filename prefix for the recorded rosbag.",
            ),
            OpaqueFunction(function=_build),
        ]
    )
