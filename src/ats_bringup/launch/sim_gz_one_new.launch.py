from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource

from ament_index_python.packages import get_package_share_directory
import os
import datetime

"""
Launch simulation with one arm. Corresponds to the implementation after refactoring.

The package can be launched with 'ros2 launch ats_bringup sim_gz_one_new.launch.py'
"""
logging = False

def generate_launch_description():
    ld = LaunchDescription()

    # Add the paths to the simulation and controller launch files
    sim_launch_path = os.path.join(get_package_share_directory('px4_uam_sim'), 'launch', 'gz_martijn_one_arm.launch.py')
    ld.add_action(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(sim_launch_path),
        launch_arguments={
            'logging': 'false', 
            'tactip_enable': 'false', 
            'major_frequency': '25.0'}.items())
        )
    # Add sim remapper node
    sim_remapper = Node(
        package='sim_remapper',
        executable='sim_remapper',
        name='sim_remapper',
        output='screen',
        parameters=[
            {'frequency': 100.0},
            {'verbose': False},
            {'operating_mode': 'velocity'}  # position or velocity
        ],
        arguments=['--ros-args', '--log-level', 'info']
    )
    ld.add_action(sim_remapper)

    # Add the logging
    if logging:
        rosbag_name = 'ros2bag_sim_'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        rosbag_path = f'/home/martijn/aerial_tactile_servoing/rosbags/{rosbag_name}'
        rosbag_record = ExecuteProcess(
            cmd=['ros2', 'bag', 'record', '-o', rosbag_path, '-a'], 
            output='screen', 
            log_cmd=True,
        )
        ld.add_action(rosbag_record)

    mission_director = Node(
        package="mission_director",
        executable="vbats_mission",
        name="vbats_mission",
        output="screen",
        parameters=[
            {'sm.frequency': 100.0},
            {'sm.position_clip': 3.0},
            {'sm.fcu_on': False},
            {'sm.sim': True},
            {'sm.manipulator_mode': 'velocity'},
            {'im.tactile_servoing_time': 30.0}
        ],
        arguments=["--ros-args", "--log-level", "info"]
    )
    ld.add_action(mission_director)

    tactip_driver = Node(
        package='tactip_ros2_driver',
        executable='tactip_ros2_driver',
        name='tactip_driver',
        output='screen',
        parameters=[
            {'source': 4},
            {'frequency': 15.},
            {'dimension': 5},
            {'verbose': False},
            {'test_model_time': False},
            {'save_debug_image': False},
            {'save_interval': 10.0},
            {'ssim_contact_threshold': 0.65},
            {'save_directory': os.path.join('/home','martijn','aerial_tactile_servoing','data','tactip_images')},
            {'zero_when_no_contact': True},
            {'fake_data': True}
        ],
        arguments=['--ros-args', '--log-level', 'info']
    )
    ld.add_action(tactip_driver)

    controller = Node(
        package='pose_based_ats',
        executable='velocity_based_ats',
        name='controller',
        output='screen',
        parameters=[
            {'frequency': 100.},
            {'Kp_depth': 40.0},
            {'Ki_depth': 2.5}, # Increase this one for door
            {'Kd_depth': 2.0},
            {'Kp_shear': 20.0},
            {'Ki_shear': 2.5},
            {'Kd_shear': 2.0},
            {'Kp_angular': 0.75},
            {'Ki_angular': 0.0},
            {'Kd_angular': 0.45},
            {'alpha': 0.1},
            {'windup_clip': 0.1},
            {'altitude_anchoring': False}, # Don't turn on with altitude damping
            {'altitude_damping': False}, # Don't turn on with altitude anchoring
            {'Kp_alt': 0.5},
            {'test_execution_time': False}
        ],
        arguments=['--ros-args', '--log-level', 'info']
    )
    ld.add_action(controller)

    planner = Node(
        package='ats_planner',
        executable='ats_planner',
        name='planner',
        output='screen',
        parameters=[
            {'frequency': 100.},
            {'default_depth': -2.3}, # default contact depth in mm
            {'mission_preset': 'slide_x'}, # mission preset to use (e.g., 'blockref_x', 'slide_x', etc.)
            {'verbose': False}
        ],
        arguments=['--ros-args', '--log-level', 'info']
    )
    ld.add_action(planner)

    return ld
