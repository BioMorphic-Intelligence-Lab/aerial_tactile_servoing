from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
import os
import datetime

"""
Launch file for testing the aerial tactile servoing system with velocity-based controller.

The package can be launched with 'ros2 launch ats_bringup vbats_mission.launch.py'
"""

logging = True
log_path = '/ros2_ws/aerial_tactile_servoing/rosbags/'
config_name = 'dxl_ros2_vbats.yaml'

def generate_launch_description():
    ld = LaunchDescription()

    param_file = os.path.join(get_package_share_directory('ats_bringup'), 'config', config_name)
    servo_driver = Node(
        package="dxl_driver",
        executable="dxl_driver_node",
        name="dxl_driver",
        output="screen",
        parameters=[param_file],
        arguments=["--ros-args", "--log-level", "info"]
    )
    ld.add_action(servo_driver)

    mission_director = Node(
        package="mission_director",
        executable="vbats_mission",
        name="vbats_mission",
        output="screen",
        parameters=[
            {'sm.frequency': 100.0},
            {'sm.position_clip': 3.0},
            {'sm.fcu_on': True},
            {'sm.sim': False},
            {'sm.manipulator_mode': 'velocity'}
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
            {'source': 0},
            {'frequency': 15.},
            {'dimension': 5},
            {'verbose': False},
            {'test_model_time': False},
            {'save_debug_image': False},
            {'ssim_contact_threshold': 0.65},
            {'save_directory': os.path.join('/home','martijn','aerial_tactile_servoing','data','tactip_images')},
            {'zero_when_no_contact': True},
            {'fake_data': False}
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
            {'Kp_linear': 40.0},
            {'Kp_angular': 0.55},
            {'Ki_linear': 0.0},
            {'Ki_angular': 0.0},
            {'windup_clip': 0.03},
            {'publish_log': False},
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
            {'default_depth': 2.5}, # default contact depth in mm
            {'varying_refs': True},
            {'verbose': True}
        ],
    )
    ld.add_action(planner)

    if logging:
        rosbag_name = 'vbats_ros2bag_'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        ros2bag = ExecuteProcess(
            cmd=['ros2', 'bag', 'record', '-o', log_path+rosbag_name, '-a'], 
            output='screen', 
            log_cmd=True,
        )
        ld.add_action(ros2bag)
    
    return ld
