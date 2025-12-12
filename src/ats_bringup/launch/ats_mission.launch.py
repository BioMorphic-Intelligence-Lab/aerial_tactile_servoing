from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
import os
import datetime

"""
Launch file for testing the ROS2 Dynamixel driver. 

The package can be launched with 'ros2 launch ats_bringup dxl_example_config.launch.py'
"""

logging = False
log_path = '/ros2_ws/aerial_tactile_servoing/rosbags/'
config_name = 'dxl_ros2_ats.yaml'

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
        executable="ats_mission",
        name="md_ats_mission",
        output="screen",
        parameters=[
            {'sm.frequency': 50.0},
            {'sm.position_clip': 3.0},
            {'sm.fcu_on': True},
            {'sm.sim': False}
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
        executable='pose_based_ats',
        name='controller',
        output='screen',
        parameters=[
            {'frequency': 50.},
            {'reference_pose': [0., 0., 0.003]},
            {'Kp_linear': 10.0},
            {'Kp_angular': 0.3},
            {'Ki_linear': 0.0}, # was 0.2
            {'Ki_angular': 0.0}, # was 0.01
            {'windup_clip': 0.1},
            {'publish_log': False},
            {'regularization_weight': 0.001},
            {'test_execution_time': False}
        ],
        arguments=['--ros-args', '--log-level', 'info']
    )
    ld.add_action(controller)

    if logging:
        rosbag_name = 'ros2bag_ats_'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        ros2bag = ExecuteProcess(
            cmd=['ros2', 'bag', 'record', '-o', log_path+rosbag_name, '-a'], 
            output='screen', 
            log_cmd=True,
        )
        ld.add_action(ros2bag)
    
    return ld
