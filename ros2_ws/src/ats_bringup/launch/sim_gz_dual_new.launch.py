from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource

from ament_index_python.packages import get_package_share_directory
import os
import datetime
from math import pi as PI

"""
Launch the dual-arm pinch grasp mission in Gazebo simulation.
"""
logging = False

def generate_launch_description():
    ld = LaunchDescription()

    # Same grasp geometry as the real launch 
    grasp_config = os.path.join(get_package_share_directory('ats_bringup'), 'config',
                                'grasp_geometry.yaml')

    # Add the paths to the simulation and controller launch files
    sim_launch_path = os.path.join(get_package_share_directory('px4_uam_sim'), 'launch', 'gz_martijn_dual_arm.launch.py')
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
        rosbag_path = os.path.join(os.path.expanduser('~'), 'aerial_tactile_servoing', 'data', 'rosbags', rosbag_name)
        rosbag_record = ExecuteProcess(
            cmd=['ros2', 'bag', 'record', '-o', rosbag_path, '-a'], 
            output='screen', 
            log_cmd=True,
        )
        ld.add_action(rosbag_record)

    mission_director = Node(
        package="mission_director",
        executable="pinch_grasp",
        name="mission_director",
        output="screen",
        parameters=[
            grasp_config,
            {'sm.frequency': 100.0},
            {'sm.position_clip': 3.0},
            {'sm.fcu_on': False},
            {'sm.sim': True},
            {'sm.manipulator_mode': 'velocity'},
            {'im.tactile_servoing_time': 200.0}
        ],
        remappings=[
            # The base class uses the driver's default names; in the dual stack those are the LEFT
            # arm. The mission makes its own clients for the right one.
            ('/tactip/pose', '/tactip_left/pose'),
            ('/tactip/contact', '/tactip_left/contact'),
            ('set_ssim_ref', 'set_ssim_ref_left'),
        ],
        arguments=["--ros-args", "--log-level", "info"]
    )
    ld.add_action(mission_director)

    # Left TacTip (B1) on arm 1.
    tactip_driver_left = Node(
        package='tactip_ros2_driver',
        executable='tactip_ros2_driver',
        name='tactip_driver_left',
        output='screen',
        parameters=[
            {'source': 0},
            {'frequency': 15.},
            {'dimension': 5},
            {'verbose': True},
            {'test_model_time': False},
            {'save_debug_image': False},
            {'save_interval': 10.0},
            {'ssim_contact_threshold': 0.65},
            {'save_directory': os.path.join(os.path.expanduser('~'),'aerial_tactile_servoing','data','tactip_images')},
            {'zero_when_no_contact': True},
            {'fake_data': True},
            # Distinct TF frames per arm, or both broadcast the same edge and lookups mix them.
            {'sensor_frame': 'present_sensor_frame_left'},
            {'contact_frame': 'present_contact_frame_tactipdriver_left'},
        ],
        remappings=[
            ('/tactip/pose', '/tactip_left/pose'),
            ('/tactip/ssim', '/tactip_left/ssim'),
            ('/tactip/contact', '/tactip_left/contact'),
            ('/tactip/force', '/tactip_left/force'),
            ('set_ssim_ref', 'set_ssim_ref_left'),
            ('tare_force', 'tare_force_left'),
        ],
        arguments=['--ros-args', '--log-level', 'info']
    )
    ld.add_action(tactip_driver_left)

    # Right TacTip (B2) on arm 2.
    tactip_driver_right = Node(
        package='tactip_ros2_driver',
        executable='tactip_ros2_driver',
        name='tactip_driver_right',
        output='screen',
        parameters=[
            {'source': 0},
            {'frequency': 15.},
            {'dimension': 5},
            {'verbose': True},
            {'test_model_time': False},
            {'save_debug_image': False},
            {'save_interval': 10.0},
            {'ssim_contact_threshold': 0.65},
            {'save_directory': os.path.join(os.path.expanduser('~'),'aerial_tactile_servoing','data','tactip_images')},
            {'zero_when_no_contact': True},
            {'fake_data': True},
            # Distinct TF frames, or both broadcast the same edge and lookups mix the sensors.
            {'sensor_frame': 'present_sensor_frame_right'},
            {'contact_frame': 'present_contact_frame_tactipdriver_right'},
        ],
        remappings=[
            ('/tactip/pose', '/tactip_right/pose'),
            ('/tactip/ssim', '/tactip_right/ssim'),
            ('/tactip/contact', '/tactip_right/contact'),
            ('/tactip/force', '/tactip_right/force'),
            ('set_ssim_ref', 'set_ssim_ref_right'),
        ],
        arguments=['--ros-args', '--log-level', 'info']
    )
    ld.add_action(tactip_driver_right)

    tactile_controller = Node(
        package='pose_based_ats',
        executable='dual_arm_tactile_controller',
        name='dual_arm_tactile_controller',
        output='screen',
        parameters=[
            grasp_config,
            {'frequency': 30.0},                     # matches the TacTip stream
            {'sim': True},                           # fake tactile data: closes on position,
                                                     # publishes no mass estimate
            # ARM 1 is the LEFT arm (body +y, TacTip B1); ARM 2 is the RIGHT arm (body -y, B2).
            {'arm1_force_topic': '/tactip_left/force'},
            {'arm2_force_topic': '/tactip_right/force'},
            {'arm1_pose_topic': '/tactip_left/pose'},
            {'arm2_pose_topic': '/tactip_right/pose'},
        ],
        arguments=['--ros-args', '--log-level', 'info']
    )
    ld.add_action(tactile_controller)

    # Torque estimator
    # torque_observer = Node(
    #     package='wrench_observer',
    #     executable='torque_observer',
    #     name='torque_observer',
    #     output='screen',
    #     parameters=[
    #         {'frequency': 100.0},
    #         {'gain_torque': 1.0}, # Should be unity following the dynamics
    #         {'alpha_torque': 0.15}, # 1 is no filtering
    #         {'alpha_angular_velocity': 0.2},
    #         {'alpha_accelerometer': 0.2},
    #         {'alpha_motor_inputs': 0.2}, # 1 is no filtering
    #         {'model_mass': 4.239}, # [kg] with 6000 mAh batteries
    #         {'torque_bias': [0.0, 0.0, 0.0]},
    #     ],
    #     arguments=["--ros-args", "--log-level", "info"] # Log level info
    # )
    # ld.add_action(torque_observer)

    planner = Node(
        package='ats_planner',
        executable='ats_planner',
        name='planner',
        output='screen',
        parameters=[
            {'frequency': 100.},
            {'default_depth': -3.0}, # default contact depth in mm
            {'mission_preset': 'default'}, # mission preset to use (e.g., 'blockref_x', 'slide_x', etc.)
            {'verbose': False}
        ],
        arguments=['--ros-args', '--log-level', 'info']
    )
    ld.add_action(planner)

    return ld
