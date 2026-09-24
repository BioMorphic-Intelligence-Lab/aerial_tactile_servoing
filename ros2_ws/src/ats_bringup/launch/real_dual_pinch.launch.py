from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

"""
Launch file for real hardware testing of the dual-arms.
Connects directly to:
  1. Physical Dynamixel servos via dxl_driver (dxl_ros2_pinch_grasp.yaml: IDs [31,32,33, 41,42,43])
  2. Two real TacTip micro-cameras via two tactip_ros2_driver instances (left/B1, right/B2)
  3. Real PX4 Autopilot via MicroXRCEAgent / micro-ROS
  4. Autonomous PinchGraspMission state machine

Usage:
  ros2 launch ats_bringup real_dual_pinch.launch.py fcu_on:=false

  # Real flight test:
  ros2 launch ats_bringup real_dual_pinch.launch.py fcu_on:=true

Before a real run: confirm which /dev/videoN index belongs to each TacTip ( `v4l2-ctl --list-devices`) 
and set left_tactip_source / right_tactip_source accordingly --
"""

def generate_launch_description():
    ld = LaunchDescription()

    pkg_bringup = get_package_share_directory('ats_bringup')
    
    servo_config = os.path.join(pkg_bringup, 'config', 'dxl_ros2_pinch_grasp.yaml')
    # Single source for the grasp pose, backstops and waypoint geometry. Loaded by both the
    # mission and the controller so the two can never disagree about where the object is.
    grasp_config = os.path.join(pkg_bringup, 'config', 'grasp_geometry.yaml')

    declare_fcu = DeclareLaunchArgument('fcu_on', default_value='true', description='Enable flight controller')
    ld.add_action(declare_fcu)

    # /dev/videoN index for each TacTip camera.
    # `v4l2-ctl --list-devices` before a real run 
    left_tactip_source = 0
    right_tactip_source = 2

    # 1. Real Dynamixel Servo Driver (USB / U2D2)
    servo_driver = Node(
        package="dxl_driver",
        executable="dxl_driver_node",
        name="dxl_driver",
        output="screen",
        parameters=[servo_config],
        arguments=["--ros-args", "--log-level", "info"]
    )
    ld.add_action(servo_driver)

    # 2. Autonomous Dual-Arm Pinch Grasp Mission
    mission_director = Node(
        package="mission_director",
        executable="pinch_grasp",
        name="mission_director",
        output="screen",
        parameters=[
            grasp_config,
            {'sm.frequency': 100.0},
            {'sm.position_clip': 3.0},
            {'sm.fcu_on': LaunchConfiguration('fcu_on')},
            {'sm.sim': False},                        # Real hardware mode
            {'sm.manipulator_mode': 'velocity'}
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

    # 3a. Real TacTip Tactile Sensor Driver -- LEFT arm (B1).
    # unremapped topics/service (/tactip/pose, /tactip/contact, set_ssim_ref).
    tactip_driver_left = Node(
        package='tactip_ros2_driver',
        executable='tactip_ros2_driver',
        name='tactip_driver_left',
        output='screen',
        parameters=[
            {'source': left_tactip_source},
            {'frequency': 25.0},
            {'dimension': 5},
            {'verbose': False},
            {'fake_data': False},                    # Real USB camera
            {'model_dir': 'simple_cnn_B1'},          # Left TacTip is B1
            # Model frame -> housing (red mark) frame. 180 deg for BOTH sensors after the
            # latest retrain: the corrected label transform flips X and Y, so the model's
            # +Y now points opposite the mark. 
            {'tactip_angle_deg': 180.0},
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

    # 3b. Real TacTip Tactile Sensor Driver -- RIGHT arm (B2). Remapped so it doesn't collide
    # with the left instance's topics/service.
    tactip_driver_right = Node(
        package='tactip_ros2_driver',
        executable='tactip_ros2_driver',
        name='tactip_driver_right',
        output='screen',
        parameters=[
            {'source': right_tactip_source},
            {'frequency': 25.0},
            {'dimension': 5},
            {'verbose': False},
            {'fake_data': False},                    # Real USB camera
            {'model_dir': 'simple_cnn_B2'},          # Right TacTip is B2
            {'tactip_angle_deg': 180.0},             # Same as the left 
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
            # Both driver instances live in the root namespace, so without this remap the two
            # would create the same /tare_force service and only one would win.
            ('tare_force', 'tare_force_right'),
        ],
        arguments=['--ros-args', '--log-level', 'info']
    )
    ld.add_action(tactip_driver_right)

    # 4. Dual-Arm Tactile Grasp & Mass Estimation Controller
    tactile_controller = Node(
        package='pose_based_ats',
        executable='dual_arm_tactile_controller',
        name='dual_arm_tactile_controller',
        output='screen',
        parameters=[
            grasp_config,                            # pose, backstops, squeeze target
            {'frequency': 30.0},                     # matches the TacTip stream (~30 Hz)
            {'sim': False},                          # Real hardware mode
            # ARM 1 is the LEFT arm (body +y, TacTip B1); ARM 2 is the RIGHT arm (body -y, B2).
            {'arm1_force_topic': '/tactip_left/force'},
            {'arm2_force_topic': '/tactip_right/force'},
            {'arm1_pose_topic': '/tactip_left/pose'},
            {'arm2_pose_topic': '/tactip_right/pose'},
        ],
        arguments=['--ros-args', '--log-level', 'info']
    )
    ld.add_action(tactile_controller)

    return ld
