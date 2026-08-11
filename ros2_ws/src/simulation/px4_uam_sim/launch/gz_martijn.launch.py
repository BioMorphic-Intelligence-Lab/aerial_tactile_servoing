# Copyright 2022 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from ament_index_python.packages import (get_package_share_directory)

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument,
                            ExecuteProcess,
                            SetEnvironmentVariable,
                            IncludeLaunchDescription,
                            SetLaunchConfiguration,
                            TimerAction)

from launch.substitutions import PathJoinSubstitution, LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource



def generate_launch_description():
    HOME = os.environ.get('HOME')    
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')   
    gz_launch_path = PathJoinSubstitution([pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py'])
    PX4_RUN_DIR = HOME + '/PX4-Autopilot'
    
    execute_microXRCEagent = ExecuteProcess(
        cmd=[
            'MicroXRCEAgent', 'udp4', '-p', '8888'
        ],
        prefix="bash -c 'sleep 5s; $0 $@'",
        output='screen')
    
    # Launch PX4 GZ Sim
    """
    execute_px4_gz_sim = ExecuteProcess(
        cmd=[
            PX4_RUN_DIR + '/build/px4_sitl_default/bin/px4',
            ])
    """

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        name='use_sim_time',
        default_value='True',
        description='Use simulation (Gazebo) clock if true')
    
    """
    declare_flight_mode_cmd = DeclareLaunchArgument(
        name='flight_mode',
        default_value='position',
        description='Which flight mode to open the ros2-px4 bridge in')
    """

    # GZ - ROS Bridge
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            # Clock (GZ -> ROS2)
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            '/model/my_custom_model/odometry@'
                'nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/shoulder_1_joint_vel@'
                'std_msgs/msg/Float64]gz.msgs.Double',
            '/elbow_1_joint_vel@'
                'std_msgs/msg/Float64]gz.msgs.Double',
            '/forearm_1_joint_vel@'
                'std_msgs/msg/Float64]gz.msgs.Double',
            '/shoulder_2_joint_vel@'
                'std_msgs/msg/Float64]gz.msgs.Double',
            '/elbow_2_joint_vel@'
                'std_msgs/msg/Float64]gz.msgs.Double',
            '/forearm_2_joint_vel@'
                'std_msgs/msg/Float64]gz.msgs.Double',
            '/world/world_demo/model/my_custom_model/joint_state@'
                'sensor_msgs/msg/JointState[gz.msgs.Model',
        ],
        remappings=[
            ('/model/my_custom_model/odometry', '/odom'),
            ('/shoulder_1_joint_vel', '/shoulder_1_vel_cmd'),
            ('/shoulder_2_joint_vel', '/shoulder_2_vel_cmd'),
            ('/elbow_1_joint_vel', '/elbow_1_vel_cmd'),
            ('/elbow_2_joint_vel', '/elbow_2_vel_cmd'),
            ('/forearm_1_joint_vel', '/forearm_1_vel_cmd'),
            ('/forearm_2_joint_vel', '/forearm_2_vel_cmd'),            
            ('world/world_demo/model/my_custom_model/joint_state', '/servo/out/state')
        ],
        output="screen"
    )

    ld = LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gz_launch_path),
            launch_arguments={
                'gz_args': 'test_world.sdf',
                'on_exit_shutdown': 'True'
            }.items(),
        ),
        Node(package='ros_gz_sim', 
             executable='create',
            arguments=[
                '-name', 'my_custom_model',
                '-file',  '/home/martijn/aerial_tactile_servoing/src/simulation/px4_uam_sim/urdf/martijn.urdf',
                '-z', ' 0.1'],
            output='screen')
    ])
        # Execute processes
    ld.add_action(execute_microXRCEagent)
    #ld.add_action(execute_px4_gz_sim)
    ld.add_action(bridge)
    
    return ld



















