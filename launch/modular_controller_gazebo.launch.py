#!/usr/bin/env python3
"""
Launch file for modular controller with external Gazebo simulator.
Allows remapping topics to match your Gazebo setup.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get the package directory
    pkg_dir = get_package_share_directory('modular_controller')
    
    # Default waypoints file
    default_waypoints = os.path.join(pkg_dir, 'config', 'waypoints.yaml')
    
    # Declare launch arguments
    waypoints_file_arg = DeclareLaunchArgument(
        'waypoints_file',
        default_value='waypoints.yaml',
        description='Name of waypoints YAML file in config directory'
    )
    
    # Topic remapping arguments
    asv_odom_topic_arg = DeclareLaunchArgument(
        'asv_odom_topic',
        default_value='/model/blueboat/odometry',
        description='Topic for ASV odometry (nav_msgs/Odometry)'
    )
    
    towfish_odom_topic_arg = DeclareLaunchArgument(
        'towfish_odom_topic',
        default_value='/model/bluerov2_heavy/odometry',
        description='Topic for towfish odometry (nav_msgs/Odometry)'
    )
    
    left_thrust_topic_arg = DeclareLaunchArgument(
        'left_thrust_topic',
        default_value='/model/blueboat/joint/motor_port_joint/cmd_thrust',
        description='Topic for left thruster command (std_msgs/Float64)'
    )
    
    right_thrust_topic_arg = DeclareLaunchArgument(
        'right_thrust_topic',
        default_value='/model/blueboat/joint/motor_stbd_joint/cmd_thrust',
        description='Topic for right thruster command (std_msgs/Float64)'
    )
    
    # Controller parameters
    los_speed_arg = DeclareLaunchArgument('los_speed', default_value='0.5')
    los_delta_arg = DeclareLaunchArgument('los_delta', default_value='10.0')
    los_k_arg = DeclareLaunchArgument('los_k', default_value='0.2')
    k_psi_arg = DeclareLaunchArgument('k_psi', default_value='25.0')
    k_r_arg = DeclareLaunchArgument('k_r', default_value='12.0')
    heading_mode_arg = DeclareLaunchArgument('heading_mode', default_value='los')
    
    # Build full path to waypoints file
    waypoints_file = LaunchConfiguration('waypoints_file')
    
    # Controller node with topic remapping
    controller_node = Node(
        package='modular_controller',
        executable='modular_controller_node',
        name='modular_controller_node',
        output='screen',
        parameters=[{
            'waypoints_file': [pkg_dir, '/config/', waypoints_file],
            'los_speed': LaunchConfiguration('los_speed'),
            'los_delta': LaunchConfiguration('los_delta'),
            'los_k': LaunchConfiguration('los_k'),
            'k_psi': LaunchConfiguration('k_psi'),
            'k_r': LaunchConfiguration('k_r'),
            'heading_mode': LaunchConfiguration('heading_mode'),
        }],
        remappings=[
            ('/model/blueboat/odometry', LaunchConfiguration('asv_odom_topic')),
            ('/model/bluerov2_heavy/odometry', LaunchConfiguration('towfish_odom_topic')),
            ('/model/blueboat/joint/motor_port_joint/cmd_thrust', LaunchConfiguration('left_thrust_topic')),
            ('/model/blueboat/joint/motor_stbd_joint/cmd_thrust', LaunchConfiguration('right_thrust_topic')),
        ]
    )
    
    return LaunchDescription([
        waypoints_file_arg,
        asv_odom_topic_arg,
        towfish_odom_topic_arg,
        left_thrust_topic_arg,
        right_thrust_topic_arg,
        los_speed_arg,
        los_delta_arg,
        los_k_arg,
        k_psi_arg,
        k_r_arg,
        heading_mode_arg,
        controller_node,
    ])
