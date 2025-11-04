#!/usr/bin/env python3
"""
Launch file for modular controller node.
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_dir = get_package_share_directory('modular_controller')
    
    # Declare arguments
    waypoints_file_arg = DeclareLaunchArgument(
        'waypoints_file',
        default_value='waypoints.yaml',
        description='Waypoints YAML file name'
    )
    
    los_speed_arg = DeclareLaunchArgument(
        'los_speed',
        default_value='1.0',
        description='LOS desired speed [m/s]'
    )
    
    los_delta_arg = DeclareLaunchArgument(
        'los_delta',
        default_value='5.0',
        description='LOS look-ahead distance [m]'
    )
    
    los_k_arg = DeclareLaunchArgument(
        'los_k',
        default_value='0.5',
        description='LOS convergence gain'
    )
    
    heading_mode_arg = DeclareLaunchArgument(
        'heading_mode',
        default_value='los',
        description='Heading control mode: path, los, or force'
    )
    
    k_psi_arg = DeclareLaunchArgument(
        'k_psi',
        default_value='10.0',
        description='Heading error gain [N⋅m/rad]'
    )
    
    k_r_arg = DeclareLaunchArgument(
        'k_r',
        default_value='5.0',
        description='Yaw rate damping [N⋅m⋅s/rad]'
    )
    
    # Controller node
    controller_node = Node(
        package='modular_controller',
        executable='modular_controller_node',
        name='modular_controller',
        output='screen',
        parameters=[{
            'waypoints_file': LaunchConfiguration('waypoints_file'),
            'los_speed': LaunchConfiguration('los_speed'),
            'los_delta': LaunchConfiguration('los_delta'),
            'los_k': LaunchConfiguration('los_k'),
            'heading_mode': LaunchConfiguration('heading_mode'),
            'k_psi': LaunchConfiguration('k_psi'),
            'k_r': LaunchConfiguration('k_r'),
        }]
    )
    
    return LaunchDescription([
        waypoints_file_arg,
        los_speed_arg,
        los_delta_arg,
        los_k_arg,
        heading_mode_arg,
        k_psi_arg,
        k_r_arg,
        controller_node,
    ])
