#!/usr/bin/env python3
"""
Launch file for modular controller with ASV-towfish simulator.
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get package directories
    controller_pkg = get_package_share_directory('modular_controller')
    sim_pkg = get_package_share_directory('asv_controller')
    
    # Declare arguments
    waypoints_file_arg = DeclareLaunchArgument(
        'waypoints_file',
        default_value='waypoints.yaml',
        description='Waypoints YAML file name'
    )
    
    # Simulator node
    sim_node = Node(
        package='asv_controller',
        executable='asv_towfish_sim',
        name='asv_towfish_sim',
        output='screen',
        parameters=[{
            'rate_hz': 100.0,
            'tether_length': 3.5,
            'tow_mass': 25.0,
            'tow_stiffness': 12.0,
            'tow_damping': 35.0,
            'enable_plot': True,
            'plot_stride': 5,
            'plot_show_vref': True,
            'plot_show_force': True,
        }]
    )
    
    # Controller node
    controller_node = Node(
        package='modular_controller',
        executable='modular_controller_node',
        name='modular_controller',
        output='screen',
        parameters=[{
            'waypoints_file': LaunchConfiguration('waypoints_file'),
            'los_speed': 1.0,
            'los_delta': 5.0,
            'los_k': 0.5,
            'heading_mode': 'los',
            'k_psi': 10.0,
            'k_r': 5.0,
        }]
    )
    
    return LaunchDescription([
        waypoints_file_arg,
        sim_node,
        controller_node,
    ])
