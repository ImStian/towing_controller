#!/usr/bin/env python3
"""
Live plotter for ASV and towfish positions from Gazebo simulator.
Subscribes to odometry topics and displays real-time trajectory visualization.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from collections import deque
import math


class LivePlotter(Node):
    def __init__(self):
        super().__init__('live_plotter')
        
        # Parameters
        self.declare_parameter('asv_odom_topic', '/model/blueboat/odometry')
        self.declare_parameter('towfish_odom_topic', '/model/bluerov2_heavy/odometry')
        self.declare_parameter('waypoints_topic', '/waypoints')
        self.declare_parameter('history_length', 5000)
        self.declare_parameter('update_rate_hz', 20)
        self.declare_parameter('show_tether', True)
        self.declare_parameter('show_boat_shape', True)
        
        asv_topic = self.get_parameter('asv_odom_topic').value
        towfish_topic = self.get_parameter('towfish_odom_topic').value
        waypoints_topic = self.get_parameter('waypoints_topic').value
        history_len = self.get_parameter('history_length').value
        update_rate = self.get_parameter('update_rate_hz').value
        self.show_tether = self.get_parameter('show_tether').value
        self.show_boat = self.get_parameter('show_boat_shape').value
        
        # State
        self.asv_pos = None
        self.asv_heading = None
        self.towfish_pos = None
        self.path_points = None
        
        # History
        self.asv_history = deque(maxlen=history_len)
        self.towfish_history = deque(maxlen=history_len)
        
        # Subscribers
        self.create_subscription(Odometry, asv_topic, self.asv_callback, 10)
        self.create_subscription(Odometry, towfish_topic, self.towfish_callback, 10)
        self.create_subscription(Float64MultiArray, waypoints_topic, self.waypoints_callback, 10)
        
        # Setup matplotlib
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        
        # Plot elements
        self.path_line, = self.ax.plot([], [], 'k--', lw=1.0, label='Reference Path', alpha=0.5)
        self.asv_trail, = self.ax.plot([], [], 'b-', lw=1.5, label='ASV Trail', alpha=0.7)
        self.towfish_trail, = self.ax.plot([], [], 'r-', lw=1.5, label='Towfish Trail', alpha=0.7)
        self.tether_line, = self.ax.plot([], [], 'gray', lw=2, linestyle=':', alpha=0.6, label='Tether')
        
        # ASV boat shape
        if self.show_boat:
            boat_length = 0.8
            boat_width = 0.4
            boat_shape = np.array([
                [boat_length, 0.0],
                [-boat_length/3, boat_width/2],
                [-boat_length/3, -boat_width/2],
            ])
            self.asv_boat = Polygon(boat_shape, closed=True, facecolor='blue', 
                                   edgecolor='darkblue', linewidth=2, alpha=0.8)
            self.ax.add_patch(self.asv_boat)
        else:
            self.asv_boat = None
            
        # Current positions
        self.asv_marker = self.ax.scatter([], [], c='blue', marker='o', s=150, 
                                         edgecolors='darkblue', linewidths=2, 
                                         label='ASV', zorder=10)
        self.towfish_marker = self.ax.scatter([], [], c='red', marker='x', s=200, 
                                             linewidths=3, label='Towfish', zorder=10)
        
        # Plot settings
        self.ax.set_xlabel('East [m]', fontsize=12)
        self.ax.set_ylabel('North [m]', fontsize=12)
        self.ax.set_title('ASV-Towfish Live Tracking', fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal', 'box')
        self.ax.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Update timer
        self.plot_timer = self.create_timer(1.0 / update_rate, self.update_plot)
        
        self.get_logger().info(f'Live plotter started')
        self.get_logger().info(f'  ASV topic: {asv_topic}')
        self.get_logger().info(f'  Towfish topic: {towfish_topic}')
        self.get_logger().info(f'  Waypoints topic: {waypoints_topic}')
        
    def asv_callback(self, msg: Odometry):
        """Process ASV odometry"""
        self.asv_pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        
        # Extract yaw from quaternion
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        self.asv_heading = 2.0 * math.atan2(qz, qw)
        
        self.asv_history.append(self.asv_pos.copy())
        
    def towfish_callback(self, msg: Odometry):
        """Process towfish odometry"""
        self.towfish_pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        self.towfish_history.append(self.towfish_pos.copy())
        
    def waypoints_callback(self, msg: Float64MultiArray):
        """Process waypoints for path visualization"""
        data = list(msg.data)
        if len(data) >= 4:
            pts = [(data[i], data[i+1]) for i in range(0, len(data), 2)]
            self.path_points = np.array(pts, dtype=float)
            
    def update_plot(self):
        """Update the plot with current data"""
        try:
            # Update reference path
            if self.path_points is not None and len(self.path_points) > 1:
                self.path_line.set_data(self.path_points[:, 0], self.path_points[:, 1])
            
            # Update trails
            if len(self.asv_history) > 0:
                asv_trail = np.array(self.asv_history)
                self.asv_trail.set_data(asv_trail[:, 0], asv_trail[:, 1])
                
            if len(self.towfish_history) > 0:
                towfish_trail = np.array(self.towfish_history)
                self.towfish_trail.set_data(towfish_trail[:, 0], towfish_trail[:, 1])
            
            # Update current positions
            if self.asv_pos is not None:
                self.asv_marker.set_offsets([self.asv_pos])
                
                # Update boat shape
                if self.show_boat and self.asv_boat is not None and self.asv_heading is not None:
                    boat_length = 0.8
                    boat_width = 0.4
                    boat_local = np.array([
                        [boat_length, 0.0],
                        [-boat_length/3, boat_width/2],
                        [-boat_length/3, -boat_width/2],
                    ])
                    # Rotation matrix
                    c, s = np.cos(self.asv_heading), np.sin(self.asv_heading)
                    R = np.array([[c, -s], [s, c]])
                    boat_rotated = (R @ boat_local.T).T
                    boat_world = boat_rotated + self.asv_pos
                    self.asv_boat.set_xy(boat_world)
                    
            if self.towfish_pos is not None:
                self.towfish_marker.set_offsets([self.towfish_pos])
                
            # Update tether line
            if self.show_tether and self.asv_pos is not None and self.towfish_pos is not None:
                self.tether_line.set_data(
                    [self.asv_pos[0], self.towfish_pos[0]], 
                    [self.asv_pos[1], self.towfish_pos[1]]
                )
            
            # Auto-scale view to fit all data
            all_x = []
            all_y = []
            
            if self.path_points is not None:
                all_x.extend(self.path_points[:, 0])
                all_y.extend(self.path_points[:, 1])
                
            if len(self.asv_history) > 0:
                asv_trail = np.array(self.asv_history)
                all_x.extend(asv_trail[:, 0])
                all_y.extend(asv_trail[:, 1])
                
            if len(self.towfish_history) > 0:
                towfish_trail = np.array(self.towfish_history)
                all_x.extend(towfish_trail[:, 0])
                all_y.extend(towfish_trail[:, 1])
            
            if all_x and all_y:
                margin = 2.0  # meters
                self.ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
                self.ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
            
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            self.get_logger().error(f'Plot update error: {e}')


def main(args=None):
    rclpy.init(args=args)
    plotter = LivePlotter()
    
    try:
        rclpy.spin(plotter)
    except KeyboardInterrupt:
        pass
    finally:
        plotter.destroy_node()
        rclpy.shutdown()
        plt.close('all')


if __name__ == '__main__':
    main()
