#!/usr/bin/env python3
"""
Generate and visualize different wave-shaped paths.
Run this to preview wave patterns before using them in simulation.
"""
import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
import os

# Add modular_controller to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modular_controller.waypoint_path import WaypointPath


def create_wave_waypoints(wavelength=10.0, amplitude=3.0, num_waves=5, points_per_wave=4):
    """
    Generate waypoints for a sinusoidal wave path.
    
    Args:
        wavelength: Distance for one complete wave cycle [m]
        amplitude: Height of wave peaks [m]
        num_waves: Number of complete wave cycles
        points_per_wave: Number of waypoints per wavelength
    
    Returns:
        List of [x, y] waypoints
    """
    total_length = wavelength * num_waves
    num_points = int(num_waves * points_per_wave) + 1
    
    x = np.linspace(0, total_length, num_points)
    y = amplitude * np.sin(2 * np.pi * x / wavelength)
    
    waypoints = [[float(xi), float(yi)] for xi, yi in zip(x, y)]
    return waypoints


def plot_wave_paths():
    """Plot all available wave path configurations."""
    
    config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
    
    # Define wave configurations to plot
    configs = {
        'Default Wave\n(waypoints.yaml)': 'waypoints.yaml',
        'High Frequency Wave': 'wave_high_freq.yaml',
        'Large Amplitude Wave': 'wave_large_amplitude.yaml',
        'Closed Loop Wave': 'wave_closed_loop.yaml',
    }
    
    # Also generate some analytical waves
    analytical = {
        'Generated: λ=8m, A=2.5m': create_wave_waypoints(wavelength=8.0, amplitude=2.5, num_waves=6),
        'Generated: λ=15m, A=4m': create_wave_waypoints(wavelength=15.0, amplitude=4.0, num_waves=4),
        'Generated: λ=5m, A=2m': create_wave_waypoints(wavelength=5.0, amplitude=2.0, num_waves=8),
    }
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    plot_idx = 0
    
    # Plot YAML configs
    for title, filename in configs.items():
        filepath = os.path.join(config_dir, filename)
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping")
            continue
            
        try:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
            
            waypoints = data.get('waypoints', [])
            path_settings = data.get('path_settings', {})
            method = path_settings.get('interpolation', 'cubic')
            closed = path_settings.get('closed_path', False)
            
            if len(waypoints) < 2:
                print(f"Warning: {filename} has insufficient waypoints")
                continue
            
            # Create path
            path = WaypointPath(waypoints, method=method, closed=closed)
            
            # Generate interpolated path
            s_max = path.get_path_length()
            s_plot = np.linspace(0, s_max, 500)
            path_points = np.array([path(s) for s in s_plot])
            
            # Plot
            ax = axes[plot_idx]
            waypoints_arr = np.array(waypoints)
            ax.plot(path_points[:, 0], path_points[:, 1], 'b-', linewidth=2, label='Interpolated Path')
            ax.plot(waypoints_arr[:, 0], waypoints_arr[:, 1], 'ro', markersize=6, label='Waypoints')
            ax.plot(waypoints_arr[0, 0], waypoints_arr[0, 1], 'go', markersize=10, label='Start')
            
            if closed:
                title += '\n(Closed Loop)'
            
            ax.set_xlabel('East [m]')
            ax.set_ylabel('North [m]')
            ax.set_title(title)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            
            # Add stats text
            stats = f"Length: {s_max:.1f}m\nPoints: {len(waypoints)}\nMethod: {method}"
            ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plot_idx += 1
            
        except Exception as e:
            print(f"Error plotting {filename}: {e}")
            continue
    
    # Plot analytical waves
    for title, waypoints in analytical.items():
        if plot_idx >= len(axes):
            break
            
        try:
            path = WaypointPath(waypoints, method='cubic', closed=False)
            
            s_max = path.get_path_length()
            s_plot = np.linspace(0, s_max, 500)
            path_points = np.array([path(s) for s in s_plot])
            
            ax = axes[plot_idx]
            waypoints_arr = np.array(waypoints)
            ax.plot(path_points[:, 0], path_points[:, 1], 'b-', linewidth=2, label='Interpolated Path')
            ax.plot(waypoints_arr[:, 0], waypoints_arr[:, 1], 'ro', markersize=4, label='Waypoints')
            ax.plot(waypoints_arr[0, 0], waypoints_arr[0, 1], 'go', markersize=10, label='Start')
            
            ax.set_xlabel('East [m]')
            ax.set_ylabel('North [m]')
            ax.set_title(title)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            
            stats = f"Length: {s_max:.1f}m\nPoints: {len(waypoints)}"
            ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plot_idx += 1
            
        except Exception as e:
            print(f"Error plotting {title}: {e}")
            continue
    
    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('wave_paths_preview.png', dpi=150, bbox_inches='tight')
    print("\nSaved: wave_paths_preview.png")
    plt.show()


def generate_custom_wave(output_file, wavelength=10.0, amplitude=3.0, num_waves=5):
    """
    Generate a custom wave YAML file.
    
    Args:
        output_file: Output YAML filename
        wavelength: Distance for one complete wave [m]
        amplitude: Wave amplitude [m]
        num_waves: Number of wave cycles
    """
    waypoints = create_wave_waypoints(wavelength, amplitude, num_waves, points_per_wave=4)
    
    data = {
        'waypoints': waypoints,
        'path_settings': {
            'interpolation': 'cubic',
            'resolution': 50,
            'closed_path': False
        }
    }
    
    config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
    filepath = os.path.join(config_dir, output_file)
    
    with open(filepath, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"Generated custom wave: {filepath}")
    print(f"  Wavelength: {wavelength}m")
    print(f"  Amplitude: {amplitude}m")
    print(f"  Number of waves: {num_waves}")
    print(f"  Total length: ~{wavelength * num_waves:.1f}m")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate and visualize wave paths')
    parser.add_argument('--plot', action='store_true', help='Plot all available wave paths')
    parser.add_argument('--generate', type=str, help='Generate custom wave YAML file')
    parser.add_argument('--wavelength', type=float, default=10.0, help='Wavelength [m]')
    parser.add_argument('--amplitude', type=float, default=3.0, help='Amplitude [m]')
    parser.add_argument('--num-waves', type=int, default=5, help='Number of waves')
    
    args = parser.parse_args()
    
    if args.generate:
        generate_custom_wave(args.generate, args.wavelength, args.amplitude, args.num_waves)
    elif args.plot:
        plot_wave_paths()
    else:
        print("Plotting all wave paths...")
        plot_wave_paths()


if __name__ == '__main__':
    main()
