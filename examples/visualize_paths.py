"""
Example: Visualize waypoint paths with different interpolation methods
"""
import numpy as np
import matplotlib.pyplot as plt
from modular_controller.waypoint_path import WaypointPath

# Define some example waypoints
waypoints = [
    [0.0, 0.0],
    [10.0, 5.0],
    [20.0, 8.0],
    [30.0, 5.0],
    [40.0, 2.0],
    [50.0, 0.0],
]

# Create paths with different interpolation methods
methods = ['linear', 'cubic', 'bspline']
paths = {}

for method in methods:
    paths[method] = WaypointPath(waypoints, method=method, resolution=50)

# Sample each path densely for plotting
n_samples = 500

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, method in enumerate(methods):
    ax = axes[idx]
    path = paths[method]
    
    # Sample the path
    s_values = np.linspace(0, path.get_path_length(), n_samples)
    x_path = []
    y_path = []
    
    for s in s_values:
        p = path(s)
        x_path.append(float(p[0]))
        y_path.append(float(p[1]))
    
    # Plot
    ax.plot(x_path, y_path, 'b-', linewidth=2, label=f'{method} path')
    
    # Plot waypoints
    wp_array = np.array(waypoints)
    ax.plot(wp_array[:, 0], wp_array[:, 1], 'ro', markersize=10, label='waypoints')
    
    # Mark start and end
    ax.plot(x_path[0], y_path[0], 'go', markersize=12, label='start')
    ax.plot(x_path[-1], y_path[-1], 'rs', markersize=12, label='end')
    
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title(f'{method.capitalize()} Interpolation\nLength: {path.get_path_length():.2f} m')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axis('equal')

plt.tight_layout()
plt.savefig('waypoint_paths.png', dpi=150)
print("Saved visualization to waypoint_paths.png")

# Print path information
print("\nPath Information:")
print("-" * 50)
for method in methods:
    path = paths[method]
    print(f"{method.capitalize():10s}: Length = {path.get_path_length():.2f} m")

# Test LOS guidance with the cubic path
print("\n" + "="*50)
print("Testing LOS Guidance with Cubic Path")
print("="*50)

from modular_controller.los_guidance import LOSGuidance

los = LOSGuidance()
los.los_parameters(U=2.0, delta=5.0, k=0.1)

# Simulate a vehicle slightly off-track
path = paths['cubic']
s = path.get_path_length() * 0.3  # 30% along the path

# Path position
path_pos = path(s)
print(f"\nPath position at s={s:.2f}: [{path_pos[0]:.2f}, {path_pos[1]:.2f}]")

# Vehicle position (slightly off track)
vehicle_pos = np.array([float(path_pos[0]) + 2.0, float(path_pos[1]) - 1.5])
print(f"Vehicle position: [{vehicle_pos[0]:.2f}, {vehicle_pos[1]:.2f}]")

# Compute LOS guidance
v_ref, s_dot = los.compute(vehicle_pos, s, path)

print(f"\nLOS Guidance Output:")
print(f"  Desired velocity: [{v_ref[0]:.3f}, {v_ref[1]:.3f}] m/s")
print(f"  Path rate: {s_dot:.3f} m/s")
print(f"  Velocity magnitude: {np.linalg.norm(v_ref):.3f} m/s")

plt.show()
