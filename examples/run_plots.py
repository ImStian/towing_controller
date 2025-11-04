"""
Standalone plotting script - run from gz_ws directory
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from modular_controller.modular_controller.waypoint_path import WaypointPath
from modular_controller.modular_controller.los_guidance import LOSGuidance
import numpy as np
import matplotlib.pyplot as plt

print("="*60)
print("WAYPOINT PATH SYSTEM - COMPREHENSIVE TESTING & PLOTTING")
print("="*60)

# Create plots directory
os.makedirs('plots', exist_ok=True)

# Define test waypoints - interesting curved path
waypoints = [
    [0.0, 0.0],
    [15.0, 10.0],
    [30.0, 12.0],
    [45.0, 8.0],
    [60.0, 3.0],
    [75.0, 0.0],
]

print("\nTest Waypoints:")
for i, wp in enumerate(waypoints):
    print(f"  WP{i}: ({wp[0]:.1f}, {wp[1]:.1f})")

# Create paths with different methods
methods = ['linear', 'cubic', 'bspline']
paths = {}

print("\nCreating paths...")
for method in methods:
    paths[method] = WaypointPath(waypoints, method=method, resolution=50)
    print(f"  {method.upper()}: length = {paths[method].get_path_length():.2f} m")

# ===== PLOT 1: Path Comparison =====
print("\n1. Plotting path comparison...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Path Interpolation Methods', fontsize=16, fontweight='bold')

for idx, method in enumerate(methods):
    ax = axes[idx]
    path = paths[method]
    
    # Sample path
    n = 300
    s_vals = np.linspace(0, path.get_path_length(), n)
    x_path, y_path = [], []
    for s in s_vals:
        p = path(s)
        x_path.append(float(p[0]))
        y_path.append(float(p[1]))
    
    ax.plot(x_path, y_path, 'b-', linewidth=2.5, label=f'{method} path')
    
    # Plot waypoints
    wp = np.array(waypoints)
    ax.scatter(wp[:, 0], wp[:, 1], s=150, c='red', marker='o',
              edgecolors='black', linewidth=2, zorder=5, label='waypoints')
    
    # Start/end
    ax.scatter(x_path[0], y_path[0], s=200, c='green', marker='*',
              edgecolors='black', linewidth=2, zorder=6, label='start')
    ax.scatter(x_path[-1], y_path[-1], s=200, c='red', marker='s',
              edgecolors='black', linewidth=2, zorder=6, label='end')
    
    ax.set_xlabel('X [m]', fontsize=12)
    ax.set_ylabel('Y [m]', fontsize=12)
    ax.set_title(f'{method.upper()}\nLength: {path.get_path_length():.2f} m',
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.axis('equal')

plt.tight_layout()
plt.savefig('plots/01_path_comparison.png', dpi=150)
print("   ✓ Saved: plots/01_path_comparison.png")
plt.close()

# ===== PLOT 2: Trajectory Simulation =====
print("\n2. Simulating trajectory...")
path = paths['cubic']
los = LOSGuidance()
U = 2.0
los.los_parameters(U=U, delta=5.0, k=0.15)

# Start slightly off-track
start_pos = path(0.0)
vehicle_pos = np.array([float(start_pos[0]) + 2.0, float(start_pos[1]) - 1.5])
s = 0.0
dt = 0.1

trajectory = [vehicle_pos.copy()]
s_history = [s]
time_history = [0.0]
cte_history = []

print(f"   Initial position: ({vehicle_pos[0]:.2f}, {vehicle_pos[1]:.2f})")
print(f"   Simulating for max 100 seconds...")

time = 0.0
for step in range(1000):
    if s >= path.get_path_length():
        break
    
    v_ref, s_dot = los.compute(vehicle_pos, s, path)
    
    # Cross-track error
    path_pos = path(s)
    cte = np.linalg.norm(vehicle_pos - np.array([float(path_pos[0]), float(path_pos[1])]))
    cte_history.append(cte)
    
    # Integrate
    vehicle_pos += np.array([float(v_ref[0]), float(v_ref[1])]) * dt
    s += s_dot * dt
    s = min(s, path.get_path_length())
    time += dt
    
    trajectory.append(vehicle_pos.copy())
    s_history.append(s)
    time_history.append(time)

trajectory = np.array(trajectory)

print(f"   Completed in {time:.2f} seconds ({len(trajectory)} steps)")
print(f"   Path completion: {s_history[-1]/path.get_path_length()*100:.1f}%")
print(f"   Mean CTE: {np.mean(cte_history):.3f} m")
print(f"   Max CTE: {np.max(cte_history):.3f} m")

# Plot trajectory
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Trajectory Simulation Results', fontsize=16, fontweight='bold')

# 1. Trajectory
ax = axes[0, 0]
n = 200
s_vals = np.linspace(0, path.get_path_length(), n)
x_path, y_path = [], []
for s in s_vals:
    p = path(s)
    x_path.append(float(p[0]))
    y_path.append(float(p[1]))

ax.plot(x_path, y_path, 'k--', linewidth=2, label='Desired Path', alpha=0.7)
ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Vehicle Trajectory')

wp = np.array(waypoints)
ax.scatter(wp[:, 0], wp[:, 1], s=100, c='orange', marker='o',
          edgecolors='black', linewidth=2, zorder=5)
ax.scatter(trajectory[0, 0], trajectory[0, 1], s=200, c='green',
          marker='*', edgecolors='black', linewidth=2, zorder=6, label='Start')
ax.scatter(trajectory[-1, 0], trajectory[-1, 1], s=200, c='red',
          marker='s', edgecolors='black', linewidth=2, zorder=6, label='End')

ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_title('Trajectory Following')
ax.grid(True, alpha=0.3)
ax.legend()
ax.axis('equal')

# 2. Cross-track error
ax = axes[0, 1]
ax.plot(time_history[1:], cte_history, 'r-', linewidth=2)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Cross-Track Error [m]')
ax.set_title(f'Cross-Track Error (mean: {np.mean(cte_history):.3f} m)')
ax.grid(True, alpha=0.3)

# 3. Path progress
ax = axes[1, 0]
ax.plot(time_history, s_history, 'g-', linewidth=2)
ax.axhline(path.get_path_length(), color='k', linestyle='--',
          label=f'Path End ({path.get_path_length():.1f} m)')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Path Parameter s [m]')
ax.set_title('Path Progress')
ax.grid(True, alpha=0.3)
ax.legend()

# 4. Statistics
ax = axes[1, 1]
ax.axis('off')
stats_text = f"""
SIMULATION STATISTICS

Waypoints: {len(waypoints)}
Path Method: CUBIC
Path Length: {path.get_path_length():.2f} m

Simulation Time: {time:.2f} s
Total Steps: {len(trajectory)}

Cross-Track Error:
  Mean: {np.mean(cte_history):.3f} m
  Max:  {np.max(cte_history):.3f} m
  Final: {cte_history[-1]:.3f} m

Path Completion: {s_history[-1]/path.get_path_length()*100:.1f}%

LOS Parameters:
  U = {U:.1f} m/s
  delta = 5.0 m
  k = 0.15
"""
ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
       verticalalignment='center')

plt.tight_layout()
plt.savefig('plots/02_trajectory_simulation.png', dpi=150)
print("   ✓ Saved: plots/02_trajectory_simulation.png")
plt.close()

# ===== PLOT 3: LOS Parameter Sensitivity =====
print("\n3. Testing parameter sensitivity...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('LOS Parameter Sensitivity', fontsize=16, fontweight='bold')

# Test point
s_test = path.get_path_length() * 0.4
path_pos = path(s_test)
vehicle_pos = np.array([float(path_pos[0]), float(path_pos[1]) + 3.0])  # 3m off track

# Sample path for background
n = 200
s_vals = np.linspace(0, path.get_path_length(), n)
x_path, y_path = [], []
for s in s_vals:
    p = path(s)
    x_path.append(float(p[0]))
    y_path.append(float(p[1]))

# Vary U
ax = axes[0]
ax.plot(x_path, y_path, 'k--', linewidth=2, alpha=0.3)
for U_val in [1.0, 2.0, 3.0]:
    los_test = LOSGuidance()
    los_test.los_parameters(U=U_val, delta=5.0, k=0.15)
    v_ref, _ = los_test.compute(vehicle_pos, s_test, path)
    scale = 2.0
    ax.arrow(vehicle_pos[0], vehicle_pos[1],
            float(v_ref[0])*scale, float(v_ref[1])*scale,
            head_width=1.0, head_length=0.8, linewidth=2, label=f'U={U_val}')
ax.scatter(*vehicle_pos, s=150, c='red', marker='x', linewidths=3)
ax.set_title('Varying Speed (U)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axis('equal')

# Vary delta
ax = axes[1]
ax.plot(x_path, y_path, 'k--', linewidth=2, alpha=0.3)
for delta_val in [3.0, 5.0, 8.0]:
    los_test = LOSGuidance()
    los_test.los_parameters(U=2.0, delta=delta_val, k=0.15)
    v_ref, _ = los_test.compute(vehicle_pos, s_test, path)
    scale = 2.0
    ax.arrow(vehicle_pos[0], vehicle_pos[1],
            float(v_ref[0])*scale, float(v_ref[1])*scale,
            head_width=1.0, head_length=0.8, linewidth=2, label=f'δ={delta_val}')
ax.scatter(*vehicle_pos, s=150, c='red', marker='x', linewidths=3)
ax.set_title('Varying Lookahead (δ)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axis('equal')

# Vary k
ax = axes[2]
ax.plot(x_path, y_path, 'k--', linewidth=2, alpha=0.3)
for k_val in [0.05, 0.15, 0.3]:
    los_test = LOSGuidance()
    los_test.los_parameters(U=2.0, delta=5.0, k=k_val)
    v_ref, _ = los_test.compute(vehicle_pos, s_test, path)
    scale = 2.0
    ax.arrow(vehicle_pos[0], vehicle_pos[1],
            float(v_ref[0])*scale, float(v_ref[1])*scale,
            head_width=1.0, head_length=0.8, linewidth=2, label=f'k={k_val}')
ax.scatter(*vehicle_pos, s=150, c='red', marker='x', linewidths=3)
ax.set_title('Varying Path Gain (k)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axis('equal')

plt.tight_layout()
plt.savefig('plots/03_parameter_sensitivity.png', dpi=150)
print("   ✓ Saved: plots/03_parameter_sensitivity.png")
plt.close()

print("\n" + "="*60)
print("✓ ALL PLOTS COMPLETE")
print("="*60)
print("\nGenerated files in plots/:")
print("  - 01_path_comparison.png")
print("  - 02_trajectory_simulation.png")
print("  - 03_parameter_sensitivity.png")
print("\nSystem verification complete!")
print("="*60 + "\n")
