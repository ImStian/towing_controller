"""
Comprehensive plotting tool for waypoint path system verification.
Creates multiple plots showing:
1. Path comparison (different interpolation methods)
2. LOS velocity vectors along path
3. Simulated trajectory following
4. Cross-track error over time
5. Path parameter evolution
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from modular_controller.waypoint_path import WaypointPath
from modular_controller.los_guidance import LOSGuidance
import jax.numpy as jnp


def plot_path_comparison(waypoints, save_path='plots/path_comparison.png'):
    """Plot different interpolation methods side by side"""
    methods = ['linear', 'cubic', 'bspline']
    colors = ['blue', 'green', 'red']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Path Interpolation Methods Comparison', fontsize=16, fontweight='bold')
    
    for idx, (method, color) in enumerate(zip(methods, colors)):
        ax = axes[idx]
        path = WaypointPath(waypoints, method=method, resolution=50)
        
        # Sample path
        n_samples = 300
        s_values = np.linspace(0, path.get_path_length(), n_samples)
        x_path, y_path = [], []
        
        for s in s_values:
            p = path(s)
            x_path.append(float(p[0]))
            y_path.append(float(p[1]))
        
        # Plot path
        ax.plot(x_path, y_path, color=color, linewidth=2.5, label=f'{method} path')
        
        # Plot waypoints
        wp = np.array(waypoints)
        ax.scatter(wp[:, 0], wp[:, 1], s=150, c='black', marker='o', 
                  edgecolors='white', linewidth=2, zorder=5, label='waypoints')
        
        # Number waypoints
        for i, (x, y) in enumerate(wp):
            ax.text(x, y+0.5, f'WP{i}', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold')
        
        # Mark start and end
        ax.scatter(x_path[0], y_path[0], s=200, c='green', marker='*', 
                  edgecolors='black', linewidth=2, zorder=6, label='start')
        ax.scatter(x_path[-1], y_path[-1], s=200, c='red', marker='s', 
                  edgecolors='black', linewidth=2, zorder=6, label='end')
        
        ax.set_xlabel('X [m]', fontsize=12)
        ax.set_ylabel('Y [m]', fontsize=12)
        ax.set_title(f'{method.upper()}\nLength: {path.get_path_length():.2f} m', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved path comparison to {save_path}")
    return fig


def plot_los_velocity_field(waypoints, method='cubic', save_path='plots/los_velocity_field.png'):
    """Plot LOS velocity vectors at various points along and around the path"""
    path = WaypointPath(waypoints, method=method, resolution=50)
    
    los = LOSGuidance()
    U = 2.0
    los.los_parameters(U=U, delta=5.0, k=0.15)
    
    # Sample path
    n_path_samples = 200
    s_values = np.linspace(0, path.get_path_length(), n_path_samples)
    x_path, y_path = [], []
    
    for s in s_values:
        p = path(s)
        x_path.append(float(p[0]))
        y_path.append(float(p[1]))
    
    # Create grid around path
    x_min, x_max = min(x_path) - 5, max(x_path) + 5
    y_min, y_max = min(y_path) - 5, max(y_path) + 5
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Plot the path
    ax.plot(x_path, y_path, 'k-', linewidth=3, label='Path', zorder=2)
    
    # Plot waypoints
    wp = np.array(waypoints)
    ax.scatter(wp[:, 0], wp[:, 1], s=150, c='orange', marker='o',
              edgecolors='black', linewidth=2, zorder=5)
    
    # Sample points along path at different offsets
    n_along = 15
    offsets = [-4, -2, 0, 2, 4]
    
    for i, s in enumerate(np.linspace(0.1 * path.get_path_length(), 
                                       0.9 * path.get_path_length(), n_along)):
        path_pos = path(s)
        path_x, path_y = float(path_pos[0]), float(path_pos[1])
        
        # Get path tangent for perpendicular
        ds = 0.01
        path_pos_next = path(min(s + ds, path.get_path_length()))
        dx = float(path_pos_next[0]) - path_x
        dy = float(path_pos_next[1]) - path_y
        norm = np.sqrt(dx**2 + dy**2)
        if norm > 1e-6:
            tx, ty = dx/norm, dy/norm
            perp_x, perp_y = -ty, tx  # perpendicular
        else:
            perp_x, perp_y = 0, 1
        
        for offset in offsets:
            test_x = path_x + offset * perp_x
            test_y = path_y + offset * perp_y
            
            vehicle_pos = np.array([test_x, test_y])
            
            try:
                v_ref, s_dot = los.compute(vehicle_pos, s, path)
                
                # Plot velocity vector
                scale = 1.5
                color = 'blue' if abs(offset) < 0.1 else 'red'
                alpha = 1.0 if abs(offset) < 0.1 else 0.5
                
                ax.arrow(test_x, test_y, 
                        float(v_ref[0]) * scale, float(v_ref[1]) * scale,
                        head_width=0.3, head_length=0.2, fc=color, ec=color,
                        alpha=alpha, width=0.05, zorder=3)
            except:
                pass
    
    ax.set_xlabel('X [m]', fontsize=12)
    ax.set_ylabel('Y [m]', fontsize=12)
    ax.set_title('LOS Velocity Field\n(Blue=on path, Red=off path)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved LOS velocity field to {save_path}")
    return fig


def simulate_and_plot_trajectory(waypoints, method='cubic', 
                                 initial_offset=(0, 0),
                                 save_path='plots/trajectory_simulation.png'):
    """Simulate vehicle following path and plot results"""
    path = WaypointPath(waypoints, method=method, resolution=50)
    
    los = LOSGuidance()
    U = 2.0
    los.los_parameters(U=U, delta=5.0, k=0.15)
    
    # Initialize
    start_pos = path(0.0)
    vehicle_pos = np.array([float(start_pos[0]) + initial_offset[0], 
                           float(start_pos[1]) + initial_offset[1]])
    s = 0.0
    dt = 0.1
    
    # Storage
    trajectory = [vehicle_pos.copy()]
    s_history = [s]
    v_ref_history = []
    cross_track_error = []
    time_history = [0.0]
    
    # Simulate
    max_steps = 1000
    time = 0.0
    
    for step in range(max_steps):
        if s >= path.get_path_length():
            break
        
        # LOS guidance
        v_ref, s_dot = los.compute(vehicle_pos, s, path)
        v_ref_history.append(np.array([float(v_ref[0]), float(v_ref[1])]))
        
        # Compute cross-track error
        path_pos = path(s)
        cte = np.linalg.norm(vehicle_pos - np.array([float(path_pos[0]), float(path_pos[1])]))
        cross_track_error.append(cte)
        
        # Simple integration (assume perfect velocity tracking)
        vehicle_pos += np.array([float(v_ref[0]), float(v_ref[1])]) * dt
        s += s_dot * dt
        s = min(s, path.get_path_length())
        time += dt
        
        trajectory.append(vehicle_pos.copy())
        s_history.append(s)
        time_history.append(time)
    
    trajectory = np.array(trajectory)
    v_ref_history = np.array(v_ref_history)
    
    # Sample path for plotting
    n_path = 200
    s_path = np.linspace(0, path.get_path_length(), n_path)
    x_path, y_path = [], []
    for s in s_path:
        p = path(s)
        x_path.append(float(p[0]))
        y_path.append(float(p[1]))
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Trajectory vs Path
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(x_path, y_path, 'k--', linewidth=2, label='Desired Path', alpha=0.7)
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Vehicle Trajectory')
    
    # Plot waypoints
    wp = np.array(waypoints)
    ax1.scatter(wp[:, 0], wp[:, 1], s=150, c='orange', marker='o',
               edgecolors='black', linewidth=2, zorder=5, label='Waypoints')
    
    # Mark start and end
    ax1.scatter(trajectory[0, 0], trajectory[0, 1], s=200, c='green', 
               marker='*', edgecolors='black', linewidth=2, zorder=6, label='Start')
    ax1.scatter(trajectory[-1, 0], trajectory[-1, 1], s=200, c='red',
               marker='s', edgecolors='black', linewidth=2, zorder=6, label='End')
    
    # Plot some velocity vectors along trajectory
    vec_interval = max(1, len(trajectory) // 20)
    for i in range(0, len(v_ref_history), vec_interval):
        scale = 2.0
        ax1.arrow(trajectory[i, 0], trajectory[i, 1],
                 v_ref_history[i, 0] * scale, v_ref_history[i, 1] * scale,
                 head_width=0.4, head_length=0.3, fc='purple', ec='purple',
                 alpha=0.4, width=0.08)
    
    ax1.set_xlabel('X [m]', fontsize=12)
    ax1.set_ylabel('Y [m]', fontsize=12)
    ax1.set_title(f'Trajectory Following ({method} interpolation)', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc='best')
    ax1.axis('equal')
    
    # 2. Cross-track error over time
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time_history[1:], cross_track_error, 'r-', linewidth=2)
    ax2.set_xlabel('Time [s]', fontsize=11)
    ax2.set_ylabel('Cross-Track Error [m]', fontsize=11)
    ax2.set_title('Cross-Track Error vs Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Path parameter evolution
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(time_history, s_history, 'g-', linewidth=2)
    ax3.axhline(path.get_path_length(), color='k', linestyle='--', 
               label=f'Path End ({path.get_path_length():.1f} m)')
    ax3.set_xlabel('Time [s]', fontsize=11)
    ax3.set_ylabel('Path Parameter s [m]', fontsize=11)
    ax3.set_title('Path Progress vs Time', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    
    # 4. Velocity magnitude over time
    ax4 = fig.add_subplot(gs[2, 0])
    v_mag = np.linalg.norm(v_ref_history, axis=1)
    ax4.plot(time_history[1:], v_mag, 'b-', linewidth=2)
    ax4.axhline(U, color='r', linestyle='--', label=f'Desired U={U} m/s')
    ax4.set_xlabel('Time [s]', fontsize=11)
    ax4.set_ylabel('Velocity Magnitude [m/s]', fontsize=11)
    ax4.set_title('Commanded Velocity Magnitude', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    
    # 5. Statistics
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    stats_text = f"""
    SIMULATION STATISTICS
    
    Path Method: {method.upper()}
    Path Length: {path.get_path_length():.2f} m
    Simulation Time: {time:.2f} s
    Total Steps: {len(trajectory)}
    
    Initial Offset: ({initial_offset[0]:.1f}, {initial_offset[1]:.1f}) m
    
    Cross-Track Error:
      Mean: {np.mean(cross_track_error):.3f} m
      Max:  {np.max(cross_track_error):.3f} m
      Final: {cross_track_error[-1]:.3f} m
    
    Path Completion: {(s_history[-1]/path.get_path_length()*100):.1f}%
    
    Velocity:
      Mean: {np.mean(v_mag):.3f} m/s
      Desired: {U:.3f} m/s
    """
    
    ax5.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved trajectory simulation to {save_path}")
    return fig, trajectory, time_history, cross_track_error


def plot_sensitivity_analysis(waypoints, save_path='plots/sensitivity_analysis.png'):
    """Test sensitivity to LOS parameters"""
    path = WaypointPath(waypoints, method='cubic', resolution=50)
    
    # Parameter variations
    U_values = [1.0, 2.0, 3.0]
    delta_values = [3.0, 5.0, 8.0]
    k_values = [0.05, 0.15, 0.3]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('LOS Parameter Sensitivity Analysis', fontsize=16, fontweight='bold')
    
    # Sample path
    n_path = 200
    s_path = np.linspace(0, path.get_path_length(), n_path)
    x_path, y_path = [], []
    for s in s_path:
        p = path(s)
        x_path.append(float(p[0]))
        y_path.append(float(p[1]))
    
    # Test point
    s_test = path.get_path_length() * 0.4
    path_pos = path(s_test)
    test_offset = 3.0
    vehicle_pos = np.array([float(path_pos[0]), float(path_pos[1]) + test_offset])
    
    # Vary U
    ax = axes[0, 0]
    ax.plot(x_path, y_path, 'k--', linewidth=2, alpha=0.5)
    for U in U_values:
        los = LOSGuidance()
        los.los_parameters(U=U, delta=5.0, k=0.15)
        v_ref, _ = los.compute(vehicle_pos, s_test, path)
        scale = 2.0
        ax.arrow(vehicle_pos[0], vehicle_pos[1],
                float(v_ref[0])*scale, float(v_ref[1])*scale,
                head_width=0.5, head_length=0.4, linewidth=2,
                label=f'U={U}')
    ax.scatter(*vehicle_pos, s=100, c='red', marker='x', linewidths=3)
    ax.set_title('Varying Surge Speed (U)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Vary delta
    ax = axes[0, 1]
    ax.plot(x_path, y_path, 'k--', linewidth=2, alpha=0.5)
    for delta in delta_values:
        los = LOSGuidance()
        los.los_parameters(U=2.0, delta=delta, k=0.15)
        v_ref, _ = los.compute(vehicle_pos, s_test, path)
        scale = 2.0
        ax.arrow(vehicle_pos[0], vehicle_pos[1],
                float(v_ref[0])*scale, float(v_ref[1])*scale,
                head_width=0.5, head_length=0.4, linewidth=2,
                label=f'δ={delta}')
    ax.scatter(*vehicle_pos, s=100, c='red', marker='x', linewidths=3)
    ax.set_title('Varying Lookahead (δ)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Vary k
    ax = axes[1, 0]
    ax.plot(x_path, y_path, 'k--', linewidth=2, alpha=0.5)
    for k in k_values:
        los = LOSGuidance()
        los.los_parameters(U=2.0, delta=5.0, k=k)
        v_ref, _ = los.compute(vehicle_pos, s_test, path)
        scale = 2.0
        ax.arrow(vehicle_pos[0], vehicle_pos[1],
                float(v_ref[0])*scale, float(v_ref[1])*scale,
                head_width=0.5, head_length=0.4, linewidth=2,
                label=f'k={k}')
    ax.scatter(*vehicle_pos, s=100, c='red', marker='x', linewidths=3)
    ax.set_title('Varying Path Gain (k)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Summary
    ax = axes[1, 1]
    ax.axis('off')
    summary = """
    PARAMETER EFFECTS
    
    U (Surge Speed):
      - Higher U → faster forward motion
      - Affects velocity magnitude
      
    δ (Lookahead Distance):
      - Higher δ → smoother convergence
      - Lower δ → tighter tracking
      - Affects lateral velocity component
      
    k (Path Gain):
      - Higher k → faster path progression
      - Affects along-track motion
      - Should be tuned with vehicle dynamics
    
    Typical Values:
      U: 1-3 m/s (vehicle dependent)
      δ: 3-8 m (2-5× vehicle length)
      k: 0.05-0.3 (tune for stability)
    """
    ax.text(0.05, 0.5, summary, fontsize=10, family='monospace',
           verticalalignment='center')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved sensitivity analysis to {save_path}")
    return fig


def run_all_plots(waypoints=None):
    """Run all plotting functions"""
    import os
    os.makedirs('plots', exist_ok=True)
    
    if waypoints is None:
        # Default test waypoints - interesting curved path
        waypoints = [
            [0.0, 0.0],
            [15.0, 10.0],
            [30.0, 12.0],
            [45.0, 8.0],
            [60.0, 3.0],
            [75.0, 0.0],
        ]
    
    print("\n" + "="*60)
    print("WAYPOINT PATH SYSTEM - COMPREHENSIVE PLOTTING")
    print("="*60)
    
    print("\n1. Path Comparison (different interpolation methods)...")
    plot_path_comparison(waypoints)
    
    print("\n2. LOS Velocity Field...")
    plot_los_velocity_field(waypoints)
    
    print("\n3. Trajectory Simulation (on-track start)...")
    simulate_and_plot_trajectory(waypoints, initial_offset=(0, 0),
                                save_path='plots/trajectory_on_track.png')
    
    print("\n4. Trajectory Simulation (off-track start)...")
    simulate_and_plot_trajectory(waypoints, initial_offset=(5, -3),
                                save_path='plots/trajectory_off_track.png')
    
    print("\n5. Sensitivity Analysis...")
    plot_sensitivity_analysis(waypoints)
    
    print("\n" + "="*60)
    print("✓ ALL PLOTS COMPLETE")
    print("="*60)
    print("\nPlots saved in: plots/")
    print("  - path_comparison.png")
    print("  - los_velocity_field.png")
    print("  - trajectory_on_track.png")
    print("  - trajectory_off_track.png")
    print("  - sensitivity_analysis.png")
    print("\n")


if __name__ == "__main__":
    run_all_plots()
    
    # Keep plots open
    plt.show()
