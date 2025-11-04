"""
Integration test: Verify LOS guidance works correctly with waypoint paths.
Tests the complete chain: waypoints → path → LOS → velocity commands
"""
import pytest
import numpy as np
import jax.numpy as jnp
from modular_controller.waypoint_path import WaypointPath
from modular_controller.los_guidance import LOSGuidance
import tempfile
import yaml


class TestLOSWithWaypoints:
    """Test LOS guidance with waypoint-based paths"""
    
    def test_straight_line_following(self):
        """Test following a straight line path"""
        # Create simple straight path
        waypoints = [[0.0, 0.0], [50.0, 0.0]]
        path = WaypointPath(waypoints, method='cubic')  # cubic for arc-length param
        
        # Setup LOS
        los = LOSGuidance()
        los.los_parameters(U=2.0, delta=5.0, k=0.1)
        
        # Vehicle on track (20% along path)
        s = 0.2 * path.get_path_length()
        path_pos = path(s)
        vehicle_pos = np.array([float(path_pos[0]), float(path_pos[1])])
        
        v_ref, s_dot = los.compute(vehicle_pos, s, path)
        
        # Should mostly point forward
        assert v_ref[0] > 1.5  # mostly forward
        assert abs(v_ref[1]) < 0.5  # minimal lateral
        assert s_dot > 0.0  # moving forward
        
    def test_off_track_convergence(self):
        """Test that vehicle off-track gets commanded back to path"""
        waypoints = [[0.0, 0.0], [50.0, 0.0]]
        path = WaypointPath(waypoints, method='cubic')  # cubic for arc-length param
        
        los = LOSGuidance()
        los.los_parameters(U=2.0, delta=5.0, k=0.1)
        
        # Vehicle above the path (20% along, but 5m off-track)
        s = 0.2 * path.get_path_length()
        path_pos = path(s)
        vehicle_pos = np.array([float(path_pos[0]), float(path_pos[1]) + 5.0])
        
        v_ref, s_dot = los.compute(vehicle_pos, s, path)
        
        # Should command negative y (back toward path)
        assert v_ref[1] < 0.0
        assert v_ref[0] > 0.0  # still moving forward
        
    def test_curved_path_following(self):
        """Test following a curved path"""
        waypoints = [
            [0.0, 0.0],
            [10.0, 5.0],
            [20.0, 5.0],
            [30.0, 0.0],
        ]
        path = WaypointPath(waypoints, method='cubic', resolution=50)
        
        los = LOSGuidance()
        los.los_parameters(U=1.5, delta=3.0, k=0.2)
        
        # Test at various points along path
        s_values = [0.1 * path.get_path_length(), 
                    0.5 * path.get_path_length(),
                    0.9 * path.get_path_length()]
        
        for s in s_values:
            path_pos = path(s)
            vehicle_pos = np.array([float(path_pos[0]), float(path_pos[1])])
            
            v_ref, s_dot = los.compute(vehicle_pos, s, path)
            
            # Should have reasonable velocity
            v_mag = float(np.linalg.norm(v_ref))
            assert 0.5 < v_mag < 2.5
            assert s_dot > 0.0
            
    def test_path_completion(self):
        """Test behavior at end of path"""
        waypoints = [[0.0, 0.0], [20.0, 0.0]]
        path = WaypointPath(waypoints, method='linear')
        
        los = LOSGuidance()
        los.los_parameters(U=2.0, delta=5.0, k=0.1)
        
        # Near end of path
        s_end = path.get_path_length() - 0.1
        path_pos = path(s_end)
        vehicle_pos = np.array([float(path_pos[0]), float(path_pos[1])])
        
        v_ref, s_dot = los.compute(vehicle_pos, s_end, path)
        
        # Should still produce valid commands
        assert np.isfinite(v_ref[0])
        assert np.isfinite(v_ref[1])
        assert np.isfinite(s_dot)
        
    def test_simulate_trajectory(self):
        """Simulate a full trajectory following a path"""
        waypoints = [
            [0.0, 0.0],
            [15.0, 10.0],
            [30.0, 5.0],
            [45.0, 0.0],
        ]
        path = WaypointPath(waypoints, method='cubic', resolution=50)
        
        los = LOSGuidance()
        U = 2.0
        los.los_parameters(U=U, delta=5.0, k=0.15)
        
        # Start at beginning
        s = 0.0
        dt = 0.1  # 10 Hz
        
        # Initialize vehicle at start
        start_pos = path(0.0)
        vehicle_pos = np.array([float(start_pos[0]), float(start_pos[1])])
        
        trajectory = [vehicle_pos.copy()]
        s_history = [s]
        
        # Simulate for limited time
        max_steps = 500
        for step in range(max_steps):
            if s >= path.get_path_length():
                break
                
            # Compute LOS guidance
            v_ref, s_dot = los.compute(vehicle_pos, s, path)
            
            # Simple integration (assuming perfect tracking)
            vehicle_pos += np.array([float(v_ref[0]), float(v_ref[1])]) * dt
            s += s_dot * dt
            s = min(s, path.get_path_length())
            
            trajectory.append(vehicle_pos.copy())
            s_history.append(s)
        
        trajectory = np.array(trajectory)
        
        # Verify trajectory makes sense
        assert len(trajectory) > 10  # Should move
        assert s > path.get_path_length() * 0.5  # Made significant progress
        
        # Check that trajectory stays reasonably close to path
        max_error = 0.0
        for i, s_val in enumerate(s_history):
            path_pos = path(min(s_val, path.get_path_length()))
            error = np.linalg.norm(trajectory[i] - np.array([float(path_pos[0]), float(path_pos[1])]))
            max_error = max(max_error, error)
        
        # Max error should be reasonable (within a few deltas)
        assert max_error < 20.0  # Allow some overshoot in curves
        
    def test_different_interpolations_consistency(self):
        """Test that different interpolation methods give consistent results"""
        waypoints = [[0.0, 0.0], [10.0, 5.0], [20.0, 0.0]]
        
        los = LOSGuidance()
        los.los_parameters(U=1.5, delta=4.0, k=0.1)
        
        results = {}
        for method in ['linear', 'cubic', 'bspline']:
            path = WaypointPath(waypoints, method=method, resolution=50)
            
            # Test at 30% along path
            s = 0.3 * path.get_path_length()
            path_pos = path(s)
            vehicle_pos = np.array([float(path_pos[0]), float(path_pos[1])])
            
            v_ref, s_dot = los.compute(vehicle_pos, s, path)
            results[method] = (v_ref, s_dot)
        
        # All methods should produce forward motion
        for method, (v_ref, s_dot) in results.items():
            assert v_ref[0] > 0.0, f"{method} should move forward"
            assert s_dot > 0.0, f"{method} should advance along path"


def test_yaml_integration():
    """Test loading path from YAML and using with LOS"""
    yaml_content = """
waypoints:
  - [0.0, 0.0]
  - [10.0, 8.0]
  - [20.0, 10.0]
  - [30.0, 5.0]

path_settings:
  interpolation: cubic
  resolution: 50
  closed_path: false
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name
    
    try:
        from modular_controller.waypoint_path import load_waypoints_from_yaml
        
        path, settings = load_waypoints_from_yaml(temp_path)
        
        # Setup LOS
        los = LOSGuidance()
        los.los_parameters(U=2.0, delta=5.0, k=0.1)
        
        # Test at middle of path
        s = path.get_path_length() * 0.5
        path_pos = path(s)
        vehicle_pos = np.array([float(path_pos[0]), float(path_pos[1])])
        
        v_ref, s_dot = los.compute(vehicle_pos, s, path)
        
        # Should produce valid output
        assert np.isfinite(v_ref[0])
        assert np.isfinite(v_ref[1])
        assert s_dot > 0.0
        
    finally:
        import os
        os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
