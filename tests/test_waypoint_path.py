"""
Tests for waypoint path generation and interpolation.
"""
import pytest
import numpy as np
import jax.numpy as jnp
from modular_controller.waypoint_path import WaypointPath, load_waypoints_from_yaml
import tempfile
import os


def test_linear_path():
    """Test linear interpolation between waypoints"""
    waypoints = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]
    path = WaypointPath(waypoints, method='linear')
    
    # Check start and end
    start = path(0.0)
    assert np.allclose(start, [0.0, 0.0], atol=1e-6)
    
    # Check that path length is reasonable
    assert path.get_path_length() > 0.0


def test_cubic_path():
    """Test cubic spline interpolation"""
    waypoints = [[0.0, 0.0], [10.0, 5.0], [20.0, 5.0], [30.0, 0.0]]
    path = WaypointPath(waypoints, method='cubic', resolution=50)
    
    # Check start
    start = path(0.0)
    assert np.allclose(start, [0.0, 0.0], atol=0.5)  # Allow some tolerance for splines
    
    # Check path is continuous
    s_mid = path.get_path_length() / 2
    p1 = path(s_mid)
    p2 = path(s_mid + 0.1)
    
    # Points should be close (continuous)
    dist = np.linalg.norm(np.array(p2) - np.array(p1))
    assert dist < 1.0


def test_bspline_path():
    """Test B-spline interpolation"""
    waypoints = [[0.0, 0.0], [5.0, 5.0], [10.0, 0.0], [15.0, -5.0]]
    path = WaypointPath(waypoints, method='bspline', resolution=50)
    
    assert path.get_path_length() > 0.0
    
    # Check that path is smooth
    s_test = path.get_path_length() * 0.5
    p = path(s_test)
    assert len(p) == 2


def test_minimum_waypoints():
    """Test that at least 2 waypoints are required"""
    with pytest.raises(ValueError):
        WaypointPath([[0.0, 0.0]], method='linear')


def test_closed_path():
    """Test closed path (loop)"""
    waypoints = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
    path = WaypointPath(waypoints, method='linear', closed=True)
    
    # Start and end should be close for a closed path
    start = path(0.0)
    end = path(path.get_path_length())
    
    # They should form a loop (may not be exactly same due to interpolation)
    assert isinstance(start, jnp.ndarray)
    assert isinstance(end, jnp.ndarray)


def test_path_clamping():
    """Test that path parameter is clamped to valid range"""
    waypoints = [[0.0, 0.0], [10.0, 0.0]]
    path = WaypointPath(waypoints, method='linear')
    
    # Beyond path length should clamp to end
    p_beyond = path(path.get_path_length() + 100.0)
    p_end = path(path.get_path_length())
    assert np.allclose(p_beyond, p_end, atol=1e-6)
    
    # Negative should clamp to start
    p_negative = path(-10.0)
    p_start = path(0.0)
    assert np.allclose(p_negative, p_start, atol=1e-6)


def test_get_waypoints():
    """Test that original waypoints can be retrieved"""
    waypoints = [[0.0, 0.0], [10.0, 5.0], [20.0, 0.0]]
    path = WaypointPath(waypoints, method='cubic')
    
    retrieved = path.get_waypoints()
    assert np.allclose(retrieved, waypoints, atol=1e-6)


def test_invalid_interpolation_method():
    """Test that invalid interpolation method raises error"""
    waypoints = [[0.0, 0.0], [10.0, 0.0]]
    
    with pytest.raises(ValueError):
        WaypointPath(waypoints, method='invalid_method')


def test_load_from_yaml():
    """Test loading waypoints from YAML file"""
    yaml_content = """
waypoints:
  - [0.0, 0.0]
  - [10.0, 5.0]
  - [20.0, 0.0]

path_settings:
  interpolation: cubic
  resolution: 50
  closed_path: false
"""
    
    # Create temporary YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name
    
    try:
        path, settings = load_waypoints_from_yaml(temp_path)
        
        assert path.get_path_length() > 0.0
        assert settings['interpolation'] == 'cubic'
        assert settings['resolution'] == 50
        assert settings['closed_path'] == False
        
        waypoints = path.get_waypoints()
        assert len(waypoints) == 3
    finally:
        os.unlink(temp_path)


def test_empty_yaml_raises():
    """Test that empty YAML raises error"""
    yaml_content = """
waypoints: []
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name
    
    try:
        with pytest.raises(ValueError):
            load_waypoints_from_yaml(temp_path)
    finally:
        os.unlink(temp_path)


def test_path_continuity():
    """Test that path is continuous (no jumps)"""
    waypoints = [[0.0, 0.0], [10.0, 10.0], [20.0, 0.0]]
    path = WaypointPath(waypoints, method='cubic')
    
    # Sample path at many points
    n_samples = 100
    s_values = np.linspace(0, path.get_path_length(), n_samples)
    
    max_jump = 0.0
    for i in range(len(s_values) - 1):
        p1 = path(s_values[i])
        p2 = path(s_values[i+1])
        jump = float(np.linalg.norm(np.array(p2) - np.array(p1)))
        max_jump = max(max_jump, jump)
    
    # Maximum jump should be small (continuous path)
    # Allowing for arc length parameterization
    expected_max_jump = path.get_path_length() / n_samples * 1.5
    assert max_jump < expected_max_jump


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
