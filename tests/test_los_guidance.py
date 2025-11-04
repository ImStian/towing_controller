import math
import pytest
import jax.numpy as jnp
from modular_controller.los_guidance import LOSGuidance

def line_path(s):
    """Straight line along x-axis"""
    return jnp.array([s, 0.0], dtype=jnp.float32)

def zero_path(_s):
    """Degenerate path (invalid): constant point, zero gradient"""
    return jnp.array([0.0, 0.0], dtype=jnp.float32)

def test_parameters_validation():
    """Test that invalid parameters raise ValueError"""
    los = LOSGuidance()
    
    # Negative U should fail
    with pytest.raises(ValueError):
        los.los_parameters(-1.0, 5.0, 0.5)
    
    # Zero delta should fail
    with pytest.raises(ValueError):
        los.los_parameters(1.0, 0.0, 0.5)
    
    # NaN k should fail
    with pytest.raises(ValueError):
        los.los_parameters(1.0, 5.0, float("nan"))
    
    # Valid parameters should work
    los.los_parameters(1.2, 5.0, 0.3)
    assert los.U == 1.2
    assert los.delta == 5.0
    assert los.k == 0.3

def test_saturation():
    """Test saturation function"""
    los = LOSGuidance()
    
    assert math.isclose(float(los.saturation(2.0, 1.0)), 1.0, abs_tol=1e-6)
    assert math.isclose(float(los.saturation(-3.0, 2.5)), -2.5, abs_tol=1e-6)
    assert math.isclose(float(los.saturation(0.3, 0.5)), 0.3, abs_tol=1e-6)

def test_straight_line_on_track():
    """Test LOS on a straight line when on-track"""
    los = LOSGuidance()
    los.los_parameters(U=1.0, delta=5.0, k=0.0)
    
    pos = jnp.array([3.0, 0.0])
    s = 3.0  # on the path
    
    v_los, s_dot = los.compute(pos, s, line_path)
    
    # On a straight x-axis path, on-track: v_los should be [U, 0], s_dot == U
    assert jnp.allclose(v_los, jnp.array([1.0, 0.0]), atol=1e-6)
    assert math.isclose(s_dot, 1.0, rel_tol=1e-3, abs_tol=1e-3)  # Relaxed for numerical diff

def test_straight_line_off_track_above():
    """Test LOS on a straight line when off-track above"""
    los = LOSGuidance()
    los.los_parameters(U=2.0, delta=5.0, k=0.0)
    
    pos = jnp.array([3.0, 2.0])  # 2 m above path
    s = 3.0
    
    v_los, s_dot = los.compute(pos, s, line_path)
    
    # Should command forward velocity, and negative y to bring it down towards path
    assert v_los[0] > 0.0
    assert v_los[1] < 0.0
    
    # s_dot will be less than U due to cross-track error affecting D
    assert s_dot > 0.0
    assert s_dot < 2.0  # less than U due to off-track position

def test_straight_line_off_track_below():
    """Test LOS on a straight line when off-track below"""
    los = LOSGuidance()
    los.los_parameters(U=1.5, delta=3.0, k=0.0)
    
    pos = jnp.array([5.0, -1.5])  # 1.5 m below path
    s = 5.0
    
    v_los, s_dot = los.compute(pos, s, line_path)
    
    # Should command forward velocity, and positive y to bring it up towards path
    assert v_los[0] > 0.0
    assert v_los[1] > 0.0

def test_zero_gradient_raises():
    """Test that zero gradient path raises ValueError"""
    los = LOSGuidance()
    los.los_parameters(U=1.0, delta=3.0, k=0.1)
    
    with pytest.raises(ValueError):
        los.compute(jnp.array([0.0, 0.0]), 0.0, zero_path)

def test_position_shape_check():
    """Test that invalid position shape raises ValueError"""
    los = LOSGuidance()
    los.los_parameters(U=1.0, delta=3.0, k=0.1)
    
    # 3D position should fail
    with pytest.raises(ValueError):
        los.compute(jnp.array([1.0, 2.0, 3.0]), 0.0, line_path)

def test_output_shapes():
    """Test that outputs have correct shapes"""
    los = LOSGuidance()
    los.los_parameters(U=1.0, delta=5.0, k=0.1)
    
    pos = jnp.array([0.0, 1.0])
    s = 0.0
    
    v_los, s_dot = los.compute(pos, s, line_path)
    
    # v_los should be shape (2,)
    assert v_los.shape == (2,)
    # s_dot should be scalar
    assert isinstance(s_dot, float)

def test_velocity_magnitude_near_speed():
    """Test that velocity magnitude is approximately U when on/near track"""
    los = LOSGuidance()
    U = 2.5
    los.los_parameters(U=U, delta=10.0, k=0.0)
    
    pos = jnp.array([5.0, 0.5])  # slightly off track
    s = 5.0
    
    v_los, _ = los.compute(pos, s, line_path)
    
    v_mag = float(jnp.linalg.norm(v_los))
    # Should be close to U
    assert math.isclose(v_mag, U, rel_tol=0.1)
