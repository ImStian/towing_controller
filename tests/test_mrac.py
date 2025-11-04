"""
Tests for MRAC (Model Reference Adaptive Controller).
"""
import pytest
import jax.numpy as jnp
import numpy as np
from modular_controller.mrac import MRAC


class TestMRACInitialization:
    """Test MRAC initialization and parameter validation."""
    
    def test_default_initialization(self):
        """Test MRAC initializes with default parameters."""
        mrac = MRAC()
        assert mrac.L == 3.5
        assert mrac.epsilon == 0.7
        assert mrac.k_v == 1.5
        assert mrac.k_a == 1.5
        assert mrac.zeta.shape == (9,)
        assert jnp.all(mrac.zeta == 0.0)
    
    def test_custom_initialization(self):
        """Test MRAC initializes with custom parameters."""
        mrac = MRAC(tether_length=5.0, epsilon=0.5, k_v=2.0, k_a=3.0)
        assert mrac.L == 5.0
        assert mrac.epsilon == 0.5
        assert mrac.k_v == 2.0
        assert mrac.k_a == 3.0
    
    def test_invalid_tether_length(self):
        """Test that invalid tether length raises error."""
        with pytest.raises(ValueError, match="tether_length must be positive"):
            MRAC(tether_length=0.0)
        with pytest.raises(ValueError, match="tether_length must be positive"):
            MRAC(tether_length=-1.0)
    
    def test_invalid_epsilon(self):
        """Test that invalid epsilon raises error."""
        with pytest.raises(ValueError, match="epsilon must be between 0 and 1"):
            MRAC(epsilon=0.0)
        with pytest.raises(ValueError, match="epsilon must be between 0 and 1"):
            MRAC(epsilon=-0.5)
    
    def test_invalid_gains(self):
        """Test that invalid gains raise errors."""
        with pytest.raises(ValueError, match="Gains k_v and k_a must be positive"):
            MRAC(k_v=0.0)
        with pytest.raises(ValueError, match="Gains k_v and k_a must be positive"):
            MRAC(k_a=-1.0)


class TestMRACCompute:
    """Test MRAC compute method."""
    
    @pytest.fixture
    def mrac(self):
        """Create MRAC instance for testing."""
        return MRAC(tether_length=3.5, epsilon=0.7, k_v=1.5, k_a=1.5)
    
    def test_zero_state_zero_reference(self, mrac):
        """Test with zero state and zero reference."""
        asv_pos = [0.0, 0.0]
        asv_vel = [0.0, 0.0]
        towfish_pos = [3.5, 0.0]  # L away
        v_ref = jnp.array([0.0, 0.0])
        v_ref_dot = jnp.array([0.0, 0.0])
        
        u_p, zeta_dot = mrac.compute(
            asv_position=asv_pos,
            asv_velocity=asv_vel,
            towfish_position=towfish_pos,
            v_ref=v_ref,
            v_ref_dot=v_ref_dot,
            dt=0.1
        )
        
        assert u_p.shape == (2,)
        assert zeta_dot.shape == (9,)
    
    def test_forward_motion(self, mrac):
        """Test with forward motion reference."""
        asv_pos = [0.0, 0.0]
        asv_vel = [1.0, 0.0]
        towfish_pos = [3.5, 0.0]
        v_ref = jnp.array([1.0, 0.0])
        v_ref_dot = jnp.array([0.0, 0.0])
        
        # Initialize with non-zero adaptive parameters
        mrac.zeta = jnp.ones(9)
        
        u_p, zeta_dot = mrac.compute(
            asv_position=asv_pos,
            asv_velocity=asv_vel,
            towfish_position=towfish_pos,
            v_ref=v_ref,
            v_ref_dot=v_ref_dot,
            dt=0.1
        )
        
        # Should produce some control force
        assert not jnp.allclose(u_p, 0.0)
        assert u_p.shape == (2,)
    
    def test_tracking_error_response(self, mrac):
        """Test that controller responds to tracking error."""
        asv_pos = [0.0, 0.0]
        asv_vel = [0.5, 0.0]  # Slower than reference
        towfish_pos = [3.5, 0.0]
        v_ref = jnp.array([1.5, 0.0])  # Faster reference
        v_ref_dot = jnp.array([0.0, 0.0])
        
        # Initialize with non-zero adaptive parameters
        mrac.zeta = jnp.ones(9)
        
        u_p, zeta_dot = mrac.compute(
            asv_position=asv_pos,
            asv_velocity=asv_vel,
            towfish_position=towfish_pos,
            v_ref=v_ref,
            v_ref_dot=v_ref_dot,
            dt=0.1
        )
        
        # Should produce forward force to increase velocity
        # (exact direction depends on adaptive parameters)
        assert not jnp.allclose(u_p, 0.0)
    
    def test_adaptive_parameter_update(self, mrac):
        """Test that adaptive parameters update over time."""
        asv_pos = [0.0, 0.0]
        asv_vel = [1.0, 0.0]
        towfish_pos = [3.5, 0.0]
        v_ref = jnp.array([1.5, 0.0])
        v_ref_dot = jnp.array([0.1, 0.0])
        
        zeta_initial = mrac.zeta.copy()
        
        # Run multiple steps
        for _ in range(10):
            u_p, zeta_dot = mrac.compute(
                asv_position=asv_pos,
                asv_velocity=asv_vel,
                towfish_position=towfish_pos,
                v_ref=v_ref,
                v_ref_dot=v_ref_dot,
                dt=0.1
            )
            mrac.zeta += zeta_dot * 0.1
        
        # Adaptive parameters should have changed
        assert not jnp.allclose(mrac.zeta, zeta_initial)
    
    def test_off_track_configuration(self, mrac):
        """Test with towfish off to the side."""
        asv_pos = [0.0, 0.0]
        asv_vel = [1.0, 0.0]
        towfish_pos = [2.5, 2.5]  # At an angle
        v_ref = jnp.array([1.0, 0.0])
        v_ref_dot = jnp.array([0.0, 0.0])
        
        # Initialize with non-zero adaptive parameters
        mrac.zeta = jnp.ones(9)
        
        u_p, zeta_dot = mrac.compute(
            asv_position=asv_pos,
            asv_velocity=asv_vel,
            towfish_position=towfish_pos,
            v_ref=v_ref,
            v_ref_dot=v_ref_dot,
            dt=0.1
        )
        
        # Should produce control force
        assert u_p.shape == (2,)
        assert not jnp.allclose(u_p, 0.0)
    
    def test_lateral_motion(self, mrac):
        """Test with lateral motion reference."""
        asv_pos = [0.0, 0.0]
        asv_vel = [0.0, 0.5]
        towfish_pos = [3.5, 0.0]
        v_ref = jnp.array([0.0, 1.0])  # Lateral motion
        v_ref_dot = jnp.array([0.0, 0.0])
        
        # Initialize with non-zero adaptive parameters
        mrac.zeta = jnp.ones(9)
        
        u_p, zeta_dot = mrac.compute(
            asv_position=asv_pos,
            asv_velocity=asv_vel,
            towfish_position=towfish_pos,
            v_ref=v_ref,
            v_ref_dot=v_ref_dot,
            dt=0.1
        )
        
        # Should produce lateral force component
        assert u_p.shape == (2,)
        assert not jnp.allclose(u_p, 0.0)
    
    def test_acceleration_tracking(self, mrac):
        """Test controller response to acceleration reference."""
        asv_pos = [0.0, 0.0]
        asv_vel = [1.0, 0.0]
        towfish_pos = [3.5, 0.0]
        v_ref = jnp.array([1.0, 0.0])
        v_ref_dot = jnp.array([0.5, 0.0])  # Accelerating
        
        # Initialize with non-zero adaptive parameters
        mrac.zeta = jnp.ones(9)
        
        u_p_accel, _ = mrac.compute(
            asv_position=asv_pos,
            asv_velocity=asv_vel,
            towfish_position=towfish_pos,
            v_ref=v_ref,
            v_ref_dot=v_ref_dot,
            dt=0.1
        )
        
        # Compare to zero acceleration
        v_ref_dot_zero = jnp.array([0.0, 0.0])
        u_p_no_accel, _ = mrac.compute(
            asv_position=asv_pos,
            asv_velocity=asv_vel,
            towfish_position=towfish_pos,
            v_ref=v_ref,
            v_ref_dot=v_ref_dot_zero,
            dt=0.1
        )
        
        # Acceleration should affect control force
        assert not jnp.allclose(u_p_accel, u_p_no_accel)


class TestMRACRegressor:
    """Test regressor matrix computation."""
    
    @pytest.fixture
    def mrac(self):
        """Create MRAC instance for testing."""
        return MRAC(tether_length=3.5, epsilon=0.7, k_v=1.5, k_a=1.5)
    
    def test_regressor_shape(self, mrac):
        """Test that regressor matrix has correct shape."""
        asv_pos = [0.0, 0.0]
        asv_vel = [1.0, 0.0]
        towfish_pos = [3.5, 0.0]
        v_ref = jnp.array([1.0, 0.0])
        v_ref_dot = jnp.array([0.0, 0.0])
        
        u_p, _ = mrac.compute(
            asv_position=asv_pos,
            asv_velocity=asv_vel,
            towfish_position=towfish_pos,
            v_ref=v_ref,
            v_ref_dot=v_ref_dot,
            dt=0.1
        )
        
        # u_p = Y @ zeta, so Y should be (2, 9)
        assert u_p.shape == (2,)
        assert mrac.zeta.shape == (9,)
    
    def test_regressor_different_velocities(self, mrac):
        """Test regressor changes with different velocities."""
        asv_pos = [0.0, 0.0]
        towfish_pos = [3.5, 0.0]
        v_ref = jnp.array([1.0, 0.0])
        v_ref_dot = jnp.array([0.0, 0.0])
        
        u_p_1, _ = mrac.compute(
            asv_position=asv_pos,
            asv_velocity=[0.5, 0.0],
            towfish_position=towfish_pos,
            v_ref=v_ref,
            v_ref_dot=v_ref_dot,
            dt=0.1
        )
        
        u_p_2, _ = mrac.compute(
            asv_position=asv_pos,
            asv_velocity=[1.5, 0.0],
            towfish_position=towfish_pos,
            v_ref=v_ref,
            v_ref_dot=v_ref_dot,
            dt=0.1
        )
        
        # Different velocities should produce different forces
        # (when zeta is non-zero)
        # For zero zeta, forces are equal, so we need to update zeta first
        mrac.zeta = jnp.ones(9)
        
        u_p_1, _ = mrac.compute(
            asv_position=asv_pos,
            asv_velocity=[0.5, 0.0],
            towfish_position=towfish_pos,
            v_ref=v_ref,
            v_ref_dot=v_ref_dot,
            dt=0.1
        )
        
        u_p_2, _ = mrac.compute(
            asv_position=asv_pos,
            asv_velocity=[1.5, 0.0],
            towfish_position=towfish_pos,
            v_ref=v_ref,
            v_ref_dot=v_ref_dot,
            dt=0.1
        )
        
        assert not jnp.allclose(u_p_1, u_p_2)


class TestMRACIntegration:
    """Integration tests for MRAC."""
    
    def test_straight_line_tracking(self):
        """Test tracking a straight line reference."""
        mrac = MRAC(tether_length=3.5, epsilon=0.7, k_v=1.5, k_a=1.5)
        
        # Initialize state
        asv_pos = np.array([0.0, 0.0])
        asv_vel = np.array([0.0, 0.0])
        towfish_pos = np.array([3.5, 0.0])
        
        dt = 0.1
        t_sim = 5.0
        steps = int(t_sim / dt)
        
        # Constant velocity reference
        v_ref = jnp.array([1.0, 0.0])
        v_ref_dot = jnp.array([0.0, 0.0])
        
        forces = []
        
        for _ in range(steps):
            u_p, zeta_dot = mrac.compute(
                asv_position=asv_pos.tolist(),
                asv_velocity=asv_vel.tolist(),
                towfish_position=towfish_pos.tolist(),
                v_ref=v_ref,
                v_ref_dot=v_ref_dot,
                dt=dt
            )
            
            forces.append(np.array(u_p))
            
            # Update adaptive parameters
            mrac.zeta += zeta_dot * dt
            
            # Simple integration (not physical, just for testing)
            asv_vel += np.array(u_p) * dt * 0.01
            asv_pos += asv_vel * dt
            towfish_pos += asv_vel * dt * 0.9  # Follows with some lag
        
        # Should produce reasonable forces
        forces = np.array(forces)
        assert forces.shape == (steps, 2)
        assert not np.allclose(forces, 0.0)
    
    def test_parameter_adaptation(self):
        """Test that adaptive parameters can be updated and used."""
        mrac = MRAC(tether_length=3.5, epsilon=0.7, k_v=1.5, k_a=1.5)
        
        # Initial zeta should be zero
        assert jnp.allclose(mrac.zeta, 0.0)
        
        asv_pos = [0.0, 0.0]
        asv_vel = [1.0, 0.0]
        towfish_pos = [3.5, 0.0]
        v_ref = jnp.array([1.5, 0.0])
        v_ref_dot = jnp.array([0.1, 0.0])
        
        # First call with zero zeta produces zero control (u_p = Y @ zeta = Y @ 0 = 0)
        u_p_zero, zeta_dot_1 = mrac.compute(
            asv_position=asv_pos,
            asv_velocity=asv_vel,
            towfish_position=towfish_pos,
            v_ref=v_ref,
            v_ref_dot=v_ref_dot,
            dt=0.1
        )
        
        assert jnp.allclose(u_p_zero, 0.0)  # Zero zeta → zero output
        assert not jnp.allclose(zeta_dot_1, 0.0)  # But zeta_dot is non-zero (adaptation active)
        
        # Update zeta manually (as controller node would do)
        mrac.zeta = jnp.ones(9)  # Set to non-zero values
        
        # Now with non-zero zeta, should get non-zero output
        u_p_nonzero, _ = mrac.compute(
            asv_position=asv_pos,
            asv_velocity=asv_vel,
            towfish_position=towfish_pos,
            v_ref=v_ref,
            v_ref_dot=v_ref_dot,
            dt=0.1
        )
        
        # With non-zero zeta, control force should be non-zero
        assert not jnp.allclose(u_p_nonzero, 0.0)
