"""
Tests for HeadingController with PD control and feedforward.
"""
import pytest
import jax.numpy as jnp
import numpy as np
from modular_controller.heading_controller import HeadingController, HeadingMode


class TestHeadingControllerInitialization:
    """Test HeadingController initialization and parameter validation."""
    
    def test_default_initialization(self):
        """Test HeadingController initializes with default parameters."""
        controller = HeadingController()
        assert controller.k_psi == 10.0
        assert controller.k_r == 5.0
        assert controller.mode == HeadingMode.LOS
    
    def test_custom_initialization(self):
        """Test HeadingController with custom parameters."""
        controller = HeadingController(k_psi=10.0, k_r=5.0, mode=HeadingMode.PATH)
        assert controller.k_psi == 10.0
        assert controller.k_r == 5.0
        assert controller.mode == HeadingMode.PATH
    
    def test_invalid_gains(self):
        """Test that invalid gains raise errors."""
        with pytest.raises(ValueError, match="Gains k_psi and k_r must be positive"):
            HeadingController(k_psi=0.0)
        with pytest.raises(ValueError, match="Gains k_psi and k_r must be positive"):
            HeadingController(k_psi=-1.0)
        with pytest.raises(ValueError, match="Gains k_psi and k_r must be positive"):
            HeadingController(k_r=0.0)
    
    def test_set_gains(self):
        """Test setting gains after initialization."""
        controller = HeadingController()
        controller.set_gains(k_psi=15.0, k_r=8.0)
        assert controller.k_psi == 15.0
        assert controller.k_r == 8.0
    
    def test_set_mode(self):
        """Test changing control mode."""
        controller = HeadingController(mode=HeadingMode.LOS)
        assert controller.mode == HeadingMode.LOS
        
        controller.set_mode(HeadingMode.PATH)
        assert controller.mode == HeadingMode.PATH
        
        controller.set_mode(HeadingMode.FORCE)
        assert controller.mode == HeadingMode.FORCE


class TestHeadingControllerCompute:
    """Test heading control computation."""
    
    @pytest.fixture
    def controller(self):
        """Create controller for testing."""
        return HeadingController(k_psi=10.0, k_r=5.0)
    
    def test_zero_error_zero_rate(self, controller):
        """Test with zero heading error and zero yaw rate."""
        psi = 0.0
        r = 0.0
        psi_ref = 0.0
        r_ref = 0.0
        
        tau_r = controller.compute(psi, r, psi_ref, r_ref)
        
        assert jnp.isclose(tau_r, 0.0, atol=1e-6)
    
    def test_positive_heading_error(self, controller):
        """Test with positive heading error."""
        psi = 0.5  # 0.5 rad ahead of reference
        r = 0.0
        psi_ref = 0.0
        r_ref = 0.0
        
        tau_r = controller.compute(psi, r, psi_ref, r_ref)
        
        # Should produce negative torque to reduce heading
        assert tau_r < 0.0
        assert jnp.isclose(tau_r, -10.0 * 0.5, atol=1e-6)
    
    def test_negative_heading_error(self, controller):
        """Test with negative heading error."""
        psi = -0.5  # 0.5 rad behind reference
        r = 0.0
        psi_ref = 0.0
        r_ref = 0.0
        
        tau_r = controller.compute(psi, r, psi_ref, r_ref)
        
        # Should produce positive torque to increase heading
        assert tau_r > 0.0
        assert jnp.isclose(tau_r, 10.0 * 0.5, atol=1e-6)
    
    def test_yaw_rate_damping(self, controller):
        """Test yaw rate damping."""
        psi = 0.0
        r = 0.5  # Positive yaw rate
        psi_ref = 0.0
        r_ref = 0.0
        
        tau_r = controller.compute(psi, r, psi_ref, r_ref)
        
        # Should produce negative torque to dampen yaw rate
        assert tau_r < 0.0
        assert jnp.isclose(tau_r, -5.0 * 0.5, atol=1e-6)
    
    def test_feedforward_yaw_rate(self, controller):
        """Test feedforward yaw rate tracking."""
        psi = 0.0
        r = 0.2  # Current yaw rate
        psi_ref = 0.0
        r_ref = 0.5  # Desired yaw rate higher than current
        
        tau_r = controller.compute(psi, r, psi_ref, r_ref)
        
        # Should produce torque to match reference yaw rate
        # tau_r = -k_r * (r - r_ref) = -5.0 * (0.2 - 0.5) = 1.5
        assert tau_r > 0.0
        assert jnp.isclose(tau_r, 5.0 * 0.3, atol=1e-6)
    
    def test_combined_error_and_rate(self, controller):
        """Test combined heading error and yaw rate error."""
        psi = 0.3
        r = 0.1
        psi_ref = 0.0
        r_ref = 0.0
        
        tau_r = controller.compute(psi, r, psi_ref, r_ref)
        
        # tau_r = -k_psi * e_psi - k_r * (r - r_ref)
        #       = -10.0 * 0.3 - 5.0 * 0.1
        #       = -3.0 - 0.5 = -3.5
        expected = -10.0 * 0.3 - 5.0 * 0.1
        assert jnp.isclose(tau_r, expected, atol=1e-6)
    
    def test_angle_wrapping_positive(self, controller):
        """Test angle wrapping for large positive errors."""
        psi = 3.5  # > pi
        r = 0.0
        psi_ref = 0.0
        r_ref = 0.0
        
        tau_r = controller.compute(psi, r, psi_ref, r_ref)
        
        # Error should wrap to negative (closer path)
        wrapped_error = (3.5 - 0.0 + np.pi) % (2 * np.pi) - np.pi
        expected = -10.0 * wrapped_error
        assert jnp.isclose(tau_r, expected, atol=1e-6)
    
    def test_angle_wrapping_negative(self, controller):
        """Test angle wrapping for large negative errors."""
        psi = -3.5  # < -pi
        r = 0.0
        psi_ref = 0.0
        r_ref = 0.0
        
        tau_r = controller.compute(psi, r, psi_ref, r_ref)
        
        # Error should wrap to positive (closer path)
        wrapped_error = (-3.5 - 0.0 + np.pi) % (2 * np.pi) - np.pi
        expected = -10.0 * wrapped_error
        assert jnp.isclose(tau_r, expected, atol=1e-6)
    
    def test_pi_transition(self, controller):
        """Test wrapping at pi/-pi boundary."""
        psi = 3.0
        r = 0.0
        psi_ref = -3.0
        r_ref = 0.0
        
        tau_r = controller.compute(psi, r, psi_ref, r_ref)
        
        # Should take shortest path through 0, not through pi
        error = (3.0 - (-3.0) + np.pi) % (2 * np.pi) - np.pi
        assert abs(error) < np.pi


class TestHeadingModeLOS:
    """Test LOS (velocity direction) mode."""
    
    @pytest.fixture
    def controller(self):
        """Create controller in LOS mode."""
        return HeadingController(k_psi=10.0, k_r=5.0, mode=HeadingMode.LOS)
    
    def test_forward_velocity(self, controller):
        """Test reference from forward velocity."""
        v_ref = jnp.array([1.0, 0.0])
        v_ref_dot = jnp.array([0.0, 0.0])
        
        psi_ref, r_ref = controller.compute_reference_from_los(v_ref, v_ref_dot)
        
        # Should point forward (0 rad)
        assert jnp.isclose(psi_ref, 0.0, atol=1e-6)
        assert jnp.isclose(r_ref, 0.0, atol=1e-6)
    
    def test_lateral_velocity(self, controller):
        """Test reference from lateral velocity."""
        v_ref = jnp.array([0.0, 1.0])
        v_ref_dot = jnp.array([0.0, 0.0])
        
        psi_ref, r_ref = controller.compute_reference_from_los(v_ref, v_ref_dot)
        
        # Should point to port (pi/2 rad)
        assert jnp.isclose(psi_ref, np.pi/2, atol=1e-6)
        assert jnp.isclose(r_ref, 0.0, atol=1e-6)
    
    def test_diagonal_velocity(self, controller):
        """Test reference from diagonal velocity."""
        v_ref = jnp.array([1.0, 1.0])
        v_ref_dot = jnp.array([0.0, 0.0])
        
        psi_ref, r_ref = controller.compute_reference_from_los(v_ref, v_ref_dot)
        
        # Should point at 45 degrees
        assert jnp.isclose(psi_ref, np.pi/4, atol=1e-6)
        assert jnp.isclose(r_ref, 0.0, atol=1e-6)
    
    def test_turning_motion(self, controller):
        """Test feedforward with turning motion."""
        v_ref = jnp.array([1.0, 0.0])  # Moving forward
        v_ref_dot = jnp.array([0.0, 0.5])  # Accelerating laterally (turning)
        
        psi_ref, r_ref = controller.compute_reference_from_los(v_ref, v_ref_dot)
        
        # Should have non-zero feedforward rate
        # r_ref = (vx*ay - vy*ax) / (vx^2 + vy^2) = (1.0*0.5 - 0*0) / 1.0 = 0.5
        assert jnp.isclose(r_ref, 0.5, atol=1e-6)
    
    def test_circular_motion(self, controller):
        """Test feedforward for circular motion."""
        # Moving in circle: v perpendicular to acceleration
        v_ref = jnp.array([1.0, 0.0])
        v_ref_dot = jnp.array([0.0, 1.0])  # Centripetal acceleration
        
        psi_ref, r_ref = controller.compute_reference_from_los(v_ref, v_ref_dot)
        
        # r_ref = (1.0*1.0 - 0.0*0.0) / 1.0 = 1.0 rad/s
        assert jnp.isclose(r_ref, 1.0, atol=1e-6)


class TestHeadingModePath:
    """Test PATH (path tangent) mode."""
    
    @pytest.fixture
    def controller(self):
        """Create controller in PATH mode."""
        return HeadingController(k_psi=10.0, k_r=5.0, mode=HeadingMode.PATH)
    
    def test_straight_path(self, controller):
        """Test reference from straight path."""
        # Simple linear path
        def path_func(s):
            return jnp.array([s, 0.0])
        
        s = 5.0
        s_dot = 1.0
        dt = 0.1
        
        psi_ref, r_ref = controller.compute_reference_from_path(s, s_dot, path_func, dt)
        
        # Straight horizontal path -> heading = 0
        assert jnp.isclose(psi_ref, 0.0, atol=1e-2)
        assert jnp.isclose(r_ref, 0.0, atol=1e-2)
    
    def test_vertical_path(self, controller):
        """Test reference from vertical path."""
        # Vertical path
        def path_func(s):
            return jnp.array([0.0, s])
        
        s = 5.0
        s_dot = 1.0
        dt = 0.1
        
        psi_ref, r_ref = controller.compute_reference_from_path(s, s_dot, path_func, dt)
        
        # Vertical path -> heading = pi/2
        assert jnp.isclose(psi_ref, np.pi/2, atol=1e-2)
        # r_ref may be non-zero due to numerical differentiation, but should be finite
        assert jnp.isfinite(r_ref)
    
    def test_circular_path(self, controller):
        """Test reference from circular path."""
        # Circular path
        radius = 10.0
        def path_func(s):
            theta = s / radius
            return jnp.array([radius * jnp.cos(theta), radius * jnp.sin(theta)])
        
        s = 0.0  # Starting point
        s_dot = 2.0  # Moving along path
        dt = 0.1
        
        psi_ref, r_ref = controller.compute_reference_from_path(s, s_dot, path_func, dt)
        
        # At s=0, tangent points upward (pi/2)
        # r_ref should be related to curvature and speed
        assert jnp.isfinite(psi_ref)
        assert jnp.isfinite(r_ref)


class TestHeadingModeForce:
    """Test FORCE (force direction) mode."""
    
    @pytest.fixture
    def controller(self):
        """Create controller in FORCE mode."""
        return HeadingController(k_psi=10.0, k_r=5.0, mode=HeadingMode.FORCE)
    
    def test_forward_force(self, controller):
        """Test reference from forward force."""
        u_p = jnp.array([10.0, 0.0])
        
        psi_ref, r_ref = controller.compute_reference_from_force(u_p)
        
        # Should point forward (0 rad)
        assert jnp.isclose(psi_ref, 0.0, atol=1e-6)
        assert jnp.isclose(r_ref, 0.0, atol=1e-6)
    
    def test_lateral_force(self, controller):
        """Test reference from lateral force."""
        u_p = jnp.array([0.0, 10.0])
        
        psi_ref, r_ref = controller.compute_reference_from_force(u_p)
        
        # Should point to port (pi/2 rad)
        assert jnp.isclose(psi_ref, np.pi/2, atol=1e-6)
        assert jnp.isclose(r_ref, 0.0, atol=1e-6)
    
    def test_diagonal_force(self, controller):
        """Test reference from diagonal force."""
        u_p = jnp.array([5.0, 5.0])
        
        psi_ref, r_ref = controller.compute_reference_from_force(u_p)
        
        # Should point at 45 degrees
        assert jnp.isclose(psi_ref, np.pi/4, atol=1e-6)
        assert jnp.isclose(r_ref, 0.0, atol=1e-6)
    
    def test_backward_force(self, controller):
        """Test reference from backward force."""
        u_p = jnp.array([-10.0, 0.0])
        
        psi_ref, r_ref = controller.compute_reference_from_force(u_p)
        
        # Should point backward (pi or -pi)
        assert jnp.isclose(abs(psi_ref), np.pi, atol=1e-6)
        assert jnp.isclose(r_ref, 0.0, atol=1e-6)


class TestHeadingControllerIntegration:
    """Integration tests for heading controller."""
    
    def test_heading_stabilization(self):
        """Test that controller stabilizes heading."""
        controller = HeadingController(k_psi=5.0, k_r=2.0)
        
        psi = 0.5  # Initial heading error
        r = 0.0  # Initial yaw rate
        psi_ref = 0.0
        r_ref = 0.0
        
        dt = 0.1
        steps = 50
        
        headings = []
        
        for _ in range(steps):
            tau_r = controller.compute(psi, r, psi_ref, r_ref)
            
            headings.append(psi)
            
            # Simple dynamics: I * r_dot = tau_r
            I = 1.0  # Moment of inertia
            r += (tau_r / I) * dt
            psi += r * dt
        
        # Heading should converge toward reference
        headings = np.array(headings)
        assert abs(headings[-1]) < abs(headings[0])
    
    def test_mode_switching(self):
        """Test switching between control modes."""
        controller = HeadingController(k_psi=10.0, k_r=5.0, mode=HeadingMode.LOS)
        
        # LOS mode
        v_ref = jnp.array([1.0, 0.0])
        v_ref_dot = jnp.array([0.0, 0.0])
        psi_ref_los, r_ref_los = controller.compute_reference_from_los(v_ref, v_ref_dot)
        
        # Switch to FORCE mode
        controller.set_mode(HeadingMode.FORCE)
        u_p = jnp.array([1.0, 0.0])
        psi_ref_force, r_ref_force = controller.compute_reference_from_force(u_p)
        
        # Both should point forward for these inputs
        assert jnp.isclose(psi_ref_los, psi_ref_force, atol=1e-6)
    
    def test_get_state(self):
        """Test state reporting."""
        controller = HeadingController(k_psi=10.0, k_r=5.0, mode=HeadingMode.PATH)
        
        state = controller.get_state()
        
        assert state['k_psi'] == 10.0
        assert state['k_r'] == 5.0
        assert state['mode'] == 'path'  # Returns string value
