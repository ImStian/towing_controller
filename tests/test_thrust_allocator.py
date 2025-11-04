"""
Tests for ThrustAllocator - differential thrust system.
"""
import pytest
import numpy as np
from modular_controller.thrust_allocator import ThrustAllocator


class TestThrustAllocatorInitialization:
    """Test ThrustAllocator initialization and parameter validation."""
    
    def test_default_initialization(self):
        """Test ThrustAllocator initializes with default parameters."""
        allocator = ThrustAllocator()
        assert allocator.B == 0.5
        assert allocator.max_thrust == 50.0
    
    def test_custom_initialization(self):
        """Test ThrustAllocator with custom parameters."""
        allocator = ThrustAllocator(thruster_separation=0.8, max_thrust=100.0)
        assert allocator.B == 0.8
        assert allocator.max_thrust == 100.0
    
    def test_invalid_separation(self):
        """Test that invalid thruster separation raises error."""
        with pytest.raises(ValueError):
            ThrustAllocator(thruster_separation=0.0)
        with pytest.raises(ValueError):
            ThrustAllocator(thruster_separation=-0.5)
    
    def test_invalid_max_thrust(self):
        """Test that invalid max thrust raises error."""
        with pytest.raises(ValueError):
            ThrustAllocator(max_thrust=0.0)
        with pytest.raises(ValueError):
            ThrustAllocator(max_thrust=-10.0)


class TestThrustAllocation:
    """Test thrust allocation computation."""
    
    @pytest.fixture
    def allocator(self):
        """Create allocator for testing."""
        return ThrustAllocator(thruster_separation=0.5, max_thrust=50.0)
    
    def test_zero_input(self, allocator):
        """Test with zero force and torque."""
        F_surge = 0.0
        tau_r = 0.0
        
        T_left, T_right = allocator.allocate(F_surge, tau_r)
        
        assert T_left == 0.0
        assert T_right == 0.0
    
    def test_pure_surge_force(self, allocator):
        """Test with pure surge force, no yaw torque."""
        F_surge = 20.0
        tau_r = 0.0
        
        T_left, T_right = allocator.allocate(F_surge, tau_r)
        
        # Both thrusters should be equal
        assert T_left == T_right
        # Sum should equal surge force
        assert np.isclose(T_left + T_right, F_surge)
        assert np.isclose(T_left, 10.0)
        assert np.isclose(T_right, 10.0)
    
    def test_pure_yaw_torque_positive(self, allocator):
        """Test with pure positive yaw torque, no surge force."""
        F_surge = 0.0
        tau_r = 5.0  # Positive torque (turn left)
        
        T_left, T_right = allocator.allocate(F_surge, tau_r)
        
        # Right thruster > left thruster for positive torque
        assert T_right > T_left
        # Should be antisymmetric
        assert np.isclose(T_left, -T_right)
        # Torque: tau_r = (T_right - T_left) * B/2
        # 5.0 = (T_right - T_left) * 0.5/2 = (T_right - T_left) * 0.25
        # T_right - T_left = 20.0
        assert np.isclose(T_right - T_left, tau_r / (allocator.B / 2.0))
    
    def test_pure_yaw_torque_negative(self, allocator):
        """Test with pure negative yaw torque, no surge force."""
        F_surge = 0.0
        tau_r = -5.0  # Negative torque (turn right)
        
        T_left, T_right = allocator.allocate(F_surge, tau_r)
        
        # Left thruster > right thruster for negative torque
        assert T_left > T_right
        # Should be antisymmetric
        assert np.isclose(T_left, -T_right)
    
    def test_combined_surge_and_yaw(self, allocator):
        """Test with both surge force and yaw torque."""
        F_surge = 20.0
        tau_r = 2.5
        
        T_left, T_right = allocator.allocate(F_surge, tau_r)
        
        # Sum should equal surge force
        assert np.isclose(T_left + T_right, F_surge)
        # Difference relates to torque: tau_r = (T_right - T_left) * B/2
        torque_check = (T_right - T_left) * (allocator.B / 2.0)
        assert np.isclose(torque_check, tau_r)
    
    def test_allocation_formulas(self, allocator):
        """Test explicit allocation formulas."""
        F_surge = 30.0
        tau_r = 5.0
        B = allocator.B
        
        T_left, T_right = allocator.allocate(F_surge, tau_r)
        
        # T_left = F_surge/2 - tau_r/B
        expected_left = F_surge / 2.0 - tau_r / B
        assert np.isclose(T_left, expected_left)
        
        # T_right = F_surge/2 + tau_r/B
        expected_right = F_surge / 2.0 + tau_r / B
        assert np.isclose(T_right, expected_right)


class TestThrustSaturation:
    """Test thrust saturation limits."""
    
    @pytest.fixture
    def allocator(self):
        """Create allocator for testing."""
        return ThrustAllocator(thruster_separation=0.5, max_thrust=50.0)
    
    def test_no_saturation_below_limit(self, allocator):
        """Test that thrusts below limit are not saturated."""
        F_surge = 20.0
        tau_r = 2.0
        
        T_left, T_right = allocator.allocate(F_surge, tau_r)
        
        # Both should be below limit
        assert abs(T_left) < allocator.max_thrust
        assert abs(T_right) < allocator.max_thrust
    
    def test_saturation_positive_limit(self, allocator):
        """Test saturation at positive limit."""
        F_surge = 200.0  # Very large force
        tau_r = 0.0
        
        T_left, T_right = allocator.allocate(F_surge, tau_r)
        
        # Both should be saturated at max
        assert T_left == allocator.max_thrust
        assert T_right == allocator.max_thrust
    
    def test_saturation_negative_limit(self, allocator):
        """Test saturation at negative limit."""
        F_surge = -200.0  # Very large negative force
        tau_r = 0.0
        
        T_left, T_right = allocator.allocate(F_surge, tau_r)
        
        # Both should be saturated at -max
        assert T_left == -allocator.max_thrust
        assert T_right == -allocator.max_thrust
    
    def test_asymmetric_saturation_right(self, allocator):
        """Test saturation when only right thruster saturates."""
        F_surge = 50.0
        tau_r = 30.0  # Large torque
        
        T_left, T_right = allocator.allocate(F_surge, tau_r)
        
        # Right thruster should saturate
        assert T_right == allocator.max_thrust
        # Left thruster may or may not saturate
        assert abs(T_left) <= allocator.max_thrust
    
    def test_asymmetric_saturation_left(self, allocator):
        """Test saturation when only left thruster saturates."""
        F_surge = 50.0
        tau_r = -30.0  # Large negative torque
        
        T_left, T_right = allocator.allocate(F_surge, tau_r)
        
        # Left thruster should saturate
        assert T_left == allocator.max_thrust
        # Right thruster may or may not saturate
        assert abs(T_right) <= allocator.max_thrust
    
    def test_reverse_with_large_torque(self, allocator):
        """Test reverse thrust with large torque."""
        F_surge = -40.0
        tau_r = 50.0  # Large torque
        
        T_left, T_right = allocator.allocate(F_surge, tau_r)
        
        # Both thrusters should be within limits
        assert abs(T_left) <= allocator.max_thrust
        assert abs(T_right) <= allocator.max_thrust


class TestDifferentConfigurations:
    """Test with different thruster configurations."""
    
    def test_wide_separation(self):
        """Test with wide thruster separation."""
        allocator = ThrustAllocator(thruster_separation=1.0, max_thrust=50.0)
        
        F_surge = 20.0
        tau_r = 10.0
        
        T_left, T_right = allocator.allocate(F_surge, tau_r)
        
        # With wider separation, same torque requires less thrust difference
        # tau_r = (T_right - T_left) * B/2 = (T_right - T_left) * 0.5
        # 10.0 = (T_right - T_left) * 0.5
        # T_right - T_left = 20.0
        assert np.isclose(T_right - T_left, 20.0)
    
    def test_narrow_separation(self):
        """Test with narrow thruster separation."""
        allocator = ThrustAllocator(thruster_separation=0.25, max_thrust=50.0)
        
        F_surge = 20.0
        tau_r = 10.0
        
        T_left, T_right = allocator.allocate(F_surge, tau_r)
        
        # With narrow separation, same torque requires larger thrust difference
        # tau_r = (T_right - T_left) * B/2 = (T_right - T_left) * 0.125
        # 10.0 = (T_right - T_left) * 0.125
        # T_right - T_left = 80.0
        # But will saturate!
        assert abs(T_left) <= allocator.max_thrust
        assert abs(T_right) <= allocator.max_thrust
    
    def test_high_thrust_limit(self):
        """Test with high thrust limit."""
        allocator = ThrustAllocator(thruster_separation=0.5, max_thrust=200.0)
        
        F_surge = 150.0
        tau_r = 50.0
        
        T_left, T_right = allocator.allocate(F_surge, tau_r)
        
        # High limits should allow large commands
        assert T_left < 200.0
        assert T_right < 200.0
        # Should not saturate
        expected_left = F_surge / 2.0 - tau_r / 0.5
        expected_right = F_surge / 2.0 + tau_r / 0.5
        assert np.isclose(T_left, expected_left)
        assert np.isclose(T_right, expected_right)


class TestPhysicalConsistency:
    """Test physical consistency of allocation."""
    
    @pytest.fixture
    def allocator(self):
        """Create allocator for testing."""
        return ThrustAllocator(thruster_separation=0.5, max_thrust=50.0)
    
    def test_surge_force_sum(self, allocator):
        """Test that thrust sum equals surge force (when not saturated)."""
        test_cases = [
            (10.0, 0.0),
            (20.0, 2.0),
            (-15.0, 1.5),
            (30.0, -5.0),
        ]
        
        for F_surge, tau_r in test_cases:
            T_left, T_right = allocator.allocate(F_surge, tau_r)
            
            # If not saturated, sum should equal surge force
            if abs(T_left) < allocator.max_thrust and abs(T_right) < allocator.max_thrust:
                assert np.isclose(T_left + T_right, F_surge)
    
    def test_yaw_torque_difference(self, allocator):
        """Test that thrust difference produces correct torque (when not saturated)."""
        test_cases = [
            (20.0, 5.0),
            (15.0, -3.0),
            (25.0, 10.0),
            (0.0, 8.0),
        ]
        
        for F_surge, tau_r in test_cases:
            T_left, T_right = allocator.allocate(F_surge, tau_r)
            
            # If not saturated, check torque relationship
            if abs(T_left) < allocator.max_thrust and abs(T_right) < allocator.max_thrust:
                torque_check = (T_right - T_left) * (allocator.B / 2.0)
                assert np.isclose(torque_check, tau_r)
    
    def test_symmetry(self, allocator):
        """Test symmetry of allocation."""
        F_surge = 20.0
        tau_r = 5.0
        
        T_left_1, T_right_1 = allocator.allocate(F_surge, tau_r)
        T_left_2, T_right_2 = allocator.allocate(F_surge, -tau_r)
        
        # Reversing torque should swap thrusters
        assert np.isclose(T_left_1, T_right_2)
        assert np.isclose(T_right_1, T_left_2)
    
    def test_zero_torque_equal_thrusters(self, allocator):
        """Test that zero torque produces equal thrusters."""
        test_surges = [-20.0, -10.0, 0.0, 10.0, 20.0]
        
        for F_surge in test_surges:
            T_left, T_right = allocator.allocate(F_surge, 0.0)
            
            # Should be equal
            assert np.isclose(T_left, T_right)
            # Each should be half of surge
            assert np.isclose(T_left, F_surge / 2.0)


class TestThrustAllocatorIntegration:
    """Integration tests for thrust allocator."""
    
    def test_realistic_scenario(self):
        """Test with realistic BlueBoat-like parameters."""
        # Approximate BlueBoat parameters
        allocator = ThrustAllocator(thruster_separation=0.6, max_thrust=40.0)
        
        # Typical mission: 20N forward, slight turn
        F_surge = 20.0
        tau_r = 3.0
        
        T_left, T_right = allocator.allocate(F_surge, tau_r)
        
        # Should be reasonable values
        assert 0 < T_left < 30.0
        assert 0 < T_right < 30.0
        assert T_right > T_left  # Turning left
    
    def test_aggressive_maneuver(self):
        """Test aggressive turning maneuver."""
        allocator = ThrustAllocator(thruster_separation=0.5, max_thrust=50.0)
        
        # Moderate forward speed with aggressive turn
        F_surge = 30.0
        tau_r = 20.0
        
        T_left, T_right = allocator.allocate(F_surge, tau_r)
        
        # One thruster should be much higher
        assert T_right > T_left
        # Both should be within limits
        assert abs(T_left) <= 50.0
        assert abs(T_right) <= 50.0
    
    def test_emergency_stop(self):
        """Test emergency stop (full reverse)."""
        allocator = ThrustAllocator(thruster_separation=0.5, max_thrust=50.0)
        
        # Full reverse thrust
        F_surge = -100.0  # More than max
        tau_r = 0.0
        
        T_left, T_right = allocator.allocate(F_surge, tau_r)
        
        # Both should be at max reverse
        assert T_left == -50.0
        assert T_right == -50.0
