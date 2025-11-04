"""
Model Reference Adaptive Controller (MRAC) for ASV-Towfish system.

This controller implements an adaptive pendulum-based controller that:
1. Models the ASV-towfish system as a pendulum in the navigation frame
2. Adapts internal parameters to handle model uncertainties
3. Outputs planar force commands in the navigation/world frame
"""

import numpy as np
from typing import Tuple

class MRAC:
    """
    Model Reference Adaptive Controller for ASV-Towfish pendulum dynamics.
    
    All computations are performed in the navigation/world frame (ENU).
    The controller outputs a planar force vector that should be mapped to 
    body frame only at the actuation stage.
    """
    def __init__(self, tether_length = 3.5, epsilon = 0.7, k_v = 1.5, k_a = 1.5, logger=None):
        """
        Initialize MRAC controller.
        
        Args:
            tether_length: Length of tether between ASV and towfish (m)
            epsilon: Fraction of tether length for pendulum model (0 < epsilon < 1)
            k_v: Velocity tracking gain (positive)
            k_a: Adaptation gain (positive)
            logger: Optional ROS logger for debug messages
        """
        # Sanity checks
        if not (0.0 < epsilon < 1.0):
            raise ValueError("epsilon must be between 0 and 1")
        if tether_length <= 0.0:
            raise ValueError("tether_length must be positive")
        if k_v <= 0.0 or k_a <= 0.0:
            raise ValueError("Gains k_v and k_a must be positive")
        
        # Controller parameters
        self.L = float(tether_length)
        self.epsilon = float(epsilon)
        self.k_v = float(k_v)
        self.k_a = float(k_a)
        self.logger = logger
        
        # Initialize adaptive parameters (9 elements)
        self.zeta = np.zeros(9, dtype=float)





    def compute(
        self,
        asv_position: np.ndarray,      
        asv_velocity: np.ndarray,      
        towfish_position: np.ndarray, 
        v_ref: np.ndarray,             
        v_ref_dot: np.ndarray,         
        dt: float                      
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute MRAC control force.
        
        All inputs and outputs are in the navigation/world frame (ENU).
        
        Args:
            asv_position: ASV position [x, y] in nav frame (m)
            asv_velocity: ASV velocity [vx, vy] in nav frame (m/s)
            towfish_position: Towfish position [x, y] in nav frame (m)
            v_ref: Reference velocity [vx, vy] in nav frame (m/s)
            v_ref_dot: Reference acceleration [ax, ay] in nav frame (m/s²)
            dt: Time step since last update (s)
            
        Returns:
            u_p: Control force vector [Fx, Fy] in nav frame (N)
            zeta_dot: Adaptive parameter derivatives for integration
        """
        # Ensure all inputs are 2D column vectors
        asv_pos = np.asarray(asv_position, dtype=float).reshape(2, 1)
        asv_vel = np.asarray(asv_velocity, dtype=float).reshape(2, 1)
        tow_pos = np.asarray(towfish_position, dtype=float).reshape(2, 1)
        v_ref = np.asarray(v_ref, dtype=float).reshape(2, 1)
        v_ref_dot = np.asarray(v_ref_dot, dtype=float).reshape(2, 1)

        # Compute pendulum angle and rate from relative positions
        dx = tow_pos - asv_pos  # Relative position vector
        distance = float(np.linalg.norm(dx))

        if distance < 1e-6:
            # Degenerate case: towfish at ASV position
            if self.logger:
                self.logger.warning("Towfish too close to ASV; returning zero force")
            return np.zeros(2, dtype=float), np.zeros(9, dtype=float)

        # Pendulum angle (theta) in navigation frame
        theta = float(np.arctan2(dx[1, 0], dx[0, 0]))

        # Pendulum angular rate (theta_dot)
        if dt > 1e-6 and hasattr(self, '_prev_theta'):
            dtheta = (theta - self._prev_theta)
            # Wrap angle difference to [-pi, pi]
            dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
            theta_dot = float(dtheta / dt)
        else:
            theta_dot = 0.0
        self._prev_theta = theta

# Pendulum basis vectors in nav frame
        Gamma = np.array([[np.cos(theta)], 
                          [np.sin(theta)]], dtype=float)  # 2x1
        dGamma = np.array([[-np.sin(theta)], 
                           [np.cos(theta)]], dtype=float)  # 2x1
        
        # Velocity of pendulum mass (towfish) in nav frame
        v = asv_vel + self.epsilon * self.L * theta_dot * dGamma
        
        # Velocity tracking error
        v_err = v - v_ref
        
        # Auxiliary velocity (full pendulum extension)
        v1 = asv_vel + self.L * theta_dot * dGamma
        
        # Projection matrix onto pendulum direction
        J = dGamma @ dGamma.T  # 2x2
        
        # --- Construct Regressor Matrix Y ---
        # Each term is a 2D column vector or matrix
        
        # Term 1: Centripetal + cross-coupling + feedforward
        term1 = self.L * theta_dot**2 * Gamma
        term2 = -theta_dot / (2 * (self.epsilon - 1)) * (
            Gamma @ dGamma.T + dGamma @ Gamma.T
        ) @ v_err
        term3 = -J @ (v_ref_dot - self.k_v * v_err) / (self.epsilon - 1)
        combined_term = term1 + term2 + term3  # 2x1
        
        # Construct Y matrix: 2x9
        # Each "column" in the regressor corresponds to one adaptive parameter
        Y = np.hstack([
            combined_term,                    # Column 0
            asv_vel,                          # Column 1 (2x1)
            self.L * theta_dot * dGamma,      # Column 2 (2x1)
            np.eye(2),                        # Columns 3-4 (2x2)
            J @ v1,                           # Column 5 (2x1)
            J,                                # Columns 6-7 (2x2)
            self.k_v * v_err - v_ref_dot      # Column 8 (2x1)
        ])
        
        # Sanity check and guard against NaN/Inf
        Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
        
        # --- Control Law ---
        u_p = -Y @ self.zeta.reshape(-1, 1)  # 2x1 force vector in nav frame
        
        # --- Adaptation Law ---
        zeta_dot = self.k_a * (Y.T @ v_err)  # 9x1 parameter update
        zeta_dot = np.nan_to_num(zeta_dot.flatten(), nan=0.0, posinf=0.0, neginf=0.0)
        
        # Integrate adaptive parameters (simple Euler step)
        if dt > 0:
            self.zeta += zeta_dot * dt
        
        return u_p.flatten(), zeta_dot


    def reset(self):
        """Reset adaptive parameters to zero."""
        self.zeta = np.zeros(9, dtype=float)
        if hasattr(self, '_prev_theta'):
            delattr(self, '_prev_theta')
    
    def set_parameters(
        self,
        tether_length: float = None,
        epsilon: float = None,
        k_v: float = None,
        k_a: float = None
    ):
        """Update controller parameters at runtime."""
        if tether_length is not None:
            if tether_length <= 0.0:
                raise ValueError("tether_length must be positive")
            self.L = float(tether_length)
        
        if epsilon is not None:
            if not (0.0 < epsilon < 1.0):
                raise ValueError("epsilon must be between 0 and 1")
            self.epsilon = float(epsilon)
        
        if k_v is not None:
            if k_v <= 0.0:
                raise ValueError("k_v must be positive")
            self.k_v = float(k_v)
        
        if k_a is not None:
            if k_a <= 0.0:
                raise ValueError("k_a must be positive")
            self.k_a = float(k_a)
    
    def get_state(self) -> dict:
        """Get current controller state for debugging/logging."""
        return {
            'zeta': self.zeta.tolist(),
            'L': self.L,
            'epsilon': self.epsilon,
            'k_v': self.k_v,
            'k_a': self.k_a
        }