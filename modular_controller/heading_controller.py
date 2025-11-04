"""
Heading Controller for ASV

Implements two heading modes:
1. Path Mode: Align heading with path tangent direction
2. LOS Mode: Align heading with LOS velocity reference direction
"""

import numpy as np
import jax.numpy as jnp
from jax import grad
from enum import Enum
from typing import Callable, Tuple
import logging


class HeadingMode(Enum):
    """Heading control modes."""
    PATH = "path"  # Follow path tangent
    LOS = "los"    # Follow LOS reference velocity direction
    FORCE = "force"  # Follow control force direction (simple)


class HeadingController:
    """
    PD heading controller with feedforward compensation.
    
    Implements: τ_r = -k_ψ * angle_error - k_r * (r - r_ref)
    where angle_error wraps to [-π, π]
    """
    
    def __init__(
        self,
        k_psi: float = 10.0,
        k_r: float = 5.0,
        mode: HeadingMode = HeadingMode.LOS,
        logger: logging.Logger = None
    ):
        """
        Initialize heading controller.
        
        Args:
            k_psi: Proportional gain on heading error (positive)
            k_r: Derivative gain on yaw rate error (positive)
            mode: Heading control mode
            logger: Optional logger for debugging
        """
        if k_psi <= 0.0 or k_r <= 0.0:
            raise ValueError("Gains k_psi and k_r must be positive")
        
        self.k_psi = float(k_psi)
        self.k_r = float(k_r)
        self.mode = mode
        self.logger = logger or logging.getLogger(__name__)
        
        # For computing derivatives
        self._prev_psi_ref = 0.0
        self._prev_s = 0.0
        self._prev_time = 0.0
    
    def compute(
        self,
        psi: float,           # Current heading (rad)
        r: float,             # Current yaw rate (rad/s)
        psi_ref: float,       # Reference heading (rad)
        r_ref: float = 0.0,   # Reference yaw rate (rad/s)
    ) -> float:
        """
        Compute yaw moment command.
        
        Args:
            psi: Current ASV heading in radians
            r: Current yaw rate in rad/s
            psi_ref: Desired heading in radians
            r_ref: Desired yaw rate in rad/s (feedforward)
            
        Returns:
            tau_r: Yaw moment command (N·m)
        """
        # Wrap heading error to [-π, π]
        angle_error = self._wrap_to_pi(psi - psi_ref)
        
        # Rate error
        rate_error = r - r_ref
        
        # PD control law
        tau_r = -self.k_psi * angle_error - self.k_r * rate_error
        
        return float(tau_r)
    
    def compute_reference_from_path(
        self,
        s: float,
        s_dot: float,
        path_function: Callable,
        dt: float
    ) -> Tuple[float, float]:
        """
        Compute reference heading and yaw rate from path tangent (Path Mode).
        
        Args:
            s: Current path parameter
            s_dot: Path parameter rate (ds/dt)
            path_function: Callable that returns [x, y] given s
            dt: Time step for numerical differentiation
            
        Returns:
            psi_ref: Reference heading (rad)
            r_ref: Reference yaw rate (rad/s)
        """
        # Compute path tangent using numerical differentiation
        ds = 1e-4  # Small step for derivative
        p_curr = jnp.asarray(path_function(s), dtype=jnp.float32)
        p_next = jnp.asarray(path_function(s + ds), dtype=jnp.float32)
        dp_ds = (p_next - p_curr) / ds
        
        # Reference heading from path tangent
        psi_ref = float(jnp.arctan2(dp_ds[1], dp_ds[0]))
        
        # Reference yaw rate: r_ref = dψ/dt = (dψ/ds) * (ds/dt)
        # Compute dψ/ds numerically
        if dt > 1e-6 and hasattr(self, '_prev_psi_ref'):
            dpsi_ds = (psi_ref - self._prev_psi_ref) / (s - self._prev_s) if abs(s - self._prev_s) > 1e-8 else 0.0
            dpsi_ds = self._wrap_to_pi(dpsi_ds)
            r_ref = float(dpsi_ds * s_dot)
        else:
            r_ref = 0.0
        
        self._prev_psi_ref = psi_ref
        self._prev_s = s
        
        return psi_ref, r_ref
    
    def compute_reference_from_los(
        self,
        v_ref: np.ndarray,
        v_ref_dot: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Compute reference heading and yaw rate from LOS velocity (LOS Mode).
        
        Args:
            v_ref: Reference velocity [vx, vy] in nav frame (m/s)
            v_ref_dot: Reference acceleration [ax, ay] in nav frame (m/s²)
            
        Returns:
            psi_ref: Reference heading (rad)
            r_ref: Reference yaw rate (rad/s)
        """
        v_ref = jnp.asarray(v_ref, dtype=jnp.float32).reshape(2)
        v_ref_dot = jnp.asarray(v_ref_dot, dtype=jnp.float32).reshape(2)
        
        # Reference heading from velocity direction
        psi_ref = float(jnp.arctan2(v_ref[1], v_ref[0]))
        
        # Reference yaw rate: r_ref = dψ/dt = d/dt[atan2(vy, vx)]
        # Using chain rule: r_ref = (vx*ay - vy*ax) / (vx² + vy²)
        v_mag_sq = float(jnp.sum(v_ref**2))
        
        if v_mag_sq > 1e-6:
            r_ref = float((v_ref[0] * v_ref_dot[1] - v_ref[1] * v_ref_dot[0]) / v_mag_sq)
        else:
            r_ref = 0.0
        
        return psi_ref, r_ref
    
    def compute_reference_from_force(
        self,
        u_p: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute reference heading from control force direction (Force Mode).
        
        This is a simplified mode where we just want to point in the direction
        of the planar force vector, with no feedforward yaw rate.
        
        Args:
            u_p: Planar control force [Fx, Fy] in nav frame (N)
            
        Returns:
            psi_ref: Reference heading (rad)
            r_ref: Reference yaw rate (0.0 for this mode)
        """
        u_p = jnp.asarray(u_p, dtype=jnp.float32).reshape(2)
        psi_ref = float(jnp.arctan2(u_p[1], u_p[0]))
        return psi_ref, 0.0
    
    def set_mode(self, mode: HeadingMode):
        """Change heading control mode."""
        self.mode = mode
        self.logger.info(f"Heading mode changed to: {mode.value}")
    
    def set_gains(self, k_psi: float = None, k_r: float = None):
        """Update controller gains."""
        if k_psi is not None:
            if k_psi <= 0.0:
                raise ValueError("k_psi must be positive")
            self.k_psi = float(k_psi)
        
        if k_r is not None:
            if k_r <= 0.0:
                raise ValueError("k_r must be positive")
            self.k_r = float(k_r)
    
    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        """Wrap angle to [-π, π]."""
        return float((angle + np.pi) % (2 * np.pi) - np.pi)
    
    def get_state(self) -> dict:
        """Get current controller state for debugging."""
        return {
            'k_psi': self.k_psi,
            'k_r': self.k_r,
            'mode': self.mode.value
        }
