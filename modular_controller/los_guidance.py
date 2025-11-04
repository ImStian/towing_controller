import math
import jax.numpy as jnp

class LOSGuidance:
    def __init__(self, U=0.0, delta=1.0, k=0.0):
        self.U = U
        self.delta = delta
        self.k = k
        self.chi_d = 0.0
        self._eps_grad = 1e-8
        self._eps_D = 1e-6

    def los_parameters(self, U, delta, k):
        '''
        Updates all LOS parameters.
        Args:
            U: float, surge speed
            delta: float, cross-track error
            k: float, Path update gain
        '''
        # Sanity checks
        if not jnp.isfinite(U) or U < 0.0:
            raise ValueError("U must be finite and >= 0")
        if not jnp.isfinite(delta) or delta <= 0.0:
            raise ValueError("delta must be finite and > 0")
        if not jnp.isfinite(k):
            raise ValueError("k must be finite")

        # Update parameters
        self.U = float(U)
        self.delta = float(delta)
        self.k = float(k)

    def saturation(self, x, limit=1.0):
        '''
        Saturation function to limit the value of x within [-limit, limit].
        Args:
            x: float, input value
            limit: float, saturation limit
        Returns:
            float, saturated value
        '''
        return jnp.clip(x, -abs(limit), abs(limit))

    def compute(self, position, s, path_function):
        '''
        LOS computes desired heading angle.
        Args:
            position: array, Position of ASV (2D)
            s: float, path variable
            path_function: function, function that defines the path
        Returns:
            chi_d: float, desired heading angle in radians
        '''
        # Validate inputs
        pos = jnp.asarray(position, dtype=jnp.float32).reshape(-1)
        if pos.shape[0] != 2:
            raise ValueError("position must be a length-2 vector")

        # Path point
        p_path = jnp.asarray(path_function(s), dtype=jnp.float32).reshape(-1)
        if p_path.shape[0] != 2:
            raise ValueError("path_function(s) must return a length-2 vector")
        
        # Compute path derivative using numerical differentiation
        # (avoids JAX tracer issues with scipy interpolators)
        ds = 0.001  # small step for numerical derivative
        p_path_behind = jnp.asarray(path_function(max(0.0, s - ds)), dtype=jnp.float32).reshape(-1)
        p_path_ahead = jnp.asarray(path_function(s + ds), dtype=jnp.float32).reshape(-1)
        # Central difference for better accuracy
        p_path_dot = (p_path_ahead - p_path_behind) / (2.0 * ds)

        grad_norm = jnp.linalg.norm(p_path_dot)
        if float(grad_norm) < self._eps_grad:
            raise ValueError("Path gradient norm is ~0; cannot compute heading. Check path_function.")

        theta_path = jnp.arctan2(p_path_dot[1], p_path_dot[0])
        c, s_th = jnp.cos(theta_path), jnp.sin(theta_path)
        R_path = jnp.array([[c, -s_th],
                            [s_th,  c]], dtype=jnp.float32)

        # Cross-track error in path frame
        e = R_path.T @ (pos - p_path)
        e_x, e_y = e[0], e[1]

        # Distances and guards
        D = jnp.sqrt(self.delta**2 + e_y**2)
        D = jnp.maximum(D, self._eps_D)

        # LOS Guidance velocity
        v_los = (self.U / D) * (R_path @ jnp.array([self.delta, -e_y], dtype=jnp.float32))

        # Path parameter update
        s_dot = float(self.U / grad_norm * (self.delta / D + self.k * self.saturation(e_x)))

        return v_los, s_dot