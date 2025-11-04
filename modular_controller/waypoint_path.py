"""
Path generator from waypoints using various interpolation methods.
Supports linear, cubic spline, and B-spline interpolation.
"""
import numpy as np
import jax.numpy as jnp
from scipy import interpolate
from typing import List, Tuple, Callable
import yaml


class WaypointPath:
    """
    Creates a smooth parametric path from waypoints.
    The path is parameterized by arc length s.
    """
    
    def __init__(self, waypoints: List[List[float]], 
                 method: str = 'cubic',
                 resolution: int = 50,
                 closed: bool = False):
        """
        Args:
            waypoints: List of [x, y] waypoint coordinates
            method: Interpolation method ('linear', 'cubic', 'bspline')
            resolution: Number of points between waypoints for interpolation
            closed: Whether to close the path (loop back to start)
        """
        if len(waypoints) < 2:
            raise ValueError("Need at least 2 waypoints")
        
        self.waypoints = np.array(waypoints, dtype=float)
        self.method = method.lower()
        self.resolution = max(resolution, 10)
        self.closed = closed
        
        # Build the interpolated path
        self._build_path()
        
    def _build_path(self):
        """Build the interpolated parametric path."""
        waypoints = self.waypoints.copy()
        
        # Close the path if requested
        if self.closed and not np.allclose(waypoints[0], waypoints[-1]):
            waypoints = np.vstack([waypoints, waypoints[0:1]])
        
        n_waypoints = len(waypoints)
        
        if self.method == 'linear':
            # Simple linear interpolation between waypoints
            self.path_x = interpolate.interp1d(
                np.arange(n_waypoints), waypoints[:, 0], 
                kind='linear', fill_value='extrapolate'
            )
            self.path_y = interpolate.interp1d(
                np.arange(n_waypoints), waypoints[:, 1],
                kind='linear', fill_value='extrapolate'
            )
            # Parameter goes from 0 to n_waypoints-1
            self.s_max = float(n_waypoints - 1)
            
        elif self.method == 'cubic':
            # Cubic spline interpolation
            if n_waypoints < 4:
                # Fall back to quadratic for fewer points
                k = min(n_waypoints - 1, 2)
            else:
                k = 3
                
            # Create parameter t based on cumulative distance
            t = np.zeros(n_waypoints)
            for i in range(1, n_waypoints):
                t[i] = t[i-1] + np.linalg.norm(waypoints[i] - waypoints[i-1])
            
            # Normalize t to [0, 1]
            if t[-1] > 0:
                t = t / t[-1]
            
            self.path_x = interpolate.UnivariateSpline(t, waypoints[:, 0], k=k, s=0)
            self.path_y = interpolate.UnivariateSpline(t, waypoints[:, 1], k=k, s=0)
            
            # Compute arc length by integration
            self._compute_arc_length_table(t)
            
        elif self.method == 'bspline':
            # B-spline interpolation
            if n_waypoints < 4:
                k = min(n_waypoints - 1, 2)
            else:
                k = 3
            
            # Create parameter array
            t = np.linspace(0, 1, n_waypoints)
            
            # Fit B-spline
            tck, _ = interpolate.splprep([waypoints[:, 0], waypoints[:, 1]], 
                                         k=k, s=0)
            self.tck = tck
            
            # Compute arc length
            self._compute_arc_length_table_bspline()
            
        else:
            raise ValueError(f"Unknown interpolation method: {self.method}")
    
    def _compute_arc_length_table(self, t):
        """Compute arc length lookup table for cubic splines."""
        # Sample the path densely
        n_samples = self.resolution * len(self.waypoints)
        t_samples = np.linspace(0, 1, n_samples)
        
        x_samples = self.path_x(t_samples)
        y_samples = self.path_y(t_samples)
        
        # Compute cumulative arc length
        dx = np.diff(x_samples)
        dy = np.diff(y_samples)
        ds = np.sqrt(dx**2 + dy**2)
        s_samples = np.zeros(n_samples)
        s_samples[1:] = np.cumsum(ds)
        
        self.s_max = float(s_samples[-1])
        
        # Create inverse mapping: s -> t
        self.s_to_t = interpolate.interp1d(
            s_samples, t_samples, 
            kind='linear', 
            bounds_error=False,
            fill_value=(0.0, 1.0)
        )
    
    def _compute_arc_length_table_bspline(self):
        """Compute arc length lookup table for B-splines."""
        n_samples = self.resolution * len(self.waypoints)
        u_samples = np.linspace(0, 1, n_samples)
        
        points = interpolate.splev(u_samples, self.tck)
        x_samples = points[0]
        y_samples = points[1]
        
        # Compute cumulative arc length
        dx = np.diff(x_samples)
        dy = np.diff(y_samples)
        ds = np.sqrt(dx**2 + dy**2)
        s_samples = np.zeros(n_samples)
        s_samples[1:] = np.cumsum(ds)
        
        self.s_max = float(s_samples[-1])
        
        # Create inverse mapping: s -> u
        self.s_to_u = interpolate.interp1d(
            s_samples, u_samples,
            kind='linear',
            bounds_error=False,
            fill_value=(0.0, 1.0)
        )
    
    def __call__(self, s: float) -> jnp.ndarray:
        """
        Evaluate path at arc length parameter s.
        
        Args:
            s: Arc length parameter
            
        Returns:
            jnp.array([x, y]) position on the path
        """
        s = float(s)
        
        # Clamp s to valid range
        s = np.clip(s, 0.0, self.s_max)
        
        if self.method == 'linear':
            x = float(self.path_x(s))
            y = float(self.path_y(s))
            
        elif self.method == 'cubic':
            t = float(self.s_to_t(s))
            x = float(self.path_x(t))
            y = float(self.path_y(t))
            
        elif self.method == 'bspline':
            u = float(self.s_to_u(s))
            points = interpolate.splev(u, self.tck)
            x = float(points[0])
            y = float(points[1])
        
        return jnp.array([x, y], dtype=jnp.float32)
    
    def get_path_length(self) -> float:
        """Return total path length."""
        return self.s_max
    
    def get_waypoints(self) -> np.ndarray:
        """Return the original waypoints."""
        return self.waypoints.copy()


def load_waypoints_from_yaml(yaml_path: str) -> Tuple[WaypointPath, dict]:
    """
    Load waypoints from YAML file and create a path.
    
    Args:
        yaml_path: Path to YAML configuration file
        
    Returns:
        (WaypointPath object, settings dictionary)
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    waypoints = config.get('waypoints', [])
    if not waypoints:
        raise ValueError("No waypoints defined in YAML file")
    
    settings = config.get('path_settings', {})
    method = settings.get('interpolation', 'cubic')
    resolution = settings.get('resolution', 50)
    closed = settings.get('closed_path', False)
    
    path = WaypointPath(waypoints, method=method, resolution=resolution, closed=closed)
    
    return path, settings


if __name__ == "__main__":
    # Example usage
    waypoints = [
        [0.0, 0.0],
        [10.0, 5.0],
        [20.0, 5.0],
        [30.0, 0.0],
    ]
    
    path = WaypointPath(waypoints, method='cubic', resolution=50)
    
    print(f"Path length: {path.get_path_length():.2f} m")
    print(f"Start: {path(0.0)}")
    print(f"Mid: {path(path.get_path_length()/2)}")
    print(f"End: {path(path.get_path_length())}")
