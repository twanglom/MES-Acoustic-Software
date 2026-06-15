"""
Radiative Energy Transfer - Configuration Module

Configuration classes for the RET solver.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict
import numpy as np


@dataclass
class RETConfig:
    """
    Configuration for the Radiative Energy Transfer Solver.
    
    Attributes
    ----------
    c : float
        Speed of sound (m/s)
    rho0 : float
        Air density (kg/m³)
    W : float
        Source power (W). Set to 0.0 for boundary/surface source mode.
    dt : float
        Time step for time-domain simulation (s)
    tmax : float
        Maximum simulation time (s)
    m_air : float
        Air absorption coefficient (1/m)
    check_obstruction : bool
        Enable ray-tracing for non-convex enclosures
    show_progress : bool
        Show progress during computation
    progress_every : int
        Print progress every N operations
    """
    c: float = 343.0
    rho0: float = 1.21
    W: float = 1.0
    dt: float = 1 / 24000
    tmax: float = 1.0
    m_air: float = 0.0
    check_obstruction: bool = True
    show_progress: bool = True
    progress_every: int = 5000
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.c > 0, "Speed of sound must be positive"
        assert self.rho0 > 0, "Air density must be positive"
        assert self.W >= 0, "Source power must be non-negative"
        assert self.dt > 0, "Time step must be positive"
        assert self.tmax > 0, "Max time must be positive"
        assert self.m_air >= 0, "Air absorption must be non-negative"
    
    @property
    def is_boundary_source(self) -> bool:
        """Check if operating in boundary/surface source mode."""
        return self.W <= 0


@dataclass
class OutputConfig:
    """
    Configuration for output and export options.
    
    Attributes
    ----------
    save_B_distribution : bool
        Save radiation density on mesh
    save_SPL_plane : bool
        Save SPL on a plane grid
    save_decay_curve : bool
        Save decay curve data
    output_dir : str
        Directory for output files
    vtk_format : str
        VTK file format ('vtk' or 'vtu')
    """
    save_B_distribution: bool = True
    save_SPL_plane: bool = False
    save_decay_curve: bool = False
    output_dir: str = "results"
    vtk_format: str = "vtk"
    
    # SPL plane options
    spl_plane: str = "XY-plane"
    spl_plane_height: float = 0.0
    spl_plane_spacing: float = 0.5
    spl_plane_offset: float = 0.1


@dataclass
class SourceConfig:
    """
    Sound source configuration.
    
    Attributes
    ----------
    position : array-like
        Source position (x, y, z)
    power : float
        Acoustic power (W)
    directivity : str
        Source directivity ('omnidirectional' only for now)
    """
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    power: float = 1.0
    directivity: str = "omnidirectional"
    
    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)


@dataclass 
class ReceiverConfig:
    """
    Receiver configuration.
    
    Attributes
    ----------
    position : array-like
        Receiver position (x, y, z)
    """
    position: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    
    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)


@dataclass
class MaterialConfig:
    """
    Surface material/absorption configuration.
    
    Attributes
    ----------
    default_alpha : float
        Default absorption coefficient
    surface_alpha : dict
        Per-surface absorption coefficients {surface_name: alpha}
    """
    default_alpha: float = 0.1
    surface_alpha: Dict[str, float] = field(default_factory=dict)
