# """
# Radiative Energy Transfer - Acoustical Energy Method for Room Acoustics

# A Python package for computing sound fields in enclosures using the 
# radiative energy transfer (acoustical radiosity) method.

# Based on:
# - Nosal, Hodgson & Ashdown (2004) JASA 116(2), 970-980
# - Kuttruff (1997) ACUSTICA 83, 622-628

# Features:
# - Steady-state energy density computation
# - Time-domain impulse response
# - Reverberation time estimation
# - SPL at arbitrary receiver positions
# - SPL mapping on planes (for ParaView)
# - Adaptive grid generation for complex geometries
# - VTK export for visualization
# - Boundary/surface source mode support

# Basic Usage:
# -----------
# ```python
# from radiative_energy_transfer import (
#     MeshProcessor, 
#     ViewFactorCalculator,
#     RadiativeEnergyTransfer,
#     RETConfig,
#     ResultExporter
# )

# # 1. Load and prepare mesh
# processor = MeshProcessor("room.vtu")
# mesh = processor.prepare_for_ret()  # Auto-checks and flips normals

# # 2. Compute or load view factors
# vf_calc = ViewFactorCalculator(mesh)
# F = vf_calc.compute()  # or vf_calc.load("viewfactors.npy")

# # 3. Create solver and run
# cfg = RETConfig(W=1.0)
# solver = RadiativeEnergyTransfer(mesh, F, alpha=0.1, cfg=cfg, volume=V)

# source = [1, 1, 1]
# B = solver.solve_steady_state(source)

# # 4. Get SPL at receiver
# receiver = [3, 2, 1.2]
# SPL = solver.spl_at_receiver(B, source, receiver)
# print(f"SPL = {SPL:.1f} dB")

# # 5. Export results
# exporter = ResultExporter("results")
# exporter.save_mesh_vtk(solver.get_results_mesh(), "B_distribution.vtk")

# # 6. Create adaptive SPL plane (for complex geometries)
# from radiative_energy_transfer import SPLPlaneCalculator
# spl_calc = SPLPlaneCalculator(solver)
# grid = spl_calc.create_adaptive_plane_grid("XY-plane", height=1.2, spacing=0.2, offset=0.1)
# grid = spl_calc.compute_spl_on_grid(grid, B, source)
# exporter.save_plane_vtk(grid, "SPL_plane.vtk")
# ```

# Author: Thanasak (PhD Research - Aero-vibro-acoustics)
# """

# __version__ = "2.0.0"
# __author__ = "Thanasak"

# # Core classes
# from .config import (
#     RETConfig,
#     OutputConfig,
#     SourceConfig,
#     ReceiverConfig,
#     MaterialConfig,
# )

# from .geometry import (
#     MeshProcessor,
#     ViewFactorCalculator,
#     compute_volume,
#     estimate_volume_from_area,
#     create_absorption_array,
#     create_alpha_by_normal,
#     create_alpha_from_physical,
#     create_alpha_array,
# )

# from .RETsolver import (
#     RadiativeEnergyTransfer
# )

# from .postprocess import (
#     SPLPlaneCalculator,
#     ResultExporter,
#     Visualizer,
#     save_vtk,
# )

# # Convenience aliases
# RET = RadiativeEnergyTransfer
# Config = RETConfig

# # Backward compatibility
# SolverConfig = RETConfig

# __all__ = [
#     # Config
#     "RETConfig",
#     "SolverConfig",  # Backward compatibility
#     "OutputConfig", 
#     "SourceConfig",
#     "ReceiverConfig",
#     "MaterialConfig",
#     "Config",
#     # Geometry
#     "MeshProcessor",
#     "ViewFactorCalculator",
#     "compute_volume",
#     "estimate_volume_from_area",
#     "create_absorption_array",
#     "create_alpha_by_normal",
#     "create_alpha_from_physical",
#     "create_alpha_array",
#     # Solver
#     "RadiativeEnergyTransfer",
#     "EnergyRadiositySolver",
#     "Solver",
#     "RET",
#     # Postprocess
#     "SPLPlaneCalculator",
#     "ResultExporter",
#     "Visualizer",
#     "save_vtk",
# ]

from .config import (
    MaterialConfig,
    OutputConfig,
    ReceiverConfig,
    RETConfig,
    SourceConfig,
)
from .geometry import (
    MeshProcessor,
    ViewFactorCalculator,
    compute_volume,
    create_absorption_array,
    create_alpha_array,
    create_alpha_by_normal,
    create_alpha_from_physical,
    estimate_volume_from_area,
)
from .postprocess import (
    EnergyFieldCalculator,
    ResultExporter,
    SPLPlaneCalculator,
    Visualizer,
)
from .RETsolver import RadiativeEnergyTransfer

RET = RadiativeEnergyTransfer
Config = RETConfig
SolverConfig = RETConfig

__all__ = [
    "Config",
    "EnergyFieldCalculator",
    "MaterialConfig",
    "MeshProcessor",
    "OutputConfig",
    "RadiativeEnergyTransfer",
    "ReceiverConfig",
    "ResultExporter",
    "RET",
    "RETConfig",
    "SolverConfig",
    "SourceConfig",
    "SPLPlaneCalculator",
    "ViewFactorCalculator",
    "Visualizer",
    "compute_volume",
    "create_absorption_array",
    "create_alpha_array",
    "create_alpha_by_normal",
    "create_alpha_from_physical",
    "estimate_volume_from_area",
]
