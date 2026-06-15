"""
Quick Start: Minimal Example
============================
ตัวอย่างการใช้งาน module แบบย่อที่สุด

Prerequisites:
- sphere_R2.msh หรือ sphere_R2.vtu ใน folder geo/
- sphere_R2_vf.npy (view factors) ใน folder geo/

ถ้ายังไม่มี view factors ให้ run:
    python 02_compute_viewfactors.py

Usage:
    python quick_start.py
"""

import numpy as np
import pyvista as pv
import sys
sys.path.insert(0, '..')

from RET import (
    MeshProcessor,
    RadiativeEnergyTransfer,
    RETConfig,
)


# ============================================================
# 1. LOAD MESH
# ============================================================
print("Loading mesh...")

# Option A: Load raw mesh and prepare
processor = MeshProcessor("geo/sphere_R2.msh")
mesh = processor.prepare_geometry()

# Option B: Load already prepared mesh
# mesh = pv.read("geo/sphere_R2_prepared.vtu")

# ============================================================
# 2. LOAD VIEW FACTORS
# ============================================================
print("Loading view factors...")
F = np.load("geo/sphere_R2_vf.npy")

# ============================================================
# 3. CREATE SOLVER
# ============================================================
print("Creating solver...")

cfg = RETConfig(
    W=0.005,    # Source power (W)
    c=343.0,    # Speed of sound (m/s)
)

solver = RadiativeEnergyTransfer(
    mesh=mesh,
    view_factors=F,
    alpha=0.2,          # Absorption coefficient
    cfg=cfg,
    volume=33.51,       # Sphere R=2: V = (4/3)πR³
    skip_preprocessing=True,
)

# ============================================================
# 4. SOLVE
# ============================================================
print("Solving...")

source = np.array([0.0, 0.0, 0.0])      # Center
receiver = np.array([1.414, 0.0, 0.0])  # r = √2

B = solver.solve_steady_state(source)
SPL = solver.spl_at_receiver(B, source, receiver)

print(f"\n✓ SPL at receiver = {SPL:.1f} dB")
print(f"  (Expected: ~92.7 dB for α=0.2)")

# ============================================================
# 5. SAVE RESULTS (Optional)
# ============================================================
results = solver.get_results_mesh()
results.save("output/quick_result.vtk")
print(f"✓ Saved to output/quick_result.vtk")
