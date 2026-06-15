"""
Room SPL Plane Computation
==========================
คำนวณ SPL บน XY และ XZ planes สำหรับห้องที่มี obstacles
ใช้ adaptive grid (Shapely) เพื่อหลีกเลี่ยงขอบและ obstacles
"""
import numpy as np
import pyvista as pv
import os
import sys
sys.path.insert(0, '..')

from ret import (
    MeshProcessor, 
    RadiativeEnergyTransfer, 
    RETConfig,
    SPLPlaneCalculator,
    create_alpha_array,  # ใช้ function จาก module!
)

# =============================================================================
# CONFIG
# =============================================================================
MESH_PATH = "geo/room.vtu"
VF_PATH = "geo/room.npy"
OUTPUT_DIR = "output"

# Physical IDs และ absorption coefficients
PHYS_ID = {"WALL": 1, "FLOOR": 2, "BLOCK": 3, "CEILING": 4}
ALPHA = {"WALL": 0.2, "FLOOR": 0.3, "BLOCK": 0.1, "CEILING": 0.6}

# Source position
SOURCE = np.array([1.0, 1.0, 1.0])

# =============================================================================
# MAIN
# =============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # 1. Load & Prepare Mesh
    # -------------------------------------------------------------------------
    print("[1] Loading mesh...")
    processor = MeshProcessor(MESH_PATH)
    mesh = processor.prepare_for_ret()
    print(f"    Cells: {mesh.n_cells}")
    
    # -------------------------------------------------------------------------
    # 2. Load View Factors & Create Alpha
    # -------------------------------------------------------------------------
    print("[2] Loading view factors...")
    F = np.load(VF_PATH)
    
    # ใช้ create_alpha_array จาก module (auto-detect physical IDs หรือ fallback)
    alpha = create_alpha_array(mesh, PHYS_ID, ALPHA)
    print(f"    Alpha mean: {alpha.mean():.3f}")
    
    # -------------------------------------------------------------------------
    # 3. Create Solver & Solve
    # -------------------------------------------------------------------------
    print("[3] Solving steady-state...")
    cfg = RETConfig(W=0.005, check_obstruction=True)
    solver = RadiativeEnergyTransfer(mesh, F, alpha=alpha, cfg=cfg, skip_preprocessing=True)
    
    B = solver.solve_steady_state(SOURCE)
    print(f"    Mean B: {np.mean(B):.4e} W/m²")
    
    # Save surface energy
    mesh_out = solver.get_results_mesh()
    mesh_out.save(f"{OUTPUT_DIR}/surface_energy.vtk")
    print(f"    Saved: {OUTPUT_DIR}/surface_energy.vtk")
    
    # -------------------------------------------------------------------------
    # 4. Generate SPL Planes
    # -------------------------------------------------------------------------
    print("[4] Generating SPL planes...")
    spl_calc = SPLPlaneCalculator(solver)
    
    # XY-plane at z=1.2m (ear level)
    print("    XY-plane (z=1.2m)...")
    xy_grid = spl_calc.create_adaptive_plane_grid("XY-plane", height=1.2, spacing=0.2, offset=0.1)
    xy_grid = spl_calc.compute_spl_on_grid(xy_grid, B, SOURCE)
    xy_grid.save(f"{OUTPUT_DIR}/spl_xy_z1.2.vtk")
    
    # XZ-plane at mid-Y
    mid_y = (mesh.bounds[2] + mesh.bounds[3]) / 2
    print(f"    XZ-plane (y={mid_y:.1f}m)...")
    xz_grid = spl_calc.create_adaptive_plane_grid("XZ-plane", height=mid_y, spacing=0.2, offset=0.1)
    xz_grid = spl_calc.compute_spl_on_grid(xz_grid, B, SOURCE)
    xz_grid.save(f"{OUTPUT_DIR}/spl_xz_y{mid_y:.1f}.vtk")
    
    print(f"\n[DONE] Files saved to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()