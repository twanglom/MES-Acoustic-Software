"""
Example 03: Run Steady-State Solver
===================================
การคำนวณ steady-state radiation density และ SPL

Steady-state หมายถึง:
- แหล่งกำเนิดเสียงเปิดตลอด (continuous source)
- ระบบถึง equilibrium แล้ว
- คำนวณเร็วกว่า time-domain มาก

Input: 
- sphere_R2_prepared.vtu (prepared mesh)
- sphere_R2_vf.npy (view factor matrix)

Output:
- SPL at receiver position
- B distribution on mesh (VTK file)
"""

import numpy as np
import pyvista as pv
import sys
sys.path.insert(0, '..')

from ret import (
    RadiativeEnergyTransfer,
    RETConfig,
    ResultExporter,
)


def main():
    # ============================================================
    # 1. LOAD MESH AND VIEW FACTORS
    # ============================================================
    print("="*60)
    print("STEP 1: Load Mesh and View Factors")
    print("="*60)
    
    mesh = pv.read("geo/sphere_R2_prepared.vtu")
    F = np.load("geo/sphere_R2_vf.npy")
    
    print(f"  Mesh: {mesh.n_cells} cells")
    print(f"  View factors: {F.shape}")
    
    # ============================================================
    # 2. SET PARAMETERS
    # ============================================================
    print("\n" + "="*60)
    print("STEP 2: Set Parameters")
    print("="*60)
    
    # Sphere parameters
    R = 2.0  # radius (m)
    V = (4/3) * np.pi * R**3  # volume
    S = 4 * np.pi * R**2      # surface area
    
    print(f"  Sphere radius: {R} m")
    print(f"  Volume: {V:.2f} m³")
    print(f"  Surface area: {S:.2f} m²")
    
    # Absorption coefficient (from Nosal Table I)
    # Case 1: α = 0.05
    # Case 2: α = 0.20
    # Case 3: α = 0.50
    alpha = 0.20  # Case 2
    
    print(f"  Absorption coefficient: {alpha}")
    
    # Source and receiver positions
    source_pos = np.array([0.0, 0.0, 0.0])  # Center of sphere
    receiver_pos = np.array([np.sqrt(2), 0.0, 0.0])  # r = √2 m
    
    print(f"  Source position: {source_pos}")
    print(f"  Receiver position: {receiver_pos} (r = √2 m)")
    
    # ============================================================
    # 3. CREATE SOLVER
    # ============================================================
    print("\n" + "="*60)
    print("STEP 3: Create Solver")
    print("="*60)
    
    # Configuration
    cfg = RETConfig(
        W=0.005,              # Source power (W) - 5 mW
        c=343.0,              # Speed of sound (m/s)
        rho0=1.21,            # Air density (kg/m³)
        check_obstruction=False,  # False for convex enclosure
    )
    
    # Create solver
    solver = RadiativeEnergyTransfer(
        mesh=mesh,
        view_factors=F,
        alpha=alpha,
        cfg=cfg,
        volume=V,
        skip_preprocessing=True,  # Mesh already prepared
    )
    
    # ============================================================
    # 4. SOLVE STEADY-STATE
    # ============================================================
    print("\n" + "="*60)
    print("STEP 4: Solve Steady-State")
    print("="*60)
    
    B_steady = solver.solve_steady_state(source_pos)
    
    # ============================================================
    # 5. COMPUTE SPL AT RECEIVER
    # ============================================================
    print("\n" + "="*60)
    print("STEP 5: Compute SPL at Receiver")
    print("="*60)
    
    SPL = solver.compute_SPL_steady(B_steady, source_pos, receiver_pos)
    
    print(f"  SPL at receiver: {SPL:.2f} dB")
    
    # ============================================================
    # 6. COMPARE WITH THEORY (Nosal Table I)
    # ============================================================
    print("\n" + "="*60)
    print("STEP 6: Compare with Theory")
    print("="*60)
    
    # Theoretical values from Nosal Table I (Case 2: α = 0.20)
    B_theory = 3.98e-4  # W/m²
    Lp_theory = 92.68   # dB
    
    B_computed = np.mean(B_steady)
    
    print(f"  B_theory:   {B_theory:.2e} W/m²")
    print(f"  B_computed: {B_computed:.2e} W/m²")
    print(f"  B error:    {abs(B_computed - B_theory) / B_theory * 100:.2f}%")
    print()
    print(f"  Lp_theory:   {Lp_theory:.2f} dB")
    print(f"  Lp_computed: {SPL:.2f} dB")
    print(f"  Lp error:    {abs(SPL - Lp_theory):.2f} dB")
    
    # ============================================================
    # 7. EXPORT RESULTS
    # ============================================================
    print("\n" + "="*60)
    print("STEP 7: Export Results")
    print("="*60)
    
    exporter = ResultExporter("output")
    
    # Get mesh with results
    results_mesh = solver.get_results_mesh()
    
    # Save to VTK
    exporter.save_mesh_vtk(results_mesh, "sphere_B_steady.vtk")
    
    # Save summary
    summary = {
        "Case": "Sphere R=2m, α=0.20",
        "Source Power": f"{cfg.W} W",
        "Volume": f"{V:.2f} m³",
        "Surface Area": f"{S:.2f} m²",
        "Mean B": f"{B_computed:.4e} W/m²",
        "SPL at receiver": f"{SPL:.2f} dB",
        "B_theory": f"{B_theory:.4e} W/m²",
        "Lp_theory": f"{Lp_theory:.2f} dB",
    }
    exporter.save_summary(summary, "steady_state_summary.txt")
    
    print("\n" + "="*60)
    print("STEADY-STATE SIMULATION COMPLETE")
    print("="*60)
    
    return solver, B_steady


if __name__ == "__main__":
    solver, B_steady = main()
