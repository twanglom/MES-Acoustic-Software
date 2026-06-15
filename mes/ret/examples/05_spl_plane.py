"""
Example 05: SPL Plane Calculation
=================================
การคำนวณ SPL บน plane สำหรับ visualization ใน ParaView

รองรับ 2 แบบ:
1. Simple grid - สำหรับ geometry เรียบๆ
2. Adaptive grid - สำหรับ geometry ซับซ้อน มี obstacles

Input: 
- sphere_R2_prepared.vtu (prepared mesh)
- sphere_R2_vf.npy (view factor matrix)

Output:
- SPL plane VTK files for ParaView
"""

import numpy as np
import pyvista as pv
import sys
sys.path.insert(0, '..')

from ret import (
    RadiativeEnergyTransfer,
    RETConfig,
    SPLPlaneCalculator,
    ResultExporter,
    Visualizer,
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
    
    # ============================================================
    # 2. CREATE SOLVER AND SOLVE
    # ============================================================
    print("\n" + "="*60)
    print("STEP 2: Create Solver and Solve Steady-State")
    print("="*60)
    
    # Sphere parameters
    R = 2.0
    V = (4/3) * np.pi * R**3
    alpha = 0.20
    
    cfg = RETConfig(W=0.005)
    solver = RadiativeEnergyTransfer(
        mesh=mesh,
        view_factors=F,
        alpha=alpha,
        cfg=cfg,
        volume=V,
        skip_preprocessing=True,
    )
    
    source_pos = np.array([0.0, 0.0, 0.0])
    B_steady = solver.solve_steady_state(source_pos)
    
    # ============================================================
    # 3. CREATE SPL PLANE CALCULATOR
    # ============================================================
    print("\n" + "="*60)
    print("STEP 3: Create SPL Plane Calculator")
    print("="*60)
    
    spl_calc = SPLPlaneCalculator(solver)
    exporter = ResultExporter("output")
    
    # ============================================================
    # 4. SIMPLE GRID (XY-Plane)
    # ============================================================
    print("\n" + "="*60)
    print("STEP 4: Simple Grid - XY Plane at z=0")
    print("="*60)
    
    # สร้าง simple structured grid
    xy_grid = spl_calc.create_plane_grid(
        plane="XY-plane",
        height=0.0,       # z = 0
        spacing=0.2,      # Grid spacing (m)
    )
    
    # คำนวณ SPL บน grid
    xy_grid = spl_calc.compute_spl_on_grid(
        grid=xy_grid,
        B=B_steady,
        source_pos=source_pos,
        show_progress=True
    )
    
    # Save
    exporter.save_plane_vtk(xy_grid, "spl_xy_z0_simple.vtk")
    
    # ============================================================
    # 5. SIMPLE GRID (XZ-Plane)
    # ============================================================
    print("\n" + "="*60)
    print("STEP 5: Simple Grid - XZ Plane at y=0")
    print("="*60)
    
    xz_grid = spl_calc.create_plane_grid(
        plane="XZ-plane",
        height=0.0,       # y = 0
        spacing=0.2,
    )
    
    xz_grid = spl_calc.compute_spl_on_grid(xz_grid, B_steady, source_pos)
    exporter.save_plane_vtk(xz_grid, "spl_xz_y0_simple.vtk")
    
    # ============================================================
    # 6. ADAPTIVE GRID (For Complex Geometries)
    # ============================================================
    print("\n" + "="*60)
    print("STEP 6: Adaptive Grid - XY Plane at z=0")
    print("="*60)
    
    # Adaptive grid จะ:
    # - Slice mesh ที่ height ที่กำหนด
    # - สร้าง polygon จาก slice
    # - Apply offset เพื่อหลีกเลี่ยงขอบ
    # - Triangulate ภายใน polygon
    
    # Note: ต้องการ shapely package
    try:
        xy_adaptive = spl_calc.create_adaptive_plane_grid(
            plane="XY-plane",
            height=0.0,
            spacing=0.15,     # Grid spacing
            offset=0.1,       # Distance from walls
        )
        
        if xy_adaptive.n_points > 0:
            xy_adaptive = spl_calc.compute_spl_on_grid(
                xy_adaptive, B_steady, source_pos
            )
            exporter.save_plane_vtk(xy_adaptive, "spl_xy_z0_adaptive.vtk")
        else:
            print("  [WARNING] Adaptive grid is empty")
            
    except ImportError:
        print("  [SKIP] Shapely not installed - cannot create adaptive grid")
        print("  Install with: pip install shapely")
    
    # ============================================================
    # 7. MULTIPLE HEIGHTS
    # ============================================================
    print("\n" + "="*60)
    print("STEP 7: Multiple Heights")
    print("="*60)
    
    heights = [-1.0, 0.0, 1.0]  # Multiple z-levels
    
    for z in heights:
        print(f"\n  Creating XY plane at z = {z}m...")
        
        grid = spl_calc.create_plane_grid(
            plane="XY-plane",
            height=z,
            spacing=0.25,
        )
        
        grid = spl_calc.compute_spl_on_grid(grid, B_steady, source_pos)
        
        filename = f"spl_xy_z{z:.1f}.vtk".replace("-", "m")
        exporter.save_plane_vtk(grid, filename)
    
    # ============================================================
    # 8. VISUALIZE (Optional)
    # ============================================================
    print("\n" + "="*60)
    print("STEP 8: Visualization")
    print("="*60)
    
    # Uncomment to visualize
    # Visualizer.plot_SPL_plane(xy_grid, show_mesh=True, mesh=mesh)
    # Visualizer.preview_mesh_with_plane(mesh, xy_grid)
    
    print("  (Uncomment Visualizer calls to view interactively)")
    print("  Or open VTK files in ParaView for better visualization")
    
    print("\n" + "="*60)
    print("SPL PLANE CALCULATION COMPLETE")
    print("="*60)
    print("\nOutput files:")
    print("  - output/spl_xy_z0_simple.vtk")
    print("  - output/spl_xz_y0_simple.vtk")
    print("  - output/spl_xy_z*.vtk (multiple heights)")
    print("\nOpen in ParaView and apply 'SPL' colormap")
    
    return solver, B_steady


if __name__ == "__main__":
    solver, B_steady = main()
