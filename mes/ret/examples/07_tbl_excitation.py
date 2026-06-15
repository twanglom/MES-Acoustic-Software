"""
TBL Excitation on Aircraft Fuselage (Realistic Example)
========================================================
การคำนวณ interior noise จาก Turbulent Boundary Layer excitation

Flow:
  TBL pressure → Panel vibration → Radiated sound → Interior field

Input ที่ต้องการ:
  1. Mesh ของ interior cavity
  2. View factors
  3. TBL-excited patches (เช่น fuselage skin)
  4. Radiated power จากแต่ละ patch (จาก vibro-acoustic analysis)

Output:
  - B distribution บน interior surfaces
  - SPL ที่ตำแหน่ง receiver (เช่น passenger head position)
"""
import numpy as np
import pyvista as pv
import os
import sys
sys.path.insert(0, '..')

from ret import MeshProcessor, RadiativeEnergyTransfer, RETConfig


def select_tbl_patches(mesh, method='z_threshold', **kwargs):
    """
    เลือก patches ที่รับ TBL excitation
    
    Methods:
    - 'z_threshold': patches ที่ z > threshold
    - 'physical_id': patches ที่มี gmsh:physical == id
    - 'custom': user-defined mask
    """
    centers = mesh.cell_centers().points
    N = mesh.n_cells
    
    if method == 'z_threshold':
        threshold = kwargs.get('threshold', 0.0)
        mask = centers[:, 2] > threshold
        
    elif method == 'physical_id':
        phys_id = kwargs.get('phys_id', 1)
        if 'gmsh:physical' in mesh.cell_data:
            mask = mesh.cell_data['gmsh:physical'] == phys_id
        else:
            print("[WARNING] No physical IDs found, using all patches")
            mask = np.ones(N, dtype=bool)
            
    elif method == 'custom':
        mask = kwargs.get('mask', np.ones(N, dtype=bool))
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return mask


def compute_tbl_radiated_power(mesh, tbl_mask, 
                                Spp=1.0,        # Wall pressure PSD (Pa²/Hz)
                                sigma=0.1,      # Radiation efficiency
                                delta_f=1000,   # Frequency bandwidth (Hz)
                                rho0=1.21,      # Air density
                                c=343.0):       # Speed of sound
    """
    คำนวณ radiated power จาก TBL excitation
    
    Simplified model:
        P_rad = σ × A × Spp × Δf / (ρ₀ × c)
    
    ในงานจริง ควรใช้:
        - TBL model (Goody, Chase, Smol'yakov, etc.)
        - Panel transfer function
        - Radiation efficiency จาก modal analysis
    
    Parameters:
    -----------
    Spp : float
        Wall pressure PSD (Pa²/Hz) - จาก TBL model
    sigma : float
        Radiation efficiency (0-1) - จาก structural analysis
    delta_f : float
        Frequency bandwidth (Hz)
    
    Returns:
    --------
    P_rad : ndarray
        Radiated power per patch (W)
    """
    areas = mesh.compute_cell_sizes(area=True).cell_data["Area"]
    
    # Radiated intensity (W/m²)
    # I = σ × Spp × Δf / (ρ₀ × c)
    I_rad = sigma * Spp * delta_f / (rho0 * c)
    
    # Radiated power per patch (W)
    P_rad = np.zeros(mesh.n_cells)
    P_rad[tbl_mask] = I_rad * areas[tbl_mask]
    
    return P_rad, I_rad


def solve_boundary_source(solver, E_source):
    """
    Solve radiosity equation with boundary source
    
    B = (I - K)^(-1) × E
    
    Parameters:
    -----------
    E_source : ndarray
        Source term per patch (W/m²)
    
    Returns:
    --------
    B : ndarray
        Radiation density per patch (W/m²)
    """
    N = solver.N
    rho = solver.rho  # reflection coefficient per patch
    K = solver.K      # radiosity operator
    
    # E_boundary = ρ × E_source
    E_boundary = rho * E_source
    
    # Solve: B = (I - K)^(-1) × E
    I = np.eye(N)
    B = np.linalg.solve(I - K, E_boundary)
    
    return B


def compute_spl_from_B(solver, B, receiver):
    """
    Compute SPL at receiver from B distribution (no direct source)
    
    I = Σ (B_i × Ω_i / π)
    SPL = 10 × log10(I × ρ₀ × c / p_ref²)
    """
    receiver = np.asarray(receiver)
    
    # Reflected contribution only
    I_total = 0.0
    for i in range(solver.N):
        Omega = solver.projected_solid_angle(i, receiver)
        if Omega > 0:
            I_total += (1.0 / np.pi) * B[i] * Omega
    
    # Convert to SPL
    p2 = I_total * solver.cfg.rho0 * solver.cfg.c
    pref = 2e-5
    if p2 <= 0:
        return -100.0
    return 10 * np.log10(p2 / pref**2)


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*60)
    print("TBL EXCITATION EXAMPLE")
    print("="*60)
    
    # -------------------------------------------------------------------------
    # 1. Load geometry
    # -------------------------------------------------------------------------
    print("\n[1] Loading geometry...")
    processor = MeshProcessor("geo/sphereR2.vtu")
    mesh = processor.prepare_geometry()
    F = np.load("geo/sphereR2_vf.npy")
    
    N = mesh.n_cells
    areas = mesh.compute_cell_sizes(area=True).cell_data["Area"]
    print(f"    Patches: {N}")
    print(f"    Total area: {np.sum(areas):.2f} m²")
    
    # -------------------------------------------------------------------------
    # 2. Select TBL region
    # -------------------------------------------------------------------------
    print("\n[2] Selecting TBL region...")
    
    # ตัวอย่าง: ครึ่งบนของ sphere เป็น TBL region
    tbl_mask = select_tbl_patches(mesh, method='z_threshold', threshold=0.5)
    
    n_tbl = np.sum(tbl_mask)
    tbl_area = np.sum(areas[tbl_mask])
    print(f"    TBL patches: {n_tbl} ({n_tbl/N*100:.1f}%)")
    print(f"    TBL area: {tbl_area:.2f} m²")
    
    # -------------------------------------------------------------------------
    # 3. Compute TBL radiated power
    # -------------------------------------------------------------------------
    print("\n[3] Computing TBL radiated power...")
    
    # TBL parameters (example values)
    # ในงานจริง ควรคำนวณจาก:
    #   - Flight condition (Mach, altitude)
    #   - TBL model (Goody, Chase, etc.)
    #   - Panel properties
    Spp = 100.0       # Wall pressure PSD (Pa²/Hz) - example
    sigma = 0.01      # Radiation efficiency - example
    delta_f = 1000    # Bandwidth (Hz)
    
    P_rad, I_rad = compute_tbl_radiated_power(
        mesh, tbl_mask,
        Spp=Spp,
        sigma=sigma,
        delta_f=delta_f
    )
    
    total_power = np.sum(P_rad)
    print(f"    Spp = {Spp} Pa²/Hz")
    print(f"    σ = {sigma}")
    print(f"    Δf = {delta_f} Hz")
    print(f"    Radiated intensity: {I_rad:.4e} W/m²")
    print(f"    Total radiated power: {total_power:.4e} W")
    
    # -------------------------------------------------------------------------
    # 4. Create solver (boundary source mode)
    # -------------------------------------------------------------------------
    print("\n[4] Creating solver...")
    
    cfg = RETConfig(
        W=0.0,  # Boundary source mode
        check_obstruction=False,
    )
    
    R = 2.0
    V = (4/3) * np.pi * R**3
    alpha = 0.2
    
    solver = RadiativeEnergyTransfer(
        mesh, F,
        alpha=alpha,
        cfg=cfg,
        volume=V,
        skip_preprocessing=True
    )
    
    # -------------------------------------------------------------------------
    # 5. Solve
    # -------------------------------------------------------------------------
    print("\n[5] Solving...")
    
    # E_source = radiated power density (W/m²)
    E_source = np.zeros(N)
    E_source[tbl_mask] = I_rad
    
    B = solve_boundary_source(solver, E_source)
    
    print(f"    Mean B: {np.mean(B):.4e} W/m²")
    print(f"    B on TBL region: {np.mean(B[tbl_mask]):.4e} W/m²")
    print(f"    B on other region: {np.mean(B[~tbl_mask]):.4e} W/m²")
    
    # -------------------------------------------------------------------------
    # 6. Compute SPL at receivers
    # -------------------------------------------------------------------------
    print("\n[6] Computing SPL at receivers...")
    
    # Receiver positions (example: passenger positions)
    receivers = {
        "Center": np.array([0.0, 0.0, 0.0]),
        "Near TBL (z=1.5)": np.array([0.0, 0.0, 1.5]),
        "Far from TBL (z=-1.5)": np.array([0.0, 0.0, -1.5]),
        "Side (y=1.0)": np.array([0.0, 1.0, 0.0]),
    }
    
    print("\n    " + "-"*45)
    print(f"    {'Position':<25} {'SPL (dB)':<10}")
    print("    " + "-"*45)
    for name, pos in receivers.items():
        spl = compute_spl_from_B(solver, B, pos)
        print(f"    {name:<25} {spl:<10.1f}")
    print("    " + "-"*45)
    
    # -------------------------------------------------------------------------
    # 7. Save results
    # -------------------------------------------------------------------------
    print("\n[7] Saving results...")
    
    os.makedirs("output", exist_ok=True)
    
    mesh_out = mesh.copy()
    mesh_out.cell_data['B'] = B
    mesh_out.cell_data['B_dB'] = 10 * np.log10(B / 1e-12 + 1e-30)
    mesh_out.cell_data['TBL_region'] = tbl_mask.astype(float)
    mesh_out.cell_data['E_source'] = E_source
    mesh_out.cell_data['P_radiated'] = P_rad
    
    mesh_out.save("output/tbl_excitation_result.vtk")
    print("    Saved: output/tbl_excitation_result.vtk")
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    
    return solver, B, tbl_mask


if __name__ == "__main__":
    solver, B, tbl_mask = main()
