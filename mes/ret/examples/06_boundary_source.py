"""
Boundary Source Example: TBL Excitation
========================================
กรณีที่ source ไม่ใช่ point source แต่เป็น surface excitation
เช่น Turbulent Boundary Layer (TBL) กระทำบน fuselage panel

Boundary Source Mode:
- W = 0 (ไม่มี point source)
- พลังงานมาจาก surface patches โดยตรง
- กำหนด B_initial หรือ E_source บน patches ที่ถูก excite

Use Cases:
- TBL excitation on aircraft fuselage
- Vibrating panel as sound source
- Distributed sound sources on surfaces
"""
import numpy as np
import pyvista as pv
import os
import sys
sys.path.insert(0, '..')

from ret import MeshProcessor, RadiativeEnergyTransfer, RETConfig

# =============================================================================
# CONFIG
# =============================================================================
MESH_PATH = "geo/sphereR2.vtu"
VF_PATH = "geo/sphereR2_vf.npy"

# =============================================================================
# 1. LOAD MESH & VIEW FACTORS
# =============================================================================
print("="*60)
print("BOUNDARY SOURCE EXAMPLE: TBL Excitation")
print("="*60)

print("\nLoading mesh...")
processor = MeshProcessor(MESH_PATH)
mesh = processor.prepare_geometry()

print("Loading view factors...")
F = np.load(VF_PATH)

N = mesh.n_cells
print(f"  Patches: {N}")

# =============================================================================
# 2. DEFINE TBL EXCITATION REGION
# =============================================================================
# ในตัวอย่างนี้ เราจะกำหนดให้ patches บางส่วนเป็น TBL source
# 
# วิธีเลือก patches:
#   1. By position (เช่น z > threshold)
#   2. By physical ID (ถ้ามี gmsh:physical)
#   3. By cell indices
#   4. Custom selection

print("\n" + "="*60)
print("STEP 2: Define TBL Excitation Region")
print("="*60)

centers = mesh.cell_centers().points
areas = mesh.compute_cell_sizes(area=True).cell_data["Area"]

# ตัวอย่าง: เลือก patches ที่ z > 1.0 (ครึ่งบนของ sphere)
# ในงานจริง อาจเป็น fuselage panel ที่รับ TBL
tbl_mask = centers[:, 2] > 1.0  # z > 1.0 m

n_tbl = np.sum(tbl_mask)
tbl_area = np.sum(areas[tbl_mask])
print(f"  TBL region: {n_tbl} patches ({n_tbl/N*100:.1f}%)")
print(f"  TBL area: {tbl_area:.2f} m²")

# =============================================================================
# 3. CALCULATE TBL SOURCE POWER
# =============================================================================
# TBL excitation สามารถคำนวณจาก:
#   - TBL wall pressure spectrum (Goody, Chase, etc.)
#   - Panel radiation efficiency
#   - หรือกำหนดค่า power density โดยตรง
#
# สมมติ: TBL power density = 1e-4 W/m² บน excited patches

print("\n" + "="*60)
print("STEP 3: Calculate TBL Source Power")
print("="*60)

# Power density จาก TBL (W/m²)
# ในงานจริง คำนวณจาก: P = σ_rad × S_pp × Δf
#   σ_rad = radiation efficiency
#   S_pp  = wall pressure PSD
#   Δf    = frequency bandwidth
tbl_power_density = 1e-4  # W/m²

# คำนวณ source term E สำหรับแต่ละ patch
# E_i = (ρ_i × P_tbl × A_i) / A_i = ρ_i × P_tbl
# หรือถ้าต้องการกำหนด B โดยตรง ก็ได้

E_source = np.zeros(N)
E_source[tbl_mask] = tbl_power_density  # W/m² บน TBL patches

total_tbl_power = tbl_power_density * tbl_area
print(f"  TBL power density: {tbl_power_density:.2e} W/m²")
print(f"  Total TBL power: {total_tbl_power:.4f} W")

# =============================================================================
# 4. CREATE SOLVER (BOUNDARY SOURCE MODE)
# =============================================================================
print("\n" + "="*60)
print("STEP 4: Create Solver (Boundary Source Mode)")
print("="*60)

# W = 0 → ไม่มี point source
# Direct contribution จะเป็น 0
# พลังงานมาจาก E_source ที่เรากำหนดเอง
cfg = RETConfig(
    W=0.0,                  # *** BOUNDARY SOURCE MODE ***
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

print(f"  Mode: Boundary Source (W=0)")
print(f"  Absorption: α = {alpha}")

# =============================================================================
# 5. SOLVE WITH BOUNDARY SOURCE
# =============================================================================
print("\n" + "="*60)
print("STEP 5: Solve with Boundary Source")
print("="*60)

# Method 1: Modify E directly และแก้สมการ
# B = (I - K)^(-1) × E
# โดย E คือ source term ที่เรากำหนด (ไม่ใช่จาก point source)

rho = 1.0 - alpha  # reflection coefficient
K = solver.K       # radiosity operator

# E_boundary = ρ × E_source (reflected portion becomes B)
E_boundary = rho * E_source

# Solve: B = (I - K)^(-1) × E
I = np.eye(N)
B = np.linalg.solve(I - K, E_boundary)

print(f"  Mean B: {np.mean(B):.4e} W/m²")
print(f"  Max B:  {np.max(B):.4e} W/m²")
print(f"  Min B:  {np.min(B):.4e} W/m²")

# =============================================================================
# 6. COMPUTE SPL AT RECEIVERS
# =============================================================================
print("\n" + "="*60)
print("STEP 6: Compute SPL at Receivers")
print("="*60)

# สำหรับ boundary source, ไม่มี direct contribution (W=0)
# SPL มาจาก reflected contribution เท่านั้น:
#   I = Σ (B_i × Ω_i / π)

def compute_spl_boundary(solver, B, receiver):
    """Compute SPL for boundary source (no direct contribution)"""
    receiver = np.asarray(receiver)
    
    # Reflected contribution only
    I_ref = 0.0
    for i in range(solver.N):
        Omega = solver.projected_solid_angle(i, receiver)
        if Omega > 0:
            I_ref += (1.0 / np.pi) * B[i] * Omega
    
    # Convert to SPL
    p2 = I_ref * solver.cfg.rho0 * solver.cfg.c
    pref = 2e-5
    if p2 <= 0:
        return -100.0
    return 10 * np.log10(p2 / pref**2)

# ตัวอย่าง receivers หลายตำแหน่ง
receivers = [
    np.array([0.0, 0.0, 0.0]),      # Center
    np.array([1.0, 0.0, 0.0]),      # x = 1m
    np.array([0.0, 0.0, 1.5]),      # z = 1.5m (ใกล้ TBL region)
    np.array([0.0, 0.0, -1.5]),     # z = -1.5m (ไกล TBL region)
]

print("\n  Receiver positions and SPL:")
print("  " + "-"*40)
for rr in receivers:
    spl = compute_spl_boundary(solver, B, rr)
    print(f"  r = [{rr[0]:5.1f}, {rr[1]:5.1f}, {rr[2]:5.1f}] → SPL = {spl:.1f} dB")

# =============================================================================
# 7. SAVE RESULTS
# =============================================================================
print("\n" + "="*60)
print("STEP 7: Save Results")
print("="*60)

os.makedirs("output", exist_ok=True)

# Save mesh with B distribution
mesh_out = mesh.copy()
mesh_out.cell_data['B'] = B
mesh_out.cell_data['B_dB'] = 10 * np.log10(B / 1e-12 + 1e-30)
mesh_out.cell_data['TBL_source'] = tbl_mask.astype(float)  # Mark TBL region
mesh_out.cell_data['E_source'] = E_source

mesh_out.save("output/boundary_source_result.vtk")
print("  Saved: output/boundary_source_result.vtk")
print("  Fields: B, B_dB, TBL_source, E_source")

print("\n" + "="*60)
print("COMPLETE")
print("="*60)
print("""
Open in ParaView:
  - Color by 'B' or 'B_dB' to see energy distribution
  - Color by 'TBL_source' to see excitation region
  
Note: SPL near TBL region (z > 1) is higher than opposite side
""")

# =============================================================================
# OPTIONAL: QUICK VISUALIZATION
# =============================================================================
# Uncomment to visualize
# import matplotlib.pyplot as plt
# 
# plotter = pv.Plotter()
# plotter.add_mesh(mesh_out, scalars='B_dB', cmap='hot', show_edges=False)
# plotter.add_scalar_bar(title='B (dB re 1e-12)')
# plotter.show()
