"""
Validation: Nosal Table I - Case 2 (Steady-State)
=================================================
Sphere R=2m, α=0.20

Steady-State หมายถึง:
- Source เปิดตลอด (continuous)
- ระบบถึง equilibrium แล้ว
- แก้สมการ: B = (I - K)^(-1) × E
  โดย K = radiosity operator, E = direct contribution
"""
import numpy as np
import sys
sys.path.insert(0, '..')

from RET import MeshProcessor, RadiativeEnergyTransfer, RETConfig

# =============================================================================
# REFERENCE VALUES (Nosal Table I, Case 2)
# =============================================================================
B_THEORY = 3.98e-4   # W/m² - Mean radiation density on surface
LP_THEORY = 92.68    # dB   - SPL at receiver

# =============================================================================
# PARAMETERS
# =============================================================================
R = 2.0                          # Sphere radius (m)
V = (4/3) * np.pi * R**3         # Volume = 33.51 m³
alpha = 0.20                     # Absorption coefficient (20%)
source = np.array([0, 0, 0])     # Source at center
receiver = np.array([np.sqrt(2), 0, 0])  # Receiver at r = √2 m from center

# =============================================================================
# 1. LOAD MESH
# =============================================================================
# ใช้ MeshProcessor เพื่อ:
#   - Load mesh
#   - ตรวจสอบและ flip normals ให้ชี้เข้าด้านใน
print("Loading mesh...")
processor = MeshProcessor("geo/sphereR2.vtu")
mesh = processor.prepare_geometry()

# =============================================================================
# 2. LOAD VIEW FACTORS
# =============================================================================
# View factors คำนวณไว้แล้วจาก compute_viewfactors.py
print("Loading view factors...")
F = np.load("geo/sphereR2_vf.npy")

# =============================================================================
# 3. CREATE SOLVER & SOLVE
# =============================================================================
# RETConfig parameters:
#   - W: Source power (Watts)
#   - check_obstruction: False สำหรับ convex enclosure
#
# RadiativeEnergyTransfer:
#   - mesh: prepared surface mesh
#   - F: view factor matrix (N×N)
#   - alpha: absorption coefficient (scalar หรือ array per cell)
#   - volume: enclosure volume (ถ้าไม่ใส่จะ estimate เอง)
#   - skip_preprocessing: True ถ้า mesh ผ่าน prepare_for_ret() มาแล้ว
print("Solving steady-state...")
cfg = RETConfig(W=0.005, check_obstruction=False)
solver = RadiativeEnergyTransfer(
    mesh, F, 
    alpha=alpha, 
    cfg=cfg, 
    volume=V, 
    skip_preprocessing=True
)

# solve_steady_state():
#   - คำนวณ direct contribution E จาก source → patches
#   - แก้ B = (I - K)^(-1) × E
#   - Return: B array (radiation density per patch)
B = solver.solve_steady_state(source)

# =============================================================================
# 4. COMPUTE SPL AT RECEIVER
# =============================================================================
# SPL ประกอบด้วย:
#   - Direct: W / (4πR²) จาก source ตรงไป receiver
#   - Reflected: Σ (B_i × Ω_i / π) จากทุก patches
#
# แปลงเป็น SPL: Lp = 10 × log10(p² / p_ref²)
#   โดย p² = I × ρ₀ × c, p_ref = 20 μPa
Lp = solver.compute_SPL_steady(B, source, receiver)

# =============================================================================
# 5. COMPARE WITH THEORY
# =============================================================================
B_mean = np.mean(B)
B_err = abs(B_mean - B_THEORY) / B_THEORY * 100
Lp_err = abs(Lp - LP_THEORY)

print("\n" + "="*50)
print("STEADY-STATE RESULTS (Case 2: α=0.20)")
print("="*50)
print(f"  B:  {B_mean:.4e} W/m²  (theory: {B_THEORY:.4e}, err: {B_err:.1f}%)")
print(f"  Lp: {Lp:.2f} dB        (theory: {LP_THEORY:.2f}, err: {Lp_err:.2f} dB)")
print("="*50)

# =============================================================================
# 6. SAVE RESULTS
# =============================================================================
# Export mesh พร้อม B distribution สำหรับ visualize ใน ParaView
solver.get_results_mesh().save("output/case2_steady.vtk")
print("\nSaved: output/case2_steady.vtk (open in ParaView)")