"""
Compute View Factors for Sphere R=2m
=====================================
View Factor F_ij = สัดส่วนพลังงานที่ออกจาก patch i แล้วไปถึง patch j

สำหรับ closed enclosure: ΣF_ij = 1 (row sum ≈ 1)
"""
import numpy as np
import pyvista as pv
import os
import sys
sys.path.insert(0, '..')

from ret import MeshProcessor, ViewFactorCalculator

# =============================================================================
# CONFIG
# =============================================================================
MESH_PATH = "geo/sphereR2.vtu"      # Input mesh
VF_PATH = "geo/sphereR2_vf.npy"     # Output/Input view factors
FORCE_RECOMPUTE = False             # True = คำนวณใหม่แม้มีไฟล์อยู่แล้ว

# =============================================================================
# 1. LOAD & PREPARE MESH
# =============================================================================
# MeshProcessor ทำหน้าที่:
#   - Load mesh จากไฟล์ (.vtu, .msh, .vtk, .stl)
#   - ตรวจสอบทิศทาง normal (ต้องชี้เข้าด้านใน)
#   - Flip normals อัตโนมัติถ้าชี้ผิดทาง
#   - Triangulate surface
print("="*60)
print("STEP 1: Load & Prepare Mesh")
print("="*60)
processor = MeshProcessor(MESH_PATH)
mesh = processor.prepare_geometry()  # auto_flip=True by default

info = processor.get_mesh_info()
print(f"  Cells: {info['n_cells']}")
print(f"  Surface Area: {info['total_area']:.2f} m² (sphere R=2: 4πR² = 50.27 m²)")
print(f"  Normals flipped: {info['normals_flipped']}")

# =============================================================================
# 2. LOAD OR COMPUTE VIEW FACTORS
# =============================================================================
vf_calc = ViewFactorCalculator(mesh)

if os.path.exists(VF_PATH) and not FORCE_RECOMPUTE:
    # ----- LOAD EXISTING -----
    print("\n" + "="*60)
    print("STEP 2: Load Existing View Factors")
    print("="*60)
    F = vf_calc.load(VF_PATH)
    print(f"  Loaded: {VF_PATH}")
    print(f"  Shape: {F.shape}")
    
else:
    # ----- COMPUTE NEW -----
    print("\n" + "="*60)
    print("STEP 2: Compute View Factors")
    print("="*60)
    # ViewFactorCalculator ใช้ pyviewfactor library
    # 
    # Parameters:
    #   - skip_obstruction=True  : ไม่ตรวจ obstruction (เร็ว, ใช้กับ convex)
    #   - skip_obstruction=False : ตรวจ obstruction (ช้า, ใช้กับ non-convex)
    #   - verbose=True           : แสดง progress
    #
    # Convention ของ pyviewfactor:
    #   F[i,j] = fraction leaving j arriving at i (receiver, emitter)
    #   Solver จะ transpose ให้เป็น paper convention
    F = vf_calc.compute(
        skip_visibility=False,
        skip_obstruction=True,  # True สำหรับ convex enclosure (sphere, box)
        verbose=True
    )
    
    print(f"\n  Matrix shape: {F.shape}")
    print(f"  Memory: {F.nbytes / 1e6:.2f} MB")
    
    # ----- SAVE -----
    vf_calc.save(VF_PATH)
    print(f"\n  Saved: {VF_PATH}")

# =============================================================================
# 3. VALIDATE VIEW FACTORS
# =============================================================================
# สำหรับ closed enclosure ที่ดี:
#   - Row sum mean ≈ 1.0 (พลังงานไม่หาย)
#   - Diagonal ≈ 0 (patch ไม่เห็นตัวเอง)
#   - Reciprocity: A_i × F_ij = A_j × F_ji
print("\n" + "="*60)
print("STEP 3: Validate View Factors")
print("="*60)
results = vf_calc.validate()

if results['row_sum_mean'] > 0.95:
    print("\n  [OK] View factors valid for closed enclosure")
elif results['row_sum_mean'] > 0.85:
    print("\n  [WARNING] Row sums slightly low - acceptable")
else:
    print("\n  [ERROR] Row sums too low - mesh may not be closed!")

print("\n" + "="*60)
print("COMPLETE")
print("="*60)
print(f"View factors ready: {VF_PATH}")
print("\nNext: python validate_steady_state.py")





# =============================================================================
#  EXTERNAL MESH OBS CASE
# =============================================================================

# MESH_PATH = "geo/A320_MODEL_CONFIG_C_CABIN.msh"
# SEAT_PATH = "geo/A320_MODEL_CONFIG_C_SEAT.msh"   # <--- เพิ่ม
# VF_PATH   = "geo/A320_vf_with_seat.npy"

# processor = MeshProcessor(MESH_PATH)
# mesh = processor.prepare_geometry()

# # --- LOAD SEAT (obstacle) ---
# seat_proc = MeshProcessor(SEAT_PATH)
# seat_mesh = seat_proc.prepare_geometry()  # หรืออ่านแล้ว triangulate ให้เป็น PolyData

# vf_calc = ViewFactorCalculator(mesh)

# F = vf_calc.compute(
#     obstacles=seat_mesh,        # <--- สำคัญ
#     skip_visibility=False,
#     skip_obstruction=False,     # <--- ต้องเป็น False เพราะมี obstacle/non-convex
#     verbose=True
# )

# vf_calc.save(VF_PATH)
