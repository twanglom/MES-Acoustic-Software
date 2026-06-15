"""
Example 01: Prepare Mesh
========================
การเตรียม mesh สำหรับ RET simulation

ขั้นตอน:
1. Load mesh จากไฟล์ (.msh, .vtu, .vtk)
2. เลือก surfaces ที่ต้องการ (ถ้ามี physical IDs)
3. ตรวจสอบและแก้ไข normal direction
4. Export mesh ที่เตรียมแล้ว

Input: sphere_R2.msh (Gmsh mesh file)
Output: sphere_R2.vtu (prepared mesh)
"""

import numpy as np
import pyvista as pv
import sys
sys.path.insert(0, '..')

from ret import MeshProcessor


def main():
    # ============================================================
    # 1. LOAD MESH
    # ============================================================
    print("="*60)
    print("STEP 1: Load Mesh")
    print("="*60)
    
    # สร้าง MeshProcessor และ load mesh
    # รองรับ: .msh (Gmsh), .vtu, .vtk, .stl
    processor = MeshProcessor("geo/ROOM_3D_NONCONVEX_SURFACE.msh")
    
    # ============================================================
    # 2. SET PHYSICAL IDs (Optional - สำหรับ Gmsh meshes)
    # ============================================================
    print("\n" + "="*60)
    print("STEP 2: Set Physical IDs (if using Gmsh)")
    print("="*60)
    
    # กำหนด mapping ระหว่างชื่อ surface กับ physical ID
    physical_ids = {
        "WALL": 1,      # ผนังทรงกลม
        # "FLOOR": 2,   # พื้น (ถ้ามี)
        # "CEILING": 3, # เพดาน (ถ้ามี)
    }
    processor.set_physical_ids(physical_ids)
    
    # เลือก surfaces ที่ต้องการ (None = เลือกทั้งหมด)
    # processor.select_surfaces(["WALL"])  # เลือกเฉพาะ WALL
    processor.select_surfaces(None)  # เลือกทั้งหมด
    
    # ============================================================
    # 3. PREPARE FOR RET (Check & Fix Normals)
    # ============================================================
    print("\n" + "="*60)
    print("STEP 3: Prepare Mesh for RET")
    print("="*60)
    
    # prepare_for_ret() จะ:
    # - Extract surface และ triangulate
    # - ตรวจสอบ normal direction
    # - Flip normals ถ้าชี้ออก (auto_flip=True)
    # - Compute cell normals
    
    mesh = processor.prepare_geometry(auto_flip=True)
    
    # ============================================================
    # 4. CHECK MESH INFO
    # ============================================================
    print("\n" + "="*60)
    print("STEP 4: Mesh Information")
    print("="*60)
    
    info = processor.get_mesh_info()
    print(f"  Number of cells: {info['n_cells']}")
    print(f"  Number of points: {info['n_points']}")
    print(f"  Total surface area: {info['total_area']:.4f} m²")
    print(f"  Mean cell area: {info['mean_cell_area']:.6f} m²")
    print(f"  Volume estimate: {info['volume_estimate']:.4f} m³")
    print(f"  Normals flipped: {info['normals_flipped']}")
    
    # สำหรับทรงกลม R=2m:
    # - Surface area ควร ≈ 4πR² = 50.27 m²
    # - Volume ควร ≈ (4/3)πR³ = 33.51 m³
    
    # ============================================================
    # 5. VISUALIZE NORMALS (Optional)
    # ============================================================
    print("\n" + "="*60)
    print("STEP 5: Visualize Normals")
    print("="*60)
    
    # Uncomment เพื่อดู normals
    processor.visualize_normals(scale=0.3)
    print("  (Uncomment processor.visualize_normals() to view)")
    
    # ============================================================
    # 6. SAVE PREPARED MESH
    # ============================================================
    print("\n" + "="*60)
    print("STEP 6: Save Prepared Mesh")
    print("="*60)
    
    output_path = "geo/sphere_R2_prepared.vtu"
    mesh.save(output_path)
    print(f"  Saved to: {output_path}")
    
    print("\n" + "="*60)
    print("MESH PREPARATION COMPLETE")
    print("="*60)
    
    return mesh, processor


if __name__ == "__main__":
    mesh, processor = main()
