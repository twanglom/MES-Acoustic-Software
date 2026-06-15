"""
Example 04: Run Time-Domain Solver (RT60)
=========================================
การคำนวณ time-domain response และ Reverberation Time (RT60)

Time-domain หมายถึง:
- แหล่งกำเนิดเสียงเป็น impulse (เปิดชั่วขณะแล้วปิด)
- ดูการสลายตัวของเสียงตามเวลา
- ใช้คำนวณ RT60 (เวลาที่เสียงลดลง 60 dB)
- คำนวณช้ากว่า steady-state

Input: 
- sphere_R2_prepared.vtu (prepared mesh)
- sphere_R2_vf.npy (view factor matrix)

Output:
- RT60 (reverberation time)
- Decay curve plot
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')

from RET import (
    RadiativeEnergyTransfer,
    RETConfig,
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
    
    # Absorption coefficient (Case 3 from Nosal Table I)
    alpha = 0.50
    
    # Source and receiver positions
    source_pos = np.array([0.0, 0.0, 0.0])  # Center
    receiver_pos = np.array([np.sqrt(2), 0.0, 0.0])  # r = √2 m
    
    print(f"  Absorption coefficient: {alpha}")
    print(f"  Source position: {source_pos}")
    print(f"  Receiver position: {receiver_pos}")
    
    # ============================================================
    # 3. CREATE SOLVER (TIME-DOMAIN CONFIG)
    # ============================================================
    print("\n" + "="*60)
    print("STEP 3: Create Solver (Time-Domain Config)")
    print("="*60)
    
    # Configuration สำหรับ time-domain
    cfg = RETConfig(
        W=0.005,              # Source power (W)
        c=343.0,              # Speed of sound (m/s)
        rho0=1.21,            # Air density (kg/m³)
        dt=1/24000,           # Time step (s) - ควรเล็กพอสำหรับความแม่นยำ
        tmax=1.0,             # Max simulation time (s) - ควรยาวพอสำหรับ decay
        check_obstruction=False,
        show_progress=True,
        progress_every=5000,
    )
    
    print(f"  Time step: {cfg.dt*1000:.4f} ms")
    print(f"  Max time: {cfg.tmax*1000:.0f} ms")
    print(f"  Number of time steps: {int(cfg.tmax/cfg.dt)}")
    
    # Create solver
    solver = RadiativeEnergyTransfer(
        mesh=mesh,
        view_factors=F,
        alpha=alpha,
        cfg=cfg,
        volume=V,
        skip_preprocessing=True,
    )
    
    # ============================================================
    # 4. SOLVE TIME-DOMAIN
    # ============================================================
    print("\n" + "="*60)
    print("STEP 4: Solve Time-Domain")
    print("="*60)
    
    # use_averaging=True จะใช้ Algorithm 2 เพื่อ extend late-time decay
    B_time = solver.solve_time_domain(source_pos, use_averaging=True)
    
    print(f"\n  B_time shape: {B_time.shape}")
    print(f"  (N_patches × N_timesteps)")
    
    # ============================================================
    # 5. COMPUTE RT60
    # ============================================================
    print("\n" + "="*60)
    print("STEP 5: Compute RT60")
    print("="*60)
    
    result = solver.compute_rt60(B_time, source_pos, receiver_pos)
    
    if result is not None:
        RT60, t, Lp_norm, slope = result
        
        # Theoretical RT60 (Eyring formula)
        RT60_theory = solver.theoretical_rt60()
        
        # From Nosal Table I (Case 3: α = 0.50)
        RT60_nosal = 0.242  # s
        
        print(f"\n  RT60 (computed):  {RT60:.3f} s ({RT60*1000:.0f} ms)")
        print(f"  RT60 (Eyring):    {RT60_theory:.3f} s ({RT60_theory*1000:.0f} ms)")
        print(f"  RT60 (Nosal):     {RT60_nosal:.3f} s ({RT60_nosal*1000:.0f} ms)")
        print(f"  Error vs Nosal:   {abs(RT60 - RT60_nosal)*1000:.1f} ms")
    else:
        print("  [ERROR] Could not compute RT60")
        RT60 = None
    
    # ============================================================
    # 6. PLOT DECAY CURVE
    # ============================================================
    print("\n" + "="*60)
    print("STEP 6: Plot Decay Curve")
    print("="*60)
    
    if result is not None:
        # Use Visualizer
        Visualizer.plot_decay_curve(
            time=t,
            spl=Lp_norm,
            rt60=RT60,
            rt_theory=RT60_theory,
            slope=slope,
            save_path="output/decay_curve.png"
        )
    
    # ============================================================
    # 7. COMPUTE SPL TIME HISTORY
    # ============================================================
    print("\n" + "="*60)
    print("STEP 7: SPL Time History")
    print("="*60)
    
    Lp_time = solver.compute_SPL_time(B_time, source_pos, receiver_pos)
    
    print(f"  Peak SPL: {np.max(Lp_time):.1f} dB")
    print(f"  Time at peak: {np.argmax(Lp_time) * cfg.dt * 1000:.1f} ms")
    
    # ============================================================
    # 8. EXPORT RESULTS
    # ============================================================
    print("\n" + "="*60)
    print("STEP 8: Export Results")
    print("="*60)
    
    exporter = ResultExporter("output")
    
    # Save decay curve data
    if result is not None:
        exporter.save_decay_curve(t, Lp_norm, "decay_curve.txt")
    
    # Save summary
    summary = {
        "Case": "Sphere R=2m, α=0.50 (Time-Domain)",
        "Source Power": f"{cfg.W} W",
        "Volume": f"{V:.2f} m³",
        "Time step": f"{cfg.dt*1000:.4f} ms",
        "Max time": f"{cfg.tmax*1000:.0f} ms",
        "RT60 (computed)": f"{RT60:.3f} s" if RT60 else "N/A",
        "RT60 (Eyring)": f"{solver.theoretical_rt60():.3f} s",
        "RT60 (Nosal)": "0.242 s",
    }
    exporter.save_summary(summary, "time_domain_summary.txt")
    
    print("\n" + "="*60)
    print("TIME-DOMAIN SIMULATION COMPLETE")
    print("="*60)
    
    return solver, B_time, RT60


if __name__ == "__main__":
    solver, B_time, RT60 = main()
