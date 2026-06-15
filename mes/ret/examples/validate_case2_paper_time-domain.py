"""
Validation: Nosal Table I - Case 2 (Time-Domain + RT60)
=======================================================
Sphere R=2m, α=0.20

Time-Domain หมายถึง:
- Source เป็น impulse (เปิดชั่วขณะแล้วปิด)
- ดูการสลายตัวของเสียงตามเวลา
- ใช้คำนวณ RT60 (เวลาที่เสียงลดลง 60 dB)

Algorithm:
1. Algorithm 1: Time propagation - คำนวณ B(t) ทีละ time step
2. Algorithm 2: Late-time averaging - extend decay curve
"""
import numpy as np
import sys
sys.path.insert(0, '..')

from RET import MeshProcessor, RadiativeEnergyTransfer, RETConfig, Visualizer

# =============================================================================
# REFERENCE VALUES (Nosal Table I, Case 2)
# =============================================================================
RT_THEORY = 0.483    # s - Reverberation time (60 dB decay)

# =============================================================================
# PARAMETERS
# =============================================================================
R = 2.0                          # Sphere radius (m)
V = (4/3) * np.pi * R**3         # Volume = 33.51 m³
alpha = 0.20                     # Absorption coefficient
source = np.array([0, 0, 0])     # Source at center
receiver = np.array([np.sqrt(2), 0, 0])  # Receiver at r = √2 m

# =============================================================================
# 1. LOAD MESH
# =============================================================================
print("Loading mesh...")
processor = MeshProcessor("geo/sphereR2.vtu")
mesh = processor.prepare_geometry()

# =============================================================================
# 2. LOAD VIEW FACTORS
# =============================================================================
print("Loading view factors...")
F = np.load("geo/sphereR2_vf.npy")

# =============================================================================
# 3. CREATE SOLVER (TIME-DOMAIN CONFIG)
# =============================================================================
# Time-domain parameters:
#   - dt: time step (s) - ควรเล็กพอ เช่น 1/24000 = 0.042 ms
#   - tmax: simulation time (s) - ควรยาวพอให้ decay 60 dB
#
# Rule of thumb: tmax ≈ 2-3 × RT60_expected
print("Creating solver...")
cfg = RETConfig(
    W=0.005,                # Source power (W)
    check_obstruction=False,
    dt=1/24000,             # Time step ≈ 0.042 ms
    tmax=1.0,               # Simulate 1 second
)
solver = RadiativeEnergyTransfer(
    mesh, F, 
    alpha=alpha, 
    cfg=cfg, 
    volume=V, 
    skip_preprocessing=True
)

# =============================================================================
# 4. SOLVE TIME-DOMAIN
# =============================================================================
# solve_time_domain():
#   - Algorithm 1: Propagate energy through patches over time
#   - Algorithm 2: Extend late-time decay using averaging (optional)
#
# Return: B(N, nt) - radiation density per patch per time step
print("Solving time-domain (this may take a while)...")
B_time = solver.solve_time_domain(source, use_averaging=True)

# =============================================================================
# 5. COMPUTE RT60
# =============================================================================
# compute_rt60():
#   - คำนวณ SPL(t) ที่ receiver
#   - หา peak และ normalize
#   - Linear fit ในช่วง -5 to -35 dB
#   - Extrapolate หา RT60 (เวลาที่ลดลง 60 dB)
#
# Return: (RT60, time_array, Lp_normalized, slope)
result = solver.compute_rt60(B_time, source, receiver)

if result is not None:
    RT60, t, Lp_norm, slope = result
    
    # Eyring RT60: RT = 0.161V / (-S × ln(1-α))
    RT_eyring = solver.theoretical_rt60()
    RT_err = abs(RT60 - RT_THEORY) / RT_THEORY * 100
    
    print("\n" + "="*50)
    print("TIME-DOMAIN RESULTS (Case 2: α=0.20)")
    print("="*50)
    print(f"  RT60 computed: {RT60:.3f} s  ({RT60*1000:.0f} ms)")
    print(f"  RT60 theory:   {RT_THEORY:.3f} s  ({RT_THEORY*1000:.0f} ms)")
    print(f"  RT60 Eyring:   {RT_eyring:.3f} s")
    print(f"  Error:         {RT_err:.1f}%")
    print("="*50)
    
    # ==========================================================================
    # 6. PLOT DECAY CURVE
    # ==========================================================================
    # Decay curve แสดง:
    #   - SPL(t) normalized to peak
    #   - Linear fit line
    #   - RT60 marker
    Visualizer.plot_decay_curve(
        t, Lp_norm, 
        rt60=RT60, 
        rt_theory=RT_eyring, 
        slope=slope,
        save_path="output/case2_decay.png"
    )
    print("\nSaved: output/case2_decay.png")
else:
    print("\n[ERROR] Could not compute RT60 - check simulation parameters")