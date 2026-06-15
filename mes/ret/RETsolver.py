"""
Radiative Energy Transfer - Solver Module

Main solver for acoustical radiative energy transfer computation.
Based on:
- Nosal, Hodgson & Ashdown (2004) JASA 116(2), 970-980
- Kuttruff (1997) ACUSTICA 83, 622-628

Supports:
- Steady-state solution
- Time-domain (impulse response) solution
- Reverberation time calculation
- Obstruction checking for non-convex enclosures
- Boundary/surface source mode (W=0)
- Hybrid Source Mode (Boundary Source + Point Source) [UPDATED]
"""

import numpy as np
import pyvista as pv
import time
from typing import Optional, Tuple, Union
from dataclasses import dataclass

from .config import RETConfig
from .geometry import compute_volume, estimate_volume_from_area


class RadiativeEnergyTransfer:
    """
    Solver for Acoustical Radiative Energy Transfer.
    
    Computes steady-state and time-domain sound fields
    assuming diffuse (Lambertian) surface reflection.
    
    Features:
    - Steady-state energy density computation
    - Time-domain impulse response
    - Reverberation time estimation
    - SPL at arbitrary receiver positions
    - Optional obstruction checking for non-convex rooms
    - Boundary/surface source mode (W=0)
    """

    def __init__(self, 
                 mesh: pv.PolyData, 
                 view_factors: np.ndarray,
                 alpha: Union[float, np.ndarray],
                 cfg: Optional[RETConfig] = None,
                 volume: Optional[float] = None,
                 obstacles: Optional[pv.PolyData] = None,
                 skip_preprocessing: bool = False):
        """
        Initialize solver.
        
        Parameters
        ----------
        mesh : pyvista.PolyData
            Triangulated surface mesh with normals
        view_factors : ndarray (N×N)
            View factor matrix from pyviewfactor
            Convention: F[i,j] = fraction leaving j arriving at i
        alpha : float or ndarray
            Absorption coefficient(s)
        cfg : RETConfig, optional
            Solver configuration
        volume : float, optional
            Enclosure volume (auto-estimated if None)
        obstacles : pyvista.PolyData, optional
            Obstacle mesh for ray tracing
        skip_preprocessing : bool
            Skip mesh preprocessing if already done
        """
        self.cfg = cfg or RETConfig()
        
        self.mesh = mesh
        self.N = mesh.n_cells
        
        # Prepare mesh
        if not skip_preprocessing:
            mesh = mesh.extract_surface().triangulate()
            # Note: Normals should be checked by MeshProcessor.prepare_for_ret()
            # Here we just ensure they're computed
            mesh = mesh.compute_normals(cell_normals=True, point_normals=False)
        
        self.mesh = mesh
        self.N = mesh.n_cells
        
        # Geometry
        self.centers = mesh.cell_centers().points
        self.areas = mesh.compute_cell_sizes(area=True).cell_data["Area"]
        self.normals = mesh.cell_data["Normals"]
        self.total_area = np.sum(self.areas)
        
        # Obstacles for ray tracing
        if obstacles is not None:
            self.obstacle_mesh = obstacles.extract_surface().triangulate()
        else:
            self.obstacle_mesh = mesh
        
        # Absorption
        if np.isscalar(alpha):
            self.alpha = np.full(self.N, float(alpha))
        else:
            self.alpha = np.asarray(alpha, dtype=float)
        self.rho = 1.0 - self.alpha  # Reflection coefficient
        
        # View factor matrix (transpose for paper convention)
        # pyviewfactor: F[receiver, emitter] = F_{emitter→receiver}
        # Paper: F_ij = fraction leaving i arriving at j
        self.F = np.asarray(view_factors, dtype=float).T
        assert self.F.shape == (self.N, self.N), \
            f"View factor shape {self.F.shape} != ({self.N}, {self.N})"
        
        # Volume
        if volume is None:
            try:
                self.volume = compute_volume(mesh)
            except:
                self.volume = estimate_volume_from_area(self.total_area)
            print(f"  Estimated volume: {self.volume:.2f} m^3")
        else:
            self.volume = volume
            
        # Mean free path
        self.mfp = 4 * self.volume / self.total_area
        
        # Radiosity operator: K_ij = ρ_i × F_ij
        self.K = self.rho[:, None] * self.F
        
        # Time grid
        self.nt = int(np.ceil(self.cfg.tmax / self.cfg.dt)) + 1
        self.time = np.arange(self.nt) * self.cfg.dt
        
        # Distance matrix
        self.distances = np.linalg.norm(
            self.centers[:, None, :] - self.centers[None, :, :], axis=2
        )
        self.delay_steps = np.ceil(self.distances / (self.cfg.c * self.cfg.dt)).astype(int)
        
        # Results storage
        self.B_steady = None
        self.B_steady_sources = None
        self.B_time = None
        self.B_source = None  # [ADDED] For manual boundary source
        
        self._print_info()

    def set_boundary_source(self, b_source_array: np.ndarray):
        """
        [ADDED] Manually set Initial Radiosity (W/m^2) on boundary elements.
        Used for Hybrid EFEA-RET coupling where walls act as sources.
        
        Parameters
        ----------
        b_source_array : ndarray
            Array of size (N,) containing initial radiosity flux (W/m^2)
        """
        if len(b_source_array) != self.N:
            raise ValueError(f"Source array size mismatch. Expected {self.N}, got {len(b_source_array)}")
            
        self.B_source = b_source_array.copy()
        print(f"[RET] Set Boundary Source manually. Total Power: {np.sum(self.B_source * self.areas):.4e} W")
        
    def _print_info(self):
        """Print solver information."""
        mode = "Boundary Source" if self.cfg.is_boundary_source else f"Point Source (W={self.cfg.W}W)"
        print(f"\n[RET Solver] Initialized")
        print(f"  Mode: {mode}")
        print(f"  Patches: {self.N}")
        print(f"  Surface area: {self.total_area:.2f} m^2")
        print(f"  Volume: {self.volume:.2f} m^3")
        print(f"  Mean free path: {self.mfp:.3f} m")
        print(f"  Mean alpha: {np.mean(self.alpha):.3f}")
        print(f"  Obstruction check: {'Enabled' if self.cfg.check_obstruction else 'Disabled'}")
        print(f"  Time steps: {self.nt} ({self.cfg.tmax*1000:.0f} ms)")
        
    # ========================================================
    # OBSTRUCTION CHECKING
    # ========================================================
    def is_obstructed(self, p1: np.ndarray, p2: np.ndarray, tolerance: float = 0.01) -> bool:
        """Check if line-of-sight between p1 and p2 is blocked."""
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        
        points, _ = self.obstacle_mesh.ray_trace(p1, p2)
        
        if len(points) == 0:
            return False
            
        d_total = np.linalg.norm(p2 - p1)
        if d_total < 1e-10:
            return False
            
        for pt in points:
            d1 = np.linalg.norm(pt - p1)
            d2 = np.linalg.norm(pt - p2)
            if d1 > tolerance * d_total and d2 > tolerance * d_total:
                return True
                
        return False
    
    def is_visible(self, p1: np.ndarray, p2: np.ndarray, tolerance: float = 0.01) -> bool:
        """Check if p2 is visible from p1 (no obstruction)."""
        return not self.is_obstructed(p1, p2, tolerance)
        
    # ========================================================
    # SOLID ANGLE COMPUTATION
    # ========================================================
    @staticmethod
    def _solid_angle_triangle(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> float:
        """Compute solid angle of spherical triangle."""
        a = np.linalg.norm(v1)
        b = np.linalg.norm(v2)
        c = np.linalg.norm(v3)
        if a < 1e-12 or b < 1e-12 or c < 1e-12:
            return 0.0
        det = np.dot(v1, np.cross(v2, v3))
        denom = a*b*c + np.dot(v1, v2)*c + np.dot(v2, v3)*a + np.dot(v3, v1)*b
        return 2 * np.arctan2(abs(det), max(denom, 1e-30))
        
    def projected_solid_angle(self, cell_id: int, point: np.ndarray, 
                               check_obstruction: Optional[bool] = None) -> float:
        """
        Compute projected solid angle of a cell from a point.
        
        ∫ (cos θ / R²) dS
        
        Parameters
        ----------
        cell_id : int
            Cell index
        point : ndarray
            Point from which to compute solid angle
        check_obstruction : bool, optional
            Override config setting for obstruction check
        """
        if check_obstruction is None:
            check_obstruction = self.cfg.check_obstruction
            
        cell = self.mesh.extract_cells(cell_id)
        pts = cell.points
        center = np.mean(pts, axis=0)
        
        # Visibility check (normal direction)
        to_point = point - center
        if np.dot(to_point, self.normals[cell_id]) < 0:
            return 0.0
            
        # Obstruction check
        if check_obstruction:
            if self.is_obstructed(center, point):
                return 0.0
                
        # Triangular fan decomposition
        v0 = pts[0] - point
        Omega = 0.0
        for i in range(1, pts.shape[0] - 1):
            Omega += self._solid_angle_triangle(v0, pts[i] - point, pts[i+1] - point)
            
        return Omega
    
    # Alias
    def solid_angle(self, cell_id: int, point: np.ndarray) -> float:
        """Alias for projected_solid_angle()."""
        return self.projected_solid_angle(cell_id, point)
        
    # ========================================================
    # DIRECT CONTRIBUTION
    # ========================================================
    def _direct_continuous(
        self,
        source_pos: np.ndarray,
        power: Optional[float] = None,
    ) -> np.ndarray:
        """Direct contribution for continuous source. Returns 0 if W=0."""
        E = np.zeros(self.N)

        source_power = self.cfg.W if power is None else float(power)
        if source_power <= 0:
            return E
             
        for i in range(self.N):
            Omega = self.projected_solid_angle(i, source_pos)
            if Omega > 0:
                E[i] = (
                    self.rho[i] * source_power / (self.areas[i] * 4 * np.pi)
                ) * Omega
        return E
        
    def _direct_impulse(self, source_pos: np.ndarray) -> np.ndarray:
        """Direct contribution for impulsive source. Returns 0 if W=0."""
        B0 = np.zeros((self.N, self.nt))
        
        # Safety check for boundary source mode
        if self.cfg.is_boundary_source:
            return B0
            
        for i in range(self.N):
            R = np.linalg.norm(self.centers[i] - source_pos)
            q = int(np.ceil(R / (self.cfg.c * self.cfg.dt)))
            if q >= self.nt:
                continue
            Omega = self.projected_solid_angle(i, source_pos)
            if Omega > 0:
                B0[i, q] = (self.rho[i] * self.cfg.W / (self.areas[i] * 4 * np.pi)) * Omega
        return B0
        
    # ========================================================
    # STEADY-STATE SOLVER
    # ========================================================
    def solve_steady_state(self, source_pos: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Solve steady-state radiosity equation.
        Auto-handles Superposition of Boundary Source + Point Source.
        
        B = (I - K)^{-1} * E_total
        """
        print("\n[RET] Computing steady-state solution...")
        t0 = time.time()
        
        # สร้าง Vector พลังงานเริ่มต้น (E) เป็น 0 ก่อน
        E_total = np.zeros(self.N)
        
        # 1. เช็ค Boundary Source (เช่น ผนังสั่นจาก EFEA)
        if self.B_source is not None:
            print("   -> Adding Boundary Source (Structure/Wall)...")
            E_total += self.B_source  # บวกเข้าไป
            
        # 2. เช็ค Point Source (เช่น เครื่องจักรกลางห้อง)
        if source_pos is not None:
            print(f"   -> Adding Point Source at {source_pos}...")
            # ใช้ฟังก์ชันภายในคำนวณ Projection ลงผนัง
            # (ต้องแน่ใจว่า cfg.W ถูกตั้งค่าเป็น Power ของ Point Source แล้ว)
            E_point = self._direct_continuous(np.asarray(source_pos))
            E_total += E_point        # บวกเพิ่มเข้าไป (Superposition)
            
        # เช็คว่ามี Source บ้างไหม?
        if np.all(E_total == 0):
            print("   [Warning] No sources defined (or fully absorbed).")
        
        # 3. แก้สมการ (I - K)B = E_total
        # ใช้ linalg.solve เร็วกว่า inv
        self.B_steady = np.linalg.solve(np.eye(self.N) - self.K, E_total)
        
        elapsed = time.time() - t0
        print(f"   Completed in {elapsed:.2f}s")
        print(f"   Mean B = {np.mean(self.B_steady):.3e} W/m^2")
        
        return self.B_steady

    def solve_steady_state_sources(
        self,
        source_positions: np.ndarray,
        source_powers: np.ndarray,
    ) -> np.ndarray:
        """
        Solve all point sources as a single multi-right-hand-side system.

        Returns one radiosity column per source. The summed field is also
        stored in ``B_steady`` for result export and visualization.
        """
        positions = np.asarray(source_positions, dtype=float)
        powers = np.asarray(source_powers, dtype=float)

        if positions.ndim == 1:
            positions = positions.reshape(1, 3)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("source_positions must have shape (n_sources, 3)")
        if powers.ndim == 0:
            powers = powers.reshape(1)
        if len(positions) != len(powers):
            raise ValueError("source_positions and source_powers must have equal length")
        if np.any(powers < 0):
            raise ValueError("source powers must be non-negative")

        E_sources = np.column_stack([
            self._direct_continuous(position, power)
            for position, power in zip(positions, powers)
        ])

        self.B_steady_sources = np.linalg.solve(
            np.eye(self.N) - self.K,
            E_sources,
        )
        self.B_steady = np.sum(self.B_steady_sources, axis=1)
        return self.B_steady_sources


    # ========================================================
    # TIME-DOMAIN SOLVER
    # ========================================================
    def _algorithm_1(self, B0: np.ndarray) -> np.ndarray:
        """Time propagation algorithm (Algorithm 1 from Nosal et al.)."""
        print("\n[RET] Running time propagation...")
        t0 = time.time()
        
        B = B0.copy()
        active = list(zip(*np.nonzero(B)))
        ptr = 0
        
        while ptr < len(active):
            j, q = active[ptr]
            ptr += 1
            
            Bjq = B[j, q]
            if Bjq == 0:
                continue
                
            qq = q + self.delay_steps[:, j]
            valid = qq < self.nt
            contrib = Bjq * self.K[valid, j]
            
            for i, qi, val in zip(np.where(valid)[0], qq[valid], contrib):
                if abs(val) < 1e-20:
                    continue
                was_zero = (B[i, qi] == 0)
                B[i, qi] += val
                if was_zero:
                    active.append((i, qi))
                    
            if self.cfg.show_progress and ptr % self.cfg.progress_every == 0:
                frac = ptr / max(len(active), 1)
                elapsed = time.time() - t0
                eta = elapsed * (1 - frac) / max(frac, 1e-6)
                print(f"  [{100*frac:5.1f}%] {ptr}/{len(active)} | "
                      f"elapsed {elapsed:.1f}s | ETA {eta:.1f}s")
                      
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s ({len(active)} operations)")
        
        return B
        
    def _algorithm_2(self, B: np.ndarray, n_extend: Optional[int] = None) -> np.ndarray:
        """Late-time averaging extension (Algorithm 2 from Nosal et al.)."""
        print("\n[RET] Applying late-time averaging...")
        
        qavg = int(np.ceil(self.mfp / (self.cfg.c * self.cfg.dt)))
        rho_avg = np.sum(self.areas * self.rho) / self.total_area
        
        n = self.nt - 1
        total_B = np.sum(B, axis=0)
        n_prime = np.max(np.where(total_B > 0)[0]) if np.any(total_B > 0) else n
        
        if n_extend is None:
            n_extend = self.nt
        nmax = min(n + n_extend, 10 * self.nt)
        
        if nmax > B.shape[1]:
            B_ext = np.zeros((self.N, nmax))
            B_ext[:, :B.shape[1]] = B
            B = B_ext
            
        B_avg = np.zeros(nmax)
        for q in range(n + 1, min(n_prime + 1, nmax)):
            B_avg[q] = np.sum(self.areas * B[:, q]) / self.total_area
            
        M_est = np.zeros(nmax)
        for q in range(n + 1 + qavg, min(n_prime + qavg + 1, nmax)):
            q_prev = q - qavg
            if q_prev >= 0:
                M_est[q] = B_avg[q_prev] + rho_avg * M_est[q_prev]
                
        for i in range(1, qavg + 1):
            j = 1
            while n_prime + i + j * qavg < nmax:
                idx = n_prime + i + j * qavg
                idx_base = n_prime + i
                if idx_base < len(M_est):
                    M_est[idx] = (rho_avg ** j) * M_est[idx_base]
                j += 1
                
        for q in range(n + 1, nmax):
            B[:, q] += self.rho * M_est[q]
            
        print(f"  Extended to {nmax} time steps")
        
        return B
        
    def solve_time_domain(self, source_pos: np.ndarray, 
                          use_averaging: bool = True) -> np.ndarray:
        """
        Solve time-domain radiosity equation.
        
        Parameters
        ----------
        source_pos : array-like
            Source position (x, y, z)
        use_averaging : bool
            Apply late-time averaging extension
            
        Returns
        -------
        B : ndarray (N × nt)
            Radiation density vs time for each patch
        """
        print("\n" + "="*50)
        print("[RET] TIME-DOMAIN SOLUTION")
        print("="*50)
        
        source_pos = np.asarray(source_pos, dtype=float)
        
        B0 = self._direct_impulse(source_pos)
        print(f"  Direct contribution: {np.sum(B0 > 0)} non-zero entries")
        
        B = self._algorithm_1(B0)
        
        if use_averaging:
            B = self._algorithm_2(B)
            
        self.B_time = B
        return B
        
    # ========================================================
    # RECEIVER RESPONSE
    # ========================================================
    def receiver_intensity_steady(self, B: np.ndarray, source_pos: np.ndarray,
                                   receiver_pos: np.ndarray) -> float:
        """
        Compute intensity at receiver position (steady-state).
        
        Handles W=0 (boundary source mode) safely.
        """
        source_pos = np.asarray(source_pos, dtype=float)
        receiver_pos = np.asarray(receiver_pos, dtype=float)
        
        # Reflected contribution
        I_ref = 0.0
        for i in range(self.N):
            Omega = self.projected_solid_angle(i, receiver_pos)
            if Omega > 0:
                I_ref += (1.0 / np.pi) * B[i] * Omega
        
        # Direct contribution (only if W > 0)
        # [UPDATED FIX]: Allow Direct Sound calculation even if Boundary Source is present
        # if and only if there is a defined Point Source Power (W > 0).
        # Old blocking logic: if not self.cfg.is_boundary_source and self.B_source is None:
        
        I_dir = 0.0
        if self.cfg.W > 0:
            Rsr = np.linalg.norm(source_pos - receiver_pos)
            if Rsr > 1e-6:  # Avoid division by zero
                visible = True
                if self.cfg.check_obstruction:
                    visible = self.is_visible(source_pos, receiver_pos)
                if visible:
                    I_dir = self.cfg.W / (4 * np.pi * Rsr**2)
        
        return I_ref + I_dir
    
    def receiver_intensity_time(self, B: np.ndarray, source_pos: np.ndarray,
                                 receiver_pos: np.ndarray) -> np.ndarray:
        """
        Compute intensity at receiver position (time-domain).
        
        Handles W=0 (boundary source mode) safely.
        """
        source_pos = np.asarray(source_pos, dtype=float)
        receiver_pos = np.asarray(receiver_pos, dtype=float)
        nt = B.shape[1]
        I = np.zeros(nt)
        
        # Reflected contribution
        for i in range(self.N):
            Omega = self.projected_solid_angle(i, receiver_pos)
            if Omega <= 0:
                continue
            R = np.linalg.norm(self.centers[i] - receiver_pos)
            q = int(np.ceil(R / (self.cfg.c * self.cfg.dt)))
            if q < nt:
                I[q:] += (1/np.pi) * Omega * B[i, :nt-q]
        
        # Direct contribution (only if W > 0)
        # [UPDATED FIX]: Allow Direct Sound for hybrid cases (W > 0)
        if self.cfg.W > 0:
            Rsr = np.linalg.norm(source_pos - receiver_pos)
            if Rsr > 1e-6:
                qsr = int(np.ceil(Rsr / (self.cfg.c * self.cfg.dt)))
                visible = True
                if self.cfg.check_obstruction:
                    visible = self.is_visible(source_pos, receiver_pos)
                if qsr < nt and visible:
                    I[qsr] += self.cfg.W / (4 * np.pi * Rsr**2)
        
        # Air absorption
        if self.cfg.m_air > 0:
            t = np.arange(nt) * self.cfg.dt
            I *= np.exp(-self.cfg.m_air * self.cfg.c * t)
        
        return I
    
    def intensity_at_receiver(self, B: np.ndarray, source_pos: np.ndarray,
                              receiver_pos: np.ndarray) -> Union[float, np.ndarray]:
        """
        Compute intensity at receiver position.
        
        Auto-detects steady-state vs time-domain from B shape.
        """
        if B.ndim == 1:
            return self.receiver_intensity_steady(B, source_pos, receiver_pos)
        else:
            return self.receiver_intensity_time(B, source_pos, receiver_pos)
        
    def compute_SPL_steady(self, B: np.ndarray, source_pos: np.ndarray,
                           receiver_pos: np.ndarray) -> float:
        """
        Compute SPL at receiver position (steady-state).
        
        Returns
        -------
        Lp : float
            Sound pressure level (dB re 20 μPa)
        """
        I = self.receiver_intensity_steady(B, source_pos, receiver_pos)
        p2 = I * self.cfg.rho0 * self.cfg.c
        pref = 2e-5
        
        if p2 <= 0:
            return -100.0
        return 10 * np.log10(p2 / pref**2)

    def compute_SPL_steady_sources(
        self,
        B_sources: np.ndarray,
        source_positions: np.ndarray,
        source_powers: np.ndarray,
        receiver_pos: np.ndarray,
    ) -> float:
        """Compute steady-state SPL from multiple incoherent point sources."""
        positions = np.asarray(source_positions, dtype=float)
        powers = np.asarray(source_powers, dtype=float)
        receiver = np.asarray(receiver_pos, dtype=float)
        source_fields = np.asarray(B_sources, dtype=float)

        if positions.ndim == 1:
            positions = positions.reshape(1, 3)
        if source_fields.ndim == 1:
            source_fields = source_fields.reshape(self.N, 1)
        if source_fields.shape != (self.N, len(positions)):
            raise ValueError(
                "B_sources must have shape (n_cells, n_sources)"
            )
        if len(powers) != len(positions):
            raise ValueError("source_positions and source_powers must have equal length")

        B_total = np.sum(source_fields, axis=1)
        I_ref = 0.0
        for i in range(self.N):
            Omega = self.projected_solid_angle(i, receiver)
            if Omega > 0:
                I_ref += (1.0 / np.pi) * B_total[i] * Omega

        I_dir = 0.0
        for source, power in zip(positions, powers):
            if power <= 0:
                continue
            distance = np.linalg.norm(source - receiver)
            if distance <= 1e-6:
                continue
            visible = (
                not self.cfg.check_obstruction
                or self.is_visible(source, receiver)
            )
            if visible:
                I_dir += power / (4 * np.pi * distance**2)

        p2 = (I_ref + I_dir) * self.cfg.rho0 * self.cfg.c
        if p2 <= 0:
            return -100.0
        return float(10 * np.log10(p2 / (2e-5)**2))
    
    def compute_SPL_time(self, B: np.ndarray, source_pos: np.ndarray,
                         receiver_pos: np.ndarray) -> np.ndarray:
        """
        Compute SPL at receiver position (time-domain).
        
        Returns
        -------
        Lp : ndarray
            Sound pressure level vs time (dB re 20 μPa)
        """
        I = self.receiver_intensity_time(B, source_pos, receiver_pos)
        p2 = I * self.cfg.rho0 * self.cfg.c
        pref = 2e-5
        
        with np.errstate(divide='ignore'):
            Lp = 10 * np.log10(p2 / pref**2)
        Lp[~np.isfinite(Lp)] = -100
        
        return Lp
    
    def spl_at_receiver(self, B: np.ndarray, source_pos: np.ndarray,
                        receiver_pos: np.ndarray) -> Union[float, np.ndarray]:
        """
        Compute SPL at receiver position.
        
        Auto-detects steady-state vs time-domain from B shape.
        """
        if B.ndim == 1:
            return self.compute_SPL_steady(B, source_pos, receiver_pos)
        else:
            return self.compute_SPL_time(B, source_pos, receiver_pos)
        
    # ========================================================
    # REVERBERATION TIME
    # ========================================================
    def compute_rt60(self, B_time: np.ndarray, source_pos: np.ndarray,
                     receiver_pos: np.ndarray) -> Optional[Tuple[float, np.ndarray, np.ndarray, float]]:
        """
        Compute RT60 from decay curve.
        
        Handles both point source (W>0) and boundary source (W=0) modes.
        
        Returns
        -------
        result : tuple or None
            (RT60, time, Lp_norm, slope) or None if computation fails
        """
        Lp = self.compute_SPL_time(B_time, source_pos, receiver_pos)
        t = np.arange(len(Lp)) * self.cfg.dt
        
        # Auto-detect start
        if not self.cfg.is_boundary_source:
            # Point source: start after direct sound arrival
            direct_time = np.linalg.norm(np.asarray(source_pos) - np.asarray(receiver_pos)) / self.cfg.c
            start_idx = int(direct_time / self.cfg.dt) + 10
        else:
            # Boundary source: find first time > -50 dB
            valid_indices = np.where(Lp > -50)[0]
            if len(valid_indices) > 0:
                start_idx = valid_indices[0] + 10
            else:
                print("  [Warning] No valid data for RT calculation")
                return None
        
        if start_idx >= len(Lp):
            print("  [Warning] Start index out of range")
            return None
        
        valid_range = Lp[start_idx:]
        if len(valid_range) == 0 or np.max(valid_range) < -90:
            print("  [Warning] No valid data for RT calculation")
            return None
            
        peak_idx = np.argmax(valid_range) + start_idx
        Lp_norm = Lp - Lp[peak_idx]
        
        # Linear fit in -5 to -35 dB range
        valid = (Lp_norm > -35) & (Lp_norm < -5) & (np.arange(len(Lp)) > peak_idx)
        
        if np.sum(valid) < 10:
            print("  [Warning] Not enough data points for RT fit")
            return None
            
        coeffs = np.polyfit(t[valid], Lp_norm[valid], 1)
        slope = coeffs[0]
        
        RT60 = -60 / slope
        
        print(f"\n[RT60] Computed: {RT60:.3f} s (slope: {slope:.1f} dB/s)")
        
        return RT60, t, Lp_norm, slope
    
    # Alias
    def compute_RT(self, B: np.ndarray, source_pos: np.ndarray,
                   receiver_pos: np.ndarray) -> Optional[Tuple[float, np.ndarray, np.ndarray, float]]:
        """Alias for compute_rt60()."""
        return self.compute_rt60(B, source_pos, receiver_pos)
        
    def theoretical_rt60(self) -> float:
        """Compute Eyring reverberation time."""
        alpha_avg = np.sum(self.alpha * self.areas) / self.total_area
        if alpha_avg >= 1:
            return 0.0
        return 0.161 * self.volume / (-self.total_area * np.log(1 - alpha_avg))
    
    # Alias
    def theoretical_RT(self) -> float:
        """Alias for theoretical_rt60()."""
        return self.theoretical_rt60()
        
    # ========================================================
    # RESULTS
    # ========================================================
    def get_results_mesh(self, include_steady: bool = True, 
                         include_time_avg: bool = False) -> pv.PolyData:
        """
        Get mesh with results as cell data.
        
        Parameters
        ----------
        include_steady : bool
            Include steady-state B
        include_time_avg : bool
            Include time-averaged B from time-domain
            
        Returns
        -------
        mesh : pyvista.PolyData
            Mesh with results
        """
        mesh = self.mesh.copy()
        mesh.cell_data['alpha'] = self.alpha
        mesh.cell_data['rho'] = self.rho
        mesh.cell_data['area'] = self.areas
        
        if include_steady and self.B_steady is not None:
            mesh.cell_data['B_steady'] = self.B_steady
            mesh.cell_data['B_steady_dB'] = 10 * np.log10(self.B_steady / 1e-12 + 1e-30)
            
        if include_time_avg and self.B_time is not None:
            mesh.cell_data['B_time_avg'] = np.mean(self.B_time, axis=1)
            mesh.cell_data['B_time_max'] = np.max(self.B_time, axis=1)
            
        # [ADDED] Export Manual Source for debugging
        if self.B_source is not None:
            mesh.cell_data['B_Source_Input'] = self.B_source
            
        return mesh
