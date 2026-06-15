

"""
Radiative Energy Transfer - Geometry Module

Mesh processing, view factor calculation, and geometry utilities.
Includes automatic normal direction detection and correction.
"""

import numpy as np
import pyvista as pv
from typing import Optional, Dict, List, Union, Tuple
from pathlib import Path


class MeshProcessor:
    """
    Process and prepare mesh for RET computation.
    
    Includes automatic detection and correction of normal directions
    to ensure they point inward for enclosed spaces.
    """
    
    def __init__(self, mesh_path: Optional[str] = None, mesh: Optional[pv.DataSet] = None):
        """
        Initialize mesh processor.
        
        Parameters
        ----------
        mesh_path : str, optional
            Path to mesh file (.vtu, .vtk, .msh, .stl)
        mesh : pyvista.DataSet, optional
            Existing PyVista mesh
        """
        self.raw_mesh = None
        self.processed_mesh = None
        self.physical_ids = {}
        self._normals_flipped = False

        if mesh_path is not None:
            self.load_mesh(mesh_path)
        elif mesh is not None:
            self.raw_mesh = mesh
        
    def load_mesh(self, mesh_path: str):
        """Load mesh from file."""
        path = Path(mesh_path)
        
        if path.suffix.lower() == '.msh':
            import meshio

            msh = meshio.read(mesh_path)
            self.raw_mesh = pv.from_meshio(msh)
            self.physical_ids = {
                name: int(data[0])
                for name, data in msh.field_data.items()
                if len(data) >= 2 and int(data[1]) == 2
            }
        else:
            self.raw_mesh = pv.read(mesh_path)
            
        print(f"[Mesh] Loaded: {self.raw_mesh.n_cells} cells, {self.raw_mesh.n_points} points")
        return self.raw_mesh
    
    def set_physical_ids(self, physical_ids: Dict[str, int]):
        """
        Set physical ID mapping for Gmsh meshes.
        
        Parameters
        ----------
        physical_ids : dict
            Mapping of surface names to physical IDs
            e.g., {"WALL": 1, "FLOOR": 2, "CEILING": 3}
        """
        self.physical_ids = physical_ids
        
    def select_surfaces(self, surface_names: Optional[List[str]] = None) -> pv.PolyData:
        """
        Select surfaces by physical ID names.
        
        Parameters
        ----------
        surface_names : list of str, optional
            List of surface names to select. If None, select all.
            
        Returns
        -------
        mesh : pyvista.PolyData
            Selected surface mesh
        """
        surf = self.raw_mesh.extract_surface().triangulate()
        
        if surface_names is None or not self.physical_ids:
            self.processed_mesh = surf
            return surf
            
        if "gmsh:physical" not in surf.cell_data:
            print("[Warning] No physical IDs found, returning all surfaces")
            self.processed_mesh = surf
            return surf
            
        phys = surf.cell_data["gmsh:physical"]
        ids = [self.physical_ids[name] for name in surface_names if name in self.physical_ids]
        mask = np.isin(phys, ids)
        
        self.processed_mesh = surf.extract_cells(np.where(mask)[0])
        print(f"[Mesh] Selected {self.processed_mesh.n_cells} cells from {surface_names}")
        
        return self.processed_mesh
    
    def check_normals_direction(self, mesh: Optional[pv.PolyData] = None) -> Tuple[bool, float]:
        """
        Check if normals point inward (toward centroid) or outward.
        
        For enclosed spaces, normals should point INWARD for correct
        view factor calculation.
        
        Parameters
        ----------
        mesh : pyvista.PolyData, optional
            Mesh to check. If None, uses processed_mesh.
            
        Returns
        -------
        needs_flip : bool
            True if normals should be flipped (currently pointing outward)
        inward_ratio : float
            Ratio of cells with inward-pointing normals (0-1)
        """
        if mesh is None:
            mesh = self.processed_mesh if self.processed_mesh is not None else self.raw_mesh
            
        if mesh is None:
            raise ValueError("No mesh available")
            
        # Ensure normals are computed
        mesh = mesh.compute_normals(cell_normals=True, point_normals=False, inplace=False)
        
        # Get centroid of the enclosure
        centroid = np.mean(mesh.points, axis=0)
        
        # Get cell centers and normals
        centers = mesh.cell_centers().points
        normals = mesh.cell_data["Normals"]
        
        # Vector from cell center to centroid
        to_centroid = centroid - centers
        to_centroid_norm = to_centroid / (np.linalg.norm(to_centroid, axis=1, keepdims=True) + 1e-10)
        
        # Dot product: positive means normal points toward centroid (inward)
        dots = np.sum(normals * to_centroid_norm, axis=1)
        
        n_inward = np.sum(dots > 0)
        n_total = len(dots)
        inward_ratio = n_inward / n_total
        
        # If more than 50% point outward, we should flip
        needs_flip = inward_ratio < 0.5
        
        direction = "outward" if needs_flip else "inward"
        print(f"[Mesh] Normal direction check: {inward_ratio*100:.1f}% inward -> normals point {direction}")
        
        return needs_flip, inward_ratio
    
    def prepare_geometry(self, auto_flip: bool = True) -> pv.PolyData:
        """
        Prepare mesh for RET computation:
        - Extract surface
        - Triangulate
        - Check and optionally flip normals to point inward
        - Compute cell normals
        
        Parameters
        ----------
        auto_flip : bool
            Automatically flip normals if they point outward
            
        Returns
        -------
        mesh : pyvista.PolyData
            Prepared mesh with inward-pointing normals
        """
        if self.processed_mesh is None:
            self.processed_mesh = self.raw_mesh.extract_surface().triangulate()
             
        mesh = self.processed_mesh.extract_surface().triangulate()
        mesh = mesh.compute_normals(cell_normals=True, point_normals=False, inplace=False)
        
        # Check normal direction
        needs_flip, inward_ratio = self.check_normals_direction(mesh)
        
        if needs_flip and auto_flip:
            print("[Mesh] Flipping normals to point inward...")
            if "Normals" in mesh.cell_data:
                del mesh.cell_data["Normals"]
            if "Normals" in mesh.point_data:
                del mesh.point_data["Normals"]
            mesh = mesh.compute_normals(
                cell_normals=True,
                point_normals=False,
                flip_normals=True,
                inplace=False,
            )
            self._normals_flipped = True
        elif needs_flip and not auto_flip:
            print("[Warning] Normals point outward but auto_flip=False")
            
        self.processed_mesh = mesh
        print(f"[Mesh] Prepared: {mesh.n_cells} triangular cells")
        
        return mesh
    
    
    def visualize_normals(self, scale: float = 0.2):
        """Visualize mesh with normal vectors."""
        if self.processed_mesh is None:
            self.prepare_geometry()
            
        mesh = self.processed_mesh
        centers = mesh.cell_centers().points
        
        arrows = pv.PolyData(centers)
        arrows["Normals"] = mesh.cell_data["Normals"]
        glyphs = arrows.glyph(orient="Normals", scale=False, factor=scale)
        
        pl = pv.Plotter()
        pl.add_mesh(mesh, opacity=0.5, color="lightgray", show_edges=True)
        pl.add_mesh(glyphs, color="red")
        pl.add_axes()
        pl.show()
        
    def get_mesh_info(self) -> dict:
        """Get mesh statistics."""
        if self.processed_mesh is None:
            return {}
            
        mesh = self.processed_mesh
        areas = mesh.compute_cell_sizes(area=True).cell_data["Area"]
        
        bounds = mesh.bounds
        volume_est = (bounds[1]-bounds[0]) * (bounds[3]-bounds[2]) * (bounds[5]-bounds[4])
        
        return {
            "n_cells": mesh.n_cells,
            "n_points": mesh.n_points,
            "total_area": np.sum(areas),
            "mean_cell_area": np.mean(areas),
            "min_cell_area": np.min(areas),
            "max_cell_area": np.max(areas),
            "bounds": bounds,
            "volume_estimate": volume_est,
            "normals_flipped": self._normals_flipped,
        }


class ViewFactorCalculator:
    """
    Calculate view factors for RET computation.
    """
    
    def __init__(self, mesh: pv.PolyData):
        """
        Initialize view factor calculator.
        
        Parameters
        ----------
        mesh : pyvista.PolyData
            Prepared triangular surface mesh with inward-pointing normals
        """
        self.mesh = mesh
        self.F = None
        
    def compute(self, 
                epsilon: float = 1e-3,
                skip_visibility: bool = False,
                skip_obstruction: bool = False,
                verbose: bool = True,
                obstacles: Optional[pv.PolyData] = None) -> np.ndarray:
        """
        Compute view factor matrix using pyviewfactor.
        
        Parameters
        ----------
        epsilon : float
            Numerical tolerance
        skip_visibility : bool
            Skip visibility check (faster but less accurate)
        skip_obstruction : bool
            Skip obstruction check (faster for convex enclosures)
        verbose : bool
            Print progress
        obstacles : pyvista.PolyData, optional
            Obstacle mesh used to occlude patch-to-patch visibility (e.g., seats).
            
        Returns
        -------
        F : ndarray (N×N)
            View factor matrix
        """
        try:
            import pyviewfactor as pvf
        except ImportError:
            raise ImportError("pyviewfactor is required. Install with: pip install pyviewfactor")
        
        print(f"[ViewFactor] Computing for {self.mesh.n_cells} cells...")
        
        if not skip_obstruction and obstacles is None:
            obstacles = self.mesh
            print("[ViewFactor] Using mesh itself for obstruction checks")

        F = pvf.compute_viewfactor_matrix(
            self.mesh,
            obstacles=obstacles,
            skip_visibility=skip_visibility,
            skip_obstruction=skip_obstruction,
            strict_visibility=False,
            strict_obstruction=False,
            rounding_decimal=8,
            epsilon=epsilon,
            verbose=verbose
        )
        
        self.F = np.asarray(F, dtype=np.float64)
        print(f"[ViewFactor] Complete: shape = {self.F.shape}")
        
        return self.F
    
    def save(self, filepath: str):
        """Save view factor matrix to file."""
        if self.F is None:
            raise ValueError("No view factors computed yet")
        np.save(filepath, self.F)
        print(f"[ViewFactor] Saved to {filepath}")
        
    def load(self, filepath: str) -> np.ndarray:
        """Load view factor matrix from file."""
        self.F = np.load(filepath)
        print(f"[ViewFactor] Loaded from {filepath}: shape = {self.F.shape}")
        return self.F
    
    def validate(self) -> dict:
        """
        Validate view factor matrix.
        
        Returns
        -------
        results : dict
            Validation results including row sums and reciprocity checks
        """
        if self.F is None:
            raise ValueError("No view factors to validate")
            
        F = self.F
        n = F.shape[0]
        
        # pyviewfactor convention:
        #   F[receiver, emitter] = fraction leaving emitter arriving at receiver
        # Therefore, closed-enclosure conservation is checked by column sums,
        # i.e. sum over all receivers for each emitting patch.
        leaving_sums = np.sum(F, axis=0)
        
        # Diagonal (should be ~0)
        diag_max = np.max(np.abs(np.diag(F)))
        
        # Reciprocity check: Ai*Fij ≈ Aj*Fji
        areas = self.mesh.compute_cell_sizes(area=True).cell_data["Area"]
        
        # Sample random pairs
        rng = np.random.default_rng(0)
        n_samples = min(10000, n*n)
        i = rng.integers(0, n, size=n_samples)
        j = rng.integers(0, n, size=n_samples)
        mask = i != j
        i, j = i[mask], j[mask]
        
        # pyviewfactor: F[receiver, emitter] = F_{emitter→receiver}
        F_i_to_j = F[j, i]
        F_j_to_i = F[i, j]
        
        lhs = areas[i] * F_i_to_j
        rhs = areas[j] * F_j_to_i
        
        denom = np.maximum(np.abs(lhs) + np.abs(rhs), 1e-30)
        rel_err = np.abs(lhs - rhs) / denom
        
        results = {
            "row_sum_mean": np.mean(leaving_sums),
            "row_sum_min": np.min(leaving_sums),
            "row_sum_max": np.max(leaving_sums),
            "leaving_sum_mean": np.mean(leaving_sums),
            "leaving_sum_min": np.min(leaving_sums),
            "leaving_sum_max": np.max(leaving_sums),
            "diagonal_max": diag_max,
            "reciprocity_median_err": np.median(rel_err),
            "reciprocity_95pct_err": np.quantile(rel_err, 0.95),
            "reciprocity_max_err": np.max(rel_err),
        }
        
        print("\n[ViewFactor] Validation Results:")
        print(f"  Leaving sums: mean={results['leaving_sum_mean']:.4f}, "
              f"min={results['row_sum_min']:.4f}, max={results['row_sum_max']:.4f}")
        print(f"  Diagonal max: {results['diagonal_max']:.2e}")
        print(f"  Reciprocity error: median={results['reciprocity_median_err']:.2e}, "
              f"95%={results['reciprocity_95pct_err']:.2e}")
        
        return results


def compute_volume(mesh: pv.PolyData) -> float:
    """
    Estimate enclosed volume from surface mesh.
    
    For closed surfaces, uses the divergence theorem.
    For open surfaces, uses bounding box estimate.
    """
    try:
        # Try to compute actual volume for closed surface
        solid = mesh.extract_surface().triangulate()
        # Check if closed
        edges = solid.extract_feature_edges(
            boundary_edges=True,
            feature_edges=False,
            manifold_edges=False,
            non_manifold_edges=False
        )
        if edges.n_cells == 0:
            # Closed surface - compute volume
            return float(solid.volume)
    except:
        pass
    
    # Fallback: bounding box
    bounds = mesh.bounds
    return (bounds[1]-bounds[0]) * (bounds[3]-bounds[2]) * (bounds[5]-bounds[4])


def estimate_volume_from_area(total_area: float) -> float:
    """
    Estimate volume assuming spherical enclosure.
    
    Parameters
    ----------
    total_area : float
        Total surface area
        
    Returns
    -------
    volume : float
        Estimated volume
    """
    r_est = np.sqrt(total_area / (4 * np.pi))
    return (4/3) * np.pi * r_est**3


def create_absorption_array(mesh: pv.PolyData, 
                           alpha: Union[float, np.ndarray, Dict[str, float]],
                           physical_ids: Optional[Dict[str, int]] = None) -> np.ndarray:
    """
    Create absorption coefficient array for mesh cells.
    
    Parameters
    ----------
    mesh : pyvista.PolyData
        Surface mesh
    alpha : float, array, or dict
        Absorption coefficient(s):
        - float: uniform absorption
        - array: per-cell absorption
        - dict: per-surface absorption {"WALL": 0.1, "FLOOR": 0.3}
    physical_ids : dict, optional
        Physical ID mapping for dict-based alpha
        
    Returns
    -------
    alpha_array : ndarray
        Absorption coefficient for each cell
    """
    n_cells = mesh.n_cells
    
    if isinstance(alpha, (int, float)):
        return np.full(n_cells, float(alpha))
        
    if isinstance(alpha, np.ndarray):
        assert len(alpha) == n_cells, f"Alpha array length {len(alpha)} != {n_cells} cells"
        return alpha.astype(float)
        
    if isinstance(alpha, dict):
        if physical_ids is None or "gmsh:physical" not in mesh.cell_data:
            raise ValueError("Physical IDs required for dict-based alpha")
            
        phys = mesh.cell_data["gmsh:physical"]
        alpha_array = np.zeros(n_cells)
        
        for name, a in alpha.items():
            if name in physical_ids:
                mask = phys == physical_ids[name]
                alpha_array[mask] = a
                
        return alpha_array
        
    raise TypeError(f"Unsupported alpha type: {type(alpha)}")


def create_alpha_from_physical(mesh: pv.PolyData,
                                phys_id: Dict[str, int],
                                alpha: Dict[str, float],
                                default: float = 0.2) -> np.ndarray:
    """
    Create absorption array from gmsh:physical IDs.
    
    Parameters
    ----------
    mesh : pyvista.PolyData
        Surface mesh with 'gmsh:physical' cell data
    phys_id : dict
        Physical ID mapping: {"WALL": 1, "FLOOR": 2, ...}
    alpha : dict
        Absorption values: {"WALL": 0.2, "FLOOR": 0.3, ...}
    default : float
        Default absorption for unmatched cells
        
    Returns
    -------
    alpha_array : ndarray
        Absorption coefficient for each cell
        
    Example
    -------
    >>> PHYS_ID = {"WALL": 1, "FLOOR": 2, "CEILING": 3}
    >>> ALPHA = {"WALL": 0.2, "FLOOR": 0.3, "CEILING": 0.6}
    >>> alpha_arr = create_alpha_from_physical(mesh, PHYS_ID, ALPHA)
    """
    if "gmsh:physical" not in mesh.cell_data:
        raise ValueError("Mesh has no 'gmsh:physical' cell data. Use create_alpha_by_normal() instead.")
    
    phys_array = mesh.cell_data["gmsh:physical"]
    alpha_arr = np.full(mesh.n_cells, default, dtype=float)
    
    for name, pid in phys_id.items():
        if name in alpha:
            mask = (phys_array == pid)
            alpha_arr[mask] = alpha[name]
    
    return alpha_arr


def create_alpha_array(mesh: pv.PolyData,
                       phys_id: Dict[str, int],
                       alpha: Dict[str, float],
                       default: float = 0.2) -> np.ndarray:
    """
    Create absorption array - auto-detect method.
    
    Uses physical IDs if available, otherwise falls back to normal-based classification.
    
    Parameters
    ----------
    mesh : pyvista.PolyData
        Surface mesh
    phys_id : dict
        Physical ID mapping: {"WALL": 1, "FLOOR": 2, ...}
    alpha : dict
        Absorption values: {"WALL": 0.2, "FLOOR": 0.3, ...}
    default : float
        Default absorption for unmatched cells
        
    Returns
    -------
    alpha_array : ndarray
    """
    if "gmsh:physical" in mesh.cell_data:
        print("[Alpha] Using gmsh:physical IDs")
        return create_alpha_from_physical(mesh, phys_id, alpha, default)
    else:
        print("[Alpha] Fallback: classifying by normal direction")
        return create_alpha_by_normal(mesh, alpha)


def create_alpha_by_normal(mesh: pv.PolyData, 
                           alpha_config: Dict[str, float],
                           tol: float = 0.7) -> np.ndarray:
    """
    Create absorption array by classifying cells based on normal direction.
    
    Useful when physical IDs are not available.
    
    Parameters
    ----------
    mesh : pyvista.PolyData
        Surface mesh
    alpha_config : dict
        Absorption values: {"WALL": 0.2, "FLOOR": 0.3, "CEILING": 0.6, "BLOCK": 0.1}
    tol : float
        Tolerance for normal direction classification
        
    Returns
    -------
    alpha_array : ndarray
        Absorption coefficient for each cell
    """
    mesh_with_normals = mesh.compute_normals(cell_normals=True, point_normals=False)
    normals = mesh_with_normals.cell_data["Normals"]
    centers = mesh.cell_centers().points
    
    n_cells = mesh.n_cells
    alpha = np.zeros(n_cells, dtype=float)
    
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    Lz = zmax - zmin
    z_ceiling_threshold = zmin + 0.7 * Lz
    z_floor_threshold = zmin + 0.3 * Lz
    
    default_alpha = alpha_config.get("WALL", 0.2)
    
    for i in range(n_cells):
        nx, ny, nz = normals[i]
        cz = centers[i][2]
        
        # Check if on boundary or interior
        on_x = (centers[i][0] < xmin + 0.1 or centers[i][0] > xmax - 0.1)
        on_y = (centers[i][1] < ymin + 0.1 or centers[i][1] > ymax - 0.1)
        on_z = (cz < zmin + 0.1 or cz > zmax - 0.1)
        
        if not (on_x or on_y or on_z):
            alpha[i] = alpha_config.get("BLOCK", 0.1)
        elif cz > z_ceiling_threshold and nz < -0.3:
            alpha[i] = alpha_config.get("CEILING", 0.6)
        elif cz < z_floor_threshold and nz > tol:
            alpha[i] = alpha_config.get("FLOOR", 0.3)
        else:
            alpha[i] = default_alpha
            
    return alpha
