"""
Radiative Energy Transfer - Postprocessing Module

SPL calculation on planes, VTK export, and visualization.
Includes adaptive grid generation for complex geometries.
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, Union, TYPE_CHECKING, Sequence
import os

if TYPE_CHECKING:
    from .RETsolver import RadiativeEnergyTransfer


class SPLPlaneCalculator:
    """
    Calculate SPL on 2D planes through the enclosure.
    
    Supports both simple rectangular grids and adaptive grids
    that conform to complex room geometries with obstacles.
    """
    
    def __init__(self, solver: "RadiativeEnergyTransfer"):
        """
        Initialize SPL plane calculator.
        
        Parameters
        ----------
        solver : RadiativeEnergyTransfer
            Initialized solver with computed B
        """
        self.solver = solver
        
    def create_plane_grid(self, 
                          plane: str = "XY-plane",
                          height: float = 0.0,
                          spacing: float = 0.5,
                          bounds: Optional[Tuple] = None) -> pv.StructuredGrid:
        """
        Create a simple structured grid on a plane.
        
        For complex geometries with obstacles, use create_adaptive_plane_grid().
        
        Parameters
        ----------
        plane : str
            "XY-plane", "XZ-plane", or "YZ-plane"
        height : float
            Position along perpendicular axis
        spacing : float
            Grid spacing (m)
        bounds : tuple, optional
            (xmin, xmax, ymin, ymax, zmin, zmax)
            If None, uses mesh bounds
            
        Returns
        -------
        grid : pyvista.StructuredGrid
        """
        if bounds is None:
            bounds = self.solver.mesh.bounds
            
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        
        if plane == "XY-plane":
            xs = np.arange(xmin, xmax + spacing, spacing)
            ys = np.arange(ymin, ymax + spacing, spacing)
            X, Y = np.meshgrid(xs, ys, indexing="xy")
            Z = np.full_like(X, height)
            
        elif plane == "XZ-plane":
            xs = np.arange(xmin, xmax + spacing, spacing)
            zs = np.arange(zmin, zmax + spacing, spacing)
            X, Z = np.meshgrid(xs, zs, indexing="xy")
            Y = np.full_like(X, height)
            
        elif plane == "YZ-plane":
            ys = np.arange(ymin, ymax + spacing, spacing)
            zs = np.arange(zmin, zmax + spacing, spacing)
            Y, Z = np.meshgrid(ys, zs, indexing="xy")
            X = np.full_like(Y, height)
            
        else:
            raise ValueError(f"Unknown plane: {plane}")
            
        grid = pv.StructuredGrid(X, Y, Z)
        print(f"[SPL Plane] {plane} at {height}m: {grid.n_points} points")
        
        return grid
    
    def _mask_points_inside_obstacles(
        self,
        points: np.ndarray,
        obstacles: Union[pv.PolyData, Sequence[pv.PolyData]],
        tolerance: float = 1e-6,
        check_surface: bool = True,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Return a boolean mask of points that are inside any obstacle volume.

        Notes
        -----
        This uses PyVista's select_enclosed_points, which is most reliable when the
        obstacle surface is closed/watertight. If an obstacle is not closed, the
        mask may be inaccurate. In that case, you may need to repair/close the mesh.
        """
        pts = np.asarray(points, dtype=float)
        if pts.size == 0:
            return np.zeros((0,), dtype=bool)

        obs_list = obstacles if isinstance(obstacles, (list, tuple)) else [obstacles]
        inside_any = np.zeros(len(pts), dtype=bool)

        cloud = pv.PolyData(pts)

        for k, obs in enumerate(obs_list):
            if obs is None or getattr(obs, "n_points", 0) == 0:
                continue
            try:
                obs2 = obs.triangulate().clean()
                sel = cloud.select_enclosed_points(obs2, tolerance=tolerance, check_surface=check_surface)
                inside = np.asarray(sel.point_data.get("SelectedPoints", np.zeros(len(pts))), dtype=bool)
                inside_any |= inside
                if verbose:
                    print(f"  [ObstacleMask] obstacle#{k}: inside={int(inside.sum())}/{len(pts)}")
            except Exception as e:
                # If select_enclosed_points fails (often due to non-watertight surfaces), skip with a warning.
                print(f"  [ObstacleMask] Warning: could not evaluate obstacle#{k} ({e}). "
                      f"Points inside this obstacle will NOT be masked.")
                continue

        return inside_any

    def create_adaptive_plane_grid(self,
                                   plane: str = "XY-plane",
                                   height: float = 0.0,
                                   spacing: float = 0.1,
                                   offset: float = 0.1,
                                   mesh: Optional[pv.PolyData] = None,
                                   obstacles: Optional[Union[pv.PolyData, Sequence[pv.PolyData]]] = None,
                                   remove_obstacle_points: bool = True,
                                   obstacle_tolerance: float = 1e-6,
                                   obstacle_check_surface: bool = True) -> pv.PolyData:
        """
        Create an adaptive triangular grid that conforms to room geometry.

        Uses mesh slicing + Shapely polygonization to build the valid plane region.
        Optionally removes points that fall inside obstacle volumes (e.g., seats),
        so you don't generate SPL points inside obstacles.

        Parameters
        ----------
        plane : str
            "XY-plane", "XZ-plane", or "YZ-plane"
        height : float
            Position along perpendicular axis
        spacing : float
            Grid point spacing (m)
        offset : float
            Distance to shrink from boundaries (m)
        mesh : pyvista.PolyData, optional
            Mesh to slice. If None, uses solver mesh.
        obstacles : pyvista.PolyData or list of PolyData, optional
            Obstacle closed surfaces (e.g., seat mesh). Used only to remove points
            inside obstacles. (RET visibility is handled elsewhere by the solver.)
        remove_obstacle_points : bool
            If True and obstacles is provided, remove grid points inside obstacles.
        obstacle_tolerance : float
            Tolerance passed to select_enclosed_points.
        obstacle_check_surface : bool
            Whether to check if obstacle surface is closed.

        Returns
        -------
        grid : pyvista.PolyData
            Triangulated plane grid
        """
        try:
            from shapely.geometry import Polygon, MultiPolygon, LineString, Point
            from shapely.ops import polygonize, unary_union
            from shapely.prepared import prep
        except ImportError:
            raise ImportError("Shapely is required for adaptive grids. Install with: pip install shapely")

        if mesh is None:
            mesh = self.solver.mesh

        print(f"[AdaptivePlane] Generating {plane} at {height:.2f}m (offset={offset}m)...")

        # Define projection functions
        if plane == "XY-plane":
            normal, origin = [0, 0, 1], [0, 0, height]
            to_2d = lambda p: p[:, 0:2]
            to_3d = lambda uv: np.column_stack([uv, np.full(len(uv), height)])
        elif plane == "XZ-plane":
            normal, origin = [0, 1, 0], [0, height, 0]
            to_2d = lambda p: p[:, [0, 2]]
            to_3d = lambda uv: np.column_stack([uv[:, 0], np.full(len(uv), height), uv[:, 1]])
        elif plane == "YZ-plane":
            normal, origin = [1, 0, 0], [height, 0, 0]
            to_2d = lambda p: p[:, [1, 2]]
            to_3d = lambda uv: np.column_stack([np.full(len(uv), height), uv[:, 0], uv[:, 1]])
        else:
            raise ValueError(f"Unknown plane: {plane}")

        # Step 1: Slice 3D geometry
        contours_3d = mesh.slice(normal=normal, origin=origin)
        if contours_3d.n_points == 0:
            print(f"  [Warning] Plane does not intersect geometry")
            return pv.PolyData()

        # Step 2: Convert to Shapely polygons
        lines_2d_pts = to_2d(contours_3d.points)
        lines_connected = contours_3d.lines.reshape(-1, 3)[:, 1:]
        shapely_lines = [LineString([lines_2d_pts[id1], lines_2d_pts[id2]])
                        for id1, id2 in lines_connected]

        polys = list(polygonize(shapely_lines))
        if not polys:
            print(f"  [Warning] Could not form polygons from slice")
            return pv.PolyData()

        # Step 3: Identify room vs holes (largest area = room)
        polys.sort(key=lambda p: p.area, reverse=True)
        room_poly = polys[0]
        holes = polys[1:]

        if holes:
            base_shape = room_poly.difference(unary_union(holes))
        else:
            base_shape = room_poly

        # Step 4: Apply offset (shrink region)
        buffered_shape = base_shape.buffer(-offset, join_style=2)  # Mitre join
        if buffered_shape.is_empty:
            print(f"  [Warning] Shape is empty after offset")
            return pv.PolyData()

        # Step 5: Densify boundary
        def densify_ring(ring, max_len):
            pts = []
            coords = list(ring.coords)
            for i in range(len(coords) - 1):
                p1, p2 = np.array(coords[i]), np.array(coords[i + 1])
                dist = np.linalg.norm(p2 - p1)
                pts.append(p1)
                if dist > max_len:
                    num = int(np.ceil(dist / max_len))
                    for kk in range(1, num):
                        pts.append(p1 + (p2 - p1) * (kk / num))
            pts.append(coords[-1])
            return pts

        def densify_geometry(geom, max_len):
            if geom.geom_type == 'Polygon':
                ext = densify_ring(geom.exterior, max_len)
                ints = [densify_ring(i, max_len) for i in geom.interiors]
                return Polygon(ext, ints)
            elif geom.geom_type == 'MultiPolygon':
                return MultiPolygon([densify_geometry(p, max_len) for p in geom.geoms])
            return geom

        dense_shape = densify_geometry(buffered_shape, spacing)

        # Step 6: Generate internal grid points (2D) and filter inside region
        b = mesh.bounds
        if plane == "XY-plane":
            xs, ys = np.arange(b[0], b[1], spacing), np.arange(b[2], b[3], spacing)
            U, V = np.meshgrid(xs, ys)
        elif plane == "XZ-plane":
            xs, zs = np.arange(b[0], b[1], spacing), np.arange(b[4], b[5], spacing)
            U, V = np.meshgrid(xs, zs)
        elif plane == "YZ-plane":
            ys, zs = np.arange(b[2], b[3], spacing), np.arange(b[4], b[5], spacing)
            U, V = np.meshgrid(ys, zs)

        grid_2d = np.column_stack([U.ravel(), V.ravel()])

        prep_shape = prep(buffered_shape)
        mask = [prep_shape.contains(Point(pt)) for pt in grid_2d]
        valid_grid_3d = to_3d(grid_2d[mask])

        # Extract boundary points
        bound_pts_2d = []

        def extract_boundary(geom):
            if geom.geom_type == 'Polygon':
                bound_pts_2d.extend(list(geom.exterior.coords))
                for interior in geom.interiors:
                    bound_pts_2d.extend(list(interior.coords))
            elif geom.geom_type == 'MultiPolygon':
                for g in geom.geoms:
                    extract_boundary(g)

        extract_boundary(dense_shape)
        bound_pts_3d = to_3d(np.array(bound_pts_2d)) if bound_pts_2d else np.empty((0, 3))

        # NEW: remove points that fall inside obstacles (e.g., seats)
        if remove_obstacle_points and (obstacles is not None):
            all_pts = np.vstack([bound_pts_3d, valid_grid_3d]) if len(valid_grid_3d) > 0 else bound_pts_3d
            inside_any = self._mask_points_inside_obstacles(
                all_pts,
                obstacles=obstacles,
                tolerance=obstacle_tolerance,
                check_surface=obstacle_check_surface,
                verbose=False,
            )
            if inside_any.any():
                # split back
                n_bound = len(bound_pts_3d)
                bound_keep = ~inside_any[:n_bound]
                grid_keep = ~inside_any[n_bound:]
                bound_pts_3d = bound_pts_3d[bound_keep] if n_bound > 0 else bound_pts_3d
                valid_grid_3d = valid_grid_3d[grid_keep] if len(valid_grid_3d) > 0 else valid_grid_3d

        # Step 7: Triangulate
        all_pts = np.vstack([bound_pts_3d, valid_grid_3d]) if len(valid_grid_3d) > 0 else bound_pts_3d
        cloud_2d = pv.PolyData(np.column_stack([to_2d(all_pts), np.zeros(len(all_pts))]))
        surf = cloud_2d.delaunay_2d()

        plane_surf = pv.PolyData(to_3d(surf.points[:, 0:2]), faces=surf.faces)

        # Final clean (remove triangles in holes)
        centers = to_2d(plane_surf.cell_centers().points)
        mask_cells = [prep_shape.contains(Point(pt)) for pt in centers]
        final_plane = plane_surf.extract_cells(np.where(mask_cells)[0]).clean()

        # (Optional) attach debug mask for the final points as well
        if remove_obstacle_points and (obstacles is not None) and final_plane.n_points > 0:
            inside_final = self._mask_points_inside_obstacles(
                final_plane.points,
                obstacles=obstacles,
                tolerance=obstacle_tolerance,
                check_surface=obstacle_check_surface,
                verbose=False,
            )
            final_plane.point_data["inside_obstacle"] = inside_final.astype(np.int8)

        print(f"  Generated: {final_plane.n_cells} cells, {final_plane.n_points} points")

        return final_plane
    def compute_spl_on_grid(self,
                            grid: Union[pv.StructuredGrid, pv.PolyData],
                            B: np.ndarray,
                            source_pos: np.ndarray,
                            show_progress: bool = True,
                            valid_mask: Optional[np.ndarray] = None,
                            obstacles: Optional[Union[pv.PolyData, Sequence[pv.PolyData]]] = None,
                            obstacle_tolerance: float = 1e-6,
                            obstacle_check_surface: bool = True) -> Union[pv.StructuredGrid, pv.PolyData]:
        """
        Compute SPL at each point on the grid.

        Parameters
        ----------
        grid : pyvista.StructuredGrid or pyvista.PolyData
            Plane grid
        B : ndarray
            Radiation density (steady-state)
        source_pos : array-like
            Source position
        show_progress : bool
            Show progress during computation
        valid_mask : ndarray, optional
            Boolean mask of points to compute. If False, SPL is left as NaN.
        obstacles : pyvista.PolyData or list of PolyData, optional
            If provided, points inside obstacles are automatically masked out.
            (Useful as an extra safety net.)
        obstacle_tolerance : float
            Tolerance passed to select_enclosed_points.
        obstacle_check_surface : bool
            Whether to check if obstacle surface is closed.

        Returns
        -------
        grid : same type as input
            Grid with SPL values
        """
        points = grid.points
        n_pts = points.shape[0]
        spl_values = np.full(n_pts, np.nan, dtype=float)

        if valid_mask is None:
            valid_mask = np.ones(n_pts, dtype=bool)
        else:
            valid_mask = np.asarray(valid_mask, dtype=bool).copy()
            if valid_mask.shape[0] != n_pts:
                raise ValueError(f"valid_mask length {valid_mask.shape[0]} does not match number of points {n_pts}")

        # Extra safety: mask points inside obstacles
        if obstacles is not None and n_pts > 0:
            inside = self._mask_points_inside_obstacles(
                points,
                obstacles=obstacles,
                tolerance=obstacle_tolerance,
                check_surface=obstacle_check_surface,
                verbose=False,
            )
            valid_mask &= ~inside
            # keep for debug
            try:
                grid.point_data["inside_obstacle"] = inside.astype(np.int8)
            except Exception:
                pass

        print(f"[SPL Calc] Computing on {n_pts} points...")

        for i, pt in enumerate(points):
            if not valid_mask[i]:
                continue
            spl_values[i] = self.solver.compute_SPL_steady(B, source_pos, pt)

            if show_progress and (i + 1) % 100 == 0:
                print(f"  {i+1}/{n_pts}", end="\r")

        if show_progress:
            print()

        try:
            grid.point_data["SPL"] = spl_values
            grid.point_data["SPL_valid"] = np.isfinite(spl_values).astype(np.int8)
        except Exception:
            # StructuredGrid sometimes prefers dict-style assignment
            grid["SPL"] = spl_values
            grid["SPL_valid"] = np.isfinite(spl_values).astype(np.int8)

        return grid
    def compute_energy_density_field(self,
                                     B: np.ndarray,
                                     source_pos: np.ndarray,
                                     n_grid: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute energy density on a grid in the x-z plane (y=0).
        
        For creating contour plots like Kuttruff Figure 3.
        """
        mesh = self.solver.mesh
        bounds = mesh.bounds
        
        r_max = min(
            (bounds[1] - bounds[0]) / 2,
            (bounds[3] - bounds[2]) / 2,
            (bounds[5] - bounds[4]) / 2
        ) * 0.95
        
        x = np.linspace(-r_max, r_max, n_grid)
        z = np.linspace(-r_max, r_max, n_grid)
        X, Z = np.meshgrid(x, z)
        
        U = np.full_like(X, np.nan)
        
        print(f"[Energy] Computing on {n_grid}x{n_grid} grid...")
        
        for i in range(n_grid):
            for j in range(n_grid):
                xi, zi = X[i, j], Z[i, j]
                r = np.sqrt(xi**2 + zi**2)
                
                if r >= r_max or r < 0.5:
                    continue
                    
                rr = np.array([xi, 0, zi])
                I = self.solver.intensity_at_receiver(B, source_pos, rr)
                U[i, j] = I / self.solver.cfg.c
                
            if (i + 1) % 10 == 0:
                print(f"  Row {i+1}/{n_grid}")
                
        return X, Z, U
    

    def compute_spl_at_point(self,
                             point: np.ndarray,
                             B: np.ndarray,
                             source_pos: Optional[np.ndarray] = None) -> float:
        """
        Compute SPL at a single 3D point.

        Parameters
        ----------
        point : array-like (3,)
            Receiver position [x,y,z]
        B : ndarray
            Radiation density (steady-state)
        source_pos : array-like or None
            Keep consistent with solver API. If your solver ignores it, pass None.

        Returns
        -------
        spl_db : float
            SPL at the point (dB)
        """
        pt = np.asarray(point, dtype=float).reshape(3,)
        return float(self.solver.compute_SPL_steady(B, source_pos, pt))



class ResultExporter:
    """
    Export results to various formats.
    """
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize exporter.
        
        Parameters
        ----------
        output_dir : str
            Output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def save_mesh_vtk(self, mesh: pv.PolyData, filename: str) -> str:
        """Save mesh with results to VTK file."""
        filepath = self.output_dir / filename
        mesh.save(str(filepath))
        print(f"[Export] Saved mesh to {filepath}")
        return str(filepath)
        
    def save_plane_vtk(self, grid: Union[pv.StructuredGrid, pv.PolyData], 
                       filename: str) -> str:
        """Save SPL plane to VTK file."""
        filepath = self.output_dir / filename
        grid.save(str(filepath))
        print(f"[Export] Saved plane to {filepath}")
        return str(filepath)
        
    def save_decay_curve(self, time: np.ndarray, spl: np.ndarray, 
                         filename: str) -> str:
        """Save decay curve data."""
        filepath = self.output_dir / filename
        data = np.column_stack([time, spl])
        np.savetxt(str(filepath), data, header="time(s) SPL(dB)", fmt="%.6e")
        print(f"[Export] Saved decay curve to {filepath}")
        return str(filepath)
        
    def save_summary(self, summary: dict, filename: str = "summary.txt") -> str:
        """Save results summary."""
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("RADIATIVE ENERGY TRANSFER RESULTS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            for key, value in summary.items():
                value_str = str(value).replace('³', '^3').replace('²', '^2')
                f.write(f"{key}: {value_str}\n")
        print(f"[Export] Saved summary to {filepath}")
        return str(filepath)


class Visualizer:
    """
    Visualization utilities.
    """
    
    @staticmethod
    def plot_decay_curve(time: np.ndarray, spl: np.ndarray,
                         rt60: Optional[float] = None,
                         rt_theory: Optional[float] = None,
                         slope: Optional[float] = None,
                         save_path: Optional[str] = None):
        """Plot decay curve."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(time * 1000, spl, 'b-', linewidth=1, label='Decay curve')
        
        if slope is not None:
            fit_line = slope * time
            ax.plot(time * 1000, fit_line, 'r--', linewidth=1.5, 
                   label=f'Fit: {slope:.1f} dB/s')
        
        ax.axhline(-60, color='gray', linestyle=':', alpha=0.7)
        
        if rt60 is not None:
            ax.axvline(rt60 * 1000, color='g', linestyle='--', alpha=0.7,
                       label=f'RT60 = {rt60*1000:.0f} ms')
                       
        title = 'Sound Decay'
        if rt60 is not None and rt_theory is not None:
            title += f'\nRT60: computed={rt60:.3f}s, Eyring={rt_theory:.3f}s'
            
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Level (dB re peak)', fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.set_ylim([-70, 5])
        if rt60 is not None:
            ax.set_xlim([0, min(rt60 * 2000, time[-1] * 1000)])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Plot] Saved to {save_path}")
            
        plt.show()
        
    @staticmethod
    def plot_energy_contours(X: np.ndarray, Z: np.ndarray, U: np.ndarray,
                             radius: Optional[float] = None,
                             save_path: Optional[str] = None):
        """Plot energy density contours."""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Convert to dB
        U_valid = U[~np.isnan(U)]
        U_ref = np.median(U_valid)
        U_dB = 10 * np.log10(U / U_ref + 1e-20)
        
        levels = np.arange(-20, 20, 2)
        
        cf = ax.contourf(X, Z, U_dB, levels=30, cmap='YlOrRd', alpha=0.6)
        cs = ax.contour(X, Z, U_dB, levels=levels, colors='blue', linewidths=1.0)
        ax.clabel(cs, inline=True, fontsize=9, fmt='%.0f')
        
        if radius is not None:
            theta = np.linspace(0, 2*np.pi, 200)
            ax.plot(radius * np.cos(theta), radius * np.sin(theta), 'k-', linewidth=2)
            
        ax.plot(0, 0, 'ko', markersize=15, label='Source')
        
        ax.set_xlabel('x (m)', fontsize=12)
        ax.set_ylabel('z (m)', fontsize=12)
        ax.set_title('Energy Density (dB re median)', fontsize=14)
        ax.set_aspect('equal')
        
        cbar = plt.colorbar(cf, ax=ax, shrink=0.8)
        cbar.set_label('Energy Density (dB)', fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Plot] Saved to {save_path}")
            
        plt.show()
        
    @staticmethod
    def plot_B_distribution(mesh: pv.PolyData, B: np.ndarray,
                            save_path: Optional[str] = None):
        """Plot radiation density on mesh."""
        mesh_copy = mesh.copy()
        mesh_copy.cell_data['B'] = B
        
        plotter = pv.Plotter()
        plotter.add_mesh(mesh_copy, scalars='B', cmap='hot', show_edges=False)
        plotter.add_scalar_bar(title='B (W/m²)')
        plotter.add_axes()
        
        if save_path:
            plotter.screenshot(save_path)
            print(f"[Plot] Saved to {save_path}")
            
        plotter.show()
        
    @staticmethod
    def plot_SPL_plane(grid: Union[pv.StructuredGrid, pv.PolyData],
                       save_path: Optional[str] = None,
                       show_mesh: bool = False,
                       mesh: Optional[pv.PolyData] = None):
        """Plot SPL distribution on plane."""
        plotter = pv.Plotter()
        
        if show_mesh and mesh is not None:
            plotter.add_mesh(mesh, color="lightgray", opacity=0.3)
        
        plotter.add_mesh(grid, scalars='SPL', cmap='jet', 
                        show_edges=False, clim=[40, 100])
        plotter.add_scalar_bar(title='SPL (dB)')
        plotter.add_axes()
        
        if save_path:
            plotter.screenshot(save_path)
            print(f"[Plot] Saved to {save_path}")
            
        plotter.show()
        
    @staticmethod
    def preview_mesh_with_plane(mesh: pv.PolyData, 
                                plane_grid: Union[pv.StructuredGrid, pv.PolyData]):
        """Preview mesh with SPL plane."""
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color="lightgray", opacity=0.3)
        plotter.add_mesh(plane_grid, style="wireframe", color="yellow", opacity=0.5)
        plotter.add_axes()
        plotter.show()


# Utility function for quick VTK saving
def save_vtk(data: Union[pv.PolyData, pv.StructuredGrid], 
             filename: str, 
             output_dir: str = "output") -> str:
    """Quick utility to save VTK files."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    data.save(filepath)
    print(f"[Saved] {filepath}")
    return filepath






class EnergyFieldCalculator:
    """
    Compute acoustic energy density w [J/m3] and energy flux vector q [W/m2]
    on a plane grid from RET steady-state radiosity B.

    Physics
    -------
    Energy density:
        w(r) = I(r) / c

    Energy flux (intensity) vector:
        q(r) = sum_i [B_i A_i cos_theta_i / (pi R_i^2)] * e_hat_i
               + [W/(4 pi Rs^2)] * e_hat_s

    ParaView usage
    --------------
    1. Open .vtk
    2. Color by "SPL" or "w_Jm3"
    3. Filters -> Glyph -> Vector="q_Wm2", Type=Arrow,
       Scale array="q_mag_Wm2"
    """

    def __init__(self, solver):
        self.solver  = solver
        self.centers = solver.centers
        self.normals = solver.normals
        self.areas   = solver.areas
        self.N       = solver.N
        self.c       = solver.cfg.c
        self.W       = solver.cfg.W
        self.check_obstruction = solver.cfg.check_obstruction

    def _flux_vector_at_point(self, r, B, source_pos):
        """Net energy flux vector q at receiver point r."""
        diff  = r[None, :] - self.centers
        R     = np.linalg.norm(diff, axis=1)
        valid = R > 1e-9
        e_hat = np.zeros_like(diff)
        e_hat[valid] = diff[valid] / R[valid, None]

        cos_theta = np.einsum("ij,ij->i", self.normals, e_hat)
        cos_theta = np.maximum(cos_theta, 0.0)

        vis = np.ones(self.N, dtype=bool)
        if self.check_obstruction:
            for i in np.where(valid & (cos_theta > 0) & (B > 0))[0]:
                vis[i] = self.solver.is_visible(self.centers[i], r)

        mask    = valid & vis & (cos_theta > 0) & (B > 0)
        weights = (B[mask] * self.areas[mask] * cos_theta[mask]
                   / (np.pi * R[mask] ** 2))
        q = np.sum(weights[:, None] * e_hat[mask], axis=0) if mask.any() else np.zeros(3)

        if source_pos is not None and self.W > 0:
            Rs = np.linalg.norm(r - source_pos)
            if Rs > 1e-6:
                vis_src = (not self.check_obstruction) or self.solver.is_visible(source_pos, r)
                if vis_src:
                    q += (self.W / (4 * np.pi * Rs**2)) * (r - source_pos) / Rs

        return q

    def compute_fields_on_grid(self, grid, B, source_pos=None, show_progress=True):
        """
        Attach w_Jm3, q_Wm2, q_mag_Wm2, q_dB to every valid point on grid.

        Inherits SPL_valid mask from compute_spl_on_grid so obstacle points
        and NaN positions are automatically skipped.

        Parameters
        ----------
        grid       : pv.PolyData  (output of SPLPlaneCalculator.compute_spl_on_grid)
        B          : (N,) steady-state radiosity
        source_pos : (3,) point source position, or None
        show_progress : bool

        Returns
        -------
        grid : same object with added point_data arrays
        """
        points = grid.points
        n_pts  = len(points)

        if "SPL_valid" in grid.point_data:
            valid_mask = grid.point_data["SPL_valid"].astype(bool)
        else:
            valid_mask = np.ones(n_pts, dtype=bool)

        w_arr = np.full(n_pts, np.nan)
        q_arr = np.full((n_pts, 3), np.nan)

        # Guard: empty grid (e.g. XZ-plane slice failed)
        if n_pts == 0:
            print("[EnergyField] WARNING: empty grid - skipping")
            return grid

        valid_idx = np.where(valid_mask)[0]
        print(f"[EnergyField] Computing w & q on {len(valid_idx)}/{n_pts} points...")

        for k, idx in enumerate(valid_idx):
            r = points[idx]
            I_scalar   = self.solver.receiver_intensity_steady(B, source_pos, r)
            w_arr[idx] = I_scalar / self.c
            q_arr[idx] = self._flux_vector_at_point(r, B, source_pos)
            if show_progress and (k + 1) % 50 == 0:
                print(f"  {k+1}/{len(valid_idx)}", end="\r")

        if show_progress:
            print()

        q_mag = np.linalg.norm(q_arr, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            q_dB = np.where(q_mag > 0, 10 * np.log10(q_mag / 1e-12), np.nan)

        # --- detect plane normal from point coordinates ---
        var = np.var(grid.points, axis=0)
        normal_axis = int(np.argmin(var))  # 0=X, 1=Y, 2=Z

        # project q onto the plane
        q_inplane = q_arr.copy()
        q_inplane[:, normal_axis] = 0.0
        q_inplane_mag = np.linalg.norm(q_inplane, axis=1)

        # unit vector for equal-size arrows in ParaView
        q_dir = np.zeros_like(q_inplane)
        nz = q_inplane_mag > 0
        q_dir[nz] = q_inplane[nz] / q_inplane_mag[nz, None]

        grid.point_data["w_Jm3"]         = w_arr
        grid.point_data["q_Wm2"]         = q_arr           # full 3D (reference)
        grid.point_data["q_inplane_Wm2"] = q_inplane       # projected onto plane
        grid.point_data["q_mag_Wm2"]     = q_inplane_mag   # in-plane magnitude
        grid.point_data["q_dB"]          = q_dB
        grid.point_data["q_dir"]         = q_dir            # unit vector in-plane

        print(f"  Plane normal: {[chr(88+normal_axis)]}  in-plane projection applied")
        print(f"  w   mean={np.nanmean(w_arr):.3e} J/m3  max={np.nanmax(w_arr):.3e} J/m3")
        print(f"  |q| mean={np.nanmean(q_inplane_mag[np.isfinite(q_inplane_mag)]):.3e} W/m2  "
              f"max={np.nanmax(q_inplane_mag[np.isfinite(q_inplane_mag)]):.3e} W/m2")
        return grid
