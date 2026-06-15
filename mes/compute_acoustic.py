"""Compute steady-state acoustic fields for mesh cases available in geo/."""

from dataclasses import dataclass
from pathlib import Path
import argparse
import sys

import numpy as np

try:
    from .ret.config import RETConfig
    from .ret.geometry import MeshProcessor, create_alpha_array
    from .ret.RETsolver import RadiativeEnergyTransfer
    from .ret.postprocess import EnergyFieldCalculator, SPLPlaneCalculator
except ImportError:
    from ret.config import RETConfig
    from ret.geometry import MeshProcessor, create_alpha_array
    from ret.RETsolver import RadiativeEnergyTransfer
    from ret.postprocess import EnergyFieldCalculator, SPLPlaneCalculator

BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class AcousticCase:
    name: str
    mesh_path: Path
    vf_path: Path
    output_dir: Path
    check_obstruction: bool


CASES = {
    "base": AcousticCase(
        name="base",
        mesh_path=BASE_DIR / "geo/base.msh",
        vf_path=BASE_DIR / "geo/base_nominal_vf.npy",
        output_dir=BASE_DIR / "output/nominal/base",
        check_obstruction=False,
    ),
    "obs": AcousticCase(
        name="obs",
        mesh_path=BASE_DIR / "geo/obs.msh",
        vf_path=BASE_DIR / "geo/obs_nominal_vf_check.npy",
        output_dir=BASE_DIR / "output/nominal/obs",
        check_obstruction=True,
    ),
}

AVAILABLE_CASES = {
    name: case
    for name, case in CASES.items()
    if case.mesh_path.exists()
}

# Physical IDs stored in geo/base.msh and geo/obs.msh.
PHYS_ID = {
    "WALL": 1,
    "FLOOR": 2,
    "TOP": 3,
    "OBS1": 4,
    "OBS2": 5,
}

# Absorption coefficients. Adjust here if you want different materials.
ALPHA = {
    "WALL": 0.2,
    "FLOOR": 0.3,
    "TOP": 0.2,
    "OBS1": 0.1,
    "OBS2": 0.1,
}

DEFAULT_SOURCE = np.array([0.5, 0.5, 0.5])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute RET steady-state results for available mesh cases."
    )
    parser.add_argument(
        "--case",
        choices=["all", *AVAILABLE_CASES.keys()],
        default="all",
        help="Which case to process.",
    )
    parser.add_argument(
        "--source",
        nargs=3,
        type=float,
        default=DEFAULT_SOURCE,
        metavar=("X", "Y", "Z"),
        help="Point-source position in meters.",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=0.005,
        help="Point-source acoustic power W.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=0.2,
        help="SPL plane grid spacing in meters.",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=0.1,
        help="Distance used to shrink SPL planes from the boundary.",
    )
    parser.add_argument(
        "--xy-height",
        type=float,
        default=0.8,
        help="Z value for the XY SPL plane.",
    )
    return parser.parse_args()


def selected_cases(case_name: str) -> list[AcousticCase]:
    if case_name == "all":
        return list(AVAILABLE_CASES.values())
    return [AVAILABLE_CASES[case_name]]


def run_case(
    case: AcousticCase,
    source: np.ndarray,
    power: float,
    spacing: float,
    offset: float,
    xy_height: float,
) -> None:
    print("\n" + "=" * 70)
    print(f"CASE: {case.name}")
    print("=" * 70)

    if not case.mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {case.mesh_path}")
    if not case.vf_path.exists():
        raise FileNotFoundError(
            f"View factors not found: {case.vf_path}. "
            f"Run: python compute_viewfactors.py --case {case.name}"
        )

    case.output_dir.mkdir(parents=True, exist_ok=True)

    print("[1] Load and prepare mesh")
    processor = MeshProcessor(str(case.mesh_path))
    mesh = processor.prepare_geometry()
    print(f"  Cells: {mesh.n_cells}")
    print(f"  Bounds: {tuple(round(v, 3) for v in mesh.bounds)}")

    print("\n[2] Load view factors and absorption")
    F = np.load(case.vf_path)
    if F.shape != (mesh.n_cells, mesh.n_cells):
        raise ValueError(
            f"View-factor shape {F.shape} does not match mesh cells {mesh.n_cells}. "
            f"Recompute with: python compute_viewfactors.py --case {case.name} --force"
        )

    alpha = create_alpha_array(mesh, PHYS_ID, ALPHA, default=ALPHA["WALL"])
    print(f"  Alpha: mean={alpha.mean():.3f}, min={alpha.min():.3f}, max={alpha.max():.3f}")

    print("\n[3] Solve steady-state RET")
    cfg = RETConfig(W=power, check_obstruction=case.check_obstruction)
    solver = RadiativeEnergyTransfer(mesh, F, alpha=alpha, cfg=cfg, skip_preprocessing=True)

    B = solver.solve_steady_state(source)
    print(f"  Mean B: {np.mean(B):.4e} W/m^2")

    surface_path = case.output_dir / "surface_energy.vtk"
    mesh_out = solver.get_results_mesh()
    mesh_out.save(surface_path)
    print(f"  Saved: {surface_path}")

    print("\n[4] Generate SPL planes")
    spl_calc = SPLPlaneCalculator(solver)
    energy_calc = EnergyFieldCalculator(solver)

    xy_grid = spl_calc.create_adaptive_plane_grid(
        "XY-plane",
        height=xy_height,
        spacing=spacing,
        offset=offset,
    )
    xy_grid = spl_calc.compute_spl_on_grid(xy_grid, B, source)
    xy_grid = energy_calc.compute_fields_on_grid(xy_grid, B, source)
    xy_path = case.output_dir / f"spl_xy_z{xy_height:.2f}.vtk"
    xy_grid.save(xy_path)
    print(f"  Saved: {xy_path}")

    mid_y = (mesh.bounds[2] + mesh.bounds[3]) / 2
    xz_grid = spl_calc.create_adaptive_plane_grid(
        "XZ-plane",
        height=mid_y,
        spacing=spacing,
        offset=offset,
    )
    xz_grid = spl_calc.compute_spl_on_grid(xz_grid, B, source)
    xz_grid = energy_calc.compute_fields_on_grid(xz_grid, B, source)
    xz_path = case.output_dir / f"spl_xz_y{mid_y:.2f}.vtk"
    xz_grid.save(xz_path)
    print(f"  Saved: {xz_path}")

    mid_x = (mesh.bounds[0] + mesh.bounds[1]) / 2
    yz_grid = spl_calc.create_adaptive_plane_grid(
        "YZ-plane",
        height=mid_x,
        spacing=spacing,
        offset=offset,
    )
    yz_grid = spl_calc.compute_spl_on_grid(yz_grid, B, source)
    yz_grid = energy_calc.compute_fields_on_grid(yz_grid, B, source)
    yz_path = case.output_dir / f"spl_yz_x{mid_x:.2f}.vtk"
    yz_grid.save(yz_path)
    print(f"  Saved: {yz_path}")

    summary_path = case.output_dir / "summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"case: {case.name}",
                f"mesh: {case.mesh_path}",
                f"view_factors: {case.vf_path}",
                f"source: {source.tolist()}",
                f"power_W: {power}",
                f"check_obstruction: {case.check_obstruction}",
                f"cells: {mesh.n_cells}",
                f"mean_B_Wm2: {np.mean(B):.8e}",
                f"mean_alpha: {alpha.mean():.8e}",
                f"surface_energy: {surface_path}",
                f"spl_xy: {xy_path}",
                f"spl_xz: {xz_path}",
                f"spl_yz: {yz_path}",
            ]
        ),
        encoding="utf-8",
    )
    print(f"\n[DONE] Files saved to {case.output_dir}")


def main() -> int:
    args = parse_args()
    source = np.asarray(args.source, dtype=float)

    for case in selected_cases(args.case):
        run_case(
            case=case,
            source=source,
            power=args.power,
            spacing=args.spacing,
            offset=args.offset,
            xy_height=args.xy_height,
        )

    print("\nAll requested acoustic cases are complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
