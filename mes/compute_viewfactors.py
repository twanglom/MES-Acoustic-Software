"""Compute view factors for the room-acoustic mesh cases available in geo/."""

from dataclasses import dataclass
from pathlib import Path
import argparse
import sys

import numpy as np

try:
    from .ret.geometry import MeshProcessor, ViewFactorCalculator
except ImportError:
    from ret.geometry import MeshProcessor, ViewFactorCalculator

BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ViewFactorCase:
    name: str
    mesh_path: Path
    vf_path: Path
    skip_obstruction: bool


CASES = {
    "base": ViewFactorCase(
        name="base",
        mesh_path=BASE_DIR / "geo/base.msh",
        vf_path=BASE_DIR / "geo/base_nominal_vf.npy",
        skip_obstruction=True,
    ),
    "obs": ViewFactorCase(
        name="obs",
        mesh_path=BASE_DIR / "geo/obs.msh",
        vf_path=BASE_DIR / "geo/obs_nominal_vf_check.npy",
        skip_obstruction=False,
    ),
}

AVAILABLE_CASES = {
    name: case
    for name, case in CASES.items()
    if case.mesh_path.exists()
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute or load RET view factors for available mesh cases."
    )
    parser.add_argument(
        "--case",
        choices=["all", *AVAILABLE_CASES.keys()],
        default="all",
        help="Which case to process.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even when the .npy view-factor file already exists.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-3,
        help="Numerical tolerance passed to pyviewfactor.",
    )
    return parser.parse_args()


def selected_cases(case_name: str) -> list[ViewFactorCase]:
    if case_name == "all":
        return list(AVAILABLE_CASES.values())
    return [AVAILABLE_CASES[case_name]]


def validate_view_factors(vf_calc: ViewFactorCalculator) -> None:
    results = vf_calc.validate()

    if results["row_sum_mean"] > 0.95:
        print("  [OK] View factors valid for closed enclosure")
    elif results["row_sum_mean"] > 0.85:
        print("  [WARNING] Row sums slightly low - acceptable")
    else:
        print("  [ERROR] Row sums too low - mesh may not be closed or visibility is wrong")


def run_case(case: ViewFactorCase, force: bool, epsilon: float) -> None:
    print("\n" + "=" * 70)
    print(f"CASE: {case.name}")
    print("=" * 70)

    if not case.mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {case.mesh_path}")

    print("[1] Load and prepare mesh")
    processor = MeshProcessor(str(case.mesh_path))
    mesh = processor.prepare_geometry()
    info = processor.get_mesh_info()

    print(f"  Mesh: {case.mesh_path}")
    print(f"  Cells: {info['n_cells']}")
    print(f"  Surface area: {info['total_area']:.3f} m^2")
    print(f"  Bounds: {tuple(round(v, 3) for v in info['bounds'])}")
    print(f"  Normals flipped: {info['normals_flipped']}")

    print("\n[2] Load or compute view factors")
    vf_calc = ViewFactorCalculator(mesh)

    if case.vf_path.exists() and not force:
        F = vf_calc.load(str(case.vf_path))
        print(f"  Loaded existing file: {case.vf_path}")
    else:
        print(f"  Obstruction check: {'disabled (convex)' if case.skip_obstruction else 'enabled (non-convex)'}")
        F = vf_calc.compute(
            epsilon=epsilon,
            skip_visibility=False,
            skip_obstruction=case.skip_obstruction,
            verbose=True,
        )
        case.vf_path.parent.mkdir(parents=True, exist_ok=True)
        vf_calc.save(str(case.vf_path))

    print(f"  Matrix shape: {F.shape}")
    print(f"  Memory: {F.nbytes / 1e6:.2f} MB")

    print("\n[3] Validate")
    validate_view_factors(vf_calc)
    print(f"\n[DONE] View factors ready: {case.vf_path}")


def main() -> int:
    args = parse_args()

    for case in selected_cases(args.case):
        run_case(case, force=args.force, epsilon=args.epsilon)

    print("\nAll requested view-factor cases are complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
