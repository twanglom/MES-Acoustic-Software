# Development Guide

## Architecture

- `main_run.py`: main PySide6 GUI and application workflow
- `SUB_CLASS/support_class.py`: dialogs, workers, progress UI, and SPL display
- `mes/ret/geometry.py`: mesh preparation and view-factor integration
- `mes/ret/RETsolver.py`: steady-state and time-domain RET solver
- `mes/ret/postprocess.py`: receiver-plane and result utilities
- `mes/ret/config.py`: solver configuration data classes
- `mes/ret/examples`: core API examples and validation cases
- `MES-Acoustic.spec`: CPU-only PyInstaller configuration
- `installer/MES-Acoustic.iss`: Inno Setup installer configuration

## Development Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main_run.py
```

The supported development interpreter is Python 3.11 on 64-bit Windows.

## Mesh Contract

The preferred input is a closed, triangulated Gmsh surface mesh containing
`gmsh:physical` cell data. Physical Surface names and IDs are read by
`MeshProcessor`.

RET view factors require inward-facing surface normals. `prepare_geometry()`
computes cell normals, estimates their direction relative to the enclosure
centroid, and flips them when necessary.

## View-Factor Convention

The view-factor integration layer stores:

```text
F[receiver, emitter] = fraction leaving emitter and arriving at receiver
```

The solver adapts this convention internally. A matrix loaded by the GUI must
match the current mesh cell count.

## Build The Application

```powershell
python -m pip install pyinstaller
pyinstaller --clean --noconfirm MES-Acoustic.spec
```

Output:

```text
dist\MES-Acoustic\MES-Acoustic.exe
```

The spec intentionally excludes CUDA, CuPy, machine-learning frameworks, and
other optional scientific packages. The current view-factor implementation
uses Numba's CPU backend.

## Build The Installer

Install Inno Setup 6, then run:

```powershell
& "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\MES-Acoustic.iss
```

Or run the complete release script:

```powershell
.\build-release.ps1
```

Output is written to `installer-output`.

## Before Submitting Changes

1. Run the GUI from source.
2. Import a Gmsh mesh with Physical Surfaces.
3. Verify surface selection and absorption assignment.
4. Compute or load a matching view-factor matrix.
5. Run at least one point receiver and one plane receiver.
6. Confirm the SPL contour display.
7. For packaging changes, build the installer and test it on a clean machine.
