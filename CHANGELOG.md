# Changelog

## Unreleased

### Changed

- Replaced legacy VTK examples with `room.msh` and `a320.msh`, both containing
  Gmsh Physical Surface IDs

## 0.1.0 - 2026-06-15

### Added

- Gmsh Physical Surface selection for material groups
- Automatic inward-normal preparation
- View-factor progress and log display in the GUI
- Point and standard-plane receivers
- Receiver selection before solver execution
- Solver progress display
- PyVista SPL contour visualization
- CPU-only Windows packaging and Inno Setup installer

### Changed

- Migrated the computation workflow to the `mes.ret` core
- Reduced the Windows installer to approximately 150 MB
