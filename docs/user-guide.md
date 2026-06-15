# User Guide

## 1. Prepare The Mesh

Create a closed, triangulated surface mesh in Gmsh. Define Physical Surfaces
for material regions such as walls, floor, ceiling, windows, and seats before
exporting the `.msh` file.

MES-Acoustic uses these Physical Surface IDs to assign material absorption.
Surface normals are checked and corrected automatically for RET view-factor
calculation.

The repository includes two ready-to-load examples:

- `geo/room.msh` for a compact room test
- `geo/a320.msh` for a larger aircraft-cabin test

Start with `room.msh` when testing view-factor computation because its mesh is
substantially smaller.

## 2. Create A Project

1. Select **File > New Project**.
2. Open `geo/room.msh`, `geo/a320.msh`, or another compatible mesh.
3. Confirm that the geometry appears in the PyVista viewport.
4. In **Initial Setup**, right-click **Geometric information** and select
   **Mesh Data** to inspect the imported mesh.
5. Right-click **Verify Geometry** and select **Calculate** to inspect the
   prepared geometry and normals.

## 3. Assign Surface Absorption

1. Right-click **Surface Properties**.
2. Select **Add Material Group** and enter a group name.
3. Right-click the new group and select **Select Physical Surface**.
4. Select one or more Physical Surface IDs.
5. Right-click the group again and select **Add Absorption**.
6. Enter an absorption coefficient between `0` and `1`.

Repeat these steps until every relevant surface has the intended material
coefficient.

## 4. Compute Or Load View Factors

To compute a new matrix:

1. Right-click **VF-Computation**.
2. Select **New Compute**.
3. Choose the destination for the `.npy` matrix.
4. Monitor progress and detailed messages in the computation dialog.

To reuse an existing matrix, right-click **VF-Computation**, select
**Load VF**, and choose a `.npy` or compatible view-factor file.

The view-factor matrix must have shape `(number of cells, number of cells)` for
the currently loaded mesh.

## 5. Add Sources

1. Open the **Run Analysis** tab.
2. Right-click **Source** and select **Add Source**.
3. Right-click the new source and select **Add source position**.
4. Enter its X, Y, and Z coordinates.
5. Right-click it again and select **Add source power**.
6. Enter source power in watts.

Multiple point sources can be added. The steady-state solver combines their
incoherent contributions.

## 6. Add Receivers

### Point Receiver

1. Right-click **Receiver** and select **Add point-receiver**.
2. Right-click the new receiver and select **Set point**.
3. Enter its X, Y, and Z coordinates.

### Plane Receiver

1. Right-click **Receiver** and select **Add plane-receiver**.
2. Right-click the new plane receiver.
3. Select **Define-(XY/XZ/YZ)**.
4. Choose `XY-plane`, `XZ-plane`, or `YZ-plane`.
5. Enter the plane height:
   - XY uses Z
   - XZ uses Y
   - YZ uses X
6. Enter the grid spacing.

The receiver grid is clipped to the valid region inside the loaded mesh.

## 7. Run The Solver

1. Right-click **Run solver**.
2. Select **Start Solver**.
3. Check the receiver groups to calculate.
4. Select **OK**.
5. Monitor the progress bar, active receiver, and solver phase in the dialog.

Geometry, a matching view-factor matrix, at least one complete source, and at
least one configured receiver are required.

## 8. View Results

Open the **Post-Process** tab after the solver completes.

For a plane receiver, right-click its `SPL - <receiver>` result and select
**Show SPL - Contour Plot**. Use **Remove SPL - Contour Plot** or
**Clear All Plots** to remove displayed results.

The toolbar provides mesh-edge, section, standard-view, axis, and contour
configuration controls.

## 9. Save The Project

Select **File > Save** and choose a JSON filename. MES-Acoustic creates a
matching `<project-name>_files` directory for associated view-factor data.

Keep the JSON file, its data directory, and the referenced mesh together.
