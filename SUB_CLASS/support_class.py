import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFrame, QHBoxLayout, QSplitter, 
    QTabWidget, QTreeView, QVBoxLayout, QWidget, QStyle, QToolBar, 
    QLabel, QMenu, QFileDialog, QDialog, QPushButton, QTableWidget, 
    QTableWidgetItem, QMessageBox, QListWidget, QListWidgetItem,
    QInputDialog, QLineEdit, QDialogButtonBox, QFileDialog, QColorDialog,
    QComboBox, QSpinBox, QSlider, QProgressBar, QTextEdit
)


from PySide6.QtGui import (
    QIcon, QStandardItemModel, QStandardItem, QAction, QTextCursor
)
from PySide6.QtCore import Qt, QThread, Signal, QObject,  QEventLoop
import queue
import contextlib
import io
import re

from pyvista import themes
import numpy as np
import time

from mes.ret.config import RETConfig
from mes.ret.geometry import ViewFactorCalculator
from mes.ret.RETsolver import RadiativeEnergyTransfer

import json
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class AnalysisWorker(QObject):
    finished = Signal()
    failed = Signal(str)
    status_changed = Signal(str)
    result_ready = Signal(int, float)
    progress_updated = Signal(int, int)  # Move this signal to the worker

    def __init__(self, source_pos_matrix, source_power, receiver_positions, F, shape, cell_center, normal_vector, cell_area, absorptivity_matrix, rho_air, sound_speed, attenuation, check_obstruction=True):
        super().__init__()
        self.source_pos_matrix = source_pos_matrix
        self.source_power = source_power
        self.receiver_positions = receiver_positions
        self.F = F
        self.shape = shape
        self.cell_center = cell_center
        self.normal_vector = normal_vector
        self.cell_area = cell_area
        self.absorptivity_matrix = absorptivity_matrix
        self.rho_air = rho_air
        self.sound_speed = sound_speed
        self.attenuation = attenuation
        self.check_obstruction = check_obstruction
        self.should_stop = False

    def run(self):
        try:
            source_positions = np.asarray(self.source_pos_matrix, dtype=float)
            source_powers = np.asarray(self.source_power, dtype=float)
            receiver_positions = np.asarray(self.receiver_positions, dtype=float)

            if source_positions.size == 0:
                raise ValueError("At least one source is required")
            if receiver_positions.size == 0:
                raise ValueError("At least one receiver is required")
            if self.F is None:
                raise ValueError("View factors have not been loaded or computed")

            self.status_changed.emit("Preparing acoustic solver...")
            cfg = RETConfig(
                c=self.sound_speed,
                rho0=self.rho_air,
                W=float(np.sum(source_powers)),
                m_air=self.attenuation,
                check_obstruction=self.check_obstruction,
                show_progress=False,
            )
            solver = RadiativeEnergyTransfer(
                mesh=self.shape,
                view_factors=np.asarray(self.F, dtype=float),
                alpha=np.asarray(self.absorptivity_matrix, dtype=float),
                cfg=cfg,
                skip_preprocessing=True,
            )
            self.status_changed.emit("Solving steady-state surface energy...")
            B_sources = solver.solve_steady_state_sources(
                source_positions,
                source_powers,
            )

            total_receivers = len(receiver_positions)
            self.status_changed.emit(
                f"Computing SPL at {total_receivers} receiver points..."
            )
            for i, pos in enumerate(receiver_positions):
                if self.should_stop:
                    break
                spl = solver.compute_SPL_steady_sources(
                    B_sources,
                    source_positions,
                    source_powers,
                    pos,
                )
                self.result_ready.emit(i, spl)
                self.progress_updated.emit(i + 1, total_receivers)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()

    def stop(self):
        self.should_stop = True




class MESProgressDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Computing SPL")
        self.setFixedSize(390, 175)
        layout = QVBoxLayout()
        
        self.label = QLabel("Process Calculating...", self)
        self.label.setWordWrap(True)
        layout.addWidget(self.label)

        self.phase_label = QLabel("Preparing solver...", self)
        self.phase_label.setWordWrap(True)
        layout.addWidget(self.phase_label)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)

        self.progress_text = QLabel("0/0 receivers (0%)", self)
        self.progress_text.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_text)
        
        self.setLayout(layout)
        self.setModal(True)

    def update_label(self, text):
        self.label.setText(text)

    def update_phase(self, text):
        self.phase_label.setText(text)

    def set_total(self, total):
        total = max(1, int(total))
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(0)
        self.progress_text.setText(f"0/{total} receivers (0%)")

    def update_progress(self, value):
        value = int(value)
        total = self.progress_bar.maximum()
        self.progress_bar.setValue(value)
        percent = round(100 * value / total) if total else 0
        self.progress_text.setText(
            f"{value}/{total} receivers ({percent}%)"
        )





class VFComputationThread(QThread):
    completed = Signal(str)
    failed = Signal(str)
    status_changed = Signal(str)
    log_message = Signal(str)
    progress_changed = Signal(int)

    def __init__(self, shape, file_path, parent=None, skip_obstruction=False):
        super().__init__(parent)
        self.shape = shape
        self.file_path = file_path
        self.skip_obstruction = skip_obstruction

    def run(self):
        try:
            stream = _VFOutputStream(self._handle_output)
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                self.status_changed.emit(
                    f"Preparing {self.shape.n_cells} surface cells..."
                )
                calculator = ViewFactorCalculator(self.shape)
                self.F = calculator.compute(
                    skip_obstruction=self.skip_obstruction,
                    verbose=True,
                )
                self.status_changed.emit("Saving view-factor matrix...")
                calculator.save(self.file_path)
                stream.flush()
            self.progress_changed.emit(100)
            self.completed.emit(self.file_path)
        except Exception as e:
            self.failed.emit(str(e))

    def _handle_output(self, text):
        text = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text).strip()
        if not text:
            return

        progress_match = re.search(r"(\d{1,3})%", text)
        if progress_match:
            self.progress_changed.emit(
                min(100, int(progress_match.group(1)))
            )
            self.status_changed.emit(text)
        else:
            self.log_message.emit(text)
            self.status_changed.emit(text)


class _VFOutputStream(io.TextIOBase):
    """Convert print/tqdm output from pyviewfactor into complete messages."""

    def __init__(self, callback):
        super().__init__()
        self.callback = callback
        self.buffer = ""

    def writable(self):
        return True

    def write(self, text):
        if not text:
            return 0

        self.buffer += str(text)
        parts = re.split(r"[\r\n]+", self.buffer)
        self.buffer = parts.pop()
        for part in parts:
            if part.strip():
                self.callback(part)

        # tqdm often refreshes a full line using carriage returns.
        if "\r" in str(text) and self.buffer.strip():
            self.callback(self.buffer)
            self.buffer = ""
        return len(text)

    def flush(self):
        if self.buffer.strip():
            self.callback(self.buffer)
        self.buffer = ""



class VFProgressDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Computing View Factors")
        self.resize(620, 320)
        layout = QVBoxLayout()

        self.label = QLabel("Preparing view-factor computation...", self)
        self.label.setWordWrap(True)
        layout.addWidget(self.label)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 0)
        layout.addWidget(self.progress_bar)

        self.log_output = QTextEdit(self)
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText(
            "View-factor progress and validation messages appear here."
        )
        layout.addWidget(self.log_output)

        self.setLayout(layout)
        self.setModal(True)

    def update_status(self, text):
        self.label.setText(text)

    def update_progress(self, value):
        if self.progress_bar.minimum() == 0 and self.progress_bar.maximum() == 0:
            self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(value)

    def append_log(self, text):
        self.log_output.append(text)
        document = self.log_output.document()
        if document.blockCount() > 200:
            cursor = self.log_output.textCursor()
            cursor.movePosition(QTextCursor.Start)
            cursor.select(QTextCursor.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()
        self.log_output.moveCursor(QTextCursor.End)



class VFload:
    def __init__(self, file_name):
        self.file_name = file_name
        
    def load_vf_file(self):
        try:
            if str(self.file_name).lower().endswith(".npy"):
                vf_file = np.load(self.file_name)
            else:
                vf_file = np.loadtxt(self.file_name)
            if vf_file.ndim != 2 or vf_file.shape[0] != vf_file.shape[1]:
                raise ValueError("View-factor matrix must be square")
            return vf_file
        except FileNotFoundError:
            print(f"Error: The file {self.file_name} was not found.")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None







class GeometryInfoDialog(QDialog):
    def __init__(self, total_elements, vertices, bound_x, bound_y, bound_z, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Geometry Information")
        self.setGeometry(100, 100, 200, 250)  # Adjust the size as needed

        layout = QVBoxLayout()

        # Create a table widget with 5 rows and 2 columns
        table = QTableWidget(5, 2)  # 5 rows for the properties, and 2 columns for label and value
        table.verticalHeader().setVisible(False)  # Hide vertical header
        table.horizontalHeader().setVisible(False)  # Optionally hide horizontal header
        table.setShowGrid(False)  # We will use stylesheet for grid
        table.setEditTriggers(QTableWidget.NoEditTriggers)  # Make table read-only

        # Use a stylesheet to set the border for cells
        table.setStyleSheet(
            "QTableWidget {"
            "    border: 1px solid black;"
            "}"
            "QTableWidget::item {"
            "    border: 1px solid black;"
            "}"
            "QTableWidget::item:first-column {"
            "    font-weight: bold;"
            "}"
        )

        # Populate the table with items
        properties = ["Total Elements", "Vertices", "Bound X", "Bound Y", "Bound Z"]
        values = [str(total_elements), str(vertices), f"{bound_x[0]}, {bound_x[1]}", 
                  f"{bound_y[0]}, {bound_y[1]}", f"{bound_z[0]}, {bound_z[1]}"]

        for i, property_name in enumerate(properties):
            header_item = QTableWidgetItem(property_name)
            header_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            table.setItem(i, 0, header_item)

            value_item = QTableWidgetItem(values[i])
            value_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            table.setItem(i, 1, value_item)

        # Resize columns to content
        table.resizeColumnToContents(0)  # Adjust the first column width
        table.horizontalHeader().setStretchLastSection(True)  # Stretch the last column to fill the dialog

        # Add the table to the layout
        layout.addWidget(table)

        # Button to close the dialog
        close_button = QPushButton("Close", self)
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)

        self.setLayout(layout)



class CellSelectionDialog(QDialog):
    selectionChanged = Signal(list)

    def __init__(self, available_cells, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Cells")
        self.setFixedSize(200, 400)
        layout = QVBoxLayout(self)

        self.list_widget = QListWidget(self)
        self.list_widget.setSelectionMode(QListWidget.MultiSelection)
        for i in available_cells:  # available_cells is already sorted
            self.list_widget.addItem(str(i))
        layout.addWidget(self.list_widget)

        self.list_widget.itemSelectionChanged.connect(self.emit_selection_changed)

        ok_btn = QPushButton("OK", self)
        ok_btn.clicked.connect(self.accept)
        layout.addWidget(ok_btn)

        cancel_btn = QPushButton("Cancel", self)
        cancel_btn.clicked.connect(self.reject)
        layout.addWidget(cancel_btn)

    def emit_selection_changed(self):
        selected_indices = self.selected_indices()
        self.selectionChanged.emit(selected_indices)

    def selected_indices(self):
        return [int(item.text()) for item in self.list_widget.selectedItems()]


class PhysicalSurfaceSelectionDialog(QDialog):
    def __init__(self, surfaces, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Physical Surfaces")
        self.setMinimumSize(320, 360)

        layout = QVBoxLayout(self)
        self.list_widget = QListWidget(self)
        self.list_widget.setSelectionMode(QListWidget.MultiSelection)

        for physical_id, name, cell_count in surfaces:
            item = QListWidgetItem(
                f"{physical_id}: {name} ({cell_count} cells)"
            )
            item.setData(Qt.UserRole, int(physical_id))
            self.list_widget.addItem(item)

        layout.addWidget(self.list_widget)

        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            self,
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def selected_ids(self):
        return [
            int(item.data(Qt.UserRole))
            for item in self.list_widget.selectedItems()
        ]


class ReceiverSelectionDialog(QDialog):
    def __init__(self, receiver_groups, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Receivers to Run")
        self.setMinimumSize(380, 360)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select receiver groups for this analysis:"))

        self.list_widget = QListWidget(self)
        for index, group in enumerate(receiver_groups):
            name = group.get("name", f"Receiver {index + 1}")
            position = group.get("position")
            if name.startswith("point-"):
                detail = "1 point" if position is not None else "not configured"
            else:
                detail = (
                    f"{len(position)} points"
                    if position is not None
                    else "not configured"
                )

            item = QListWidgetItem(f"{name} ({detail})")
            item.setData(Qt.UserRole, index)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget)

        selection_buttons = QHBoxLayout()
        select_all_button = QPushButton("Select All", self)
        select_all_button.clicked.connect(
            lambda: self.set_all_checked(True)
        )
        selection_buttons.addWidget(select_all_button)

        clear_button = QPushButton("Clear", self)
        clear_button.clicked.connect(
            lambda: self.set_all_checked(False)
        )
        selection_buttons.addWidget(clear_button)
        layout.addLayout(selection_buttons)

        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            self,
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def set_all_checked(self, checked):
        state = Qt.Checked if checked else Qt.Unchecked
        for row in range(self.list_widget.count()):
            self.list_widget.item(row).setCheckState(state)

    def selected_indices(self):
        return [
            int(item.data(Qt.UserRole))
            for row in range(self.list_widget.count())
            if (item := self.list_widget.item(row)).checkState() == Qt.Checked
        ]




class SourcePositionDialog(QDialog):
    # Static class attributes to hold the default/last entered values
    last_values = [0.1, 0.1, 0.1]  # Updated to use a default that makes sense contextually

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Input Source Position")
        self.setModal(False)

        layout = QVBoxLayout(self)

        self.x_input = QLineEdit(self)
        self.y_input = QLineEdit(self)
        self.z_input = QLineEdit(self)

        # Initialize with last saved values from static class attributes
        self.x_input.setText(str(SourcePositionDialog.last_values[0]))
        self.y_input.setText(str(SourcePositionDialog.last_values[1]))
        self.z_input.setText(str(SourcePositionDialog.last_values[2]))

        layout.addWidget(QLabel("X:"))
        layout.addWidget(self.x_input)
        layout.addWidget(QLabel("Y:"))
        layout.addWidget(self.y_input)
        layout.addWidget(QLabel("Z:"))
        layout.addWidget(self.z_input)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        layout.addWidget(buttonBox)

    def accept(self):
        # Update the class attribute with the current values before accepting
        SourcePositionDialog.last_values = [float(self.x_input.text()), float(self.y_input.text()), float(self.z_input.text())]
        super().accept()

    def getValues(self):
        return (SourcePositionDialog.last_values[0], SourcePositionDialog.last_values[1], SourcePositionDialog.last_values[2])




class Point_ReceiverPositionDialog(QDialog):
    # Static class attributes to hold the default/last entered values
    last_values = [0.0, 0.0, 0.0]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Input Receiver Position")
        self.setModal(False)

        layout = QVBoxLayout(self)

        self.x_input = QLineEdit(self)
        self.y_input = QLineEdit(self)
        self.z_input = QLineEdit(self)

        # Initialize with last saved values from static class attributes
        self.x_input.setText(str(Point_ReceiverPositionDialog.last_values[0]))
        self.y_input.setText(str(Point_ReceiverPositionDialog.last_values[1]))
        self.z_input.setText(str(Point_ReceiverPositionDialog.last_values[2]))

        layout.addWidget(QLabel("X:"))
        layout.addWidget(self.x_input)
        layout.addWidget(QLabel("Y:"))
        layout.addWidget(self.y_input)
        layout.addWidget(QLabel("Z:"))
        layout.addWidget(self.z_input)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        layout.addWidget(buttonBox)

    def accept(self):
        # Update the class attribute with the current values before accepting
        Point_ReceiverPositionDialog.last_values = [float(self.x_input.text()), float(self.y_input.text()), float(self.z_input.text())]
        super().accept()

    def getValues(self):
        return (Point_ReceiverPositionDialog.last_values[0], Point_ReceiverPositionDialog.last_values[1], Point_ReceiverPositionDialog.last_values[2])




class Plane_GridPositionDialog_vertics(QDialog):
    # Static class attributes to hold the default/last entered values
    last_vertices = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    last_spacing = 0.1

    def __init__(self, parent=None, ):
        super().__init__(parent)
        self.setWindowTitle("Input Plane Grid Parameters")
        self.setModal(False)

        layout = QVBoxLayout(self)

        # Vertex inputs are organized by vertex
        self.vertex_inputs = []
        for i in range(4):
            vertex_layout = QHBoxLayout()
            vertex_label = QLabel(f"V{i}:")
            vertex_layout.addWidget(vertex_label)
            x_input = QLineEdit(self)
            y_input = QLineEdit(self)
            z_input = QLineEdit(self)
            # Initialize with last saved values from static class attributes
            x_input.setText(str(Plane_GridPositionDialog_vertics.last_vertices[i][0]))
            y_input.setText(str(Plane_GridPositionDialog_vertics.last_vertices[i][1]))
            z_input.setText(str(Plane_GridPositionDialog_vertics.last_vertices[i][2]))
            vertex_layout.addWidget(QLabel("X:"))
            vertex_layout.addWidget(x_input)
            vertex_layout.addWidget(QLabel("Y:"))
            vertex_layout.addWidget(y_input)
            vertex_layout.addWidget(QLabel("Z:"))
            vertex_layout.addWidget(z_input)
            self.vertex_inputs.append((x_input, y_input, z_input))
            layout.addLayout(vertex_layout)

        # Grid spacing input
        spacing_layout = QHBoxLayout()
        self.spacing_input = QLineEdit(self)
        self.spacing_input.setText(str(Plane_GridPositionDialog_vertics.last_spacing))
        spacing_layout.addWidget(QLabel("Grid Space:"))
        spacing_layout.addWidget(self.spacing_input)
        layout.addLayout(spacing_layout)

        # Buttons
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        layout.addWidget(buttonBox)

    def accept(self):
        # Update the class attribute with the current values before accepting
        Plane_GridPositionDialog_vertics.last_vertices = [(float(x.text()), float(y.text()), float(z.text())) for x, y, z in self.vertex_inputs]
        Plane_GridPositionDialog_vertics.last_spacing = float(self.spacing_input.text())
        super().accept()

    def getValues(self):
        return Plane_GridPositionDialog_vertics.last_vertices, Plane_GridPositionDialog_vertics.last_spacing





class Plane_GridPositionDialog_standard_plane(QDialog):
    last_plane = "XY-plane"
    last_height = 0.0
    last_spacing = 0.1

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Plane and Grid Parameters")
        self.setModal(False)

        layout = QVBoxLayout(self)

        # Plane selection dropdown
        plane_layout = QHBoxLayout()
        plane_label = QLabel("Select Plane:")
        self.plane_select = QComboBox(self)
        self.plane_select.addItems(["XY-plane", "XZ-plane", "YZ-plane"])
        self.plane_select.setCurrentText(Plane_GridPositionDialog_standard_plane.last_plane)
        plane_layout.addWidget(plane_label)
        plane_layout.addWidget(self.plane_select)
        layout.addLayout(plane_layout)

        # Plane height input (Z for XY, Y for XZ, X for YZ)
        height_layout = QHBoxLayout()
        self.height_input = QLineEdit(self)
        self.height_input.setText(str(Plane_GridPositionDialog_standard_plane.last_height))
        height_layout.addWidget(QLabel("Height (Z/Y/X):"))
        height_layout.addWidget(self.height_input)
        layout.addLayout(height_layout)

        # Grid spacing input
        spacing_layout = QHBoxLayout()
        self.spacing_input = QLineEdit(self)
        self.spacing_input.setText(str(Plane_GridPositionDialog_standard_plane.last_spacing))
        spacing_layout.addWidget(QLabel("Grid Space:"))
        spacing_layout.addWidget(self.spacing_input)
        layout.addLayout(spacing_layout)

        # Buttons
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        layout.addWidget(buttonBox)

    def accept(self):
        # Update class attributes with the current values
        Plane_GridPositionDialog_standard_plane.last_plane = self.plane_select.currentText()
        Plane_GridPositionDialog_standard_plane.last_height = float(self.height_input.text())
        Plane_GridPositionDialog_standard_plane.last_spacing = float(self.spacing_input.text())
        super().accept()

    def getValues(self):
        return (
            Plane_GridPositionDialog_standard_plane.last_plane,
            Plane_GridPositionDialog_standard_plane.last_height,
            Plane_GridPositionDialog_standard_plane.last_spacing,
        )




import numpy as np
from scipy import interpolate
import pyvista as pv

class SPLVisualization:
    def __init__(self, shape, plot_interactor):
        self.shape = shape
        self.plot_interactor = plot_interactor
        self.mesh_actors = {}
        self.contour_actors = {}

    def create_2d_grid(self, receiver_positions, density_factor=50):
        x, y, z = receiver_positions.T
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        z_min, z_max = z.min(), z.max()

        variances = [x_max - x_min, y_max - y_min, z_max - z_min]
        dims = np.argsort(variances)[-2:]
        dims.sort()

        grid_min = [x_min, y_min, z_min]
        grid_max = [x_max, y_max, z_max]
        
        nx = max(int((grid_max[dims[0]] - grid_min[dims[0]]) * density_factor), 2)
        ny = max(int((grid_max[dims[1]] - grid_min[dims[1]]) * density_factor), 2)
        
        grid_x = np.linspace(grid_min[dims[0]], grid_max[dims[0]], nx)
        grid_y = np.linspace(grid_min[dims[1]], grid_max[dims[1]], ny)
        
        xx, yy = np.meshgrid(grid_x, grid_y)
        
        grid_points = np.zeros((xx.size, 3))
        grid_points[:, dims[0]] = xx.ravel()
        grid_points[:, dims[1]] = yy.ravel()
        grid_points[:, 3 - dims[0] - dims[1]] = np.mean([grid_min[3 - dims[0] - dims[1]], grid_max[3 - dims[0] - dims[1]]])

        plane_grid = pv.StructuredGrid(grid_points[:, 0].reshape(ny, nx),
                                       grid_points[:, 1].reshape(ny, nx),
                                       grid_points[:, 2].reshape(ny, nx))
        
        print(f"Created grid with {plane_grid.n_points} points")
        return plane_grid

    def interpolate_spl_data(self, receiver_positions, SPL_data, plane_grid_points, method='cubic'):
        try:
            variances = np.var(receiver_positions, axis=0)
            dims = np.argsort(variances)[-2:]
            dims.sort()
            
            interpolated_spl = griddata(receiver_positions[:, dims], SPL_data, 
                                        plane_grid_points[:, dims], method=method)
            
            if np.any(np.isnan(interpolated_spl)):
                nn_interpolated = griddata(receiver_positions[:, dims], SPL_data, 
                                           plane_grid_points[:, dims], method='nearest')
                interpolated_spl[np.isnan(interpolated_spl)] = nn_interpolated[np.isnan(interpolated_spl)]
            
            return interpolated_spl
        except Exception as e:
            print(f"Error during interpolation: {str(e)}")
            return None

    def mask_obstacles(self, plane_grid):
        distance = plane_grid.compute_implicit_distance(self.shape)
        return distance['implicit_distance'] >= 0






    def add_contour_plot(self, plane_grid, mask, contour_config, group_name):
        cmap_name = contour_config.get("cmap", "jet")
        contour_lines = contour_config.get("contour_lines", 10)
        opacity = contour_config.get("opacity", 1.0)
        
        spl_data = plane_grid['SPL']
        spl_data[~mask] = np.nan
        plane_grid['SPL'] = spl_data

        print(f"Total points: {plane_grid.n_points}")
        print(f"Valid points: {np.sum(~np.isnan(spl_data))}")

        valid_data = np.any(~np.isnan(spl_data))

        if valid_data:
            # Create a custom colormap with white for NaN values
            cmap = plt.get_cmap(cmap_name)
            colors = cmap(np.linspace(0, 1, 256))
            custom_cmap = ListedColormap(colors)
            custom_cmap.set_bad('white')  # Set NaN values to white

            # Add mesh with custom colormap
            mesh_actor = self.plot_interactor.add_mesh(plane_grid, scalars="SPL", cmap=custom_cmap, 
                                                       show_edges=False, opacity=opacity,
                                                       nan_color='white') 
            self.mesh_actors[group_name] = mesh_actor
            
            # Add contour lines
            # contour = plane_grid.contour(isosurfaces=contour_lines, scalars="SPL")
            # if contour.n_points > 0:
            #     contour_actor = self.plot_interactor.add_mesh(contour, color="k", line_width=1.5, opacity=1.0)
            #     self.contour_actors[group_name] = contour_actor
            # else:
            #     print(f"Warning: Contour plot could not be created for {group_name} due to insufficient valid data points.")

        else:
            print(f"Warning: No valid data points for plotting {group_name}. The plot will be empty.")


        # ######## For transparent contour
        # if valid_data:
        #     # Create a custom colormap with white for NaN values
        #     cmap = plt.get_cmap(cmap_name)
        #     colors = cmap(np.linspace(0, 1, 256))
        #     custom_cmap = ListedColormap(colors)
        #     custom_cmap.set_bad('white', alpha=0)  # Set NaN values to white

        #     # Add mesh with custom colormap
        #     mesh_actor = self.plot_interactor.add_mesh(plane_grid, scalars="SPL", cmap=custom_cmap, 
        #                                                show_edges=False, opacity=opacity,
        #                                                nan_color='white', nan_opacity=0) 
        #     self.mesh_actors[group_name] = mesh_actor
            
        #     # Add contour lines
        #     contour = plane_grid.contour(isosurfaces=contour_lines, scalars="SPL")
        #     if contour.n_points > 0:
        #         contour_actor = self.plot_interactor.add_mesh(contour, color="k", line_width=1.5, opacity=1.0)
        #         self.contour_actors[group_name] = contour_actor
        #     else:
        #         print(f"Warning: Contour plot could not be created for {group_name} due to insufficient valid data points.")

        # else:
        #     print(f"Warning: No valid data points for plotting {group_name}. The plot will be empty.")





    def remove_contour_plot(self, group_name):
        if group_name in self.mesh_actors:
            self.plot_interactor.remove_actor(self.mesh_actors[group_name])
            del self.mesh_actors[group_name]
        if group_name in self.contour_actors:
            self.plot_interactor.remove_actor(self.contour_actors[group_name])
            del self.contour_actors[group_name]

    def clear_plot(self):
        for actor in self.mesh_actors.values():
            self.plot_interactor.remove_actor(actor)
        for actor in self.contour_actors.values():
            self.plot_interactor.remove_actor(actor)
        self.mesh_actors.clear()
        self.contour_actors.clear()

    def update_plot(self):
        self.plot_interactor.update()















class ContourEditorDialog(QDialog):
    values_changed = Signal(dict)

    def __init__(self, parent=None, initial_values=None):
        super().__init__(parent)
        self.setWindowTitle("Contour Config")
        self.setFixedSize(200, 250)
        self.setWindowFlags(self.windowFlags() | Qt.Window)  # Make the dialog a window

        layout = QVBoxLayout(self)

        cmap_label = QLabel("Colormap:", self)
        self.cmap_combo = QComboBox(self)
        self.cmap_combo.addItems(["jet", "viridis", "plasma", "inferno", "magma", "cividis"])
        layout.addWidget(cmap_label)
        layout.addWidget(self.cmap_combo)

        contour_label = QLabel("Number of Contour Lines:", self)
        self.contour_spinbox = QSpinBox(self)
        self.contour_spinbox.setRange(1, 100)
        self.contour_spinbox.setValue(10)
        layout.addWidget(contour_label)
        layout.addWidget(self.contour_spinbox)

        opacity_label = QLabel("Opacity:", self)
        self.opacity_slider = QSlider(Qt.Horizontal, self)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        layout.addWidget(opacity_label)
        layout.addWidget(self.opacity_slider)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.cmap_combo.currentIndexChanged.connect(self.emit_values_changed)
        self.contour_spinbox.valueChanged.connect(self.emit_values_changed)
        self.opacity_slider.valueChanged.connect(self.emit_values_changed)

        # Set initial values if provided
        if initial_values:
            self.set_values(initial_values)

    def emit_values_changed(self):
        self.values_changed.emit(self.get_values())

    def get_values(self):
        return {
            "cmap": self.cmap_combo.currentText(),
            "contour_lines": self.contour_spinbox.value(),
            "opacity": self.opacity_slider.value() / 100.0
        }

    def set_values(self, values):
        self.cmap_combo.setCurrentText(values.get("cmap", "jet"))
        self.contour_spinbox.setValue(values.get("contour_lines", 10))
        self.opacity_slider.setValue(int(values.get("opacity", 1.0) * 100))





class BlackgroundColor:
    def __init__(self, parent=None):
        self.parent = parent
        self.main_color = None
        self.light_color = None

    def select_colors(self):
        # Show a color picker dialog for the main color
        main_color_dialog = QColorDialog(self.parent)
        main_color_dialog.setWindowTitle("Main Color")
        self.main_color = main_color_dialog.getColor()

        if not self.main_color.isValid():
            # If the user cancels or doesn't select a color
            QMessageBox.warning(self.parent, "No Color Selected", "Please select a valid main color!")
            return False

        # Show a color picker dialog for the light (top) color
        light_color_dialog = QColorDialog(self.parent)
        light_color_dialog.setWindowTitle("Top Color")
        self.light_color = light_color_dialog.getColor()

        if not self.light_color.isValid():
            # If the user cancels or doesn't select a color
            QMessageBox.warning(self.parent, "No Color Selected", "Please select a valid light color!")
            return False

        return True

    def get_main_color(self):
        return self.main_color.name() if self.main_color else None

    def get_light_color(self):
        return self.light_color.name() if self.light_color else None
    




    


class VectorScaleDialog(QDialog):
    # Signal to emit the updated scale factor value
    scale_changed = Signal(float)

    def __init__(self, initial_scale=0.1, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Adjust Vector Scale")
        self.setFixedSize(300, 150)

        # Layout
        layout = QVBoxLayout(self)

        # Slider label
        self.scale_label = QLabel(f"Scale Factor: {initial_scale}", self)
        layout.addWidget(self.scale_label)

        # Slider to adjust the scale factor
        self.scale_slider = QSlider(Qt.Horizontal, self)
        self.scale_slider.setRange(1, 100)  # Slider values from 1 to 100
        self.scale_slider.setValue(int(initial_scale * 100))  # Set initial value scaled by 100
        self.scale_slider.valueChanged.connect(self.update_scale_label)  # Update label when slider moves
        layout.addWidget(self.scale_slider)

        # OK and Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def update_scale_label(self):
        """Update the scale label and emit the new scale factor."""
        scale_value = self.scale_slider.value() / 100.0  # Convert slider value back to a scale factor
        self.scale_label.setText(f"Scale Factor: {scale_value}")
        self.scale_changed.emit(scale_value)  # Emit the new scale factor
