import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFrame, QHBoxLayout, QSplitter, 
    QTabWidget, QTreeView, QVBoxLayout, QWidget, QStyle, QToolBar, 
    QLabel, QMenu, QFileDialog, QDialog, QPushButton, QTableWidget, 
    QTableWidgetItem, QMessageBox, QListWidget, QInputDialog, QLineEdit,
    QDialogButtonBox, QFileDialog
)
from PySide6.QtGui import QIcon, QStandardItemModel, QStandardItem, QAction
from PySide6.QtCore import Qt, QThread, Signal, QObject

import pyvista as pv
from pyvista import themes
import numpy as np

from MES.VF_COMPUTATION import calculate_view_factors, save_view_factors_to_file
from MES.MES_COMPUTATION import AcousticAnalysis

from scipy.interpolate import griddata



class VFComputationThread(QThread):
    finished = Signal(str)  # Signal to emit when computation is done

    def __init__(self, shape, filepath, parent=None):
        super().__init__(parent)
        self.shape = shape
        self.filepath = filepath

    def run(self):
        try:
            self.F = calculate_view_factors(self.shape)
            filename_save = self.filepath
            save_view_factors_to_file(self.F, filename_save)
            self.finished.emit(filename_save)  # Emit the signal with the filename
        except Exception as e:
            print(f"Error during computation: {e}")
            self.finished.emit("")  # Emit an empty string on failure



class VFProgressDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Computing View Factors")
        self.setFixedSize(200, 80)
        layout = QVBoxLayout()

        self.label = QLabel("Process Calculating...", self)
        layout.addWidget(self.label)

        self.setLayout(layout)
        self.setModal(True)



class MESProgressDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Computing SPL")
        self.setFixedSize(200, 80)
        layout = QVBoxLayout()

        self.label = QLabel("Process Calculating...", self)
        layout.addWidget(self.label)

        self.setLayout(layout)
        self.setModal(True)




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
    selectionChanged = Signal(list)  # Define a new signal to emit selected indices

    def __init__(self, num_cells, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Cells")
        self.setFixedSize(200, 400)
        self.setModal(False)  # Make this dialog non-modal (สามารถ interqction กับ widget อื่นๆได้ ขณะที่หน้าต่างนี้ยังเปิดอยู่)
        layout = QVBoxLayout(self)

        # List widget for cell indices
        self.list_widget = QListWidget(self)
        self.list_widget.setSelectionMode(QListWidget.MultiSelection)
        for i in range(num_cells):
            self.list_widget.addItem(str(i))
        layout.addWidget(self.list_widget)

        self.list_widget.itemSelectionChanged.connect(self.emit_selection_changed)  # Connect selection change

        # OK and Cancel buttons
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK", self)
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel", self)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)

        layout.addLayout(btn_layout)

    def emit_selection_changed(self):
        selected_indices = self.selected_indices()
        self.selectionChanged.emit(selected_indices)  # Emit the currently selected indices

    def selected_indices(self):
        return [int(item.text()) for item in self.list_widget.selectedItems()]



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




class Plane_GridPositionDialog(QDialog):
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
            x_input.setText(str(Plane_GridPositionDialog.last_vertices[i][0]))
            y_input.setText(str(Plane_GridPositionDialog.last_vertices[i][1]))
            z_input.setText(str(Plane_GridPositionDialog.last_vertices[i][2]))
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
        self.spacing_input.setText(str(Plane_GridPositionDialog.last_spacing))
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
        Plane_GridPositionDialog.last_vertices = [(float(x.text()), float(y.text()), float(z.text())) for x, y, z in self.vertex_inputs]
        Plane_GridPositionDialog.last_spacing = float(self.spacing_input.text())
        super().accept()

    def getValues(self):
        return Plane_GridPositionDialog.last_vertices, Plane_GridPositionDialog.last_spacing



class AnalysisWorker(QObject):
    finished = Signal()  # Signal to indicate completion of all work

    def __init__(self, source_pos_matrix, source_power, receiver_positions, F, shape, cell_center, normal_vector, cell_area, absorptivity_matrix, rho_air, sound_speed, callback=None):
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
        self.callback = callback

    def run(self):
        try:
            for i, pos in enumerate(self.receiver_positions):
                analysis = AcousticAnalysis(self.source_pos_matrix, self.source_power, pos, self.F,
                                            self.shape, self.cell_center, self.normal_vector, self.cell_area, self.absorptivity_matrix, self.rho_air, self.sound_speed)
                result = analysis.run_analysis()
                SPL = result['SPL (dB)']
                if self.callback:
                    self.callback(i, SPL)  # Execute the callback with SPL data
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            self.finished.emit()



class SPLVisualization:
    def __init__(self, v0, v1, v2, v3, receiver_positions, SPL_Matrix):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.receiver_pos_matrix = receiver_positions
        self.SPL_Matrix = SPL_Matrix

    def prepare_spl_data(self):
        # Unpack vertices
        v0, v1, v2, v3 = self.v0, self.v1, self.v2, self.v3

        # Calculate the plane's basis vectors
        edge1 = v1 - v0
        edge2 = v3 - v0

        # Project the receiver positions onto the plane to get their 2D coordinates
        proj_on_edge1 = np.dot(self.receiver_pos_matrix - v0, edge1) / np.dot(edge1, edge1)
        proj_on_edge2 = np.dot(self.receiver_pos_matrix - v0, edge2) / np.dot(edge2, edge2)
        points2D = np.vstack((proj_on_edge1, proj_on_edge2)).T

        # Create the grid for interpolation
        grid_space = np.linspace(0, 1, 100)
        grid_u, grid_v = np.meshgrid(grid_space, grid_space)
        grid_points2D = np.outer(grid_u.ravel(), edge1) + np.outer(grid_v.ravel(), edge2) + v0

        # Interpolation using 2D points
        grid_spl = griddata(points2D, self.SPL_Matrix.ravel(), (grid_u.ravel(), grid_v.ravel()), method='cubic')
        grid_spl = grid_spl.reshape(100, 100)  # Reshape back to grid shape after interpolation

        # Pack the grid data into a dictionary to return
        grid_data = {
            'grid_points': grid_points2D.reshape(10000, 3),
            'grid_spl': grid_spl.ravel(),  # Flatten to match the number of points
            'grid_shape': grid_u.shape
        }
        return grid_data