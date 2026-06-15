import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFrame, QHBoxLayout, QSplitter, 
    QTabWidget, QTreeView, QVBoxLayout, QWidget, QStyle, QToolBar, 
    QMenu, QFileDialog, QMessageBox, QInputDialog,
    QFileDialog, QInputDialog, QProgressDialog
)
from PySide6.QtGui import QIcon, QStandardItemModel, QStandardItem, QAction
from PySide6.QtCore import Qt, QThread, QLocale, Slot, QTimer

import pyvista as pv
from pyvista import themes
from pyvistaqt import QtInteractor
import numpy as np
from datetime import datetime


## import many Class
from SUB_CLASS.support_class import (
    VFComputationThread, VFProgressDialog, MESProgressDialog,
    GeometryInfoDialog, CellSelectionDialog, PhysicalSurfaceSelectionDialog,
    ReceiverSelectionDialog, SourcePositionDialog,
    Point_ReceiverPositionDialog, Plane_GridPositionDialog_vertics, Plane_GridPositionDialog_standard_plane,
    AnalysisWorker, VFload, BlackgroundColor, ContourEditorDialog, VectorScaleDialog, SPLVisualization)
from mes.ret.geometry import MeshProcessor
from mes.ret.postprocess import SPLPlaneCalculator


import json
import os
import shutil
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

if getattr(sys, "frozen", False):
    os.chdir(os.path.dirname(sys.executable))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MES-Acoustic Demo-Version-2024")
        self.setWindowIcon(QIcon('PIC_MANU/MES.png'))
        self.resize(1200, 720)  # Set initial size but allow resizing
        

        
        self.directory = None
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.progress_dialog = None
        self.worker = None
        self.thread = None
        self.current_group = 0
        self.total_groups = 0
        ## =========================== Geo metry

        self.flip_normals = False
        self.shape = 0
        self.vectors_visible = 0
        self.cell_edge = 0
        self.feature_edge = 0
        self.cell_normal = 0
        self.cell_center = 0
        self.normal_vector = 0
        self.factor_normal_vector = None

        ## =========================== Surface propoties
        
        self.item_select = 0
        self.surface_item = 0
        self.cell_sizes = 0
        self.cell_area = 0
        self.absorptivity_matrix = 0
        self.absorptivity_value = 0
        self.default_absorption = 0.2
        self.record_surface_group = []
        self.available_cells = []  
        self.physical_surface_names = {}


        ## ============================ VF - Cumputation
        self.VF_item = 0
        self.VF_sub_item = 0
        self.F = 0
        self.VF_filename = None
        self.VF_filepath = None


        ## ============================ Source pos & power
        self.source_item = 0
        self.source_pos_matrix = []  # matrix of source must [[],[],[]]
        self.source_power_matrix = []  # matrix of power must [, , ,]
        self.record_source_groups = []


        ## ============================ Receiver
        self.receiver_item = 0
        self.receiver_pos_matrix = []  # matrix of receiver must [[],[],[]]
        self.SPL_Matrix = []
        self.SPL_Result_Store = []
        self.index_SPL_Result_Store = 0

        self.v0, self.v1, self.v2, self.v3 = 0,0,0,0
        self.grid_space = 0

        self.record_receiver_groups = []


        ## ============================ post - process
        self.result_folder_item = 0
        self.record_result_groups = []
        self.contour_config = {
            "cmap": "jet",
            "contour_lines": 7,
            "opacity": 1.0,
            "density_factor": 100,
            "interpolation_method": "cubic"
        }
        self.current_receiver = None
        self.active_contours = {} 


        ## =========================== model parameter 
        self.plot_interactor = 0
        self.cell_color = 'silver'
        self.show_edge = True
        self.axis_displayed = False
        self.model_filename = 0

    
        # Set initial the plot theme
        #pv.global_theme.background = 'darkblue'
        #pv.set_plot_theme(themes.ParaViewTheme())

        # Create a splitter to separate the tab widget and the PyVista interactor
        splitter = QSplitter()

        # main tab
        main_tab = self.add_main_tab()
       

        # Create the PyVista interactor
        plot_widget = self.setup_plot_widget()

        # Add the tab widget and the plot widget to the splitter
        splitter.addWidget(main_tab)
        splitter.addWidget(plot_widget)

        # Adjust splitter's initial sizes
        splitter.setSizes([int(self.width() * 0.25), int(self.width() * 0.75)])

        # Set the splitter as the central widget
        self.setCentralWidget(splitter)

        # Create the menu bar
        self.menuBar = self.menuBar()
        self.create_menus()

        # Add a toolbar with a button
        self.setup_toolbar()


## ==================================== Manu bar

    def create_menus(self):
        """Create the File, Edit, Window, and Settings menus."""

        # File menu
        file_menu = self.menuBar.addMenu("File")
        open_action = QAction("New Project", self)
        open_action.triggered.connect(self.handle_open_file)
        file_menu.addAction(open_action)
        save_action = QAction("Save", self)
        save_action.triggered.connect(self.handle_save_project) 
        file_menu.addAction(save_action)
        load_project_action = QAction("Load Project", self)
        load_project_action.triggered.connect(self.handle_load_project) 
        file_menu.addAction(load_project_action)
        # Add more actions to the File menu as needed

        # Edit menu
        edit_menu = self.menuBar.addMenu("Edit")
        # Add actions for editing functionality here (e.g., undo, redo, copy, paste)

        # Window menu
        window_menu = self.menuBar.addMenu("Window")
        # Add actions to control window behavior here (e.g., maximize, minimize, tile windows)

        # Settings menu
        settings_menu = self.menuBar.addMenu("Settings")
        change_themes = QAction("Blackground themes", self)
        change_themes.triggered.connect(self.change_background_themes)
        settings_menu.addAction(change_themes)
        reset_change_themes = QAction("Default themes", self)
        reset_change_themes.triggered.connect(self.reset_background_themes)
        settings_menu.addAction(reset_change_themes)




    def handle_open_file(self):
        """Open a Gmsh or VTK mesh and prepare it for RET computation."""
        self.model_filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open File",
            "",
            "Mesh Files (*.msh *.vtk *.vtu);;All Files (*.*)",
        )
        if self.model_filename:
            try:
                # Get the directory of the current script or executable
                base_dir = os.path.dirname(os.path.abspath(__file__))
                
                # Create a relative path
                rel_path = os.path.relpath(self.model_filename, base_dir)
                
                # Store the relative path
                self.model_filename_relative = rel_path
                
                processor = MeshProcessor(self.model_filename)
                self.shape = processor.prepare_geometry()
                self.physical_surface_names = {
                    physical_id: name
                    for name, physical_id in processor.physical_ids.items()
                }
                self.cell_edge = self.shape.extract_all_edges()
                self.feature_edge = self.shape.extract_feature_edges()
                self.cell_center = np.array([cell for cell in self.shape.cell_centers().points])

                # Initialize geometry (this will compute normals, cell areas, etc.)
                self.initialize_geometry()

                # Clear any existing plot in the interactor
                self.plot_interactor.clear()
                
                # Add the mesh to the interactor for visualization
                self.plot_interactor.add_mesh(self.shape, show_edges=self.show_edge, color=self.cell_color)

                print(f"File loaded successfully: {self.model_filename_relative}")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading file: {str(e)}")
                print(f"Error loading file: {e}")


    def handle_save_project(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "Save Project", "",
                                                "JSON Files (*.json);;All Files (*)", options=options)
        if fileName:
            # Create project directory
            project_dir = os.path.splitext(fileName)[0] + "_files"
            os.makedirs(project_dir, exist_ok=True)
            
            # Create VF directory in project folder
            vf_dir = os.path.join(project_dir, 'VF_Files')
            os.makedirs(vf_dir, exist_ok=True)
            
            # Update VF paths to point to project directory
            if hasattr(self, 'VF_filepath') and self.VF_filepath:
                source_vf_path = os.path.abspath(self.VF_filepath)
                vf_filename = os.path.basename(source_vf_path)
                destination_vf_path = os.path.abspath(
                    os.path.join(vf_dir, vf_filename)
                )
                if (
                    os.path.exists(source_vf_path)
                    and source_vf_path != destination_vf_path
                ):
                    shutil.copy2(source_vf_path, destination_vf_path)
                self.VF_filepath = destination_vf_path.replace('\\', '/')
                self.VF_filepath_relative = os.path.join(
                    os.path.basename(project_dir),
                    'VF_Files',
                    vf_filename
                )
            
            project_data = {
                'model_filename_relative': self.model_filename_relative,
                'VF_filepath_relative': self.VF_filepath_relative,
                'record_surface_group': self.record_surface_group,
                'record_source_groups': self.record_source_groups,
                'record_receiver_groups': self.record_receiver_groups,
                'record_result_groups': self.record_result_groups,
                'flip_normals': self.flip_normals
            }
            
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return json.JSONEncoder.default(self, obj)

            with open(fileName, 'w') as file:
                json.dump(project_data, file, cls=NumpyEncoder)
            print(f"Project saved to: {fileName}")


    def handle_load_project(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Load Project", "",
                                                "JSON Files (*.json);;All Files (*)", options=options)
        if fileName:
            try:
                with open(fileName, 'r') as file:
                    project_data = json.load(file)

                # Get the directory of the current script or executable
                self.base_dir = os.path.dirname(os.path.abspath(__file__))

                # Load all data from project
                self.model_filename_relative = project_data.get('model_filename_relative')
                self.VF_filepath_relative = project_data.get('VF_filepath_relative')
                
                # Reconstruct the absolute paths
                self.model_filename = os.path.join(self.base_dir, self.model_filename_relative)
                self.VF_filepath = os.path.join(self.base_dir, self.VF_filepath_relative) if self.VF_filepath_relative else None
                
                self.record_surface_group = project_data.get('record_surface_group', [])  
                self.record_source_groups = project_data.get('record_source_groups', [])  
                self.record_receiver_groups = project_data.get('record_receiver_groups', []) 
                self.record_result_groups = project_data.get('record_result_groups', [])  
                self.flip_normals = project_data.get('flip_normals', False)


                processor = MeshProcessor(self.model_filename)
                self.shape = processor.prepare_geometry()
                self.physical_surface_names = {
                    physical_id: name
                    for name, physical_id in processor.physical_ids.items()
                }
                self.cell_center = np.array([cell for cell in self.shape.cell_centers().points])
                self.cell_edge = self.shape.extract_all_edges()
                self.feature_edge = self.shape.extract_feature_edges()

                # Initialize geometry (this will compute normals, cell areas, etc.)
                self.initialize_geometry()

                # Add normal vectors to the plot if they were visible before
                if hasattr(self, 'show_normal_vector'):
                    self.shape.set_active_scalars('Normals', preference='cell')
                    self.show_normal_vector = self.shape.glyph(orient='Normals')
                    self.plot_interactor.add_mesh(self.show_normal_vector, color='b')

                # Setup view factor
                if self.VF_filepath and os.path.exists(self.VF_filepath):
                    self.F = VFload(self.VF_filepath).load_vf_file()
                    self.VF_filename = os.path.splitext(os.path.basename(self.VF_filepath))[0]
                    
                    # Clear all existing children from VF_item
                    self.VF_item.removeRows(0, self.VF_item.rowCount())

                    # Add a new child with the updated filename
                    self.VF_sub_item = self.add_tree_child(self.VF_item, self.VF_filename, 'PIC_VF/VF.png')
                    print(f"View factors loaded from: {self.VF_filepath_relative}")
                else:
                    print("No view factors file found or specified.")

                print(f"Project loaded successfully. Model file: {self.model_filename_relative}")

            except FileNotFoundError as e:
                QMessageBox.critical(self, "Error", f"File not found: {str(e)}")
            except json.JSONDecodeError:
                QMessageBox.critical(self, "Error", "Invalid project file format.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred while loading the project: {str(e)}")



             # =============== Load Surface Groups
            def update_surface_item():
                # Clear all existing children from VF_item
                self.surface_item.removeRows(0, self.surface_item.rowCount())
                
                for i in range(len(self.record_surface_group)):
                    name = self.record_surface_group[i]['name']
                    absorption = self.record_surface_group[i]['absorption']

                    surface_item = self.add_tree_parent(self.surface_item, name)
                    physical_ids = self.record_surface_group[i].get('physical_ids')
                    if physical_ids and "gmsh:physical" in self.shape.cell_data:
                        mask = np.isin(
                            self.shape.cell_data["gmsh:physical"],
                            physical_ids,
                        )
                        self.absorptivity_matrix[mask] = absorption
                        ids_text = ", ".join(map(str, physical_ids))
                        self.add_tree_child(
                            surface_item,
                            f'Physical Surface: {ids_text}',
                            'PIC_SURFACE/Cell.png',
                        )
                    else:
                        start_cell = self.record_surface_group[i].get('start_cell', 0)
                        stop_cell = self.record_surface_group[i].get('stop_cell', 0)
                        self.absorptivity_matrix[start_cell:stop_cell + 1] = absorption
                        self.add_tree_child(
                            surface_item,
                            f'Cell: {start_cell}:{stop_cell}',
                            'PIC_SURFACE/Cell.png',
                        )
                    self.add_tree_child(surface_item, f'Absorption: {absorption}', 'PIC_SURFACE/Alpha.png' )

                # Alternatively, expand only the Surface Properties item and its children
                self.geometry_treeView.expand(self.surface_item.index())
                for row in range(self.surface_item.rowCount()):
                    child_index = self.surface_item.child(row).index()
                    self.geometry_treeView.expand(child_index)

            ## call function 
            update_surface_item()




             # =============== Load Source Groups
            def update_source_item():
                self.source_item.removeRows(0, self.source_item.rowCount())

                for i in range(len(self.record_source_groups)):
                    name = self.record_source_groups[i]['name']
                    source_position = self.record_source_groups[i]['position']
                    power = self.record_source_groups[i]['power']

                    # update sorce matrix
                    source_item = self.add_tree_parent(self.source_item, name)
                    self.add_tree_child(source_item, f'pos: {source_position}', 'PIC_RUN/source_point.png'  )
                    self.add_tree_child(source_item, f'power: {power}', 'PIC_RUN/power.png'  )
                
                # Alternatively, expand only the source item and its children
                self.calculation_treeView.expand(self.source_item.index())
                for row in range(self.source_item.rowCount()):
                    child_index = self.source_item.child(row).index()
                    self.calculation_treeView.expand(child_index)

            ## call function 
            update_source_item()


             # =============== Load revceiver Groups
            def update_receiver_item():
                self.receiver_item.removeRows(0, self.receiver_item.rowCount())

                for i in range(len(self.record_receiver_groups)):
                    name = self.record_receiver_groups[i]['name']
                    print(name)
                    # for point receiver
                    if name[0:5] == 'point':
                        point_receiver_position = self.record_receiver_groups[i]['position']
                        receiver_item = self.add_tree_parent(self.receiver_item, name)
                        self.add_tree_child(receiver_item, f'pos: {point_receiver_position}', 'PIC_RUN/point_reciver.png')

                    # for plane receiver
                    if name[0:5] == 'plane':
                        grid_points = self.record_receiver_groups[i].get('position') or []
                        receiver_item = self.add_tree_parent(self.receiver_item, name)
                        self.add_tree_child(receiver_item, f'pos: {len(grid_points)} points', 'PIC_RUN/plane_reciver.png')

                # Alternatively, expand only the source item and its children
                self.calculation_treeView.expand(self.receiver_item.index())
                for row in range(self.receiver_item.rowCount()):
                    child_index = self.receiver_item.child(row).index()
                    self.calculation_treeView.expand(child_index)

            ## call function 
            update_receiver_item()


            # =============== Load Result Groups
            def update_result_item():
                # Clear existing results
                self.result_model.removeRows(0, self.result_model.rowCount())
                self.SPL_Result_Store.clear()  # Clear existing SPL results

                if self.record_result_groups:
                    result_datetime = self.record_result_groups[0].get('result_datetime', 'Unknown Date')
                    result_folder_item = self.add_tree_parent(self.result_model, f'Result - {result_datetime}')

                    for result in self.record_result_groups[1:]:  # Skip the first item which is the datetime
                        name = result['name']
                        self.add_tree_child(result_folder_item, f'SPL - {name}')
                        
                        # Update SPL_Result_Store
                        spl_result = {
                            'name': name,
                            'position': result['receiver_positions'],
                            'data': result['SPL_data'],
                            'faces': result.get('faces'),
                            'plane': result.get('plane'),
                            'height': result.get('height'),
                            'grid_spaces': result.get('grid_spaces'),
                        }
                        self.SPL_Result_Store.append(spl_result)

                    # Expand the result folder
                    index = self.result_model.indexFromItem(result_folder_item)
                    self.result_treeView.expand(index)

            # Call function to update result items
            update_result_item()

            print("Project loaded successfully")
            print(f"Loaded {len(self.SPL_Result_Store)} SPL results")


    def change_background_themes(self):
        # Create an instance of BlackgroundColor class
        color_selector = BlackgroundColor(parent=self)

        # Let the user select the colors
        if color_selector.select_colors():
            # Set the background of the plot_interactor with the selected colors
            if self.plot_interactor is not None:
                self.plot_interactor.set_background(color_selector.get_main_color(), top=color_selector.get_light_color())

            # Optionally print out or handle confirmation
            print(f"Background changed to {color_selector.get_main_color()} with top color {color_selector.get_light_color()}")


    def reset_background_themes(self):
        self.plot_interactor.set_background("lightblue", top="white")






## ============================= Main Tab

    def add_main_tab(self):
        # Create the main tab widget
        self.tab_widget = QTabWidget()

        # =========== Geometry Tab with Tree View
        self.geometry_treeView = QTreeView()
        self.geometry_model = QStandardItemModel()
        self.geometry_treeView.setModel(self.geometry_model)
        self.geometry_treeView.setHeaderHidden(True)
        # for interaction with right click
        self.geometry_treeView.setContextMenuPolicy(Qt.CustomContextMenu)
        self.geometry_treeView.customContextMenuRequested.connect(self.on_context_menu_geometry)

        # Layout for 'Geometry' Tab
        geometry_layout = QHBoxLayout()
        geometry_layout.addWidget(self.geometry_treeView)
        tab_geometry = QWidget()
        tab_geometry.setLayout(geometry_layout)

        # Add parent and child items to 'Geometry' Tab
        self.configuration_item = self.add_tree_parent(self.geometry_model, 'Configuration')
        self.add_tree_child(self.configuration_item, 'Verify Geometry', 'PIC_CON/shapes.png')
        self.add_tree_child(self.configuration_item, 'Vector direction', 'PIC_CON/construction.png')
        self.add_tree_child(self.configuration_item, 'Geometric information', 'PIC_CON/information.png')

        self.VF_item = self.add_tree_parent(self.geometry_model, 'VF-Computation')
        self.surface_item = self.add_tree_parent(self.geometry_model, 'Surface Properties')
        
        # Expand all items in Geometry TreeView
        self.geometry_treeView.expandAll()
        
        # Add the 'Geometry' tab
        self.tab_widget.addTab(tab_geometry, 'Initial Setup')





        # ========== Run Analysis Tab with Tree View
        self.calculation_treeView = QTreeView()
        self.calculation_model = QStandardItemModel()
        self.calculation_treeView.setModel(self.calculation_model)
        self.calculation_treeView.setHeaderHidden(True)
        # for interaction with right click
        self.calculation_treeView.setContextMenuPolicy(Qt.CustomContextMenu)
        self.calculation_treeView.customContextMenuRequested.connect(self.on_context_menu_simulation)


        # Layout for 'Calculation' Tab
        calculation_layout = QHBoxLayout()
        calculation_layout.addWidget(self.calculation_treeView)
        tab_calculation = QWidget()
        tab_calculation.setLayout(calculation_layout)

        # Add 'Source' parent item to the 'Calculation' tab
        self.source_item = self.add_tree_parent(self.calculation_model, 'Source')
        self.receiver_item = self.add_tree_parent(self.calculation_model, 'Receiver')

        self.add_tree_child(self.calculation_model, 'Run solver', 'PIC_RUN/run_solver.png')

        # Expand all items in Calculation TreeView
        self.calculation_treeView.expandAll()

        # Add the 'Calculation' tab
        self.tab_widget.addTab(tab_calculation, 'Run Analysis')




        # ========== Post result Tab with Tree View

        self.result_treeView = QTreeView()
        self.result_model = QStandardItemModel()
        self.result_treeView.setModel(self.result_model)
        self.result_treeView.setHeaderHidden(True)

        # for interaction with right click
        self.result_treeView.setContextMenuPolicy(Qt.CustomContextMenu)
        self.result_treeView.customContextMenuRequested.connect(self.on_context_menu_result)



        # Layout for 'Post result ' Tab
        result_layout = QHBoxLayout()
        result_layout.addWidget(self.result_treeView)
        tab_result = QWidget()
        tab_result.setLayout(result_layout)


        # Expand all items in Result TreeView
        self.result_treeView.expandAll()

        # Add the 'post result' tab
        self.tab_widget.addTab(tab_result, 'Post-Process')


        return self.tab_widget



    def add_tree_parent(self, model, title):
        # 'parent' argument removed, no longer needed because 'model' is specified
        item = QStandardItem(title)
        item.setIcon(self.style().standardIcon(QStyle.SP_DirClosedIcon))
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        model.appendRow(item)
        return item


    def add_tree_child(self, parent, title, icon_path=None):
        item = QStandardItem(title)
        if icon_path:
            item.setIcon(QIcon(icon_path))
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        parent.appendRow(item)


    def remove_item(self, item):
        # Get the parent of the item
        parent = item.parent()
        if parent is None:
            # If the item is a top-level item
            self.geometry_model.removeRow(item.row())
        else:
            # If the item is a child
            parent.removeRow(item.row())







### =============================== main calculation in tab geometry

    def on_context_menu_geometry(self, point):
        index = self.geometry_treeView.indexAt(point)
        if not index.isValid():
            return

        item = self.geometry_model.itemFromIndex(index)
        self.item_select = item
        if item is None:
            return

        menu = QMenu(self.geometry_treeView)

        if 'Group' in item.text():
            action_add_cell = QAction('Select Physical Surface', self.geometry_treeView)
            action_add_cell.triggered.connect(lambda: self.Cell_set(item.text()))
            menu.addAction(action_add_cell)
            
            action_add_absorption = QAction('Add Absorption', self.geometry_treeView)
            action_add_absorption.triggered.connect(lambda: self.Absorption_set(item.text()))
            menu.addAction(action_add_absorption)
  
            action_remove_group = QAction('Delete Group', self.geometry_treeView)
            action_remove_group.triggered.connect(lambda: self.remove_surface_group(item, item.text()))  
            menu.addAction(action_remove_group)
  

        elif item.text() == 'Verify Geometry':
            action_calculate = QAction('Calculate', self)
            action_calculate.triggered.connect(self.Verify_Geometry)
            menu.addAction(action_calculate)
            

        elif item.text() == 'Vector direction':
            action_scaling_vector = QAction('Scaling Vector', self)
            action_scaling_vector.triggered.connect(self.Scale_vector)
            menu.addAction(action_scaling_vector)



        elif item.text() == 'Geometric information':
            action_mesh_data = QAction('Mesh Data', self)
            action_mesh_data.triggered.connect(self.Show_geometry_information)
            menu.addAction(action_mesh_data)

        elif item.text() == 'VF-Computation':
            action_compute_vf = QAction('New Compute', self)
            action_compute_vf.triggered.connect(self.VF_compute)
            menu.addAction(action_compute_vf)

            action_load_vf = QAction('Load VF', self)
            action_load_vf.triggered.connect(self.VF_load)
            menu.addAction(action_load_vf)


        elif item.text() == 'Surface Properties':
            action_add_surface_group = QAction('Add Material Group', self)
            action_add_surface_group.triggered.connect(self.add_new_surface_group)
            menu.addAction(action_add_surface_group)

            menu.addSeparator()

            action_copy = QAction('Copy', self)
            # Connect to copy functionality or use dummy function if not implemented
            # action_copy.triggered.connect(self.copy_functionality)
            menu.addAction(action_copy)

            action_paste = QAction('Paste', self)
            # Connect to paste functionality or use dummy function if not implemented
            # action_paste.triggered.connect(self.paste_functionality)
            menu.addAction(action_paste)

        menu.exec(self.geometry_treeView.viewport().mapToGlobal(point))




## ======= for Confriguration parent

    def Verify_Geometry(self):
        if not self.shape:
            QMessageBox.warning(self, "Warning", "No Geometry loaded.")
            return False

        self.plot_interactor.clear()
        self.plot_interactor.add_mesh(self.cell_edge, show_edges=True, color='k')

        # Display current normal vectors
        self.update_normal_vectors()

        return True

    def update_normal_vectors(self):
        self.shape.set_active_scalars('Normals', preference='cell')
        self.show_normal_vector = self.shape.glyph(orient='Normals')
        self.plot_interactor.add_mesh(self.show_normal_vector, color='b')

    def Flip_vector(self):
        QMessageBox.information(
            self,
            "Automatic Normals",
            "Normal directions are prepared automatically from the input mesh.",
        )


    def initialize_geometry(self):
        if not self.shape:
            QMessageBox.warning(self, "Warning", "No Geometry loaded.")
            return

        # MeshProcessor already orients normals inward. Only compute missing data.
        if "Normals" not in self.shape.cell_data:
            self.shape.compute_normals(
                cell_normals=True,
                point_normals=False,
                inplace=True,
            )
        self.normal_vector = self.shape.cell_normals

        # Compute cell area
        self.cell_sizes = self.shape.compute_cell_sizes()
        self.cell_area = self.cell_sizes.cell_data['Area']

        # Prepare absorption matrix
        self.absorptivity_matrix = np.full(
            self.shape.n_cells,
            self.default_absorption,
            dtype=float,
        )

        # Initial display
        self.plot_interactor.clear()
        self.plot_interactor.add_mesh(self.shape, show_edges=self.show_edge, color=self.cell_color)
        




    def Scale_vector(self):
        # Create the vector scale dialog and set the initial scale to 0.1
        dialog = VectorScaleDialog(initial_scale=0.1, parent=self)

        # Connect the signal to dynamically update the vector scale in real-time
        dialog.scale_changed.connect(self.update_vector_scale)

        # Show the dialog (non-blocking to allow real-time updates)
        dialog.setModal(False)
        dialog.show()



    def update_vector_scale(self, factor):
        """Update the vector scale in real-time."""
        print(f"Updating vector scale to: {factor}")

        # Remove the previous vector actor if it exists
        if hasattr(self, 'vector_actor') and self.vector_actor is not None:
            self.plot_interactor.remove_actor(self.vector_actor)
            self.vector_actor = None  # Reset to ensure it won't hold the old actor

        # Create the new glyph with the updated scale factor
        glyph = self.shape.glyph(orient='Normals', scale=True, factor=factor)

        # Add the updated glyph to the plot and store the actor (not the data itself)
        self.vector_actor = self.plot_interactor.add_mesh(glyph, color="blue", name="vector_glyph")

        # Redraw the plot to reflect changes
        self.plot_interactor.render()





    def Show_geometry_information(self):
        if not self.shape:
            QMessageBox.warning(self, "Warning", "No Geomatry loaded.")
            return

        total_elements = self.shape.n_cells
        vertices = self.shape.n_points

        # Fetch bounds for X, Y, and Z axes directly from the mesh
        bound_x = (round(self.shape.bounds[0],3), round(self.shape.bounds[1],3))
        bound_y = (round(self.shape.bounds[2],3), round(self.shape.bounds[3],3))
        bound_z = (round(self.shape.bounds[4],3), round(self.shape.bounds[5],3))
        

        # Create and display the dialog with dynamic information
        dialog = GeometryInfoDialog(total_elements, vertices, bound_x, bound_y, bound_z, self)
        dialog.exec()



## ======= for VF compute

    def VF_compute(self):
        self.VF_filename, ok = QInputDialog.getText(self, 'New Compute VF', 'Name:')
        if ok and self.VF_filename:
            try:
                # Get the directory of the current VTK file
                vtk_dir = os.path.dirname(self.model_filename)
                
                # Create VF_Files directory in the same location as VTK file
                vf_dir = os.path.join(vtk_dir, 'VF_Files')
                os.makedirs(vf_dir, exist_ok=True)
                
                # Set up file paths relative to VTK file location
                self.VF_filepath_relative = os.path.join(
                    os.path.dirname(self.model_filename_relative),
                    'VF_Files',
                    self.VF_filename + '.npy'
                )
                self.VF_filepath = os.path.join(vtk_dir, 'VF_Files', self.VF_filename + '.npy')
                
                # Create and show progress dialog
                self.progress_dialog = VFProgressDialog(self)
                self.progress_dialog.show()

                # Start computation thread
                self.computation_thread = VFComputationThread(self.shape, self.VF_filepath, self)
                self.computation_thread.completed.connect(self.on_computation_finished)
                self.computation_thread.failed.connect(self.on_vf_computation_failed)
                self.computation_thread.status_changed.connect(
                    self.progress_dialog.update_status
                )
                self.computation_thread.log_message.connect(
                    self.progress_dialog.append_log
                )
                self.computation_thread.progress_changed.connect(
                    self.progress_dialog.update_progress
                )
                self.computation_thread.start()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to set up VF computation: {str(e)}")
                return
        else:
            print("Cancel VF Compute.")

    def on_computation_finished(self, file_path):
        self.progress_dialog.update_status("View-factor computation complete.")
        self.progress_dialog.update_progress(100)
        self.progress_dialog.accept()  # Close the progress dialog
        self.VF_sub_item = self.add_tree_child(self.VF_item, self.VF_filename, 'PIC_VF/VF.png')
        self.F = VFload(file_path).load_vf_file()
        print(f"View factors saved to {self.VF_filepath_relative}")

    def on_vf_computation_failed(self, message):
        self.progress_dialog.reject()
        QMessageBox.critical(self, "View Factor Error", message)

    def VF_load(self):
        try:
            options = QFileDialog.Options()
            self.VF_filepath, _ = QFileDialog.getOpenFileName(self, "Select a file", "",
                                                    "View Factors (*.npy *.txt);;All Files (*)", options=options)
            if self.VF_filepath:
                # Create relative path
                self.VF_filepath_relative = os.path.relpath(self.VF_filepath, self.base_dir)
                self.F = VFload(self.VF_filepath).load_vf_file() 
            else:
                return  # Cancel if no file is selected

            self.VF_filename = os.path.splitext(os.path.basename(self.VF_filepath))[0]

            # Clear all existing children from VF_item
            self.VF_item.removeRows(0, self.VF_item.rowCount())

            # Add a new child with the updated filename
            self.VF_sub_item = self.add_tree_child(self.VF_item, self.VF_filename, 'PIC_VF/VF.png')

        except FileNotFoundError:
            QMessageBox.critical(self, "Computation Error", "No view factors found.")
        except Exception as e:
            QMessageBox.critical(self, "Computation Error", f"An error occurred: {str(e)}")

        








## ======= for surface parrent

    def add_new_surface_group(self):
        name, ok = QInputDialog.getText(self, 'Input Dialog', 'Enter name for new surface group:')
        if ok and name:
            new_surface_group = {
                'name': 'Group-' + name,
                'physical_ids': [],
                'absorption': self.default_absorption
            }
            self.record_surface_group.append(new_surface_group)
            self.add_tree_parent(self.surface_item, new_surface_group['name'])
        else:
            print("No name provided or operation cancelled.")


    def update_tree_child_for_group_surface(self, search_string, new_text):
        for i in range(self.item_select.rowCount()):
            if search_string in self.item_select.child(i).text():
                if i == 0:
                    self.item_select.child(i).setText(new_text)
                    self.item_select.child(i).setIcon(QIcon('PIC_SURFACE/Cell.png'))
                elif i == 1:
                    self.item_select.child(i).setText(new_text)
                    self.item_select.child(i).setIcon(QIcon('PIC_SURFACE/Alpha.png'))
                return
            
        if search_string in ("Cell:", "Physical Surface:"):
            self.add_tree_child(self.item_select, new_text, 'PIC_SURFACE/Cell.png')
        elif search_string == "Absorption:":
            self.add_tree_child(self.item_select, new_text, 'PIC_SURFACE/Alpha.png')



    def remove_surface_group(self, item, item_text):
        self.remove_item(item) 
        group_name = item_text  
        for i in range(len(self.record_surface_group)):
            if self.record_surface_group[i]['name'] == group_name:
                start = self.record_surface_group[i].get('start_cell')
                stop = self.record_surface_group[i].get('stop_cell')
                if start is not None and stop is not None:
                    self.available_cells.extend(range(start, stop + 1))
                    self.available_cells.sort()
                del self.record_surface_group[i]
                break


    def Cell_set(self, item_text):
        if not self.shape:
            QMessageBox.warning(self, "Warning", "No Geometry loaded.")
            return
        if "gmsh:physical" in self.shape.cell_data:
            physical_values = np.asarray(
                self.shape.cell_data["gmsh:physical"],
                dtype=int,
            )
            assigned_ids = {
                physical_id
                for group in self.record_surface_group
                if group.get("name") != item_text
                for physical_id in group.get("physical_ids", [])
            }
            surfaces = [
                (
                    physical_id,
                    self.physical_surface_names.get(
                        physical_id,
                        f"Physical Surface {physical_id}",
                    ),
                    int(np.count_nonzero(physical_values == physical_id)),
                )
                for physical_id in sorted(np.unique(physical_values))
                if physical_id not in assigned_ids
            ]
            dialog = PhysicalSurfaceSelectionDialog(surfaces, self)
            if dialog.exec():
                selected_ids = dialog.selected_ids()
                if not selected_ids:
                    QMessageBox.information(
                        self,
                        "No Selection",
                        "No physical surfaces were selected.",
                    )
                    return

                for group in self.record_surface_group:
                    if group["name"] == item_text:
                        group["physical_ids"] = selected_ids
                        group.pop("start_cell", None)
                        group.pop("stop_cell", None)
                        break

                self.highlight_physical_surfaces(selected_ids)
                ids_text = ", ".join(map(str, selected_ids))
                self.update_tree_child_for_group_surface(
                    "Physical Surface:",
                    f"Physical Surface: {ids_text}",
                )
            return

        if not self.available_cells:
            self.available_cells = list(range(self.shape.n_cells))
        self.cell_dialog = CellSelectionDialog(self.available_cells, self)
        self.cell_dialog.selectionChanged.connect(self.highlight_cells)
        self.cell_dialog.finished.connect(lambda: self.handle_cell_selection(item_text))
        self.cell_dialog.show()


    def highlight_cells(self, selected_indices):
        if not hasattr(self, 'shape') or not self.shape:
            print("Shape not loaded yet.")
            return
        
        if not selected_indices:
            print("No cells selected.")
            self.plot_interactor.clear()
            self.plot_interactor.add_mesh(self.shape, show_edges=True, color='silver')
            self.plot_interactor.reset_camera()
            return

        selected_indices = [idx for idx in selected_indices if idx <= self.shape.n_cells]
        selected_mesh = self.shape.extract_cells(selected_indices)
        
        self.plot_interactor.clear()
        self.plot_interactor.add_mesh(self.cell_edge, show_edges=True, color='k', line_width=2)
        self.plot_interactor.add_mesh(selected_mesh, show_edges=True, color='lime', opacity=1)
        self.plot_interactor.reset_camera()



    def handle_cell_selection(self, item_text):
        selected_indices = self.cell_dialog.selected_indices()
        if selected_indices:
            self.highlight_cells(selected_indices)
            start_cell, stop_cell = min(selected_indices), max(selected_indices)

            group_name = item_text  
            for i in range(len(self.record_surface_group)):
                if self.record_surface_group[i]['name'] == group_name:
                    self.record_surface_group[i]['start_cell'] = start_cell
                    self.record_surface_group[i]['stop_cell'] = stop_cell
                    break

            # Remove selected cells from available_cells
            self.available_cells = [cell for cell in self.available_cells if cell < start_cell or cell > stop_cell]

            self.update_tree_child_for_group_surface('Cell:', f'Cell: {start_cell}:{stop_cell}')
        else:
            QMessageBox.information(self, "No Selection", "No cells were selected.")

    def highlight_physical_surfaces(self, physical_ids):
        physical_values = np.asarray(
            self.shape.cell_data["gmsh:physical"],
            dtype=int,
        )
        selected_indices = np.flatnonzero(
            np.isin(physical_values, physical_ids)
        )
        selected_mesh = self.shape.extract_cells(selected_indices)

        self.plot_interactor.clear()
        self.plot_interactor.add_mesh(
            self.cell_edge,
            show_edges=True,
            color='k',
            line_width=2,
        )
        self.plot_interactor.add_mesh(
            selected_mesh,
            show_edges=True,
            color='lime',
            opacity=1,
        )
        self.plot_interactor.reset_camera()



    def Absorption_set(self, item_text):
        if not self.shape:
            QMessageBox.warning(self, "Warning", "No shape loaded.")
            return
        
        self.absorptivity_value, ok = QInputDialog.getDouble(
            self,
            "Input Absorption Value",
            "Enter absorption value:",
            value=0.0,
            minValue=0.0,
            maxValue=1.0,
            decimals=4
        )
        
        if ok:
            group_name = item_text  
            for i in range(len(self.record_surface_group)):
                if self.record_surface_group[i]['name'] == group_name:
                    self.record_surface_group[i]['absorption'] = self.absorptivity_value
                    physical_ids = self.record_surface_group[i].get('physical_ids')
                    if physical_ids and "gmsh:physical" in self.shape.cell_data:
                        mask = np.isin(
                            self.shape.cell_data["gmsh:physical"],
                            physical_ids,
                        )
                        self.absorptivity_matrix[mask] = self.absorptivity_value
                    else:
                        start = self.record_surface_group[i].get('start_cell', 0)
                        stop = self.record_surface_group[i].get('stop_cell', 0)
                        self.absorptivity_matrix[start : stop + 1] = self.absorptivity_value
                    break
            
            self.update_tree_child_for_group_surface('Absorption:', f'Absorption: {self.absorptivity_value}')
        else:
            print("Absorption input canceled.")

















### =============================== main calculation in tab run analysis

    def on_context_menu_simulation(self, point):
        index = self.calculation_treeView.indexAt(point)
        if not index.isValid():
            return

        item = self.calculation_model.itemFromIndex(index)
        self.item_select = item
        if item is None:
            return

        menu = QMenu(self)

        ### Source options

        if item.text() == 'Source':
            action_add_source = QAction('Add Source', self)
            action_add_source.triggered.connect(self.add_new_source_group)
            menu.addAction(action_add_source)

        elif 'source' in item.text():
            action_add_source_pos = QAction('Add source position', self)
            action_add_source_pos.triggered.connect(lambda: self.source_pos_set(item.text()))
            menu.addAction(action_add_source_pos)
            
            action_add_power = QAction('Add source power', self)
            action_add_power.triggered.connect(lambda: self.source_power_set(item.text()))
            menu.addAction(action_add_power)

            action_remove_source = QAction('Delete', self)
            action_remove_source.triggered.connect(lambda: self.remove_source_group(item, item.text()))
            menu.addAction(action_remove_source)



        ### ======== Receiver
        if item.text() == 'Receiver':
            action_add_point_reciver = QAction('Add point-receiver', self)
            action_add_point_reciver.triggered.connect(self.add_new_point_receiver_group)
            menu.addAction(action_add_point_reciver)

            action_add_plane_reciver = QAction('Add plane-receiver', self)
            action_add_plane_reciver.triggered.connect(self.add_new_plane_receiver_group)
            menu.addAction(action_add_plane_reciver)

        elif 'point' in item.text():
            action_add_point_reciver_pos = QAction('Set point', self)
            action_add_point_reciver_pos.triggered.connect(lambda: self.point_receiver_pos_set(item.text()))
            menu.addAction(action_add_point_reciver_pos)

            action_remove_receiver = QAction('Delete', self)
            action_remove_receiver.triggered.connect(lambda: self.remove_receiver_group(item, item.text()))
            menu.addAction(action_remove_receiver)


        elif 'plane' in item.text():
            action_add_plane_reciver_pos_standard_plane = QAction('Define-(XY/XZ/YZ)', self)
            action_add_plane_reciver_pos_standard_plane.triggered.connect(lambda: self.plane_receiver_pos_set_standard_plane(item.text()))
            menu.addAction(action_add_plane_reciver_pos_standard_plane)

            action_remove_receiver = QAction('Delete', self)
            action_remove_receiver.triggered.connect(lambda: self.remove_receiver_group(item, item.text()))
            menu.addAction(action_remove_receiver)


    
         ### ======== Runsover
        if item.text() == 'Run solver':
            action_add_runsolver = QAction('Start Solver', self)
            action_add_runsolver.triggered.connect(
                self.select_receivers_and_run
            )
            menu.addAction(action_add_runsolver)


        menu.exec(self.calculation_treeView.viewport().mapToGlobal(point))




    ## ============= source

    def add_new_source_group(self):
        # Similar to adding a new surface group
        name, ok = QInputDialog.getText(self, 'Add New Source', 'Enter name for new source:')
        if ok and name:
            group_name = 'source-' + name  # Name formatting to match surface groups
            new_source_group = {
                'name': group_name,
                'position': None,
                'power': None
            }
            self.record_source_groups.append(new_source_group)
            self.add_tree_parent(self.source_item, group_name)  # Assuming a method to add to UI
        else:
            print("No name provided or operation cancelled.")



    def remove_source_group(self, item, item_text):
        self.remove_item(item)  
        group_name = item_text
        for i in range(len(self.record_source_groups)):
            if self.record_source_groups[i]['name'] == group_name:
                del self.record_source_groups[i]  
                break  

        print("Updated source groups:", self.record_source_groups)



    def update_tree_child_for_group_source(self, search_string, new_text):
        # Search for the child with the given search_string
        for i in range(self.item_select.rowCount()):
            if search_string in self.item_select.child(i).text():
                # If found, update the text
                if i == 0:
                    self.item_select.child(i).setText(new_text)
                    self.item_select.child(i).setIcon(QIcon('PIC_RUN/source_point.png'))
                elif i == 1:
                    self.item_select.child(i).setText(new_text)
                    self.item_select.child(i).setIcon(QIcon('PIC_RUN/power.png'))

                return
        # If not found, add a new child
        if search_string == "pos:":
            self.add_tree_child(self.item_select, new_text, 'PIC_RUN/source_point.png' )
        elif search_string == "power:":
            self.add_tree_child(self.item_select, new_text, 'PIC_RUN/power.png' )



    def source_pos_set(self, item_text):
        dialog = SourcePositionDialog(self)
        dialog.show() ## make non model (can interact with other widget)
        if dialog.exec():
            x, y, z = dialog.getValues()
            source_pos = [x, y, z]

            group_name = item_text  
            for i in range(len(self.record_source_groups)):
                if self.record_source_groups[i]['name'] == group_name:
                    self.record_source_groups[i]['position'] = source_pos
                    break

            self.plot_interactor.add_points(np.array(source_pos, dtype=np.float32), show_edges=True, opacity=1, color='red', point_size=10)
            self.update_tree_child_for_group_source(
                'pos:',
                f'pos: {source_pos}'
            )
        else:
            print("source pos input canceled.")


    def source_power_set(self, item_text):
        source_power, ok = QInputDialog.getDouble(
            self,
            "Input Power",
            "Source power value (W):",
            value=0.5,  # Default value
            minValue=0.0,
            maxValue=1.0,
            decimals=4
        )
        
        if ok:
            group_name = item_text  
            for i in range(len(self.record_source_groups)):
                if self.record_source_groups[i]['name'] == group_name:
                    self.record_source_groups[i]['power'] =  source_power

            self.update_tree_child_for_group_source('power:', f'power: {source_power}')
        else:
            print("Power input canceled.")






    ## ================ receiver


    def remove_receiver_group(self, item, item_text):
        self.remove_item(item)  # Assume a method to remove from UI
        group_name = item_text
        for i in range(len(self.record_receiver_groups)):
            if self.record_receiver_groups[i]['name'] == group_name:
                del self.record_receiver_groups[i]
                break
        print("Updated receiver groups:", self.record_receiver_groups)



    ## ================ point receiver

    def add_new_point_receiver_group(self):
        # Get the name for the new receiver group
        name, ok = QInputDialog.getText(self, 'Add New Point Receiver', 'Name for new point:')
        if ok and name:
            group_name = 'point-' + name
            new_receiver_group = {
                'name': group_name,
                'position': None  # Initialize position to None
            }
            self.record_receiver_groups.append(new_receiver_group)
            self.add_tree_parent(self.receiver_item, group_name)  # Adding to UI
        else:
            print("No name provided or operation cancelled.")


    def update_tree_child_for_point_receiver_group(self, search_string, new_text):
        # Search and update the tree child with the new text based on search string
        for i in range(self.item_select.rowCount()):
            if search_string in self.item_select.child(i).text():
                self.item_select.child(i).setText(new_text)
                self.item_select.child(i).setIcon(QIcon('PIC_RUN/point_reciver.png'))
                return
        # If not found, add as a new child
        if search_string == "pos:":
            self.add_tree_child(self.item_select, new_text, 'PIC_RUN/point_reciver.png')


    def point_receiver_pos_set(self, item_text):
        # Function to set receiver position with dialog input
        print("Setting receiver position")
        dialog = Point_ReceiverPositionDialog(self)
        dialog.show()
        if dialog.exec():
            x, y, z = dialog.getValues()
            receiver_pos = [x, y, z]

            ## add data 
            group_name = item_text  
            for i in range(len(self.record_receiver_groups)):
                if self.record_receiver_groups[i]['name'] == group_name:
                    self.record_receiver_groups[i]['position'] = receiver_pos
                    break

            self.plot_interactor.add_points(np.array(receiver_pos, dtype=np.float32), show_edges=True, opacity=1, color='gold', point_size=10)
            self.update_tree_child_for_point_receiver_group(
                'pos:',
                f'pos: {receiver_pos}'
            )
        else:
            print("Receiver position input canceled.")



    ## =============== plane receiver

    def add_new_plane_receiver_group(self):
        # Get the name for the new receiver group
        name, ok = QInputDialog.getText(self, 'Add New Point Receiver', 'Name for new point:')
        if ok and name:
            group_name = 'plane-' + name
            new_receiver_group = {
                'name': group_name,
                'position': None,
                'faces': None,
                'plane': None,
                'height': None,
                'grid_spaces': None
            }
            self.record_receiver_groups.append(new_receiver_group)
            self.add_tree_parent(self.receiver_item, group_name)  # Adding to UI
        else:
            print("No name provided or operation cancelled.")


    def update_tree_child_for_plane_receiver_group(self, search_string, new_text):
        # Search and update the tree child with the new text based on search string
        for i in range(self.item_select.rowCount()):
            if search_string in self.item_select.child(i).text():
                self.item_select.child(i).setText(new_text)
                self.item_select.child(i).setIcon(QIcon('PIC_RUN/plane_reciver.png'))
                return
        # If not found, add as a new child
        if search_string == "pos:":
            self.add_tree_child(self.item_select, new_text, 'PIC_RUN/plane_reciver.png')



    def plane_receiver_pos_set_vertics(self, item_text):
        print("Setting receiver position with grid")
        dialog = Plane_GridPositionDialog_vertics(self)  # Assuming a new dialog for grid input
        dialog.show()
        if dialog.exec():
            vertices, grid_spacing = dialog.getValues()  # This now correctly expects two values
            self.v0, self.v1, self.v2, self.v3 = vertices  # Unpack the vertices here

            # Calculate the points in the grid
            d1 = np.array(self.v1) - np.array(self.v0)
            d2 = np.array(self.v3) - np.array(self.v0)
            num_points_d1 = int(np.linalg.norm(d1) / grid_spacing) + 1
            num_points_d2 = int(np.linalg.norm(d2) / grid_spacing) + 1

            u, v = np.meshgrid(np.linspace(0, 1, num_points_d1), np.linspace(0, 1, num_points_d2))
            u = u.flatten()
            v = v.flatten()

            grid_points = np.array(self.v0) + np.outer(u, d1) + np.outer(v, d2)
            

            ## add data 
            # group_name = item_text  
            # for i in range(len(self.record_receiver_groups)):
            #     if self.record_receiver_groups[i]['name'] == group_name:
            #         self.record_receiver_groups[i]['position'] = grid_points
            #         self.record_receiver_groups[i]['vertices'] = vertices
            #         self.record_receiver_groups[i]['grid_spaces'] = grid_spacing
            #         break

            # Convert grid_points (numpy array) to a list of lists

            grid_points_list = grid_points.tolist()

            # Store the valid points, vertices, and grid spacing in the receiver group
            group_name = item_text
            for i in range(len(self.record_receiver_groups)):
                if self.record_receiver_groups[i]['name'] == group_name:
                    self.record_receiver_groups[i]['position'] = grid_points_list
                    self.record_receiver_groups[i]['vertices'] = [list(v) for v in vertices]  # Convert vertices to list of lists
                    self.record_receiver_groups[i]['grid_spaces'] = grid_spacing
                    break

            
            self.plot_interactor.add_points(grid_points, show_edges=True, opacity=1, color='red', point_size=4)
            # Update the tree to show new grid positions
            self.update_tree_child_for_plane_receiver_group(
                'pos:',
                f'pos: {len(grid_points)} points'
            )
        else:
            print("Plane grid position input canceled.")





    def plane_receiver_pos_set_standard_plane(self, item_text):
        print("Setting receiver position with standard plane")

        if not isinstance(self.shape, pv.DataSet):
            QMessageBox.warning(self, "Warning", "Load a mesh first.")
            return

        dialog = Plane_GridPositionDialog_standard_plane(self)
        if not dialog.exec():
            print("Plane grid position input canceled.")
            return

        plane, height, grid_spacing = dialog.getValues()
        solver_stub = type("PlaneSolverStub", (), {"mesh": self.shape})()
        plane_calculator = SPLPlaneCalculator(solver_stub)

        try:
            plane_grid = plane_calculator.create_adaptive_plane_grid(
                plane=plane,
                height=height,
                spacing=grid_spacing,
                offset=min(0.1, grid_spacing / 2),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Plane Error", str(exc))
            return

        if plane_grid.n_points == 0:
            QMessageBox.warning(
                self,
                "Plane Error",
                "The selected plane does not create a valid region inside the mesh.",
            )
            return

        plane_surface = plane_grid.extract_surface().triangulate().clean()
        for receiver_group in self.record_receiver_groups:
            if receiver_group['name'] == item_text:
                receiver_group['position'] = plane_surface.points.tolist()
                receiver_group['faces'] = plane_surface.faces.tolist()
                receiver_group['plane'] = plane
                receiver_group['height'] = height
                receiver_group['grid_spaces'] = grid_spacing
                break

        self.plot_interactor.add_mesh(
            plane_surface,
            style='wireframe',
            color='red',
            line_width=1,
            name=f"preview-{item_text}",
        )
        self.update_tree_child_for_plane_receiver_group(
            'pos:',
            f'pos: {plane_surface.n_points} points',
        )






























    def select_receivers_and_run(self):
        if not self.record_receiver_groups:
            QMessageBox.warning(
                self,
                "Missing Receiver",
                "Define at least one receiver before running.",
            )
            return

        dialog = ReceiverSelectionDialog(
            self.record_receiver_groups,
            self,
        )
        if not dialog.exec():
            return

        selected_indices = dialog.selected_indices()
        if not selected_indices:
            QMessageBox.information(
                self,
                "No Receiver Selected",
                "Select at least one receiver group to run.",
            )
            return

        selected_groups = [
            self.record_receiver_groups[index]
            for index in selected_indices
        ]
        self.main_simulation(selected_groups)

    def main_simulation(self, receiver_groups=None):
        if not isinstance(self.shape, pv.DataSet):
            QMessageBox.warning(self, "Missing Geometry", "Load a mesh before running.")
            return
        if self.F is None or not isinstance(self.F, np.ndarray):
            QMessageBox.warning(self, "Missing View Factors", "Compute or load view factors first.")
            return
        if self.F.shape != (self.shape.n_cells, self.shape.n_cells):
            QMessageBox.warning(
                self,
                "Invalid View Factors",
                "The view-factor matrix does not match the loaded mesh.",
            )
            return
        if not self.record_source_groups or any(
            source.get('position') is None or source.get('power') is None
            for source in self.record_source_groups
        ):
            QMessageBox.warning(self, "Missing Source", "Define at least one complete source.")
            return

        selected_receiver_groups = (
            list(receiver_groups)
            if receiver_groups is not None
            else list(self.record_receiver_groups)
        )
        if not selected_receiver_groups or any(
            receiver.get('position') is None
            for receiver in selected_receiver_groups
        ):
            QMessageBox.warning(
                self,
                "Incomplete Receiver",
                "Every selected receiver must have a position or plane.",
            )
            return

        self.active_receiver_groups = selected_receiver_groups

        self.SPL_Result_Store.clear()
        self.source_pos_matrix.clear()
        self.source_power_matrix.clear()

        result_datetime = datetime.now().strftime("%m/%d/%Y-%H:%M:%S")
        self.record_result_groups = [{'result_datetime': result_datetime}]

        # Handle source data
        for source_group in self.record_source_groups:
            self.source_pos_matrix.append(source_group['position'])
            self.source_power_matrix.append(source_group['power'])

        # Create and show progress dialog
        self.progress_dialog = MESProgressDialog(self)
        self.receiver_group_sizes = [
            1 if group['name'].startswith('point-') else len(group['position'])
            for group in self.active_receiver_groups
        ]
        self.total_receiver_points = sum(self.receiver_group_sizes)
        self.progress_dialog.set_total(self.total_receiver_points)
        self.progress_dialog.show()

        # Set up simulation for all groups
        print(self.active_receiver_groups)

        self.total_groups = len(self.active_receiver_groups)
        self.current_group = 0
        
        # Start the simulation process
        QTimer.singleShot(0, self.process_next_group)

    def process_next_group(self):
        if self.current_group < self.total_groups:
            receiver_group = self.active_receiver_groups[self.current_group]
            name = receiver_group['name']
            if name.startswith('point-'):
                positions = [receiver_group['position']]
            else:
                positions = receiver_group['position']
            
            self.progress_dialog.update_label(f"Processing group: {name}\nGroup {self.current_group + 1} of {self.total_groups}")
            self.progress_dialog.update_phase("Preparing acoustic solver...")
            
            self.SPL_Matrix.clear()
            self.run_single_simulation(name, positions)
        else:
            self.finalize_simulation()

    def run_single_simulation(self, name, positions):
        self.analysis_error = None
        self.thread = QThread()
        self.worker = AnalysisWorker(
            self.source_pos_matrix,
            self.source_power_matrix,
            positions,
            self.F,
            self.shape,
            self.cell_center,
            self.normal_vector,
            self.cell_area,
            self.absorptivity_matrix,
            1.21,  # rho_air
            343,   # sound_speed
            0.0, # Attenuation
        )

        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.on_group_finished)

        self.worker.result_ready.connect(self.on_spl_received)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.status_changed.connect(self.progress_dialog.update_phase)
        self.worker.failed.connect(self.on_analysis_failed)

        self.thread.start()

    @Slot()
    def on_group_finished(self):
        if self.analysis_error:
            self.progress_dialog.reject()
            QMessageBox.critical(
                self,
                "Solver Error",
                self.analysis_error,
            )
            return

        # Store results for this group
        receiver_group = self.active_receiver_groups[self.current_group]
        group_results = {
            'name': receiver_group['name'],
            'position': receiver_group['position'],
            'data': self.SPL_Matrix.copy(),
            'faces': receiver_group.get('faces'),
            'plane': receiver_group.get('plane'),
            'height': receiver_group.get('height'),
            'grid_spaces': receiver_group.get('grid_spaces'),
        }
        self.SPL_Result_Store.append(group_results)
        
        self.current_group += 1
        QTimer.singleShot(0, self.process_next_group)

    @Slot(str)
    def on_analysis_failed(self, message):
        self.analysis_error = message

    @Slot(int, float)
    def on_spl_received(self, index, spl):
        self.SPL_Matrix.append(spl)
        print(f"Result received for receiver {index}: SPL = {spl}")

    @Slot(int, int)
    def update_progress(self, current, total):
        completed_before_group = sum(
            self.receiver_group_sizes[:self.current_group]
        )
        overall_current = completed_before_group + current
        group_name = self.active_receiver_groups[self.current_group]['name']
        self.progress_dialog.update_label(
            f"Processing group: {group_name} "
            f"({self.current_group + 1}/{self.total_groups})\n"
            f"Receiver {current}/{total} | "
            f"Overall {overall_current}/{self.total_receiver_points}"
        )
        self.progress_dialog.update_progress(overall_current)
        print(f"Progress: Group {self.current_group + 1}/{self.total_groups}, Receiver {current}/{total}")

    def finalize_simulation(self):
        self.progress_dialog.update_label("Finalizing results...")
        self.progress_dialog.update_phase("Storing SPL results...")
        self.progress_dialog.update_progress(self.total_receiver_points)
        QApplication.processEvents()  # Force GUI update
        
        # Use QTimer to allow GUI to update before storing results
        QTimer.singleShot(0, self.store_simulation_results)


    def store_simulation_results(self):
        for result in self.SPL_Result_Store:
            stored_result = {
                'name': result['name'],
                'receiver_positions': result['position'],
                'SPL_data': result['data'],
                'faces': result.get('faces'),
                'plane': result.get('plane'),
                'height': result.get('height'),
                'grid_spaces': result.get('grid_spaces'),
            }
            self.record_result_groups.append(stored_result)
            print(f"Stored simulation result for group: {result['name']}")
            
            # Force GUI update after each result is stored
            QApplication.processEvents()

        # Create the result folder after all simulations
        result_datetime = self.record_result_groups[0]['result_datetime']
        self.result_folder_item = self.add_tree_parent(self.result_model, f'Result - {result_datetime}')

        # Add children to the result folder
        for result in self.SPL_Result_Store:
            self.add_tree_child(self.result_folder_item, f'SPL - {result["name"]}')
            QApplication.processEvents()  # Force GUI update

        # Close progress dialog and print completion message
        QTimer.singleShot(0, self.complete_simulation)

    def complete_simulation(self):
        if self.progress_dialog:
            self.progress_dialog.close()
        print("Simulation completed")














### =============================== main result in tab post process

    def on_context_menu_result(self, point):
        index = self.result_treeView.indexAt(point)
        if not index.isValid():
            return

        item = self.result_model.itemFromIndex(index)
        if item is None:
            return

        menu = QMenu(self)

        if item.text().startswith('SPL - '):
            receiver_name = item.text()[6:]  # Extract the receiver name
            receiver_type = self.get_receiver_type(receiver_name)

            if receiver_type == 'plane':
                action_show_contour = QAction('Show SPL - Contour Plot', self)
                action_show_contour.triggered.connect(lambda: self.show_spl_contour_plot(receiver_name))
                menu.addAction(action_show_contour)

                action_remove_contour = QAction('Remove SPL - Contour Plot', self)
                action_remove_contour.triggered.connect(lambda: self.remove_spl_contour_plot(receiver_name))
                menu.addAction(action_remove_contour)

        if item.text().startswith('Result - '):
            action_clear_all = QAction('Clear All Plots', self)
            action_clear_all.triggered.connect(self.clear_all_spl_plots)
            menu.addAction(action_clear_all)

        # Add delete action for all items
        action_delete = QAction('Delete', self)
        action_delete.triggered.connect(lambda: self.remove_result_item(item))
        menu.addAction(action_delete)

        menu.exec(self.result_treeView.viewport().mapToGlobal(point))

    def show_spl_contour_plot(self, receiver_name):
        print(f"Showing contour plot for {receiver_name}")
        result = next((r for r in self.SPL_Result_Store if r['name'] == receiver_name), None)
        if result:
            print(f"Found result for {receiver_name}:")
            print(f"  Position shape: {np.array(result['position']).shape}")
            print(f"  Data shape: {np.array(result['data']).shape}")
            self.SPL_contour_display(receiver_result=result)
        else:
            print(f"No SPL data available for {receiver_name}")


    def remove_spl_contour_plot(self, receiver_name):
        if hasattr(self, 'spl_viz'):
            self.spl_viz.remove_contour_plot(receiver_name)
            self.spl_viz.update_plot()

    def clear_all_spl_plots(self):
        if hasattr(self, 'spl_viz'):
            self.spl_viz.clear_plot()
            self.spl_viz.update_plot()

    def get_receiver_type(self, receiver_name):
        for receiver in self.record_receiver_groups:
            if receiver['name'] == receiver_name:
                if receiver_name.startswith('plane'):
                    return 'plane'
                elif receiver_name.startswith('line'):
                    return 'line'
                elif receiver_name.startswith('point'):
                    return 'point'
        return 'unknown'

    def remove_result_item(self, item):
        parent = item.parent()
        if parent:
            row = parent.takeRow(item.row())
        else:
            row = self.result_model.takeRow(item.row())

        if item.text().startswith('Result - '):
            result_datetime = item.text()[9:]
            if hasattr(self, 'results'):
                self.results.pop(result_datetime, None)
        elif item.text().startswith('SPL - '):
            receiver_name = item.text()[6:]
            if hasattr(self, 'results'):
                for result in self.results.values():
                    result.pop(receiver_name, None)

        print(f"Removed item: {item.text()}")



    ## method for disply SPL

    def SPL_contour_display(self, temp_config=None, receiver_result=None):
        config = temp_config or self.contour_config

        if not receiver_result:
            print("No SPL data provided.")
            return

        receiver_positions = np.array(receiver_result['position'])
        SPL_data = np.array(receiver_result['data'])
        group_name = receiver_result['name']

        print(f"SPL_contour_display for {group_name}:")
        print(f"  Receiver positions shape: {receiver_positions.shape}")
        print(f"  SPL data shape: {SPL_data.shape}")
        print(f"  Receiver positions min: {receiver_positions.min(axis=0)}")
        print(f"  Receiver positions max: {receiver_positions.max(axis=0)}")

        if not hasattr(self, 'spl_viz'):
            self.spl_viz = SPLVisualization(self.shape, self.plot_interactor)

        density_factor = config.get("density_factor", 10) 
        interpolation_method = config.get("interpolation_method", "cubic") 
        contour_lines = config.get("contour_lines", 10)  

        faces = receiver_result.get('faces')
        if faces:
            plane_grid = pv.PolyData(
                receiver_positions,
                faces=np.asarray(faces, dtype=np.int64),
            )
            plane_grid['SPL'] = SPL_data
            mask = np.isfinite(SPL_data)
        else:
            plane_grid = self.spl_viz.create_2d_grid(
                receiver_positions,
                density_factor=density_factor,
            )
            interpolated_spl = self.spl_viz.interpolate_spl_data(
                receiver_positions,
                SPL_data,
                plane_grid.points,
                method=interpolation_method,
            )
            if interpolated_spl is None:
                print("Interpolation failed.")
                return
            interpolated_spl = gaussian_filter(interpolated_spl, sigma=1)
            plane_grid['SPL'] = interpolated_spl
            mask = self.spl_viz.mask_obstacles(plane_grid)

        config["contour_lines"] = contour_lines
        self.spl_viz.add_contour_plot(plane_grid, mask, config, group_name)
        self.spl_viz.update_plot()







## ================================== Pyvista View



    def setup_plot_widget(self):
        """Set up the PyVista interactor widget."""
        frame = QFrame()
        layout = QVBoxLayout(frame)

        # Create the interactor and add a mesh
        self.plot_interactor = QtInteractor(frame)

        # Set the gradient background
        self.plot_interactor.set_background('lightblue', top='white')

        # Optionally add axes and enable parallel projection
        self.plot_interactor.add_axes()
        self.plot_interactor.enable_parallel_projection()

        layout.addWidget(self.plot_interactor)
        return frame



### =================================== Toolbar 

    def setup_toolbar(self):
        """Create a toolbar with actions."""
        toolbar = QToolBar("My main toolbar")
        self.addToolBar(toolbar)


        button_new_project = QAction(QIcon('PIC_MANU/new_project.png'),"New Project", self)
        #button_new_project.triggered.connect(self.reset_view)
        toolbar.addAction(button_new_project)

        button_import_project = QAction(QIcon('PIC_MANU/import_project.png'),"import Project", self)
        #button_new_project.triggered.connect(self.reset_view)
        toolbar.addAction(button_import_project)


        button_save_project = QAction(QIcon('PIC_MANU/save_project.png'),"Save Project", self)
        #button_new_project.triggered.connect(self.reset_view)
        toolbar.addAction(button_save_project)

        toolbar.addSeparator()

        button_reset_view = QAction(QIcon('PIC_MANU/base.png'),"Reset View", self)
        button_reset_view.triggered.connect(self.reset_view)
        toolbar.addAction(button_reset_view)

        button_plane = QAction(QIcon('PIC_MANU/section.png'), "Section", self)
        button_plane.triggered.connect(self.section_view)
        toolbar.addAction(button_plane)

        button_show_cell = QAction(QIcon('PIC_MANU/cell.png'),"Cell", self)
        button_show_cell.triggered.connect(self.cell_view)
        toolbar.addAction(button_show_cell)

        button_show_feature_edge = QAction(QIcon('PIC_MANU/edge.png'),"feature edges", self)
        button_show_feature_edge.triggered.connect(self.feature_edge_view)
        toolbar.addAction(button_show_feature_edge)

        toolbar.addSeparator()

        button_show_xy_plane = QAction(QIcon('PIC_MANU/xy_plane.png'),"X-Y plane", self)
        button_show_xy_plane.triggered.connect(self.xy_view)
        toolbar.addAction(button_show_xy_plane)

        button_show_xz_plane = QAction(QIcon('PIC_MANU/xz_plane.png'),"X-Z plane", self)
        button_show_xz_plane.triggered.connect(self.xz_view)
        toolbar.addAction(button_show_xz_plane)

        button_show_yz_plane = QAction(QIcon('PIC_MANU/yz_plane.png'),"Y-Z plane", self)
        button_show_yz_plane.triggered.connect(self.yz_view)
        toolbar.addAction(button_show_yz_plane)

        button_axis = QAction(QIcon('PIC_MANU/axes_scale.png'),"Axis", self)
        button_axis.triggered.connect(self.toggle_axis_visibility)
        toolbar.addAction(button_axis)


        toolbar.addSeparator()
        result_contour_config = QAction(QIcon('PIC_MANU/Contour.png'),"Contour Config", self)
        result_contour_config.triggered.connect(self.contour_editor)
        toolbar.addAction(result_contour_config)


    def section_view(self):
        self.plot_interactor.clear()
        self.plot_interactor.add_mesh_clip_plane(self.shape, show_edges=True, color='silver')

    def reset_view(self):
        self.plot_interactor.clear()
        self.plot_interactor.add_mesh(self.shape, show_edges=True, color='silver')

    def cell_view(self):
        self.plot_interactor.clear()
        self.plot_interactor.add_mesh(self.cell_edge, show_edges=True, color='k')

    def feature_edge_view(self):
        self.plot_interactor.clear()
        self.plot_interactor.add_mesh(self.feature_edge, show_edges=True, color='k', line_width = 2)

    def xy_view(self):
        self.plot_interactor.view_xy()
        
    def xz_view(self):
        self.plot_interactor.view_xz()

    def yz_view(self):
        self.plot_interactor.view_yz()
        

    def toggle_axis_visibility(self):
        if self.axis_displayed:
            self.plot_interactor.show_axes = False
            self.plot_interactor.remove_bounding_box()
        else:
            self.plot_interactor.show_axes = True
            self.plot_interactor.show_grid(xtitle="X (m)", ytitle="Y (m)", ztitle="Z (m)", color = 'k')
     

    def contour_editor(self):
        print("TEST")



# Start the application
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Set the global locale to English (for Arabic numerals)
    QLocale.setDefault(QLocale(QLocale.English))

    app.setStyle('Fusion')
    #app.setStyle('Windows')
    window = MainWindow()
    window.show()
    app.exec()
