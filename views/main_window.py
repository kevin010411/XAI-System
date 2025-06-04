from PySide6.QtWidgets import QMainWindow, QFileDialog
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt
from views import VolumeDock
from views import SegmentationDock
from views import ExplainDock
from views import SliceDock
from views import DataManager
from views import ModelConfigDialog


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("研究平台介面")
        self.resize(1200, 800)
        self.set_dock_config()

        self.data_manager = DataManager()

        # 建立 dock
        self.volume_dock = VolumeDock()
        self.segmentation_dock = SegmentationDock()
        self.explain_dock = ExplainDock()
        self.slice_axial = SliceDock("axial", self.volume_dock.update_slice_plane)
        self.slice_coronal = SliceDock("coronal", self.volume_dock.update_slice_plane)
        self.slice_sagittal = SliceDock("sagittal", self.volume_dock.update_slice_plane)

        self.docks = [
            ("CT Volume Viewer", self.volume_dock, Qt.LeftDockWidgetArea),
            ("Segmentation Viewer", self.segmentation_dock, Qt.RightDockWidgetArea),
            ("Explain Viewer", self.explain_dock, Qt.BottomDockWidgetArea),
            ("Slice Axial (Z)", self.slice_axial, Qt.RightDockWidgetArea),
            ("Slice Coronal (Y)", self.slice_coronal, Qt.RightDockWidgetArea),
            ("Slice Sagittal (X)", self.slice_sagittal, Qt.RightDockWidgetArea),
        ]

        # 訂閱data_manager
        self.data_manager.register(self.volume_dock)
        self.data_manager.register(self.slice_axial)
        self.data_manager.register(self.slice_coronal)
        self.data_manager.register(self.slice_sagittal)
        self.data_manager.register(self.segmentation_dock)

        menu_bar = self.menuBar()
        # 增加新增讀取NIfTI功能
        file_menu = menu_bar.addMenu("File")
        open_action = QAction("Open NIfTI", self)
        open_action.triggered.connect(self.load_nifti)
        file_menu.addAction(open_action)

        # 建立 menu 讓使用者可以重新打開 dock
        view_menu = menu_bar.addMenu("視窗")

        setting_menu = menu_bar.addMenu("設定")
        model_set_action = QAction("設定模型", self, checkable=True, checked=True)
        setting_menu.addAction(model_set_action)
        model_set_action.triggered.connect(self.open_model_config)

        self.dock_actions = []
        self.register_menu_bar(view_menu)

    def register_menu_bar(self, view_menu):
        def visibility_handler(dock, visible):
            if visible:
                dock.update(self.data_manager.get_current())
            else:  # TODO 清除視窗內容
                pass

        for label, dock, area in self.docks:
            self.addDockWidget(area, dock)
            action = QAction(label, self, checkable=True, checked=True)
            view_menu.addAction(action)
            action.toggled.connect(dock.setVisible)
            dock.visibilityChanged.connect(action.setChecked)
            dock.visibilityChanged.connect(
                lambda visible, d=dock: visibility_handler(d, visible)
            )
            self.dock_actions.append(action)

    def set_dock_config(self):
        self.setDockOptions(
            QMainWindow.AllowTabbedDocks
            | QMainWindow.AllowNestedDocks
            | QMainWindow.AnimatedDocks
        )

    def load_nifti(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "選擇 NIfTI 檔案", "", "NIfTI Files (*.nii.gz *.nii)"
        )
        if file_path:
            try:
                self.data_manager.load_nifti(file_path)
            except Exception as e:
                print("讀取失敗:", e)

    def open_model_config(self):
        dlg = ModelConfigDialog(parent=self)
        if dlg.exec():
            model, config = dlg.get_model_and_config()
            if model is None:
                print("不讀入模型")
            else:
                print("拿到模型物件：", model.__class__.__name__)
                self.model = model
                self.segmentation_dock.update_model_and_config(model, config)
