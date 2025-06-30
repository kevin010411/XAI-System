from PySide6.QtWidgets import (
    QMainWindow,
    QFileDialog,
    QToolBar,
    QComboBox,
    QSplitter,
    QLabel,
)
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt
from views import VolumeDock
from views import SegmentationDock
from views import ExplainDock
from views import SliceDock
from views import DataManager
from views import ModelConfigDialog
from views import Split2x2Window
from views import InitPanel, VolumePanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("研究平台介面")
        self.resize(1200, 800)
        self.set_dock_config()

        self.data_manager = DataManager()

        # 主畫面
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        left_widget = InitPanel(data_manager=self.data_manager)
        right_widget = Split2x2Window()
        self.splitter.addWidget(left_widget)
        self.splitter.addWidget(right_widget)
        self.splitter.setSizes([2000, 8000])
        self.splitter.setStyleSheet(
            """
            QSplitter::handle {
                background: #5F5F5F;   /* 想要多細的線，請搭配 handleWidth */
            }
        """
        )
        self.splitter.setHandleWidth(2)

        self.setCentralWidget(self.splitter)

        #  建立 dock

        # 放入Slice & Volume View
        # self.volume_dock = VolumeDock()
        # self.segmentation_dock = SegmentationDock()
        # self.explain_dock = ExplainDock()
        # self.slice_docks = [
        #     SliceDock(view_type) for view_type in ("axial", "coronal", "sagittal")
        # ]
        # right_widget.set_pane(0, 0, self.slice_docks[0])
        # right_widget.set_pane(0, 1, self.volume_dock)
        # right_widget.set_pane(1, 0, self.slice_docks[1])
        # right_widget.set_pane(1, 1, self.slice_docks[2])

        # self.docks = [
        #     ("CT Volume Viewer", self.volume_dock, Qt.LeftDockWidgetArea),
        #     ("Segmentation Viewer", self.segmentation_dock, Qt.RightDockWidgetArea),
        #     ("Explain Viewer", self.explain_dock, Qt.BottomDockWidgetArea),
        # ]

        #     # 訂閱data_manager
        #     self.data_manager.register(self.volume_dock)
        #     self.data_manager.register(self.segmentation_dock)

        menu_bar = self.menuBar()
        # 增加新增讀取NIfTI功能
        file_menu = menu_bar.addMenu("File")
        open_action = QAction("Open NIfTI", self)
        open_action.triggered.connect(self.load_nifti)
        file_menu.addAction(open_action)

        setting_menu = menu_bar.addMenu("設定")
        model_set_action = QAction("設定模型", self, checkable=False, checked=False)
        setting_menu.addAction(model_set_action)
        model_set_action.triggered.connect(self.open_model_config)

        self.dock_actions = []
        # self.register_menu_bar(view_menu)

        self.tool_bar = QToolBar("Dock Selector", self)
        self.tool_bar.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, self.tool_bar)

        self.dock_selector = QComboBox(self)
        self.select_panel = [
            {"label": "初始化面板", "factory": InitPanel},
            {"label": "Volume Control Panel", "factory": VolumePanel},
        ]
        self._panel_cache = {"0": left_widget}
        self.dock_selector.addItems([item["label"] for item in self.select_panel])
        self.dock_selector.currentIndexChanged.connect(self.change_control_panel)
        self.tool_bar.addWidget(self.dock_selector)

    def register_menu_bar(self, view_menu):
        def visibility_handler(dock, visible):
            if visible and self.data_manager.get_current() is not None:
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

    def change_control_panel(self, index):
        panel_info = self.select_panel[index]
        factory = panel_info["factory"]
        if index not in self._panel_cache:
            self._panel_cache[index] = factory(parent=self)
        widget = self._panel_cache[index]
        if 0 >= self.splitter.count():
            self.splitter.insertWidget(0, widget)
        else:
            self.splitter.replaceWidget(0, widget)

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

    def resizeEvent(self, event):
        if event.oldSize().height() > 0:  # 避免第一次 show() 觸發
            total = self.splitter.height()
            self.splitter.setSizes([total * 0.3, total * 0.7])
        super().resizeEvent(event)

    # def closeEvent(self, ev):
    #     """程式退出前，先讓各 Dock 釋放 VTK 相關資源。"""
    #     for _, dock, _ in self.docks:
    #         if hasattr(dock, "prepare_for_exit"):
    #             dock.prepare_for_exit()
    #     super().closeEvent(ev)  # 交回 Qt 正常結束
