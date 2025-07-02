from functools import partial
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
from views import SliceView
from views import DataManager
from views import ModelConfigDialog
from views import Split2x2Window
from views import InitPanel, VolumePanel, SlicePanel, ModelPanel
from views import VolumeRenderer


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("研究平台介面")
        self.resize(1200, 800)

        self.data_manager = DataManager()

        # 主畫面
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        left_widget = QLabel(
            "左側面板 (可放置其他工具或資訊)", alignment=Qt.AlignmentFlag.AlignCenter
        )
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

        volume_renderer = VolumeRenderer()
        self.slice_views = [
            SliceView(
                view_type, show_in_volume_callback=volume_renderer.update_slice_plane
            )
            for view_type in ("axial", "coronal", "sagittal")
        ]
        right_widget.set_pane(0, 0, self.slice_views[0])
        right_widget.set_pane(0, 1, volume_renderer)
        right_widget.set_pane(1, 0, self.slice_views[1])
        right_widget.set_pane(1, 1, self.slice_views[2])
        # self.data_manager.register(volume_renderer)

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
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.tool_bar)

        self.panel_selector = QComboBox(self)
        self.select_panel = [
            {
                "label": "初始化面板",
                "widget": InitPanel(data_manager=self.data_manager),
            },
            {
                "label": "Volume Control Panel",
                "widget": VolumePanel(
                    volume_renderer=volume_renderer,
                    data_manager=self.data_manager,
                ),
            },
            {
                "label": "Slice Control Panel",
                "widget": SlicePanel(
                    slice_viewers=self.slice_views,
                    data_manager=self.data_manager,
                ),
            },
            {
                "label": "Modle Control Panel",
                "widget": ModelPanel(data_manager=self.data_manager),
            },
        ]
        for panel in self.select_panel:
            self.data_manager.register(panel["widget"])
        self.panel_selector.addItems([item["label"] for item in self.select_panel])
        self.panel_selector.currentIndexChanged.connect(self.change_control_panel)
        self.tool_bar.addWidget(self.panel_selector)
        self.change_control_panel(0)  # 預設載入 InitPanel

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
        widget = panel_info["widget"]
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
