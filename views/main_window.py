from PySide6.QtWidgets import (
    QMainWindow,
    QFileDialog,
    QToolBar,
    QComboBox,
    QSplitter,
    QLabel,
)
from PySide6.QtCore import Qt
from views import SliceView, SliceToolBar
from views import DataManager
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
        self.init_panel = InitPanel(data_manager=self.data_manager)
        self.volume_panel = VolumePanel(
            volume_renderer=volume_renderer,
            data_manager=self.data_manager,
        )
        self.slice_panel = SlicePanel(
            data_manager=self.data_manager,
        )
        self.model_panel = ModelPanel(data_manager=self.data_manager)
        self.select_panel = [
            {
                "label": "初始化面板",
                "widget": self.init_panel,
            },
            {
                "label": "Volume Control Panel",
                "widget": self.volume_panel,
            },
            {
                "label": "Slice Control Panel",
                "widget": self.slice_panel,
            },
            {
                "label": "Modle Control Panel",
                "widget": self.model_panel,
            },
        ]
        self.slice_toolbar = [
            SliceToolBar(self.slice_views[i], self.slice_panel, self)
            for i in range(len(self.slice_views))
        ]
        right_widget.set_pane(0, 0, self.slice_views[0], self.slice_toolbar[0])
        right_widget.set_pane(0, 1, volume_renderer)
        right_widget.set_pane(1, 0, self.slice_views[1], self.slice_toolbar[1])
        right_widget.set_pane(1, 1, self.slice_views[2], self.slice_toolbar[2])
        # self.data_manager.register(volume_renderer)

        # Menu Bar設定
        # menu_bar = self.menuBar()
        # # 增加新增讀取NIfTI功能
        # file_menu = menu_bar.addMenu("File")
        # open_action = QAction("Open NIfTI", self)
        # open_action.triggered.connect(self.load_nifti)
        # file_menu.addAction(open_action)

        # setting_menu = menu_bar.addMenu("設定")
        # model_set_action = QAction("設定模型", self, checkable=False, checked=False)
        # setting_menu.addAction(model_set_action)
        # model_set_action.triggered.connect(self.open_model_config)

        # self.register_menu_bar(view_menu)

        self.tool_bar = QToolBar("Dock Selector", self)
        self.tool_bar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.tool_bar)

        self.panel_selector = QComboBox(self)
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

    def resizeEvent(self, event):
        if event.oldSize().height() > 0:  # 避免第一次 show() 觸發
            total = self.splitter.height()
            self.splitter.setSizes([total * 0.3, total * 0.7])
        super().resizeEvent(event)
