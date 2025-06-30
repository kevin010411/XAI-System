from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
)
from views.volume.transfer_editor import OpacityEditor
from views import HistogramViewer


class VolumePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("Volume Control Panel")
        self.setMinimumSize(300, 200)
        self.setStyleSheet("background-color: #f0f0f0;")
        self._init_control_panel()

    def _init_control_panel(self):
        self.histogram_viewer = HistogramViewer()
        self.opacity_editor = OpacityEditor(self.update_transfer_function)
        self.control_panel_layout = QVBoxLayout(self)
        self.control_panel_layout.addWidget(self.histogram_viewer)
        self.control_panel_layout.addWidget(self.opacity_editor)

    def update_transfer_function(self):
        """
        需要把 OpacityEditor 的透明度曲線傳遞給 VolumeDock，
        """
        pass
