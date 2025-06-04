from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QDockWidget
from .utils import wrap_with_frame


class ExplainDock(QDockWidget):
    def __init__(self):
        super().__init__("Explain Viewer")
        widget = QWidget()
        layout = QVBoxLayout()

        self.label = QLabel("這裡將顯示 XAI 熱力圖 (Grad-CAM)")
        layout.addWidget(self.label)

        widget.setLayout(layout)
        self.setWidget(wrap_with_frame(widget))
