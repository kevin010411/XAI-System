from PySide6.QtWidgets import QWidget, QVBoxLayout, QComboBox, QHBoxLayout, QLabel
from PySide6.QtCore import Qt, QSignalBlocker


class BasePanel(QWidget):

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)

        self.setStyleSheet("background-color: #f0f0f0;")
        self.setMinimumSize(300, 200)

        self.data_manager = data_manager
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(8)

        self.selector_row = QHBoxLayout()
        self.selector_label = QLabel("顯示影像：")
        self.selector_row.addWidget(self.selector_label)

        self.img_selector = QComboBox()
        self.img_selector.setModel(data_manager.img_name_list_model)
        self.img_selector.setPlaceholderText("尚未載入影像")
        self.selector_row.addWidget(
            self.img_selector, 1
        )  # stretch=1 讓下拉佔滿剩餘寬度
        self.layout.addLayout(self.selector_row)

        self._stretch_idx = self.layout.count()
        self.layout.addStretch()

    def update(self, img_name, img):
        # self.refresh_img_selector(select_last=True)
        pass

    # ---------- utility for subclasses ----------
    def add_row_above_stretch(self, layout: QHBoxLayout):
        """Insert a row *above* the stretch so it stays visible."""
        self.layout.insertLayout(self._stretch_idx, layout)
        self._stretch_idx += 1  # keep stretch at the bottom
