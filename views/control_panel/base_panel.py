from PySide6.QtWidgets import QWidget, QVBoxLayout, QComboBox, QHBoxLayout, QLabel


class BasePanel(QWidget):

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)

        self.setStyleSheet("background-color: #f0f0f0;")
        self.setMinimumSize(300, 200)

        self.data_manager = data_manager
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(8)

        selector_row = QHBoxLayout()
        self.selector_label = QLabel("顯示影像：")
        selector_row.addWidget(self.selector_label)

        self.img_selector = QComboBox()
        self.img_selector.setPlaceholderText("(尚未載入影像)")
        selector_row.addWidget(self.img_selector, 1)  # stretch=1 讓下拉佔滿剩餘寬度
        self.layout.addLayout(selector_row)

        self._stretch_idx = self.layout.count()
        self.layout.addStretch()
        self.refresh_img_selector()

    def update(self, img):
        self.refresh_img_selector()

    # ---------- 下拉選單 ----------
    def refresh_img_selector(self, select_last: bool = False):
        """重新把 data_manager.imgs 填進下拉清單。"""
        self.img_selector.blockSignals(True)  # 暫停 signal，避免多次觸發
        self.img_selector.clear()

        for idx, (img_name, img_data) in enumerate(self.data_manager.imgs.items()):
            # 預設用檔名 (若 DataManager 有別名屬性可自行替換)
            self.img_selector.addItem(img_name, userData=idx)  # userData 存 index

        if select_last and self.data_manager.imgs:
            self.img_selector.setCurrentIndex(len(self.data_manager.imgs) - 1)

        self.img_selector.blockSignals(False)

    # ---------- utility for subclasses ----------
    def add_row_above_stretch(self, layout: QHBoxLayout):
        """Insert a row *above* the stretch so it stays visible."""
        self.layout.insertLayout(self._stretch_idx, layout)
        self._stretch_idx += 1  # keep stretch at the bottom
