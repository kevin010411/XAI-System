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
        self.img_selector.setPlaceholderText("(尚未載入影像)")
        self.selector_row.addWidget(
            self.img_selector, 1
        )  # stretch=1 讓下拉佔滿剩餘寬度
        self.layout.addLayout(self.selector_row)

        self._stretch_idx = self.layout.count()
        self.layout.addStretch()
        self.refresh_img_selector()

    def update(self, img_name, img):
        self.refresh_img_selector(select_last=True)

    # ---------- 下拉選單 ----------
    def refresh_img_selector(
        self, keep_current: bool = True, select_last: bool = False
    ):
        """重新把 data_manager.imgs 填進下拉清單。"""
        remembered_data = (
            self.img_selector.currentData(Qt.ItemDataRole.UserRole)
            if keep_current
            else None
        )
        with QSignalBlocker(self.img_selector):
            self.img_selector.clear()

            # 2. 按鍵名排序填入 (你也可改成其它排序規則)
            for idx, (img_name, img_data) in enumerate(self.data_manager.imgs.items()):
                self.img_selector.addItem(img_name, userData=idx)  # userData 存 index

            # 清單為空 → 顯示 placeholder
            if self.img_selector.count() == 0:
                self.img_selector.setCurrentIndex(-1)
            else:
                target_index = -1

                if select_last:  # 明確要求「最後一筆」
                    target_index = self.img_selector.count() - 1

                elif remembered_data is not None:  # 嘗試找回舊選項
                    target_index = self.img_selector.findData(
                        remembered_data, role=Qt.ItemDataRole.UserRole
                    )

                # 找不到 (== -1) 就預設選第一筆
                if target_index == -1:
                    target_index = 0

                self.img_selector.setCurrentIndex(target_index)

    # ---------- utility for subclasses ----------
    def add_row_above_stretch(self, layout: QHBoxLayout):
        """Insert a row *above* the stretch so it stays visible."""
        self.layout.insertLayout(self._stretch_idx, layout)
        self._stretch_idx += 1  # keep stretch at the bottom
