from PySide6.QtWidgets import QWidget, QVBoxLayout, QComboBox, QHBoxLayout, QLabel
from PySide6.QtCore import Qt, QSignalBlocker


class BasePanel(QWidget):

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)

        self.setStyleSheet("background-color: #f0f0f0;")
        self.setMinimumSize(300, 200)

        self.data_manager = data_manager
        self._row_name = []
        # data_manager資料改變時的觸發邏輯
        self.data_manager.img_name_list_model.rowsInserted.connect(
            self._on_rows_inserted
        )
        self.data_manager.img_name_list_model.rowsRemoved.connect(self._on_rows_removed)
        self.data_manager.img_name_list_model.dataChanged.connect(self._on_data_changed)
        self.data_manager.img_name_list_model.modelReset.connect(self._on_model_reset)

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

    def _on_rows_inserted(self, parent, first: int, last: int):
        """
        data_manager中的first~last+1的中間插入資料時觸發
        """
        for row in range(first, last + 1):
            img_name = self.data_manager.img_name_list_model.data(
                self.data_manager.img_name_list_model.index(row, 0), Qt.DisplayRole
            )
            self._row_name.append(img_name)

    def _on_rows_removed(self, parent, first: int, last: int):
        """
        data_manager中的first~last+1的資料被刪除時觸發
        """
        del self._row_name[first : last + 1]

    def _on_data_changed(self, topLeft, bottomRight, roles):
        """
        data_manager中的有資料被重新命名時觸發
        """
        for row in range(topLeft.row(), bottomRight.row() + 1):
            new_img_name = self.data_manager.img_name_list_model.data(
                self.data_manager.img_name_list_model.index(row, 0), Qt.DisplayRole
            )
            self._row_name[row] = new_img_name

    def _on_model_reset(self):
        """
        重設data_manager中的資料時觸發
        """
        pass

    # ---------- utility for subclasses ----------
    def add_row_above_stretch(self, layout: QHBoxLayout):
        """Insert a row *above* the stretch so it stays visible."""
        self.layout.insertLayout(self._stretch_idx, layout)
        self._stretch_idx += 1  # keep stretch at the bottom
