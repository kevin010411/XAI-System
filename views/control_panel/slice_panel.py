from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QComboBox,
    QDoubleSpinBox,
)
from PySide6.QtCore import Qt, QSignalBlocker

from .base_panel import BasePanel


class SlicePanel(BasePanel):

    _DISPLAY_MODES = ["gray", "cold_to_hot", "heatmap"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.setWindowTitle("Slice Control Panel")

        self._current_key: str | None = None
        self._current_img = None
        self.settings = {}
        self._row_name = []

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("顯示模式："))
        self.display_mode_selector = QComboBox()
        self.display_mode_selector.addItems(self._DISPLAY_MODES)
        self.display_mode_selector.currentIndexChanged.connect(
            self.change_slice_display_mode
        )
        mode_row.addWidget(self.display_mode_selector, 1)
        self.add_row_above_stretch(mode_row)

        # -------------------------- window row
        window_row = QHBoxLayout()
        window_row.addWidget(QLabel("Window Min-Max："))

        self.min_val_spin = QDoubleSpinBox()
        self.min_val_spin.setDecimals(2)
        # self.min_val_spin.valueChanged.connect(self.on_window_changed)
        window_row.addWidget(self.min_val_spin)

        self.max_val_spin = QDoubleSpinBox()
        self.max_val_spin.setDecimals(2)
        # self.max_val_spin.valueChanged.connect(self.on_window_changed)
        window_row.addWidget(self.max_val_spin)

        self.add_row_above_stretch(window_row)

        self.img_selector.currentIndexChanged.connect(self.on_img_selected)
        self.data_manager.img_name_list_model.rowsInserted.connect(
            self._on_rows_inserted
        )
        self.data_manager.img_name_list_model.rowsRemoved.connect(self._on_rows_removed)
        self.data_manager.img_name_list_model.dataChanged.connect(self._on_data_changed)
        self.data_manager.img_name_list_model.modelReset.connect(self._on_model_reset)

    def _apply_settings_to_widgets(self, settings) -> None:
        """Push a settings‑dict into all widgets *without* triggering signals."""
        with QSignalBlocker(self.display_mode_selector):
            mode_index = self.display_mode_selector.findText(settings["mode"])
            if mode_index != -1:
                self.display_mode_selector.setCurrentIndex(mode_index)
        with QSignalBlocker(self.min_val_spin), QSignalBlocker(self.max_val_spin):
            self.min_val_spin.setValue(settings["vmin"])
            self.max_val_spin.setValue(settings["vmax"])

    def on_img_selected(self, index):
        """使用者在下拉選了新影像 → 通知所有 SliceViewer 更新。"""
        new_key = self.img_selector.currentText()
        if not new_key:
            return  # nothing loaded
        # 1) Persist current settings before switching away
        if self._current_key is not None:
            self.save_setting()

        # 2) Update current refs
        self._current_key = new_key
        self._current_img = self.data_manager.get_img(new_key)

        # 3) Restore previous settings *if any*, else defaults
        if new_key in self.settings:
            self._apply_settings_to_widgets(self.settings[new_key])
        else:
            self._create_slice_init_setting(self._current_img, new_key)

    def save_setting(self, img_name=None):
        if img_name is not None:
            self.settings[img_name] = {
                "mode": self.display_mode_selector.currentText(),
                "vmin": self.min_val_spin.value(),
                "vmax": self.max_val_spin.value(),
            }
        else:
            self.settings[self._current_key] = {
                "mode": self.display_mode_selector.currentText(),
                "vmin": self.min_val_spin.value(),
                "vmax": self.max_val_spin.value(),
            }
        return self.settings

    def _create_slice_init_setting(self, img, img_name) -> None:
        data = img.get_fdata()
        v_min, v_max = float(data.min()), float(data.max())
        with QSignalBlocker(self.min_val_spin), QSignalBlocker(self.max_val_spin):
            self.min_val_spin.setRange(v_min, v_max)
            self.max_val_spin.setRange(v_min, v_max)
            self.min_val_spin.setValue(v_min)
            self.max_val_spin.setValue(v_max)
        self.save_setting(img_name)

    def _on_rows_inserted(self, parent, first: int, last: int):
        """
        新增行：為 [first, last] 每一行插入一份預設設定
        """
        for row in range(first, last + 1):
            img_name = self.data_manager.img_name_list_model.data(
                self.data_manager.img_name_list_model.index(row, 0), Qt.DisplayRole
            )
            self._row_name.append(img_name)
            if img_name == "":
                return
            img = self.data_manager.get_img(img_name)
            self._create_slice_init_setting(img, img_name)

    def _on_rows_removed(self, parent, first: int, last: int):
        """
        刪除行：同步移除對應設定
        """
        for img_name in self._row_name[first : last + 1]:
            del self.settings[img_name]

    def _on_data_changed(self, topLeft, bottomRight, roles):
        """
        編輯既有字串：若你的設定內容和文字有關，這裡更新
        """
        for row in range(topLeft.row(), bottomRight.row() + 1):
            new_img_name = self.data_manager.img_name_list_model.data(
                self.data_manager.img_name_list_model.index(row, 0), Qt.DisplayRole
            )
            old_name = self._row_name[row]
            if old_name != "":
                self._row_name[row] = new_img_name
                self.settings[new_img_name] = self.settings[old_name]
                del self.save_setting[old_name]
            else:
                self._row_name[row] = new_img_name
                img = self.data_manager.get_img(new_img_name)
                self._create_slice_init_setting(img, new_img_name)

    def _on_model_reset(self):
        """
        重設 model：整份設定也重建
        """
        self.settings = [
            self._create_slice_init_setting(s)
            for s in self.data_manager.img_name_list_model.stringList()
        ]

    def change_slice_display_mode(self, index: int) -> None:
        """Change the display mode of all slice viewers."""
        pass
