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
        self._settings = {}

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
        if new_key in self._settings:
            self._apply_settings_to_widgets(self._settings[new_key])
        else:
            self._initialise_window_spins(self._current_img)

    def save_setting(self):
        self._settings[self._current_key] = {
            "mode": self.display_mode_selector.currentText(),
            "vmin": self.min_val_spin.value(),
            "vmax": self.max_val_spin.value(),
        }
        return self._settings

    def _initialise_window_spins(self, img) -> None:  # noqa: ANN001
        data = img.get_fdata()
        v_min, v_max = float(data.min()), float(data.max())
        with QSignalBlocker(self.min_val_spin), QSignalBlocker(self.max_val_spin):
            self.min_val_spin.setRange(v_min, v_max)
            self.max_val_spin.setRange(v_min, v_max)
            self.min_val_spin.setValue(v_min)
            self.max_val_spin.setValue(v_max)

    def change_slice_display_mode(self, index: int) -> None:
        """Change the display mode of all slice viewers."""
        new_mode = self.display_mode_selector.currentText()

    def update(self, img_name, img):
        super().update(img_name, img)
        self._initialise_window_spins(img)
