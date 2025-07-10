from PySide6.QtCore import Qt, QStringListModel
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QSlider,
    QAbstractSpinBox,
    QVBoxLayout,
    QWidget,
)


class SliceControlRow(QWidget):
    """A single row in the vertical list (spinbox + combo)."""

    def __init__(self, row_string_list_model, parent: QWidget | None = None):
        super().__init__(parent)
        self.spin = QDoubleSpinBox()
        self.spin.setRange(0.0, 100.0)
        self.spin.setDecimals(2)
        self.spin.setValue(0.0)
        self.spin.setFixedWidth(60)
        self.spin.setButtonSymbols(QAbstractSpinBox.NoButtons)

        self.combo = QComboBox()
        self.combo.setModel(row_string_list_model)
        self.combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(4)
        lay.addWidget(self.spin)
        lay.addWidget(self.combo, 1)

    def get_select(self):
        return {"img_name": self.combo.currentText(), "opacity": self.spin.value()}


class SliceToolBar(QWidget):
    """Widget that imitates the screenshot layout."""

    def __init__(self, slice_view, slice_panel, parent: QWidget | None = None):
        super().__init__(parent)

        self.slice_view = slice_view
        self.slice_panel = slice_panel

        self.slider = QSlider(Qt.Vertical)
        self.slider.setRange(0, 100)
        self.slider.setValue(100)
        self.slider.setFixedHeight(80)
        self.slider.valueChanged.connect(self._on_slider_changed)

        self.img_name_list_model = QStringListModel()
        for sig in [
            slice_panel.data_manager.img_name_list_model.dataChanged,
            slice_panel.data_manager.img_name_list_model.rowsRemoved,
            slice_panel.data_manager.img_name_list_model.modelReset,
        ]:
            sig.connect(self._on_model_data_changed)

        # Three placeholder rows, could be dynamic.
        self.row1 = SliceControlRow(self.img_name_list_model)
        self.row1.spin.setEnabled(False)
        self.row1.combo.setEnabled(False)
        self.row2 = SliceControlRow(self.img_name_list_model)
        self.row3 = SliceControlRow(self.img_name_list_model)
        self.row1.combo.currentIndexChanged.connect(self._on_img_row_change)
        self.row2.combo.currentIndexChanged.connect(self._on_img_row_change)
        self.row3.combo.currentIndexChanged.connect(self._on_img_row_change)

        # Frame for visual separation (optional)
        frame = QFrame()
        frame.setFrameShape(QFrame.HLine)
        frame.setFrameShadow(QFrame.Sunken)

        right_container = QWidget()
        row_layout = QVBoxLayout(right_container)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(0)
        row_layout.addWidget(frame)
        row_layout.addWidget(self.row1)
        row_layout.addWidget(self.row2)
        row_layout.addWidget(self.row3)
        row_layout.addStretch(1)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.slider)
        layout.addWidget(right_container)
        layout.setStretch(0, 1)
        layout.setStretch(1, 9)

        self._sync_rows(self.slider.value())

    def _on_slider_changed(self, val: int):
        """滑桿改動：調整透明度 + 同步 row2/row3。"""
        self._sync_rows(val)
        if hasattr(self.slice_view, "set_opacity"):
            self.slice_view.set_opacity(val / 100.0)

    def _on_model_data_changed(self):
        data_list = [
            ""
        ] + self.slice_panel.data_manager.img_name_list_model.stringList()
        self.img_name_list_model.setStringList(data_list)

    # ----------------- 新增的 slot -----------------
    def _sync_rows(self, value: int):
        """保持 row2 + row3 == 100，row2 跟滑桿同步。"""
        self.row2.spin.blockSignals(True)
        self.row3.spin.blockSignals(True)

        self.row2.spin.setValue(value)  # row2 = slider
        self.row3.spin.setValue(100 - value)  # row3 = 補足 100

        self.row2.spin.blockSignals(False)
        self.row3.spin.blockSignals(False)

    def _on_img_row_change(self):
        setting = self.slice_panel.save_setting()
        row_data = [
            self.row1.get_select(),
            self.row2.get_select(),
            self.row3.get_select(),
        ]
        show_data = [
            {
                "img": self.slice_panel.data_manager.get_img(data["img_name"]),
                "opacity": data["opacity"],
                "cmap": setting[data["img_name"]],
            }
            for data in row_data
            if data["img_name"] != ""
        ]
        self.slice_view.update(show_data)
