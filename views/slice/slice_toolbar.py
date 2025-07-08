from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class SliceSideBar(QWidget):
    """The red toolbar at the very top (reset btn + slider + label)."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        # --- Red background -------------------------------------------------
        pal: QPalette = self.palette()
        pal.setColor(QPalette.Window, Qt.red)
        self.setAutoFillBackground(True)
        self.setPalette(pal)

        # --- Widgets --------------------------------------------------------
        self.btn_reset = QToolButton(text="R", toolTip="Reset camera")
        self.btn_reset.setFixedSize(24, 24)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(-500)
        self.slider.setMaximum(500)
        self.slider.setValue(0)
        self.slider.setSingleStep(1)

        self.lbl_value = QLabel("S: 0.0 mm")
        self.lbl_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        # --- Layout ---------------------------------------------------------
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 2, 8, 2)
        lay.setSpacing(6)
        lay.addWidget(self.btn_reset)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.lbl_value)

        # Sync slider → label for demo purposes
        self.slider.valueChanged.connect(
            lambda v: self.lbl_value.setText(f"S: {v:.1f} mm")
        )


class SliceControlRow(QWidget):
    """A single row in the vertical list (spinbox + combo)."""

    def __init__(self, default_label: str = "None", parent: QWidget | None = None):
        super().__init__(parent)
        self.spin = QDoubleSpinBox()
        self.spin.setRange(-1000.0, 1000.0)
        self.spin.setDecimals(2)
        self.spin.setValue(0.0)
        self.spin.setFixedWidth(60)

        self.combo = QComboBox()
        self.combo.addItem(default_label)
        self.combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(4)
        lay.addWidget(self.spin)
        lay.addWidget(self.combo, 1)


class SliceToolBar(QWidget):
    """Widget that imitates the screenshot layout."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self.toolbar = SliceSideBar(self)

        # Three placeholder rows, could be dynamic.
        self.row1 = SliceControlRow("None")
        self.row2 = SliceControlRow("None")
        self.row3 = SliceControlRow("87")

        # Frame for visual separation (optional)
        frame = QFrame()
        frame.setFrameShape(QFrame.HLine)
        frame.setFrameShadow(QFrame.Sunken)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.toolbar)
        lay.addWidget(frame)
        lay.addWidget(self.row1)
        lay.addWidget(self.row2)
        lay.addWidget(self.row3)
        lay.addStretch(1)
