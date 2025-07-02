from PySide6.QtCore import Qt, QPropertyAnimation
from PySide6.QtWidgets import (
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QScrollArea,
)
from PySide6.QtGui import QGuiApplication


class CollapsibleBox(QWidget):
    """可折疊容器（無捲動、L abel 自動換行）。"""

    def __init__(
        self, title: str, *, duration: int = 200, parent: QWidget | None = None
    ):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        # Header
        self.toggle_btn = QToolButton(text=title, checkable=True, checked=False)
        self.toggle_btn.setStyleSheet("QToolButton { border:none; font-weight:bold; }")
        self.toggle_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_btn.setArrowType(Qt.RightArrow)
        self.toggle_btn.toggled.connect(self._on_toggle)

        # Body container
        self.body_widget = QWidget()
        self.body_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.body_layout = QVBoxLayout(self.body_widget)
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        self.body_widget.setMaximumHeight(0)  # collapsed

        # Animation
        self._anim = QPropertyAnimation(self.body_widget, b"maximumHeight", self)
        self._anim.setDuration(duration)
        self._anim.finished.connect(self._after_anim)

        # Layout
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toggle_btn)
        lay.addWidget(self.body_widget)

    # ---------- public ----------
    def add_widget(self, w: QWidget):
        self.body_layout.addWidget(w)
        if self.toggle_btn.isChecked():
            self._resize_body()

    def clear(self):
        while self.body_layout.count():
            item = self.body_layout.takeAt(0)
            if c := item.widget():
                c.deleteLater()
        if self.toggle_btn.isChecked():
            self._resize_body()

    # ---------- private ----------
    def _resize_body(self):
        self.body_widget.setMaximumHeight(self.body_layout.sizeHint().height())

    def _on_toggle(self, expand: bool):
        self.toggle_btn.setArrowType(Qt.DownArrow if expand else Qt.RightArrow)
        start = self.body_widget.maximumHeight()
        end = self.body_layout.sizeHint().height() if expand else 0
        self._anim.stop()
        self._anim.setStartValue(start)
        self._anim.setEndValue(end)
        self._anim.start()

    def _after_anim(self):
        if self.toggle_btn.isChecked():
            self.body_widget.setMaximumHeight(16777215)
