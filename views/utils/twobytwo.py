from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QGridLayout,
    QToolBar,
)
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt, QPoint, QEvent


class PaneWrapper(QWidget):
    """包裝內容 widget + Toolbar（含最大化/還原按鈕）"""

    def __init__(
        self,
        content: QWidget,
        owner: "Split2x2Window",
        pane_id: tuple[int, int],
        pin_widget: QWidget = None,
    ):
        super().__init__()
        self._owner = owner
        self._pane_id = pane_id
        self.pin_widget = None

        self.setStyleSheet("border: 1px solid #888;")

        self._toolbar = QToolBar()
        self._toolbar.setIconSize(self._toolbar.iconSize() * 0.8)

        self._pin_toggle = QAction("🏷️", self, checkable=True)
        self._pin_toggle.setStatusTip("懸浮視窗")
        self._pin_toggle.toggled.connect(self._toggle_pin_widget)
        if pin_widget is not None:
            flags = Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
            self.pin_widget = pin_widget
            self.pin_widget.setWindowFlags(self.pin_widget.windowFlags() | flags)
            self.pin_widget.hide()
        self._toolbar.addAction(self._pin_toggle)

        self._act_toggle = QAction("↗", self, checkable=True)
        self._act_toggle.setStatusTip("最大化 / 還原")
        self._act_toggle.toggled.connect(self._toggle_fullscreen)
        self._toolbar.addAction(self._act_toggle)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._toolbar)
        lay.addWidget(content)

        self._top_widget = self._toolbar.window()
        self._top_widget.installEventFilter(self)

    # ----- 切換全螢幕 -----
    def _toggle_fullscreen(self, checked: bool):
        if checked:
            self._owner.maximize_pane(self._pane_id)
            self._act_toggle.setText("↙")
        else:
            self._owner.restore_panes()
            self._act_toggle.setText("↗")

    # ----- 切換pin_widget -----
    def _toggle_pin_widget(self, checked: bool):
        if self.pin_widget is None:
            return
        if checked:
            self.pin_widget.show()
            self.pin_widget.raise_()
            self._reposition_pin_widget()
        else:
            self.pin_widget.hide()

    def _reposition_pin_widget(self):  # 🔹
        """把 pin_widget 黏在 📌 按鈕正下方 (偏移 4px)"""
        if self.pin_widget is None or not self.pin_widget.isVisible():
            return
        btn = self._toolbar.widgetForAction(self._pin_toggle)
        if btn:
            origin = btn.mapToGlobal(btn.rect().bottomLeft())
        else:  # 理論上不會執行
            rect = self._toolbar.actionGeometry(self._pin_toggle)
            origin = self._toolbar.mapToGlobal(rect.bottomLeft())
        self.pin_widget.move(origin + QPoint(0, 4))

    def moveEvent(self, ev):
        super().moveEvent(ev)
        self._reposition_pin_widget()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._reposition_pin_widget()

    def showEvent(self, ev):
        super().showEvent(ev)
        # 確保監聽到真正的 window (第一次 show 之後才穩定)
        top = self.window()
        if top is not self._top_widget:
            self._top_widget.removeEventFilter(self)
            self._top_widget = top
            self._top_widget.installEventFilter(self)

    def eventFilter(self, obj, ev):
        if obj is not self._top_widget:
            return super().eventFilter(obj, ev)

        match ev.type():
            case QEvent.Move:
                self._reposition_pin_widget()
            case QEvent.Resize:
                self._reposition_pin_widget()
            case QEvent.WindowDeactivate:
                if self.pin_widget is not None:
                    self.pin_widget.hide()
            case QEvent.WindowActivate:
                if self._pin_toggle.isChecked():
                    self.pin_widget.show()
                    self._reposition_pin_widget()

        return super().eventFilter(obj, ev)


class Split2x2Window(QWidget):
    """用 QGridLayout 實現等大 2×2；支援單格最大化/還原"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # ── 建立 2×2 Grid ──
        self._grid = QGridLayout()
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(0)

        # 每列、欄伸縮比例 = 1 → 等大
        for i in range(2):
            self._grid.setRowStretch(i, 1)
            self._grid.setColumnStretch(i, 1)

        main = QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.addLayout(self._grid)

        # 先放 4 個 placeholder
        self._panes: dict[tuple[int, int], PaneWrapper] = {}
        for r in range(2):
            for c in range(2):
                self.set_pane(r, c, QLabel(f"Pane {r},{c}", alignment=Qt.AlignCenter))

        self._maximized: tuple[int, int] | None = None  # 目前是否全螢幕

    # ---------- 公開 API ----------
    def set_pane(
        self, row: int, col: int, widget: QWidget, pin_widget: QWidget | None = None
    ):
        """把外部 widget 裝進 (row, col)；自動附 Toolbar"""
        pane = PaneWrapper(
            widget, owner=self, pane_id=(row, col), pin_widget=pin_widget
        )

        # 若已有舊 pane，先移除
        if (row, col) in self._panes:
            old = self._panes[(row, col)]
            old.setParent(None)
            old.deleteLater()

        self._panes[(row, col)] = pane
        self._grid.addWidget(pane, row, col)

    # ---------- 最大化 / 還原 ----------
    def maximize_pane(self, pane_id: tuple[int, int]):
        if self._maximized is not None:  # 已在最大化狀態
            return
        self._maximized = pane_id

        # 1. 隱藏其餘三格
        for pos, pane in self._panes.items():
            if pos != pane_id:
                pane.hide()

        # 2. 把目標 pane 重新加入，佔滿 2×2（rowSpan=colSpan=2）
        pane = self._panes[pane_id]
        self._grid.addWidget(pane, 0, 0, 2, 2)

    def restore_panes(self):
        if self._maximized is None:
            return

        # 取出目前全螢幕 pane、記住內容 widget
        max_pos = self._maximized
        pane_full = self._panes[max_pos]
        self._maximized = None

        # 1. 先移除整格，再放回原 row,col（span=1,1）
        self._grid.removeWidget(pane_full)
        row, col = max_pos
        self._grid.addWidget(pane_full, row, col)

        # 2. 顯示其餘三格
        for pos, pane in self._panes.items():
            if pos != (row, col):
                pane.show()

    # ---------- 方便覆蓋 sizeHint（可選） ----------
    def sizeHint(self):
        # 給一個舒適的預設大小
        return QWidget.sizeHint(self)  # 也可固定回 QSize(900, 600)
