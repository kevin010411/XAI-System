from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QGridLayout,
    QToolBar,
)
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt


class PaneWrapper(QWidget):
    """包裝內容 widget + Toolbar（含最大化/還原按鈕）"""

    def __init__(
        self, content: QWidget, owner: "Split2x2Window", pane_id: tuple[int, int]
    ):
        super().__init__()
        self._owner = owner
        self._pane_id = pane_id

        self.setStyleSheet("border: 1px solid #888;")

        toolbar = QToolBar()
        toolbar.setIconSize(toolbar.iconSize() * 0.8)

        self._act_toggle = QAction("↗", self, checkable=True)
        self._act_toggle.setStatusTip("最大化 / 還原")
        self._act_toggle.toggled.connect(self._toggle_fullscreen)
        toolbar.addAction(self._act_toggle)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(toolbar)
        lay.addWidget(content)

    # ----- 切換全螢幕 -----
    def _toggle_fullscreen(self, checked: bool):
        if checked:
            self._owner.maximize_pane(self._pane_id)
            self._act_toggle.setText("↙")
        else:
            self._owner.restore_panes()
            self._act_toggle.setText("↗")


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
    def set_pane(self, row: int, col: int, widget: QWidget):
        """把外部 widget 裝進 (row, col)；自動附 Toolbar"""
        pane = PaneWrapper(widget, owner=self, pane_id=(row, col))

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
