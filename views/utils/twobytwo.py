from PySide6.QtWidgets import (
    QSplitter,
    QToolBar,
    QWidget,
    QVBoxLayout,
    QLabel,
)
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt


class PaneWrapper(QWidget):
    """包裝一個『內容 widget』並在上方附加 Toolbar。"""

    def __init__(self, content: QWidget, owner, pane_id: tuple[int, int]):
        """
        :param content: 您想顯示的部件 (可任意 QWidget / QDockWidget)
        :param owner:   MainWindow，用來呼叫 maximize / restore
        :param pane_id: (row, col) 方便回報自己是哪格
        """
        super().__init__()
        self._owner = owner
        self._pane_id = pane_id

        # ── Toolbar (可加更多按鈕) ──
        self.toolbar = QToolBar()
        self.toolbar.setIconSize(self.toolbar.iconSize() * 0.8)

        self._act_toggle = QAction("↗", self, checkable=True)
        self._act_toggle.setStatusTip("最大化 / 還原")
        self._act_toggle.toggled.connect(self._toggle_fullscreen)
        self.toolbar.addAction(self._act_toggle)

        # ── 組 layout ──
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.toolbar)
        lay.addWidget(content)

    # ----- toolbar slot -----
    def _toggle_fullscreen(self, checked: bool):
        if checked:
            self._owner.maximize_pane(self._pane_id)
            self._act_toggle.setText("↙")
        else:
            self._owner.restore_panes()
            self._act_toggle.setText("↗")


class Split2x2Window(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pluggable 2×2 Split View")
        self.resize(900, 600)

        # ──────────── splitter 結構 ────────────
        self._v_split = QSplitter(Qt.Orientation.Vertical)  # 上 / 下
        self._top_h = QSplitter(Qt.Orientation.Horizontal)  # TL / TR
        self._bot_h = QSplitter(Qt.Orientation.Horizontal)  # BL / BR
        self._v_split.addWidget(self._top_h)
        self._v_split.addWidget(self._bot_h)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self._v_split)

        # 先放四個 placeholder
        for r in range(2):
            for c in range(2):
                placeholder = QLabel(f"Pane {r},{c}", alignment=Qt.AlignCenter)
                self.set_pane(r, c, placeholder)

        # split 初始平均
        self._v_split.setSizes([1, 1])
        self._top_h.setSizes([1, 1])
        self._bot_h.setSizes([1, 1])

        # 儲存 splitter 尺寸用
        self._saved_sizes: dict[str, list[list[int]]] = {}
        self._is_maximized = False

    # ---------- 公開 API ----------
    def set_pane(self, row: int, col: int, widget: QWidget):
        """把外部 widget 裝進指定格 (0/1, 0/1)；自動加 toolbar。"""
        wrapper = PaneWrapper(widget, owner=self, pane_id=(row, col))
        if row == 0:
            self._replace_in_split(self._top_h, col, wrapper)
        else:
            self._replace_in_split(self._bot_h, col, wrapper)

    # ---------- maximize / restore ----------
    def maximize_pane(self, pane_id: tuple[int, int]):
        """把指定 pane 放大到全螢幕；其餘縮到 0。"""
        if self._is_maximized:
            return
        self._is_maximized = True

        # 存目前三層 splitter 的尺寸
        self._saved_sizes["v"] = self._v_split.sizes()
        self._saved_sizes["top"] = self._top_h.sizes()
        self._saved_sizes["bot"] = self._bot_h.sizes()

        # 先把所有 size 設 0
        self._v_split.setSizes([0, 0])
        self._top_h.setSizes([0, 0])
        self._bot_h.setSizes([0, 0])

        row, col = pane_id
        if row == 0:  # 要放大上方
            self._v_split.setSizes([1, 0])
            self._top_h.setSizes([1, 0] if col == 0 else [0, 1])
        else:  # 放大下方
            self._v_split.setSizes([0, 1])
            self._bot_h.setSizes([1, 0] if col == 0 else [0, 1])

    def restore_panes(self):
        """還原之前儲存的 splitter 尺寸。"""
        if not self._is_maximized:
            return
        self._v_split.setSizes(self._saved_sizes["v"])
        self._top_h.setSizes(self._saved_sizes["top"])
        self._bot_h.setSizes(self._saved_sizes["bot"])
        self._is_maximized = False

    # ---------- utils ----------
    @staticmethod
    def _replace_in_split(splitter: QSplitter, index: int, widget: QWidget):
        """把 splitter 第 index 位子的元件換掉。"""
        old = splitter.widget(index)
        if old is not None:
            old.setParent(None)  # ← 先把舊元件從 splitter 拔掉
            old.hide()  #   讓畫面馬上消失
            old.deleteLater()  #   等事件迴圈空檔再真正釋放

        splitter.insertWidget(index, widget)  # 插入新元件
