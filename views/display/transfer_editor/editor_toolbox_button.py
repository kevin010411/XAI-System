from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QWidget,
    QToolBox,
    QVBoxLayout,
    QHBoxLayout,
)
from PySide6.QtCore import Qt, QPoint, QEvent, Signal


class EditorToolBox(QWidget):
    clear_points = Signal()
    generate_normal_distribution = Signal()
    toggle_merge = Signal()
    save_transfer = Signal()
    load_transfer = Signal()

    def __init__(self, target_btn: QPushButton):
        flags = (
            Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.FramelessWindowHint
        )
        super().__init__(None, flags)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)  # 關閉就自毀
        self.btn = target_btn
        self.owner = target_btn.window()  # 取得主視窗

        self.toolbox = QToolBox(self)
        # === 曲線工具 ===
        box_group = QWidget()
        groupbox_vbox = QVBoxLayout(box_group)
        clear_button = QPushButton("清除所有中間點")
        clear_button.clicked.connect(self.clear_points)
        groupbox_vbox.addWidget(clear_button)
        gen_normal_button = QPushButton("產生常態分佈")
        gen_normal_button.clicked.connect(self.generate_normal_distribution)
        groupbox_vbox.addWidget(gen_normal_button)
        combine_button = QPushButton("合併所有曲線")
        combine_button.clicked.connect(self.toggle_merge)
        groupbox_vbox.addWidget(combine_button)
        self.toolbox.addItem(box_group, "曲線工具")

        # === Load & Save transfer function Preset ===
        tranfer_widget = QWidget()
        tranfer__vbox = QVBoxLayout(tranfer_widget)
        save_transfer_button = QPushButton("儲存Transfer Function")
        save_transfer_button.clicked.connect(self.save_transfer)
        tranfer__vbox.addWidget(save_transfer_button)
        load_transfer_button = QPushButton("載入Transfer Function")
        load_transfer_button.clicked.connect(self.load_transfer)
        tranfer__vbox.addWidget(load_transfer_button)
        self.toolbox.addItem(tranfer_widget, "Transfer Function")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toolbox)

        # ==== 讓工具箱跟著跑 ====
        self.owner.installEventFilter(self)  # 監聽 Move/Resize
        self.reposition()  # 第一次定位

    # --- 重新對齊到按鈕旁 ---
    def reposition(self):
        gp = self.btn.mapToGlobal(QPoint(self.btn.width(), 0))
        self.move(gp)

    # --- 監聽主視窗事件 ---
    def eventFilter(self, obj, ev: QEvent):
        if obj is self.owner and ev.type() in (QEvent.Type.Move, QEvent.Type.Resize):
            self.reposition()
        return super().eventFilter(obj, ev)

    # ---- Toolbox 自身尺寸改變時 ----
    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.reposition()

    # ---- 若外部用特殊手法拖動整窗 (少見) ----
    def moveEvent(self, ev):
        super().moveEvent(ev)
        self.reposition()
