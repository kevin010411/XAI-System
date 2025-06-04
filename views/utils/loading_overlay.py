from PySide6.QtWidgets import QWidget, QVBoxLayout, QProgressBar
from PySide6.QtCore import Qt, QTimer, QEvent


class LoadingOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)  # 遮罩層要擋事件
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 120);")
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        # 旋轉型 indeterminate ProgressBar
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # Indeterminate 模式
        self.progress.setFixedSize(80, 80)
        self.progress.setTextVisible(False)
        layout.addWidget(self.progress)

        # 自動填滿 parent
        if parent:
            self.resize(parent.size())
            self.move(0, 0)
            parent.installEventFilter(self)

        self.show()

    # 跟隨 parent resize
    def eventFilter(self, obj, event: QEvent):
        if obj is self.parent():
            if event.type() == QEvent.Type.Resize:
                self.resize(obj.size())
        return super().eventFilter(obj, event)
