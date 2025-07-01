from PySide6.QtWidgets import QWidget, QVBoxLayout


class BasePanel(QWidget):

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)

        self.setStyleSheet("background-color: #f0f0f0;")
        self.setMinimumSize(300, 200)

        self.data_manager = data_manager
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

    def update(self, img):
        raise NotImplementedError("子類必須實作 some_method")
