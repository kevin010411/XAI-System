from PySide6.QtWidgets import (
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
)
from PySide6.QtCore import Slot, Qt


class TargetLayerList(QListWidget):
    list_style = f"""
        QListWidget {{
            background: #000000;
            border: 1px solid #d0d0d0;
            border-radius: 6px;
            padding: 4px;
            color: #dedcdc;
        }}
        QListWidget::item {{
            height: 24px;
            padding-left: 4px;  /* space between checkbox and text */
        }}
        QListWidget::item:hover {{
            background: #636262;
        }}
        QListWidget::indicator {{
            width: 18px;
            height: 18px;
        }}
        QScrollBar:vertical {{
            width: 12px;
            background: transparent;
            margin: 0px;
        }}
        QScrollBar::handle:vertical {{
            background: #c0c0c0;
            border-radius: 6px;
            min-height: 20px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: #a6a6a6;
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        """

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QAbstractItemView.NoSelection)
        self.setDragEnabled(False)
        self.setDragDropMode(QAbstractItemView.NoDragDrop)
        self.setStyleSheet(self.list_style)

        self.target_layer = []
        self._initializing = True

        for idx, (name, layer) in enumerate(model.named_children()):
            item = QListWidgetItem(f"[{idx}] {name}: {layer.__class__.__name__}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            # Keep a reference to the actual layer for convenience
            item.setData(Qt.UserRole, name)
            self.addItem(item)

        self.itemChanged.connect(self._on_item_changed)
        self._initializing = False

    def _on_item_changed(self, _):
        """勾選狀態改變 → 立即回調。"""
        if self._initializing:
            return
        selected: list[str] = []
        for i in range(self.count()):
            item = self.item(i)
            if item.checkState() == Qt.Checked:
                layer_name: str = item.data(Qt.UserRole)
                selected.append(layer_name)
        self.target_layer = selected
