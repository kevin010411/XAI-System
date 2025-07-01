from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QColorDialog,
)
from PySide6.QtGui import QColor
from .color_gradient import ColorGradientCanvas


class ColorPickerWidget(QWidget):
    def __init__(
        self, get_points_callback, set_color_callback, get_selected_point_index_callback
    ):
        super().__init__()
        self.get_points_callback = get_points_callback
        self.set_color_callback = set_color_callback
        self.get_selected_point_index_callback = get_selected_point_index_callback

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.gradient_canvas = ColorGradientCanvas(points=self.get_points_callback())
        layout.addWidget(self.gradient_canvas)

        self.pick_color_btn = QPushButton("選擇顏色")
        self.pick_color_btn.clicked.connect(self.pick_color)
        layout.addWidget(self.pick_color_btn)

    def pick_color(self):
        points = self.get_points_callback()
        index = self.get_selected_point_index_callback()
        if not points or index is None:
            return
        current_color = points[index][2]
        dlg = QColorDialog()
        dlg.setCurrentColor(
            QColor(
                int(current_color[0] * 255),
                int(current_color[1] * 255),
                int(current_color[2] * 255),
            )
        )
        if dlg.exec():
            color = dlg.currentColor()
            rgb = (color.redF(), color.greenF(), color.blueF())
            self.set_color_callback(index, rgb)
            # 變色後即時更新漸層
            self.gradient_canvas.update_gradient(self.get_points_callback())
