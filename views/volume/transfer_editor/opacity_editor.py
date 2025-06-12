from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QToolBox,
)

from .opacity_canva import OpacityCurveCanvas
from .color_picker import ColorPickerWidget
from .editor_toolbox_button import EditorToolBox


class OpacityEditor(QWidget):
    def __init__(self, on_update_callback):
        super().__init__()
        self.on_update_callback = on_update_callback
        self.min_val = 0
        self.max_val = 1
        self.points = [(self.min_val, 0.0, (0, 0, 1)), (self.max_val, 1.0, (1, 0, 0))]
        self.curve_list = []
        self.is_merged = False
        self.original_points = self.points.copy()

        self._init_ui()
        self._update_plot()

    def _init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.label = QLabel("Control Curve")
        layout.addWidget(self.label)

        self.canvas = OpacityCurveCanvas(
            self.points,
            curve_list=self.curve_list,
            on_change_callback=self._on_canvas_updated,
        )
        layout.addWidget(self.canvas)

        self.color_picker = ColorPickerWidget(
            self.get_points, self._set_point_color, self.get_select_points
        )
        layout.addWidget(self.color_picker)

        self.toolbox_popup = None
        self.toggle_btn = QPushButton("≡ 工具箱")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.toggled.connect(self.toggle_toolbox)
        layout.addWidget(self.toggle_btn)

    def toggle_toolbox(self, checked: bool):
        if checked:
            if self.toolbox_popup is None:
                self.toolbox_popup = EditorToolBox(self.toggle_btn)
                self.toolbox_popup.clear_points.connect(self.clear_points)
                self.toolbox_popup.generate_normal_distribution.connect(
                    self.generate_normal_distribution
                )
                self.toolbox_popup.toggle_merge.connect(self.toggle_merge)

            self.toolbox_popup.show()
        else:
            self.toolbox_popup.close()
            self.toolbox_popup = None

    def _update_plot(self):
        self.canvas.update_points(self.points, self.curve_list)
        if self.on_update_callback:
            self.on_update_callback()

    def _on_canvas_updated(self, updated_points):
        self.points = updated_points
        self.color_picker.gradient_canvas.update_gradient(self.points)
        if self.on_update_callback:
            self.on_update_callback()

    def set_range(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
        self.canvas.set_range(min_val, max_val)
        self.points = self.canvas.points

    def get_points(self):
        return self.points

    def get_select_points(self):
        return self.canvas.selected_point_index

    def _set_point_color(self, index, rgb):
        if 0 <= index < len(self.points):
            x, y = self.points[index][:2]
            self.points[index] = (x, y, rgb)
            self._update_plot()

    def clear_points(self):
        if self.is_merged:
            return
        self.points = [self.points[0], self.points[-1]]
        self._update_plot()

    def generate_normal_distribution(self):
        mean = (self.min_val + self.max_val) / 2
        std = (self.max_val - self.min_val) / 6
        self.curve_list.append({"type": "normal_curve", "mean": mean, "std": std})
        self._update_plot()

    def toggle_merge(self):
        self.canvas.display_points = not self.canvas.display_points
        self._update_plot()
        # if not self.curve_list:
        #     return
        # if not self.is_merged:
        #     self.original_points = self.points.copy()
        #     ctrl_x, ctrl_y = zip(*[(x, y) for x, y, *_ in self.points])
        #     curve_x = np.linspace(self.min_val, self.max_val, 200)
        #     interp_ctrl_y = np.interp(curve_x, ctrl_x, ctrl_y)

        #     norm_x, norm_y = zip(*self.curve_list)
        #     interp_norm_y = np.interp(curve_x, norm_x, norm_y)

        #     merged_y = np.clip(np.array(interp_ctrl_y) + np.array(interp_norm_y), 0, 1)
        #     self.points = list(zip(curve_x, merged_y))

        #     self.canvas.disable_point_display()
        #     self.curve_list = []
        #     self.is_merged = True
        # else:
        #     self.points = self.original_points.copy()
        #     self.canvas.enable_point_display()
        #     self.curve_list = []
        #     self.is_merged = False
