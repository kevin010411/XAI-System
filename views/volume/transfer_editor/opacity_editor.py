import json
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
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
                self.toolbox_popup.save_transfer.connect(self.save_state)
                self.toolbox_popup.load_transfer.connect(self.load_state)

            self.toolbox_popup.show()
        else:
            self.toolbox_popup.close()
            self.toolbox_popup = None

    def _update_plot(self):
        """
        從外部改變點須通知canva改變
        """
        self.canvas.update_points(self.points, self.curve_list)
        self.color_picker.gradient_canvas.update_gradient(self.points)
        if self.on_update_callback:
            self.on_update_callback()

    def _on_canvas_updated(self, updated_points):
        """
        從canva裡面call改變
        """
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
        self.canvas.merge_mode = not self.canvas.merge_mode
        self._update_plot()
        # 還沒實現

    def get_init_state(self, min_val, max_val):
        self.canvas.set_range(min_val, max_val)
        points = self.canvas.points
        init_state = {
            "points": [
                {"x": x, "y": y, "color": rgb} for x, y, rgb in [points[0], points[-1]]
            ],
            "curves": [],
            "min_val": min_val,
            "max_val": max_val,
        }
        self.canvas.set_range(self.min_val, self.max_val)
        return init_state

    def save_state(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "儲存 Transfer Function", "", "TF JSON (*.tf.json)"
        )
        if path is None:
            return

        state = self.get_state()
        with open(path, "w", encoding="utf-8") as file:
            json.dump(state, file)

    def get_state(self):
        return {
            "points": [{"x": x, "y": y, "color": rgb} for x, y, rgb in self.points],
            "curves": [curve for curve in self.curve_list],
            "min_val": self.min_val,
            "max_val": self.max_val,
        }

    def load_state(self, state=None):
        if state is None:  # 如果沒有傳入state，則從檔案讀取
            path, _ = QFileDialog.getOpenFileName(
                self, "讀取 Transfer Function", "", "TF JSON (*.tf.json)"
            )
            if path is None:
                return

            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
            ori_min, ori_max = self.min_val, self.max_val
            self.set_range(state["min_val"], state["max_val"])
            self.points = [(d["x"], d["y"], tuple(d["color"])) for d in state["points"]]
            self.curve_list = state.get("curves", [])
            self.set_range(ori_min, ori_max)

        else:
            self.set_range(state["min_val"], state["max_val"])
            self.points = [(d["x"], d["y"], tuple(d["color"])) for d in state["points"]]
            self.curve_list = state.get("curves", [])

        self._update_plot()
