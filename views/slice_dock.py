from PySide6.QtWidgets import (
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
)
from PySide6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np

from .utils import wrap_with_frame


class SliceDock(QDockWidget):

    def __init__(self, view_type="axial", show_in_volume_callback=None):
        view_title = {
            "axial": "Axial (Z)",
            "coronal": "Coronal (Y)",
            "sagittal": "Sagittal (X)",
        }.get(view_type, "Slice")
        super().__init__(view_title)

        self.show_in_volume_callback = show_in_volume_callback
        self.view_type = view_type

        self.setFixedSize(200, 200)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.widget = QWidget()
        self.layout = QVBoxLayout(self.widget)
        self.layout.setContentsMargins(0, 0, 0, 0)

        control_layout = QHBoxLayout()
        self.rotate_button = QPushButton("↻ Rotate")
        self.rotate_button.clicked.connect(self.rotate_view)
        self.show_in_volume_button = QPushButton("Show in Volume")
        self.show_in_volume_button.clicked.connect(self.toggle_show_slice_in_volume)
        control_layout.addWidget(self.show_in_volume_button)
        control_layout.addWidget(self.rotate_button)
        self.layout.addLayout(control_layout)

        self.fig = plt.Figure(figsize=(2, 2), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.layout.addWidget(self.canvas)
        self.setWidget(wrap_with_frame(self.widget))

        self.volume = None
        self.slice_index = 0
        self.rotation = 0
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.show_slice = False

        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)

        self.dragging = False
        self.last_event = None

    def update(self, img):
        volume = img.get_fdata()
        self.volume = np.transpose(volume, (2, 1, 0))
        self.img = img
        self.spacing = img.header.get_zooms()
        shape = volume.shape
        idx = {"axial": 0, "coronal": 1, "sagittal": 2}.get(self.view_type, 0)
        self.slice_index = shape[idx] // 2
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.render()

    def render(self):
        if self.volume is None:
            return

        # 切片選取
        if self.view_type == "axial":
            img = self.volume[self.slice_index, :, :]
            spacing = (self.spacing[1], self.spacing[2])
        elif self.view_type == "coronal":
            img = self.volume[:, self.slice_index, :]
            spacing = (self.spacing[0], self.spacing[2])
        elif self.view_type == "sagittal":
            img = self.volume[:, :, self.slice_index]
            spacing = (self.spacing[0], self.spacing[1])
        else:
            img = np.zeros((10, 10))
            spacing = (1.0, 1.0)

        self.ax.clear()
        self.ax.set_facecolor("black")
        self.ax.axis("off")
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # 補 padding 成正方形
        height, width = img.shape
        physical_w = width * spacing[0]
        physical_h = height * spacing[1]
        max_side = max(physical_w, physical_h)
        pad_w = int((max_side - physical_w) / (2 * spacing[0]))
        pad_h = int((max_side - physical_h) / (2 * spacing[1]))
        padded_img = np.pad(
            img,
            ((pad_h, pad_h), (pad_w, pad_w)),
            mode="constant",
            constant_values=np.min(img),
        )
        padded_height, padded_width = padded_img.shape

        # 顯示圖片
        self.ax.imshow(
            np.rot90(padded_img, self.rotation),
            cmap="gray",
            origin="upper",
            extent=[0, padded_width, 0, padded_height],
        )

        # 計算中心與 zoom
        center_x = padded_width / 2 - self.pan_x
        center_y = padded_height / 2 - self.pan_y
        self.zoom = max(padded_width, padded_height) / min(padded_width, padded_height)
        dx = (padded_width / 2) / self.zoom
        dy = (padded_height / 2) / self.zoom
        self.ax.set_xlim(center_x - dx, center_x + dx)
        self.ax.set_ylim(center_y - dy, center_y + dy)
        self.ax.set_aspect("equal")

        self.canvas.draw()

    def toggle_show_slice_in_volume(self):
        self.show_slice = not self.show_slice
        self.render()
        self.show_slice_in_volume()

    def show_slice_in_volume(self):
        if self.show_in_volume_callback and self.volume is not None:
            if self.view_type == "axial":
                img2d = self.volume[self.slice_index, :, :]
            elif self.view_type == "coronal":
                img2d = self.volume[:, self.slice_index, :]
            elif self.view_type == "sagittal":
                img2d = self.volume[:, :, self.slice_index]
            else:
                img2d = None
            self.show_in_volume_callback(
                self.view_type, self.slice_index, img2d, remove=not self.show_slice
            )

    def on_scroll(self, event):
        if self.volume is None:
            return
        idx = {"axial": 0, "coronal": 1, "sagittal": 2}[self.view_type]
        max_idx = self.volume.shape[idx] - 1
        self.slice_index = np.clip(
            self.slice_index + (10 if event.step > 0 else -10), 0, max_idx
        )
        self.render()
        self.show_slice_in_volume()

    def on_press(self, event):
        if event.button in [1, 3]:
            self.dragging = True
            self.last_event = event

    def on_motion(self, event):
        if (
            not self.dragging
            or self.last_event is None
            or event.x is None
            or event.y is None
        ):
            return
        dx = event.x - self.last_event.x
        dy = event.y - self.last_event.y
        if self.last_event.button == 1:
            self.pan_x += dx / self.zoom
            self.pan_y += dy / self.zoom
        elif self.last_event.button == 3:
            scale_factor = 1 + dy * 0.01
            self.zoom *= scale_factor
            self.zoom = max(self.zoom, 0.1)
        self.last_event = event
        self.render()

    def on_release(self, event):
        self.dragging = False
        self.last_event = None

    def rotate_view(self):
        self.rotation = (self.rotation + 1) % 4
        self.render()
