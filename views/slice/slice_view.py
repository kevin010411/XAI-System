from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
)
from PySide6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import nibabel as nib
from ..utils import wrap_with_frame

BTN_STYLE = """
QPushButton {
    background-color: #e4e4e4;
    border: 1px solid #999;
    border-radius: 4px;
    padding: 4px 8px;
}
QPushButton:hover {
    background-color: #d0d0d0;
}
QPushButton:pressed {
    background-color: #a8a8a8;
    border-style: inset;          /* 讓按下時有凹陷感 */
}
"""


class SliceView(QWidget):
    """A standalone slice viewer widget (formerly QDockWidget)."""

    def __init__(
        self,
        view_type: str = "axial",
        show_in_volume_callback=None,
        display_mode: str = "grey",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)

        # ---------- meta ----------
        self.show_in_volume_callback = show_in_volume_callback
        self.view_type = view_type
        self.display_mode = display_mode

        # ---------- outer layout ----------
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        # ---------- control bar ----------
        ctrl = QHBoxLayout()
        self.title_label = QLabel(
            {
                "axial": "Axial (Z)",
                "coronal": "Coronal (Y)",
                "sagittal": "Sagittal (X)",
            }.get(view_type, "Slice")
        )
        self.title_label.setAlignment(Qt.AlignCenter)
        ctrl.addWidget(self.title_label)

        self.rotate_button = QPushButton("↻ Rotate")
        self.rotate_button.setStyleSheet(BTN_STYLE)
        self.rotate_button.clicked.connect(self.rotate_view)
        self.show_in_volume_button = QPushButton("Show in Volume")
        self.show_in_volume_button.setStyleSheet(BTN_STYLE)
        self.show_in_volume_button.clicked.connect(self.toggle_show_slice_in_volume)
        ctrl.addWidget(self.show_in_volume_button)
        ctrl.addWidget(self.rotate_button)
        outer.addLayout(ctrl)

        # ---------- Matplotlib canvas ----------
        self.fig = plt.Figure(figsize=(3, 3), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        outer.addWidget(self.canvas, stretch=1)

        # ---------- state ----------
        self.volume = None
        self.slice_index = 0
        self.rotation = 0
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.show_slice = False

        # ---------- Matplotlib events ----------
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)

        self.dragging = False
        self.last_event = None

    # ===================== helpers =====================
    def get_cmap(self):
        match self.display_mode:
            case "gray":
                return "gray"
            case "heatmap":
                return "hot"
            case "cold_to_hot":
                return "coolwarm"
            case _:
                return "gray"

    # ===================== public API =====================
    def update(self, img):
        img_ras = nib.as_closest_canonical(img)
        self.volume = np.transpose(img_ras.get_fdata(), (2, 1, 0))
        self.img = img
        self.spacing = img.header.get_zooms()
        shape = self.volume.shape
        idx = {"axial": 0, "coronal": 1, "sagittal": 2}.get(self.view_type, 0)
        self.slice_index = shape[idx] // 2
        self.zoom = 1.0
        self.pan_x = self.pan_y = 0.0
        self.render()

    def change_display_mode(self, mode: str):
        """Change the display mode of this slice viewer."""
        self.display_mode = mode
        self.render()

    # ===================== rendering =====================
    def render(self):
        if self.volume is None:
            return

        # --- choose slice ---
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

        # --- pad to square ---
        h, w = img.shape
        physical_w, physical_h = w * spacing[0], h * spacing[1]
        max_side = max(physical_w, physical_h)
        pad_w = int((max_side - physical_w) / (2 * spacing[0]))
        pad_h = int((max_side - physical_h) / (2 * spacing[1]))
        padded = np.pad(
            img,
            ((pad_h, pad_h), (pad_w, pad_w)),
            mode="constant",
            constant_values=np.min(img),
        )

        cmap = self.get_cmap()
        self.ax.imshow(np.rot90(padded, self.rotation), cmap=cmap, origin="upper")

        # --- zoom & pan ---
        ph, pw = padded.shape
        cx, cy = pw / 2 - self.pan_x, ph / 2 - self.pan_y
        dx, dy = (pw / 2) / self.zoom, (ph / 2) / self.zoom
        self.ax.set_xlim(cx - dx, cx + dx)
        self.ax.set_ylim(cy - dy, cy + dy)
        self.ax.set_aspect("equal")

        self.canvas.draw_idle()

    # ===================== callbacks =====================
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
        self.slice_index = int(
            np.clip(self.slice_index + (1 if event.step > 0 else -1), 0, max_idx)
        )
        self.render()
        self.show_slice_in_volume()

    def on_press(self, event):
        if event.button in [1, 3]:
            self.dragging = True
            self.last_event = event

    def on_motion(self, event):
        if not (
            self.dragging
            and self.last_event
            and event.x is not None
            and event.y is not None
        ):
            return
        dx, dy = event.x - self.last_event.x, event.y - self.last_event.y
        if self.last_event.button == 1:
            self.pan_x += dx / self.zoom
            self.pan_y += dy / self.zoom
        elif self.last_event.button == 3:
            self.zoom = max(self.zoom * (1 + dy * 0.01), 0.1)
        self.last_event = event
        self.render()

    def on_release(self, event):
        self.dragging = False
        self.last_event = None

    def rotate_view(self):
        self.rotation = (self.rotation + 1) % 4
        self.render()
