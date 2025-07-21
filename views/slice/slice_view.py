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

_AXIS_IDX = {"axial": 0, "coronal": 1, "sagittal": 2}

DisplayMode = ["gray", "heatmap", "cold_to_hot"]
ViewType = ["axial", "coronal", "sagittal"]


class SliceView(QWidget):
    """A standalone slice viewer widget (formerly QDockWidget)."""

    def __init__(
        self,
        view_type: str = "axial",
        show_in_volume_callback=None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)

        # ---------- meta ----------
        self.show_in_volume_callback = show_in_volume_callback
        self.view_type = view_type

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
        self.slice_idx = 0
        self.rotation = 0
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.show_slice = False
        self.layers = []

        # ---------- Matplotlib events ----------
        self.canvas.mpl_connect("scroll_event", self.on_scroll)

        self.dragging = False
        self.last_event = None

    # ===================== helpers =====================
    def _extract_slice(self, vol: np.ndarray) -> np.ndarray:
        ax = self.view_type
        idx = self.slice_idx
        if ax == "axial":
            return vol[idx, :, :]
        if ax == "coronal":
            return vol[:, idx, :]
        return vol[:, :, idx]

    @staticmethod
    def _alpha(opacity):
        return opacity / 100.0 if opacity > 1 else opacity

    @staticmethod
    def _pad_or_crop_center(img: np.ndarray, side: int) -> np.ndarray:
        """Return square array of size (side, side) by centered pad or crop."""
        h, w = img.shape
        # crop if larger
        if h > side:
            top = (h - side) // 2
            img = img[top : top + side, :]
            h = side
        if w > side:
            left = (w - side) // 2
            img = img[:, left : left + side]
            w = side
        # pad if smaller
        pad_h = side - h
        pad_w = side - w
        if pad_h or pad_w:
            img = np.pad(
                img,
                ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)),
                constant_values=img.min(),
            )
        return img

    # ===================== public API =====================
    def update(self, layers: list[dict[str, any]] | nib.Nifti1Image):
        """Accept list[dict] or single nib image."""
        if isinstance(layers, list):
            if not layers:
                self.ax.clear()  # 或 self.ax.clear()
                self.fig.canvas.draw_idle()
                return
            prev_layers = {data["img_name"] for data in self.layers}
            change_slice_index = False
            self.layers = []
            for lyr in layers:
                img: nib.Nifti1Image = lyr["img"]
                arr = np.transpose(nib.as_closest_canonical(img).get_fdata(), (2, 1, 0))
                self.layers.append(
                    {
                        "img_name": lyr["img_name"],
                        "data": arr,
                        "opacity": float(lyr.get("opacity", 1.0)),
                        "cmap": lyr.get("cmap", "gray"),
                    }
                )
                if lyr["img_name"] not in prev_layers:
                    change_slice_index = True

            # spacing 取第一層
            self.spacing = nib.as_closest_canonical(
                layers[0]["img"]
            ).header.get_zooms()[:3]
            shape0 = self.layers[0]["data"].shape
            if change_slice_index:
                self.slice_idx = shape0[_AXIS_IDX[self.view_type]] // 2

        else:  # 單張 image
            self.update([{"img": layers, "opacity": 1.0, "cmap": "gray"}])
            return
        self.render()

    # ===================== rendering =====================
    def render(self):
        self.ax.clear()

        if not self.layers:  # 沒有任何 layer ⇒ 直接 refresh 畫布
            self.fig.canvas.draw_idle()
            return

        # ---------- 第一層 ----------
        base_slice = self._extract_slice(self.layers[0]["data"])
        h0, w0 = base_slice.shape
        side = max(h0, w0)
        base_sq = self._pad_or_crop_center(base_slice, side)

        cmap = {
            "gray": "gray",
            "heatmap": "hot",
            "cold_to_hot": "coolwarm",
        }
        self.ax.imshow(
            np.rot90(base_sq, self.rotation),
            cmap=cmap[self.layers[0]["cmap"]],
            origin="upper",
            alpha=self._alpha(self.layers[0]["opacity"]),
        )

        # ---------- 其餘層 ----------
        for lyr in self.layers[1:]:
            slc = self._pad_or_crop_center(self._extract_slice(lyr["data"]), side)
            self.ax.imshow(
                np.rot90(slc, self.rotation),
                cmap=cmap[lyr["cmap"]],
                origin="upper",
                alpha=self._alpha(lyr["opacity"]),
            )

        self.fig.canvas.draw_idle()

    # ===================== callbacks =====================
    def toggle_show_slice_in_volume(self):
        self.show_slice = not self.show_slice
        self.render()
        self.show_slice_in_volume()

    def show_slice_in_volume(self):
        if self.show_in_volume_callback and self.volume is not None:
            if self.view_type == "axial":
                img2d = self.volume[self.slice_idx, :, :]
            elif self.view_type == "coronal":
                img2d = self.volume[:, self.slice_idx, :]
            elif self.view_type == "sagittal":
                img2d = self.volume[:, :, self.slice_idx]
            else:
                img2d = None
            self.show_in_volume_callback(
                self.view_type, self.slice_idx, img2d, remove=not self.show_slice
            )

    def on_scroll(self, event):
        if not hasattr(self, "layers") or self.layers is None:
            return
        idx = {"axial": 0, "coronal": 1, "sagittal": 2}[self.view_type]
        max_idx = self.layers[0]["data"].shape[idx] - 1
        self.slice_idx = int(
            np.clip(self.slice_idx + (1 if event.step > 0 else -1), 0, max_idx)
        )
        self.render()
        self.show_slice_in_volume()

    def rotate_view(self):
        self.rotation = (self.rotation + 1) % 4
        self.render()
