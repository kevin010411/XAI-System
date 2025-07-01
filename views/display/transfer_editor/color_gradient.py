from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


class ColorGradientCanvas(FigureCanvas):
    def __init__(self, points=None):
        self.fig = Figure(figsize=(4, 1.5))
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.points = points if points is not None else []
        self.setMinimumHeight(60)
        self.setMaximumHeight(80)
        self._init_plot()
        self.update_gradient(self.points)

    def _init_plot(self):
        self.fig.tight_layout(pad=0)
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.ax.axis("off")

    def update_gradient(self, points):
        self.ax.clear()
        if not points or len(points) < 2:
            self.draw()
            return

        # 準備漸層資料
        points = sorted(points, key=lambda p: p[0])
        xs = [p[0] for p in points]
        colors = [p[2] for p in points]

        width = 256
        height = 40

        x_img = np.linspace(xs[0], xs[-1], width)
        color_img = np.zeros((height, width, 3))

        for i in range(3):
            row = np.interp(x_img, xs, [c[i] for c in colors])
            color_img[:, :, i] = row[None, :]

        self.ax.imshow(
            color_img, aspect="auto", extent=[xs[0], xs[-1], 0, 1], origin="lower"
        )

        self.draw()
