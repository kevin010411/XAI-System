from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np


class HistogramViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.canvas = FigureCanvas(plt.Figure(figsize=(4, 1.5)))
        self.ax = self.canvas.figure.subplots()
        self._init_plot()
        self.layout.addWidget(QLabel("Value Density Histogram"))
        self.layout.addWidget(self.canvas)

    def _init_plot(self):
        self.canvas.figure.tight_layout(pad=0)
        self.canvas.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.ax.axis("off")

    def set_histogram(self, bins, counts):
        self.ax.clear()
        self.ax.fill_between(
            bins[:-1], counts / np.max(counts), step="pre", alpha=0.3, color="gray"
        )
        self.ax.set_xlim(bins[0], bins[-1])
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel("Scalar Value")
        self.ax.set_ylabel("Density")
        self.canvas.draw()
