from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import QRunnable, QThreadPool, Signal, QObject, Slot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np


class HistogramComputeWorkerSignals(QObject):
    finished = Signal(np.ndarray, np.ndarray)


class HistogramComputeWorker(QRunnable):
    def __init__(self, volume):
        super().__init__()
        self.volume = volume
        self.signals = HistogramComputeWorkerSignals()

    def run(self):
        counts, bins = np.histogram(self.volume, bins=100)
        self.signals.finished.emit(bins, counts)


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
        self.thread_pool = QThreadPool.globalInstance()

    def _init_plot(self):
        self.canvas.figure.tight_layout(pad=0)
        self.canvas.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.ax.axis("off")

    def set_histogram(self, volume):
        worker = HistogramComputeWorker(volume)
        worker.signals.finished.connect(self._draw_histogram)
        self.thread_pool.start(worker)

    @Slot(np.ndarray, np.ndarray)
    def _draw_histogram(self, bins, counts):
        self.ax.clear()
        self.ax.fill_between(
            bins[:-1], counts / np.max(counts), step="pre", alpha=0.3, color="gray"
        )
        self.ax.set_xlim(bins[0], bins[-1])
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel("Scalar Value")
        self.ax.set_ylabel("Density")
        self.canvas.draw()
