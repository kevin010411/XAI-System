from PySide6.QtWidgets import QFrame, QVBoxLayout
import cProfile
import pstats
import io
from contextlib import contextmanager


def wrap_with_frame(widget):
    frame = QFrame()
    frame.setFrameShape(QFrame.Box)
    frame.setLineWidth(2)
    layout = QVBoxLayout()
    layout.setContentsMargins(5, 5, 5, 5)
    layout.addWidget(widget)
    frame.setLayout(layout)
    return frame


@contextmanager
def profile(sort_by="cumtime", limit=20):
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        yield
    finally:
        profiler.disable()
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats(sort_by)
        ps.print_stats(limit)
        print(s.getvalue())
