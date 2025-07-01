from views.display.transfer_editor import OpacityEditor
from views import HistogramViewer

from .base_panel import BasePanel


class VolumePanel(BasePanel):
    def __init__(self, volume_renderer, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Volume Control Panel")

        self.volume_renderer = volume_renderer
        self.histogram_viewer = HistogramViewer()
        self.opacity_editor = OpacityEditor(self.update_transfer_function)
        self.layout.addWidget(self.histogram_viewer)
        self.layout.addWidget(self.opacity_editor)

    def update_transfer_function(self):
        """
        需要把 OpacityEditor 的透明度曲線傳遞給 VolumeDock，
        """
        if not hasattr(self, "opacity_editor"):
            return

        points = self.opacity_editor.get_points()
        self.volume_renderer.update_transfer_function(points)

    def update(self, img):
        volume = img.get_fdata()

        self.opacity_editor.set_range(volume.min(), volume.max())
        self.histogram_viewer.set_histogram(volume)
