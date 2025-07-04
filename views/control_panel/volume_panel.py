from views.volume.transfer_editor import OpacityEditor
from views.volume import HistogramViewer

from .base_panel import BasePanel


class VolumePanel(BasePanel):

    def __init__(self, volume_renderer, **kwargs):
        super().__init__(**kwargs)
        self.setWindowTitle("Volume Control Panel")

        self._current_key: str | None = None
        self._current_img = None
        self._settings = {}

        self.volume_renderer = volume_renderer
        self.histogram_viewer = HistogramViewer()
        self.opacity_editor = OpacityEditor(self.update_transfer_function)
        self.layout.addWidget(self.histogram_viewer)
        self.layout.addWidget(self.opacity_editor)

        self.img_selector.currentIndexChanged.connect(self.on_img_selected)

    def update_transfer_function(self):
        """
        需要把 OpacityEditor 的透明度曲線傳遞給 VolumeDock，
        """
        if not hasattr(self, "opacity_editor"):
            return

        points = self.opacity_editor.get_points()
        self.volume_renderer.update_transfer_function(points)

    def update(self, img):
        super().update(img)
        volume = img.get_fdata()
        self.opacity_editor.clear_points()
        self.opacity_editor.set_range(volume.min(), volume.max())
        self.histogram_viewer.set_histogram(volume)
        points = self.opacity_editor.get_points()
        self.volume_renderer.update(img, points)

    def on_img_selected(self, index):
        new_key = self.img_selector.currentText()
        if not new_key:
            return  # nothing loaded
        # 1) Persist current settings before switching away
        if self._current_key is not None:
            self._settings[self._current_key] = {
                "histogram_viewer": self.histogram_viewer.get_results(),
                "opacity_editor": self.opacity_editor.get_state(),
            }

        self._current_key = new_key
        self._current_img = self.data_manager.get_img(new_key)

        if new_key in self._settings:
            self._load_settings(self._settings[new_key])
        else:
            volume = self._current_img.get_fdata()
            self.opacity_editor.clear_points()
            self.opacity_editor.set_range(volume.min(), volume.max())
            self.histogram_viewer.set_histogram(volume)

    def _load_settings(self, settings):
        """Load settings from the settings dict."""
        self.histogram_viewer.load_results(settings["histogram_viewer"])
        self.opacity_editor.load_state(settings["opacity_editor"])
