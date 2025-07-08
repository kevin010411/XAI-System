from functools import partial
from PySide6.QtWidgets import QPushButton
from PySide6.QtCore import Qt

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

        self.visibility_button = QPushButton("👁️")
        self.visibility_button.setFixedSize(30, 30)
        self.visibility_button.setCheckable(True)
        self.visibility_button.setChecked(False)
        self.visibility_button.clicked.connect(self.set_volume_visible)
        self.selector_row.insertWidget(0, self.visibility_button)

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
        img_name = self.img_selector.currentText()
        self.volume_renderer.update_transfer_function(points, img_name)

    def update(self, img_name, img):
        super().update(img_name, img)
        volume = img.get_fdata()
        self.volume_renderer.update(img, img_name)

        self.opacity_editor.clear_points()
        self.opacity_editor.set_range(volume.min(), volume.max())

        self.img_selector.setEnabled(False)
        self.histogram_viewer.histogram_ready.connect(
            partial(self.first_histogram_ready, img_name),
            Qt.ConnectionType.SingleShotConnection,
        )
        self.histogram_viewer.set_histogram(volume)

        points = self.opacity_editor.get_points()
        self.volume_renderer.update_transfer_function(points, img_name)

    def on_img_selected(self, index):
        new_key = self.img_selector.currentText()
        if not new_key:
            return  # nothing loaded
        # Persist current settings before switching away
        if self._current_key is not None:
            self._save_settings(self._current_key)

        self._current_key = new_key
        self._current_img = self.data_manager.get_img(new_key)

        if new_key in self._settings:
            self._load_settings(self._settings[new_key])
        else:
            print(
                "⚠️新的img_name，進入這裡會預設資料但會切換過早會造成BUG請找到為什麼並避免"
            )
            volume = self._current_img.get_fdata()
            self.opacity_editor.clear_points()
            self.opacity_editor.set_range(volume.min(), volume.max())
            self.histogram_viewer.set_histogram(volume)

    def first_histogram_ready(self, img_name):
        self._save_settings(img_name)
        self.img_selector.setEnabled(True)

    def _save_settings(self, img_name):
        self._settings[img_name] = {
            "histogram_viewer": self.histogram_viewer.get_results(),
            "opacity_editor": self.opacity_editor.get_state(),
            "visibility": self.visibility_button.isChecked(),
        }

    def _load_settings(self, settings):
        """Load settings from the settings dict."""

        def select_enable():
            self.img_selector.setEnabled(True)

        self.img_selector.setEnabled(False)
        self.histogram_viewer.histogram_ready.connect(
            select_enable,
            Qt.ConnectionType.SingleShotConnection,
        )
        self.histogram_viewer.load_results(settings["histogram_viewer"])
        self.opacity_editor.load_state(settings["opacity_editor"])
        self.visibility_button.setChecked(settings["visibility"])

    def set_volume_visible(self, visible: bool):
        """Override to update visibility of the volume renderer."""
        img_name = self.img_selector.currentText()
        self.volume_renderer.show_volume(img_name, visible)
