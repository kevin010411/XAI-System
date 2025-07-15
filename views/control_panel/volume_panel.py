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
        需要把 OpacityEditor 的透明度曲線傳遞給 Volume Renderer，
        """
        if not hasattr(self, "opacity_editor"):
            return

        points = self.opacity_editor.get_points()
        img_name = self.img_selector.currentText()
        if img_name != "":
            self.volume_renderer.update_transfer_function(points, img_name)

    def _create_init_setting(self, img, img_name):
        volume = img.get_fdata()
        self.volume_renderer.update(img, img_name)

        init_opacity_editor = self.opacity_editor.get_init_state(
            volume.min(), volume.max()
        )
        init_vis = False

        self.img_selector.setEnabled(False)
        self.histogram_viewer.calculate_done.connect(
            partial(
                self.histogram_done,
                img_name=img_name,
                init_opacity_editor=init_opacity_editor,
                init_vis=init_vis,
            ),
            Qt.ConnectionType.SingleShotConnection,
        )
        self.histogram_viewer.calculate_histogram(volume)

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

    def histogram_done(self, calculated_histo, img_name, init_opacity_editor, init_vis):
        init_setting = {
            "histogram_viewer": calculated_histo,
            "opacity_editor": init_opacity_editor,
            "visibility": init_vis,
        }
        self._save_settings(img_name, init_setting)
        self.img_selector.setEnabled(True)

    def _save_settings(self, img_name, setting=None):
        if setting is None:  # 沒給setting時預設儲存目前版面
            self._settings[img_name] = {
                "histogram_viewer": self.histogram_viewer.get_results(),
                "opacity_editor": self.opacity_editor.get_state(),
                "visibility": self.visibility_button.isChecked(),
            }
        else:
            self._settings[img_name] = setting

    def _load_settings(self, settings):
        """Load settings from the settings dict."""
        self.histogram_viewer.load_results(settings["histogram_viewer"])
        self.opacity_editor.load_state(settings["opacity_editor"])
        self.visibility_button.setChecked(settings["visibility"])

    def set_volume_visible(self, visible: bool):
        """Override to update visibility of the volume renderer."""
        img_name = self.img_selector.currentText()
        self.volume_renderer.show_volume(img_name, visible)

    def _on_rows_inserted(self, parent, first: int, last: int):
        """
        新增行：為 [first, last] 每一行插入一份預設設定
        """
        for row in range(first, last + 1):
            img_name = self.data_manager.img_name_list_model.data(
                self.data_manager.img_name_list_model.index(row, 0), Qt.DisplayRole
            )
            self._row_name.append(img_name)
            if img_name == "":
                return
            img = self.data_manager.get_img(img_name)
            self._create_init_setting(img, img_name)

    def _on_rows_removed(self, parent, first: int, last: int):
        """
        刪除行：同步移除對應設定
        """
        for img_name in self._row_name[first : last + 1]:
            del self._settings[img_name]
            self.volume_renderer.remove_volume(img_name)
        del self._row_name[first : last + 1]

    def _on_data_changed(self, topLeft, bottomRight, roles):
        """
        編輯既有字串：若你的設定內容和文字有關，這裡更新
        """
        for row in range(topLeft.row(), bottomRight.row() + 1):
            new_img_name = self.data_manager.img_name_list_model.data(
                self.data_manager.img_name_list_model.index(row, 0), Qt.DisplayRole
            )
            old_name = self._row_name[row]
            if old_name != "":
                self._row_name[row] = new_img_name
                self._settings[new_img_name] = self._settings[old_name]
                del self._settings[old_name]
            else:
                self._row_name[row] = new_img_name
                img = self.data_manager.get_img(new_img_name)
                self._create_init_setting(img, new_img_name)

    def _on_model_reset(self):
        """
        重設 model：整份設定也重建
        """
        self.settings = [
            self._create_init_setting(self.data_manager.get_img(s), s)
            for s in self.data_manager.img_name_list_model.stringList()
        ]
