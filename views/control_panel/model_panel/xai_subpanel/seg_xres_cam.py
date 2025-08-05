from logging import warning
from PySide6.QtWidgets import QWidget, QSpinBox, QLabel, QFormLayout
from .xai_config_list import register_panel


@register_panel("SlidingSegXResCAM")
class SegXResCAMPanel(QWidget):
    """
    A panel for displaying and interacting with the SegXResCAM model.
    This panel is designed to work with the PredictWorker to visualize
    segmentation results and explainable AI (XAI) outputs.
    """

    def __init__(self, xai_config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SegXResCAM Panel")
        form = QFormLayout(self)
        self.pool_spin = QSpinBox(self)
        self.pool_spin.setRange(1, 10)
        self.pool_spin.setValue(xai_config.get("pool_size", 1))
        self.pool_spin.setToolTip("Global-average-pool size in XResCAM")
        form.addRow(QLabel("pool_size :"), self.pool_spin)

        self.xai_config = dict(xai_config)
        self.config_update_callback = None

        self.pool_spin.valueChanged.connect(self.on_config_changed)

    def on_config_changed(self, _):
        """
        Update the panel based on the new configuration.
        This method can be extended to update UI elements or settings
        based on the provided configuration.
        """
        self.xai_config["pool_size"] = self.pool_spin.value()
        if self.config_update_callback is None:
            warning("No config update callback 所以沒辦法更新")
        else:
            self.config_update_callback(self.xai_config)

    def set_change_callback(self, callback):
        """
        Set a callback function that will be called when the configuration changes.
        This is useful for updating the model or triggering re-evaluation.
        """
        self.config_update_callback = callback
