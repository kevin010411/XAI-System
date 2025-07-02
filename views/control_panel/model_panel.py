from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QGroupBox, QVBoxLayout
from mmengine import Config
import torch

from .base_panel import BasePanel
from views.utils import CollapsibleBox
from ..segmentation_models.custom_module import build_model


class ModelPanel(BasePanel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.selector_label.setText("辨識圖像:")

        model_row = QHBoxLayout()
        self.model_select_label = QLabel("辨識模型：")
        model_row.addWidget(self.model_select_label)

        self.model_select = QComboBox()
        self.model_select.setPlaceholderText("(尚未載入模型)")
        self.model_select.currentIndexChanged.connect(self.on_model_selected)
        model_row.addWidget(self.model_select, 1)  # stretch=1 讓下拉佔滿剩餘寬度
        self.add_row_above_stretch(model_row)
        self.load_models_list("./views/segmentation_models/configs")

        self.model_object = None
        self.config: Config | None = None

        self.info_box = CollapsibleBox("模型資訊")
        self.layout.insertWidget(self._stretch_idx, self.info_box)
        self._stretch_idx += 1

    def load_models_list(self, model_path):
        from pathlib import Path

        if isinstance(model_path, str):
            model_path = Path(model_path)
        for path in model_path.glob("*.py"):
            self.model_select.addItem(path.stem, userData=path)

    def on_model_selected(self, index):
        """當選擇模型時，更新模型設定。"""
        config_path = self.model_select.itemData(index)
        try:
            self.config = Config.fromfile(config_path)
            model = build_model(self.config.model)
            state = torch.load(
                self.config.pretrain_path,
                map_location="cuda" if torch.cuda.is_available() else "cpu",
            )
            if "state_dict" in state:
                state = state["state_dict"]
                model.load_state_dict(state)
            else:
                model.load(state)
            model.eval()
            self.model_object = model
            self.populate_info_box(config_path, model)
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.critical(self, "模型載入失敗", f"錯誤: {e}")
            self.model_object = None

    def populate_info_box(self, config_path, model) -> None:  # type: ignore[no-any-unimported]
        """Fill collapsible box with human‑readable model details."""
        self.info_box.clear()

        total_params = sum(p.numel() for p in model.parameters())
        self.info_box.add_widget(QLabel(f"config位置：{config_path}", wordWrap=True))
        self.info_box.add_widget(
            QLabel(f"模型權重位置：{self.config.pretrain_path}", wordWrap=True)
        )
        self.info_box.add_widget(QLabel(f"參數量：{total_params:,}", wordWrap=True))
        self.info_box.add_widget(
            QLabel(f"架構：{self.config.model['type']}", wordWrap=True)
        )

        self.info_box.toggle_btn.setChecked(True)

    def update(self, img):
        """更新下拉選單，顯示目前載入的影像。"""
        self.refresh_img_selector()
