from logging import warning
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QGroupBox,
)
from PySide6.QtCore import Slot
from mmengine import Config
import torch

from ..base_panel import BasePanel
from views.utils import CollapsibleBox, ProgressBar
from ...segmentation_models.custom_module import (
    build_model,
    build_transform,
)
from .predict_worker import PredictWorker
from .target_layer_list import TargetLayerList


class ModelPanel(BasePanel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.selector_label.setText("辨識圖像:")
        self.img_selector.currentIndexChanged.connect(self.on_img_change)

        model_row = QHBoxLayout()
        self.model_select_label = QLabel("辨識模型：")
        model_row.addWidget(self.model_select_label)

        self.model_select = QComboBox()
        self.model_select.setPlaceholderText("(尚未載入模型)")
        self.model_select.currentIndexChanged.connect(self.on_model_selected)
        model_row.addWidget(self.model_select, 1)  # stretch=1 讓下拉佔滿剩餘寬度
        self.add_row_above_stretch(model_row)
        self.load_config_list("./views/segmentation_models/configs", self.model_select)

        xai_row = QHBoxLayout()
        self.xai_select_label = QLabel("xai採用方法：")
        xai_row.addWidget(self.xai_select_label)

        self.xai_select = QComboBox()
        self.xai_select.setPlaceholderText("(尚未選擇xai方法)")
        self.xai_select.currentIndexChanged.connect(self.on_xai_selected)
        xai_row.addWidget(self.xai_select, 1)  # stretch=1 讓下拉佔滿剩餘寬度
        self.add_row_above_stretch(xai_row)
        self.load_config_list(
            "./views/segmentation_models/configs/xai", self.xai_select
        )

        self.img = None  # 用於預測的影像
        self.model_object = None
        self.transform = None
        self.config: Config | None = None
        self.xai_config: Config | None = None

        self.info_box = CollapsibleBox("模型資訊")
        self.layout.insertWidget(self._stretch_idx, self.info_box)
        self._stretch_idx += 1

        predict_group = QVBoxLayout()
        self.predict_button = QPushButton("開始預測")
        self.predict_button.clicked.connect(self.on_predict_clicked)
        predict_group.addWidget(self.predict_button)
        self.progress_bar = ProgressBar()
        predict_group.addWidget(self.progress_bar)
        self.add_row_above_stretch(predict_group)

    def load_config_list(self, model_path, selector: QComboBox):
        from pathlib import Path

        if isinstance(model_path, str):
            model_path = Path(model_path)
        for path in model_path.glob("*.py"):
            selector.addItem(path.stem, userData=path)

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
            self.transform = build_transform(self.config.valid_transform)
            self.populate_info_box(config_path, model)
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.critical(self, "模型載入失敗", f"錯誤: {e}")
            self.model_object = None

    def on_xai_selected(self, index: int):
        xai_config_path = self.xai_select.itemData(index)
        if xai_config_path is not None:
            self.xai_config = Config.fromfile(xai_config_path)["xai_method"]

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

        self.info_box.add_widget(self.create_target_layer_widget(model))

        self.info_box.toggle_btn.setChecked(True)

    def create_target_layer_widget(self, model):
        group = QGroupBox("Target Layer")
        group.setCheckable(True)  # 允許收合/展開
        group.setChecked(True)

        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(4, 4, 4, 4)

        self.list_widget = TargetLayerList(model)

        group_layout.addWidget(self.list_widget)

        group.toggled.connect(self.list_widget.setVisible)

        return group

    def on_img_change(self, index: int):
        new_img_name = self.img_selector.currentText()
        self.img = self.data_manager.get_img(new_img_name)

    def on_predict_clicked(self):
        """當按下預測按鈕時，開始進行預測。"""
        if self.model_object is None or self.transform is None or self.img is None:
            warning("請先選擇模型和影像。")
            return
        self.predict_button.setEnabled(False)
        self.worker = PredictWorker(
            model=self.model_object,
            transform=self.transform,
            input_data=self.img,
            config=self.config,
            target_layer=self.list_widget.target_layer,
            xai_config=self.xai_config,
        )
        self.worker.pred_done.connect(self.on_predict_done)
        self.worker.finished.connect(self._cleanup_worker)
        self.worker.update.connect(self.progress_bar.update)
        self.worker.reset.connect(self.progress_bar.reset)
        self.worker.start()

    def on_predict_done(self, result):
        """處理預測結果"""
        self.predict_button.setEnabled(True)
        pred_img, heat_maps = result
        if heat_maps is not None:
            for layer_name, heatmap in heat_maps:
                self.data_manager.add_img(f"heatmap-{layer_name}", heatmap)
        self.data_manager.add_img("predicted", pred_img)

    @Slot(object)
    def _cleanup_worker(self, *_):
        # 執行緒結束 → 釋放引用，Qt 也可 deleteLater
        self.worker.finished.disconnect(self._cleanup_worker)
        self.worker.deleteLater()
        self.worker = None
