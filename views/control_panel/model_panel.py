from logging import warning
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout
from PySide6.QtCore import QThread, Signal, Slot, QObject
from mmengine import Config
import numpy as np
import torch
import nibabel as nib

from .base_panel import BasePanel
from views.utils import CollapsibleBox, ProgressBar
from ..segmentation_models.custom_module import (
    build_model,
    sliding_window_inference,
    build_transform,
    SlidingGradCAM3D,
)


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

        self.img = None  # 用於預測的影像
        self.model_object = None
        self.transform = None
        self.config: Config | None = None

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
            self.transform = build_transform(self.config.valid_transform)
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
        for layer_name, heatmap in heat_maps:
            self.data_manager.add_img(f"heatmap-{layer_name}", heatmap)
        self.data_manager.add_img("predicted", pred_img)

    @Slot(object)
    def _cleanup_worker(self, *_):
        # 執行緒結束 → 釋放引用，Qt 也可 deleteLater
        self.worker.finished.disconnect(self._cleanup_worker)
        self.worker.deleteLater()
        self.worker = None

    def update(self, img):
        """更新下拉選單，顯示目前載入的影像。"""
        super().update(img)
        self.img = img


class PredictWorker(QThread):

    class ProgressProxy:
        def __init__(self, update_signal=None, reset_signal=None):
            self._update_signal = update_signal
            self._reset_signal = reset_signal

        def update(self, n=1):
            self._update_signal.emit(n)

        def reset(self, total=1, desc="Processing"):
            self._reset_signal.emit(total, desc)

        def close(self):
            """Close the progress bar, if applicable."""
            pass

    pred_done = Signal(object)  # emit 結果物件
    update = Signal(int)  # 送出「已完成 n 步」或「目前進度值」
    reset = Signal(int, str)  # 送出「請把進度條重設到 0」

    def __init__(self, model, transform, input_data, config):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.transform = transform
        self.input_data = input_data
        self.config = config
        self.xai = SlidingGradCAM3D(
            model=model,
            target_layers=[model.bottleneck, model.out_block],
            class_selector=lambda p: torch.tensor([[1]]),  # 產兩個類別 heatmap
        )
        self.pbar = self.ProgressProxy(self.update, self.reset)

    def run(self):
        try:
            img_t, data_t = self.transform(self.input_data, self.input_data.get_fdata())
            data_t = data_t[None, None, ...].to(torch.float32).to(self.device)

            with torch.enable_grad():
                pred = sliding_window_inference(
                    inputs=data_t,
                    roi_size=[self.config.roi_x, self.config.roi_y, self.config.roi_z],
                    sw_batch_size=1,
                    predictor=self.model,
                    overlap=0.25,
                    # progress=True,
                    pbar=self.pbar,
                    xai_pre_patch=self.xai.pre,
                    xai_post_patch=self.xai.post,
                    xai_final=self.xai.final,
                )
                mask = (
                    torch.argmax(pred, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                )
                pred_img = nib.Nifti1Image(mask, img_t.affine, header=img_t.header)
                _, heatmaps = self.xai.output
                heat_imgs = [
                    (
                        layer_name,
                        nib.Nifti1Image(
                            heatmap.squeeze(0).squeeze(0),
                            img_t.affine,
                            header=img_t.header,
                        ),
                    )
                    for layer_name, heatmap in heatmaps.items()
                ]
                pred_img = (pred_img, heat_imgs)
        except Exception as e:
            print("PredictWorker error:", e)
            pred_img = (None, None)
        finally:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        self.pred_done.emit(pred_img)
