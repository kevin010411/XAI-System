from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QDockWidget,
    QTabWidget,
    QComboBox,
)
from PySide6.QtCore import QThread, Signal, Slot

from .modules import sliding_window_inference
import nibabel as nib
import numpy as np
import torch
from .utils import wrap_with_frame, WaitingSpinner
from .modules import build_transform, SlidingGradCAM3D
from .volume import VolumeDock


class SegmentationDock(QDockWidget):
    def __init__(self):
        super().__init__("Segmentation Viewer")

        self.model = None
        self.config = None
        self.transform = None
        self.worker = None

        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.label = QLabel("這裡將顯示 AI預測與可解釋化 結果")
        layout.addWidget(self.label)

        tab = QTabWidget()
        self._init_segment_tab(tab)
        self._init_xai_tab(tab)

        layout.addWidget(tab)
        self.setWidget(wrap_with_frame(widget))

        self.loading_spinner = WaitingSpinner(
            self,
            disable_parent_when_spinning=True,
            roundness=100.0,
            fade=80.0,
            radius=10,
            lines=25,
            line_length=15,
            line_width=3,
            speed=1.5707963267948966,
            color=(127, 127, 127),
        )
        self.loading_spinner.stop()

    def _init_segment_tab(self, tab):
        self.segment_volume_dock = VolumeDock()
        self.segment_volume_dock.setTitleBarWidget(QWidget(self.segment_volume_dock))
        self.segment_volume_dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        tab.addTab(self.segment_volume_dock, "Volume Viewer")

    def _init_xai_tab(self, tab):
        self.xai_volume_dock = VolumeDock(slice_mode="cold_to_hot")
        self.xai_volume_dock.setTitleBarWidget(QWidget(self.xai_volume_dock))
        self.xai_volume_dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.xai_combo = QComboBox()
        self.xai_combo.currentIndexChanged.connect(self.change_heat_map)
        layout.addWidget(self.xai_combo)
        layout.addWidget(self.xai_volume_dock)
        tab.addTab(widget, "XAI Viewer")

    def change_heat_map(self, idx):
        layer_name = self.xai_combo.itemText(idx)
        heat_map = self.heat_dict.get(layer_name, None)
        if heat_map is None:
            self.xai_volume_dock.status_label.setText(f"讀取heat map-{layer_name}失敗")
            return
        self.xai_volume_dock.update(heat_map)

    def update_model_and_config(self, model, config):
        """更新顯示的文字"""
        self.label.setText(f'以載入模型:{config.model["type"]}')
        self.model = model.to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.config = config
        self.transform = build_transform(config.valid_transform)

    def update(self, img):
        if self.model is None or self.transform is None:
            self.label.setText("模型或轉換尚未設定")
            return

        if self.worker and self.worker.isRunning():
            self.label.setText("已有推論執行中")
            return

        self.loading_spinner.start()

        self.worker = PredictWorker(
            model=self.model,
            transform=self.transform,
            input_data=img,
            config=self.config,
        )
        self.worker.pred_done.connect(self.on_predict_done)
        self.worker.finished.connect(self._cleanup_worker)
        self.worker.start()

    def on_predict_done(self, result):
        """處理預測結果"""
        pred_img, heat_maps = result
        self.xai_combo.clear()
        self.heat_dict = {}
        for layer_name, heatmap in heat_maps:
            self.xai_combo.addItems([layer_name])
            self.heat_dict[layer_name] = heatmap
        self.loading_spinner.stop()

        if isinstance(pred_img, nib.Nifti1Image):
            self.label.setText("預測結果成功")
            self.segment_volume_dock.update(pred_img)
        else:
            self.label.setText("預測結果格式錯誤")

    @Slot(object)
    def _cleanup_worker(self, *_):
        # 執行緒結束 → 釋放引用，Qt 也可 deleteLater
        self.worker.finished.disconnect(self._cleanup_worker)
        self.worker.deleteLater()
        self.worker = None

    def prepare_for_exit(self):
        self.segment_volume_dock.prepare_for_exit()
        self.xai_volume_dock.prepare_for_exit()


class PredictWorker(QThread):
    pred_done = Signal(object)  # emit 結果物件

    def __init__(self, model, transform, input_data, config):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.transform = transform
        self.input_data = input_data
        self.config = config
        self.xai = SlidingGradCAM3D(
            model=model,
            target_layers=[model.bottleneck, model.out_block],
            class_selector=lambda p: torch.tensor([[1]]),  # 產兩個類別 heatmap
        )

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
                    progress=True,
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
