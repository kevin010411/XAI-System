from functools import partial
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QDockWidget, QTabWidget
from PySide6.QtCore import QThread, Signal, QObject

from monai.inferers import sliding_window_inference
import nibabel as nib
import numpy as np
import torch
from .utils import wrap_with_frame, LoadingOverlay
from .modules import build_transform
from .volume import VolumeDock


class SegmentationDock(QDockWidget):
    def __init__(self):
        super().__init__("Segmentation Viewer")

        self.model = None
        self.config = None
        self.transform = None

        self.loading_dialog = LoadingOverlay(self)
        widget = QWidget()
        layout = QVBoxLayout()
        self.label = QLabel("這裡將顯示 segmentation 結果")
        layout.addWidget(self.label)
        widget.setLayout(layout)

        tab = QTabWidget()
        # self.volume_dock = VolumeDock()
        # self.volume_dock.setTitleBarWidget(QWidget(self.volume_dock))
        # self.volume_dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        # tab.addTab(self.volume_dock, "Volume Viewer")

        layout.addWidget(tab)
        self.setWidget(wrap_with_frame(widget))

    def update_model_and_config(self, model, config):
        """更新顯示的文字"""
        self.label.setText(f'以載入模型:{config.model["type"]}')
        self.model = model
        self.config = config
        self.transform = build_transform(config.valid_transform)

    def update(self, img):
        if self.model is None or self.transform is None:
            self.label.setText("模型或轉換尚未設定")
            return
        transform_img, transform_data = self.transform(img, img.get_fdata())
        transform_data = (transform_data[None, None, ...]).to(
            torch.float32
        )  # 增加一個維度
        self.model.eval()
        with torch.no_grad():  # 禁用梯度計算
            self.loading_dialog.show()
            # worker = PredictWorker(self.model, transform_data, self.config)
            # worker.finished.connect(
            #     partial(self.on_predict_done, input_img=transform_img)
            # )
            # worker.start()

    def on_predict_done(self, result, input_img):
        """處理預測結果"""

        if self.loading_dialog:  # 關閉讀取中
            self.loading_dialog.hide()

        # 將預測結果轉換為 NIfTI 格式
        result = result[0][1].cpu().numpy().astype(np.uint8)
        pred_img = nib.Nifti1Image(result, input_img.affine)

        breakpoint()
        if isinstance(result, nib.Nifti1Image):
            self.label.setText("預測結果成功")
            # self.volume_dock.update(result)
        else:
            self.label.setText("預測結果格式錯誤")


class PredictWorker(QThread):
    finished = Signal(object)  # emit 結果物件

    def __init__(self, model, input_data, config):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device)
        self.config = config
        self.input_data = input_data.to(device)

    def run(self):
        if self.model is None or self.input_data is None:
            self.finished.emit(None)
            return
        # 這裡在背景執行
        pred = sliding_window_inference(
            inputs=self.input_data,
            roi_size=[self.config.roi_x, self.config.roi_y, self.config.roi_z],
            sw_batch_size=1,
            predictor=self.model,
            overlap=0.25,
        )
        result = torch.where(torch.nn.functional.softmax(pred, dim=1) > 0.5, 1, 0).cpu()
        self.finished.emit(result)
