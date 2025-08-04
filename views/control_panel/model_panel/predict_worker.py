from PySide6.QtCore import QThread, Signal

import numpy as np
import torch
import nibabel as nib

# from monai.inferers import sliding_window_inference

from ...segmentation_models.custom_module import (
    sliding_window_inference,
    build_xai,
    SlidingSegGradCAM,
)


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

    def __init__(
        self, model, transform, input_data, target_layer, config, xai_config=None
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.transform = transform
        self.input_data = input_data
        self.config = config
        if xai_config is not None:
            self.xai = build_xai(xai_config)
        else:
            self.xai = build_xai(config.xai_method)
        try:
            self.xai.set_model(self.model)
            self.xai.set_target_layers(target_layer)
            self.xai.set_class(lambda p: torch.tensor([[1]]))
        except:
            print("XAI method not set correctly, using SlidingGradCAM3D as default.")
            self.xai = SlidingSegGradCAM(
                model=model,
                target_layers=[model.bottleneck, model.out_block],
                class_selector=lambda p: torch.tensor([[1]]),  # 產兩個類別 heatmap
            )
        self.pbar = self.ProgressProxy(self.update, self.reset)

    def run(self):
        """
        執行模型預測與XAI分析
        """
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
                pred_img, _ = self.transform.inverse(pred_img, pred_img.get_fdata())
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
                heat_imgs = [
                    (layer_name, self.transform.inverse(img, img.get_fdata())[0])
                    for (layer_name, img) in heat_imgs
                ]
                pred_img = (pred_img, heat_imgs)
        except Exception as e:
            print("PredictWorker error:", e)
            pred_img = (None, None)
        finally:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        self.pred_done.emit(pred_img)
