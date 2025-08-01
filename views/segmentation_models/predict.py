from matplotlib import pyplot as plt
from mmengine import Config
import numpy as np
import torch
import nibabel as nib
from .custom_module.utils import build_model, build_transform
from .custom_module.utils import sliding_window_inference
from .custom_module.xai import SlidingSegGradCAM

# from monai.inferers import sliding_window_inference


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Predict with a model")
    parser.add_argument("config", help="Path to the config file")
    args = parser.parse_args()
    return args


def load_nii_gz(path):
    img = nib.load(path)
    return img


def save_nii_gz(pred, file_name):
    pred_np = pred[0].cpu().numpy()
    for channel_idx in range(len(pred_np)):
        # pred_img, pred_bk = transform.inverse(img, pred[0][channel_idx])
        # breakpoint()
        mask_data = pred_np[channel_idx].astype(np.float32)
        pred_img = nib.Nifti1Image(mask_data, transform_img.affine)
        nib.save(pred_img, f"{file_name}_{channel_idx}.nii.gz")
        # volume_np = pred_np[channel_idx]
        print(f"Saved {file_name}_{channel_idx}.nii.gz")


def fp16_predictor(x):
    from torch.amp import autocast

    # 所有 forward 都在 autocast 內
    with autocast(device_type="cuda", dtype=torch.float16):  # 或 bfloat16 視顯卡而定
        return model(x)


if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    model = build_model(cfg.model).to("cuda")
    transform = build_transform(cfg.valid_transform)
    checkpoint_path = "./weights/exp_12_4_4_3/best_model.pth"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    # img = load_nii_gz("../../../data/dataset_3/pid_02/pid_02.nii.gz")
    img = load_nii_gz("../../../data/87.nii.gz")
    transform_img, transform_data = transform(img, img.get_fdata())
    transform_data = (
        (transform_data[None, None, ...]).to("cuda").to(torch.float32)
    )  # 增加一個維度
    xai = SlidingSegGradCAM(
        model=model,
        target_layers=[model.bottleneck, model.out_block],
        class_selector=lambda p: torch.tensor([[1]]),  # 產兩個類別 heatmap
    )
    with torch.enable_grad():
        pred = sliding_window_inference(
            inputs=transform_data,
            # mode="gaussian",
            roi_size=[cfg.roi_x, cfg.roi_y, cfg.roi_z],
            sw_batch_size=1,
            predictor=model,
            overlap=0.25,
            progress=True,
            xai_pre_patch=xai.pre,
            xai_post_patch=xai.post,
            xai_final=xai.final,
        )
        save_nii_gz(
            torch.where(torch.nn.functional.softmax(pred, dim=1) > 0.5, 1, 0).cpu(),
            "output",
        )
    preds, heatmaps = xai.output
    for k, v in heatmaps.items():
        save_nii_gz(
            v.cpu(),
            f"gradCAM_{k}",
        )
