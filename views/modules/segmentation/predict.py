from matplotlib import pyplot as plt
from mmengine import Config
import numpy as np
import torch
import nibabel as nib
from custom_module.utils import build_model, build_transform

# from monai.inferers import sliding_window_inference

from custom_module.utils.inference import sliding_window_inference


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Predict with a model")
    parser.add_argument("config", help="Path to the config file")
    args = parser.parse_args()
    return args


def load_nii_gz(path):
    img = nib.load(path)
    # ornt = nib.orientations.io_orientation(img.affine)
    # img_data = img.get_fdata()
    # transformed_data = nib.orientations.apply_orientation(img_data, ornt)
    # img = nib.Nifti1Image(transformed_data, np.eye(4))  # canonical affine
    return img


if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    model = build_model(cfg.model).to("cuda")
    model.eval()
    transform = build_transform(cfg.valid_transform)
    checkpoint_path = "./weights/exp_12_4_4_3/best_model.pth"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    img = load_nii_gz("../../../data/dataset_3/pid_02/pid_02.nii.gz")
    transform_img, transform_data = transform(img, img.get_fdata())
    transform_data = (
        (transform_data[None, None, ...]).to("cuda").to(torch.float32)
    )  # 增加一個維度
    with torch.no_grad():  # 禁用梯度計算
        pred = sliding_window_inference(
            inputs=transform_data,
            roi_size=[cfg.roi_x, cfg.roi_y, cfg.roi_z],
            sw_batch_size=1,
            predictor=model,
            overlap=0.25,
            progress=True,
        )
        pred = torch.where(torch.nn.functional.softmax(pred, dim=1) > 0.5, 1, 0).cpu()
        pred_np = pred[0].cpu().numpy()
        for channel_idx in range(len(pred_np)):
            # pred_img, pred_bk = transform.inverse(img, pred[0][channel_idx])
            # breakpoint()
            mask_data = pred[0][channel_idx].cpu().numpy().astype(np.uint8)
            pred_img = nib.Nifti1Image(mask_data, transform_img.affine)
            nib.save(pred_img, f"output_channel{channel_idx}.nii.gz")
            # volume_np = pred_np[channel_idx]
            print(f"Saved output_channel{channel_idx}.nii.gz")
