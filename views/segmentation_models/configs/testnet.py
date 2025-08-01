_base_ = ["./xai/sliding_seg_grad_cam.py"]

pretrain_path = "./views/segmentation_models/weights/exp_12_4_4_3/best_model.pth"
model = dict(
    type="testnet",
    out_channels=2,
    patch_size=2,
    kernel_size=7,
    exp_rate=4,
    feature_size=48,
    depths=[3, 3, 9, 3],
    drop_path_rate=0.1,
    deep_sup=True,
)

valid_transform = dict(
    type="Compose",
    transforms=[
        dict(type="ReOrientation", target="RAS"),
        dict(type="ReSpace", space_x=0.7, space_y=0.7, space_z=1.0),
        dict(
            type="ScaleIntensityRanged",
            a_min=-42,
            a_max=423,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        dict(type="ToTensor"),
    ],
)


norm_name = "layer"
roi_x = 64
roi_y = 64
roi_z = 64
