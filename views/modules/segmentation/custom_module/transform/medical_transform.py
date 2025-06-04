from warnings import warn
import numpy as np
from torch import nn
import nibabel as nib
from nibabel.processing import resample_from_to
from ..utils import TRANSFORMS


@TRANSFORMS.register_module()
class ScaleIntensityRanged(nn.Module):
    """
    把圖片的強度值從[a_min, a_max]範圍映射到[b_min, b_max]範圍。
    超過[a_min, a_max]的值會被設為0，除非clip=False。
    args:
    -----
    a_min：float，強度原範圍最小值。可以理解為需要被歸一化的最小值，如我們這個例子中的-300(需要寫成小數，-300.0)
    a_max： float， 強度原範圍最大值。可以理解為需要被歸一化的最大值，如我們這個例子中的300(需要寫成小數，300.0)
    b_min： float, 強度目標範圍最小值。可以理解為歸一化後的最小值，通常設定為0.0
    b_max： float, 強度目標範圍最大值。可以理解為歸一化後的最大值，通常設定為1.0
    clip： 布林值。設定為True, 才會將[-300,+300]之外的值都設為0.通常為True
    """

    def __init__(self, a_min, a_max, b_min, b_max, clip=False):
        super(ScaleIntensityRanged, self).__init__()
        self.a_min = a_min
        self.a_max = a_max
        self.b_min = b_min
        self.b_max = b_max
        self.clip = clip

    def forward(
        self, img: nib.nifti1.Nifti1Image, input: np.ndarray
    ) -> tuple[nib.nifti1.Nifti1Image, np.ndarray]:
        if self.a_max - self.a_min == 0:
            warn(
                "ScaleIntensityRanged: a_max and a_min are equal, which will lead to division by zero."
            )
            return img, input - self.a_min + self.b_min
        input = (input - self.a_min) / (self.a_max - self.a_min)
        input = input * (self.b_max - self.b_min) + self.b_min
        if self.clip:
            input = input.clip(self.b_min, self.b_max)
        return img, input

    def inverse(self, img: nib.nifti1.Nifti1Image, input: np.ndarray):
        if self.a_max - self.a_min == 0:
            return img, input - self.b_min + self.a_min
        # 先反推回標準化前
        input_rescaled = (input - self.b_min) / (self.b_max - self.b_min)
        input_rescaled = input_rescaled * (self.a_max - self.a_min) + self.a_min
        return img, input_rescaled


@TRANSFORMS.register_module()
class ReOrientation(nn.Module):
    """
    將圖片的方向改為target方向。
    ⚡注意:這個轉換會把input的空間改變，所以如果在這個轉換前對input做的其他處理會消失。
    args:
    -----
    target: str, 目標方向，預設為RAS。
    """

    def __init__(self, target="RAS"):
        super(ReOrientation, self).__init__()
        self.target = target
        self._last_ornt = None  # 儲存最後一次的方向

    def forward(self, img: nib.nifti1.Nifti1Image, input: np.ndarray) -> np.ndarray:
        affine = img.affine

        current_ornt = nib.io_orientation(affine)  # 計算目前方向
        target_ornt = nib.orientations.axcodes2ornt(
            self.target
        )  # 目標方向，例如要轉成 RAS
        transform = nib.orientations.ornt_transform(
            current_ornt, target_ornt
        )  # 轉換矩陣
        reoriented_data = nib.orientations.apply_orientation(
            input, transform
        )  # 重新定向資料 (numpy array)
        new_affine = affine.dot(
            nib.orientations.inv_ornt_aff(transform, input.shape)
        )  # 計算新的 affine 矩陣

        img = nib.Nifti1Image(reoriented_data, new_affine)  # 建立新的 Nifti1Image
        self._last_ornt = (target_ornt, current_ornt)  # 存下目前方向與之前方向
        return img, img.get_fdata()  # 返回改變方向後的圖片數據

    def inverse(self, img: nib.nifti1.Nifti1Image, input: np.ndarray):
        # 反轉: target→current
        if not self._last_ornt:
            raise RuntimeError("Must call forward() before inverse()")
        target_ornt, current_ornt = self._last_ornt
        transform = nib.orientations.ornt_transform(target_ornt, current_ornt)
        inv_data = nib.orientations.apply_orientation(input, transform)
        new_affine = img.affine.dot(
            nib.orientations.inv_ornt_aff(transform, input.shape)
        )
        img = nib.Nifti1Image(inv_data, new_affine, header=img.header)
        return img, img.get_fdata()


@TRANSFORMS.register_module()
class ReSpace(nn.Module):
    """
    將圖片的空間從原始空間轉換到目標空間。
    ⚡注意:這個轉換會把input的空間改變，所以如果在這個轉換前對input做的其他處理會消失。
    args:
    -----
    space_x: float, 目標空間在x軸的間距。
    space_y: float, 目標空間在y軸的間距。
    space_z: float, 目標空間在z軸的間距。
    """

    def __init__(self, space_x, space_y, space_z):
        super(ReSpace, self).__init__()
        self.target_zoom = np.array((space_x, space_y, space_z))
        self._orig_affine = None
        self._orig_shape = None

    def forward(self, img: nib.nifti1.Nifti1Image, input: np.ndarray) -> np.ndarray:
        self._orig_affine = img.affine.copy()
        self._orig_shape = img.shape

        old_zooms = img.header.get_zooms()[:3]
        old_shape = img.shape[:3]

        scale_factors = old_zooms / self.target_zoom
        new_shape = np.ceil(np.array(old_shape) * scale_factors).astype(int)

        # 建立新的 affine：維持原始方向，只改 spacing
        new_affine = np.copy(img.affine)
        for i in range(3):
            # 對角元素乘上原縮放比率 / 新縮放比率
            direction = img.affine[:3, i] / old_zooms[i]
            new_affine[:3, i] = direction * self.target_zoom[i]

        # 建立目標影像參考用
        target_img = nib.Nifti1Image(np.zeros(new_shape), new_affine)

        # 進行重採樣
        resampled_img = resample_from_to(img, target_img, order=3)
        return resampled_img, resampled_img.get_fdata()

    def inverse(self, img: nib.nifti1.Nifti1Image, input: np.ndarray):
        if self._orig_affine is None or self._orig_shape is None:
            raise RuntimeError("Must call forward() before inverse()")
        target_img = nib.Nifti1Image(np.zeros(self._orig_shape), self._orig_affine)
        orig_img = nib.Nifti1Image(input, img.affine, header=img.header)
        resampled_img = resample_from_to(orig_img, target_img, order=3)
        return resampled_img, resampled_img.get_fdata()
