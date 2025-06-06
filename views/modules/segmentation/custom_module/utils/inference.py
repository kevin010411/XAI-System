# sliding_window_inference_pytorch.py
"""
A feature-complete, MONAI-compatible sliding_window_inference implemented
purely with PyTorch (>=1.13).  --Kevin Fu 2025-06-06
"""
from __future__ import annotations

import math
import warnings
from itertools import product
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
import torch.nn.functional as F

try:
    from tqdm.auto import tqdm  # 自動決定 console / notebook
except ImportError:  # 沒裝 tqdm 亦可運作
    tqdm = None

__all__ = ["sliding_window_inference"]

# -------------------------------------------------------------------------- #
# Internal helpers                                                           #
# -------------------------------------------------------------------------- #


# A. fall back ROI size: treat <=0 或 None 為沿用對應影像大小
def _fallback_roi_size(
    roi_size: Sequence[int] | int,
    img_size: Sequence[int],
) -> Tuple[int, ...]:
    if isinstance(roi_size, int):
        roi_size = (roi_size,) * len(img_size)
    if len(roi_size) != len(img_size):
        raise ValueError(
            f"roi_size 長度 ({len(roi_size)}) 與影像空間維度 ({len(img_size)}) 不符"
        )
    roi_size = tuple(
        img if (r is None or r <= 0) else int(r) for r, img in zip(roi_size, img_size)
    )
    return roi_size


# B. importance map（constant / gaussian）──與 MONAI 同語義
def _compute_importance_map(
    roi_size: Sequence[int],
    mode: str = "constant",
    sigma_scale: Sequence[float] | float = 0.125,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    if mode.lower() == "constant":
        return torch.ones(
            (1, 1, *roi_size), dtype=dtype, device=device, requires_grad=False
        )

    # gaussian
    if isinstance(sigma_scale, (int, float)):
        sigma_scale = [float(sigma_scale)] * len(roi_size)
    center = [(s - 1) / 2.0 for s in roi_size]
    sigmas = [s * sc for s, sc in zip(roi_size, sigma_scale)]

    grids = torch.meshgrid(
        *[torch.arange(s, dtype=dtype, device=device) for s in roi_size], indexing="ij"
    )
    exponent = sum(
        ((g - c) ** 2) / (2 * (sig**2)) for g, c, sig in zip(grids, center, sigmas)
    )
    g_map = torch.exp(-exponent)
    g_map /= g_map.max()
    eps = torch.finfo(dtype).eps
    g_map = torch.clamp(g_map, min=eps)
    return g_map[None, None, ...]  # shape (1,1,*roi)


# C. 取得掃描起點（interval = (1-overlap)*roi，最少 1）
def _scan_intervals(
    roi_size: Sequence[int],
    overlap: float,
) -> Tuple[int, ...]:
    intervals = []
    for r in roi_size:
        if overlap < 0 or overlap >= 1:
            raise ValueError("overlap 必須在 [0,1) 區間內")
        iv = int(r * (1.0 - overlap))
        intervals.append(max(iv, 1))
    return tuple(intervals)


# D. 建立所有 patch 的 slice 列表
def _dense_patch_slices(
    img_size: Sequence[int],
    roi_size: Sequence[int],
    interval: Sequence[int],
) -> List[Tuple[slice, ...]]:
    starts_per_dim = []
    for im, r, step in zip(img_size, roi_size, interval):
        stops = im - r
        if stops < 0:
            raise ValueError("ROI 大於影像（已先 pad，不應出現）")
        s = list(range(0, stops + 1, step))
        if s[-1] != stops:
            s.append(stops)
        starts_per_dim.append(s)
    slice_list = []
    for starts in product(*starts_per_dim):
        slice_obj = tuple(slice(st, st + r) for st, r in zip(starts, roi_size))
        slice_list.append(slice_obj)
    return slice_list


# E. 重新取樣 importance_map 至 seg_prob 尺寸（nearest，無 alias）
def _resize_importance(
    imp: torch.Tensor,
    new_spatial: Sequence[int],
) -> torch.Tensor:
    if list(imp.shape[2:]) == list(new_spatial):
        return imp
    return F.interpolate(imp, size=new_spatial, mode="nearest")


# -------------------------------------------------------------------------- #
# Public API                                                                 #
# -------------------------------------------------------------------------- #
@torch.inference_mode()
def sliding_window_inference(
    inputs: torch.Tensor,
    roi_size: Sequence[int] | int,
    sw_batch_size: int,
    predictor: Callable[
        ..., Union[torch.Tensor, Sequence[torch.Tensor], Dict[Any, torch.Tensor]]
    ],
    *,
    overlap: float = 0.25,
    mode: str = "constant",
    sigma_scale: Sequence[float] | float = 0.125,
    padding_mode: str = "constant",
    cval: float = 0.0,
    sw_device: torch.device | str | None = None,
    device: torch.device | str | None = None,
    progress: bool = False,
    roi_weight_map: Optional[torch.Tensor] = None,
    process_fn: Optional[
        Callable[
            [Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor],
            Tuple[Tuple[torch.Tensor, ...], torch.Tensor],
        ]
    ] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    progress_unit: str = "patch",  # "patch" or "batch"
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[Any, torch.Tensor]]:
    """
    Pure-PyTorch sliding window inference with feature-parity to MONAI 1.x.

    新增：
    -------
    * ``progress_callback(done, total)``：每完成一個 patch / batch 即呼叫，方便 PySide6 更新 GUI。
    * ``progress_unit``            ："patch"（預設）或 "batch"，決定 callback 與 tqdm 更新顆粒度。
    """
    if inputs.dim() < 4:
        raise ValueError(
            "inputs 必須包含 batch 與 channel，shape 至少 (N,C,H,W,[D...])"
        )

    # ──初始化裝置──────────────────────────────────────────────────────────────
    if device is None:
        device = inputs.device
    if sw_device is None:
        sw_device = inputs.device

    batch_size, in_channels, *img_size = inputs.shape
    spatial_dims = len(img_size)

    # ──roi_size fallback & pad──────────────────────────────────────────────
    roi_size = _fallback_roi_size(roi_size, img_size)
    need_pad = [max(r - s, 0) for r, s in zip(roi_size, img_size)]
    if any(need_pad):
        # F.pad 的填充順序是從最後一維開始
        pad = []
        for diff in reversed(need_pad):
            half = diff // 2
            pad.extend([half, diff - half])
        inputs = F.pad(inputs, pad=pad, mode=padding_mode, value=cval)
        img_size = list(inputs.shape[2:])  # 更新為 padding 後大小
    else:
        pad = [0] * (spatial_dims * 2)  # 用於後續裁切

    # ──scan slices──────────────────────────────────────────────────────────
    interval = _scan_intervals(roi_size, overlap)
    slices = _dense_patch_slices(img_size, roi_size, interval)
    num_win = len(slices)
    total_patches = num_win * batch_size
    unit_is_patch = progress_unit == "patch"
    total_units = (
        total_patches if unit_is_patch else math.ceil(total_patches / sw_batch_size)
    )

    # ──importance map───────────────────────────────────────────────────────
    if roi_weight_map is not None and tuple(roi_weight_map.shape[2:]) == tuple(
        roi_size
    ):
        imp_map = roi_weight_map.to(
            dtype=inputs.dtype, device=device, non_blocking=True
        )
    else:
        imp_map = _compute_importance_map(
            roi_size,
            mode,
            sigma_scale,
            dtype=inputs.dtype,
            device=device,
        )

    # ──進度條───────────────────────────────────────────────────────────────
    if progress and tqdm is None:
        warnings.warn("tqdm 未安裝，無法顯示進度列")
    pbar = (
        tqdm(total=total_units, desc="Sliding-window", unit=progress_unit)
        if (progress and tqdm is not None)
        else None
    )

    # ──bufffers（延遲建立以便知道 C_out 與多輸出）──────────────────────────
    out_buffers: List[torch.Tensor] = []
    cnt_buffers: List[torch.Tensor] = []
    dict_keys: Optional[List[Any]] = None
    is_tensor_output = True  # predictor 是否僅回傳 Tensor
    buffers_init = False

    # ──主迴圈───────────────────────────────────────────────────────────────
    done_units = 0
    for g in range(0, total_patches, sw_batch_size):
        patch_range = range(g, min(g + sw_batch_size, total_patches))

        # 取得每個切片 (batch_idx, roi_slice)
        patch_slices = []
        for idx in patch_range:
            b = idx // num_win
            s = slices[idx % num_win]
            patch_slices.append((b, s))

        # 堆疊成 (B,C,*roi)
        patches = torch.stack(
            [inputs[b, :, *s].to(sw_device, non_blocking=True) for b, s in patch_slices]
        )

        # 推論
        pred_out = predictor(patches)

        # 統一成 tuple
        if isinstance(pred_out, torch.Tensor):
            pred_tuple = (pred_out,)
        elif isinstance(pred_out, Mapping):
            if dict_keys is None:
                dict_keys = sorted(pred_out.keys())
            pred_tuple = tuple(pred_out[k] for k in dict_keys)
            is_tensor_output = False
        else:  # Sequence
            pred_tuple = tuple(pred_out)
            is_tensor_output = False

        # 自訂 process_fn（與 MONAI 同介面）
        imp_curr = imp_map
        if process_fn is not None:
            pred_tuple, imp_curr = process_fn(pred_tuple, patches, imp_map)

        # 建立 / 檢查 buffers（針對多輸出）
        if not buffers_init:
            for ss, seg in enumerate(pred_tuple):
                zoom = [seg.shape[2 + d] / roi_size[d] for d in range(spatial_dims)]
                out_shape = [batch_size, seg.shape[1]] + [
                    int(img_size[d] * zoom[d]) for d in range(spatial_dims)
                ]
                out_buffers.append(
                    torch.zeros(out_shape, dtype=inputs.dtype, device=device)
                )
                cnt_buffers.append(
                    torch.zeros(
                        [1, 1] + out_shape[2:], dtype=inputs.dtype, device=device
                    )
                )
            buffers_init = True

        # 將 patch 倒回大圖
        for ss, seg in enumerate(pred_tuple):
            zoom = [seg.shape[2 + d] / roi_size[d] for d in range(spatial_dims)]
            imp_resized = _resize_importance(imp_curr, seg.shape[2:]).to(seg.dtype)
            for local_idx, (b, s) in enumerate(patch_slices):
                # 對應到 output 畫布的 slice（含 batch & channel）
                zoomed_slices: List[slice] = [slice(b, b + 1), slice(None)]
                for d, sl in enumerate(s):
                    zs = int(sl.start * zoom[d])
                    ze = int(sl.stop * zoom[d])
                    zoomed_slices.append(slice(zs, ze))
                zoomed_slices = tuple(zoomed_slices)

                out_buffers[ss][zoomed_slices] += seg[local_idx] * imp_resized
                cnt_buffers[ss][
                    (slice(None), slice(None)) + zoomed_slices[2:]
                ] += imp_resized

        # ──進度更新─────────────────────────────────────────────────────────
        done_units += len(patch_range) if unit_is_patch else 1
        if pbar is not None:
            pbar.update(len(patch_range) if unit_is_patch else 1)
        if progress_callback is not None:
            progress_callback(done_units, total_units)

    if pbar is not None:
        pbar.close()

    # ──重疊區域平均────────────────────────────────────────────────────────
    outputs: List[torch.Tensor] = []
    for out, cnt in zip(out_buffers, cnt_buffers):
        outputs.append(out / torch.clamp_min(cnt, torch.finfo(out.dtype).eps))

    # ──移除 padding────────────────────────────────────────────────────────
    if any(need_pad):
        crop_slices: List[slice] = []
        for d, diff in enumerate(need_pad):
            half = diff // 2
            end = half + img_size[d] - diff  # img_size 已是 pad 後，減 diff 取原長
            crop_slices.append(slice(half, end))
        crop_slices_full = (slice(None), slice(None), *crop_slices)
        outputs = [o[crop_slices_full] for o in outputs]

    # ──整理回傳格式（tensor / tuple / dict）───────────────────────────────
    if dict_keys is not None:
        final = dict(zip(dict_keys, outputs))
    else:
        final = outputs[0] if is_tensor_output else tuple(outputs)

    return final
