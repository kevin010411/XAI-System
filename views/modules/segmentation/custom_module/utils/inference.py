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
    # Automatically chooses rich progress bar in notebook / console.
    from tqdm.auto import tqdm
except ImportError:  # Fallback: still works without tqdm.
    tqdm = None  # type: ignore

__all__ = ["sliding_window_inference"]

# -------------------------------------------------------------------------- #
# Internal helpers                                                           #
# -------------------------------------------------------------------------- #


def _fallback_roi_size(
    roi_size: Sequence[int] | int,
    img_size: Sequence[int],
) -> Tuple[int, ...]:
    """Replace non‑positive entries in ``roi_size`` with ``img_size``.

    Args:
        roi_size: Target ROI size per spatial dimension.  If an ``int`` is
            given it is broadcast to all spatial dimensions.  Values that are
            ``None`` or ``<= 0`` mean *use the full image size* for that
            dimension.
        img_size: Spatial shape of the input image (H, W, [D, ...]).

    Returns:
        A tuple of ints with the same length as ``img_size``.
    """
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


def _compute_importance_map(
    roi_size: Sequence[int],
    mode: str = "constant",
    sigma_scale: Sequence[float] | float = 0.125,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return a 1×1×*roi_size* importance map.

    Two modes are supported:
      * ``"constant"`` – a map of ones.
      * ``"gaussian"`` – normalised Gaussian centred in the patch.
    """
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


def _scan_intervals(
    roi_size: Sequence[int],
    overlap: float,
) -> Tuple[int, ...]:
    """Compute scan step (stride) for each spatial dimension.

    Step = ``(1 - overlap) * roi_size``, but always at least 1.
    """
    intervals = []
    for r in roi_size:
        if overlap < 0 or overlap >= 1:
            raise ValueError("overlap 必須在 [0,1) 區間內")
        iv = int(r * (1.0 - overlap))
        intervals.append(max(iv, 1))
    return tuple(intervals)


def _dense_patch_slices(
    img_size: Sequence[int],
    roi_size: Sequence[int],
    interval: Sequence[int],
) -> List[Tuple[slice, ...]]:
    """Generate *all* sliding‑window slices that densely cover the image."""
    starts_per_dim = []
    for im, r, step in zip(img_size, roi_size, interval):
        stops = im - r
        if stops < 0:
            raise ValueError("ROI 大於影像（需先 pad，不應出現）")
        s = list(range(0, stops + 1, step))
        if s[-1] != stops:
            s.append(stops)
        starts_per_dim.append(s)
    slice_list = []
    for starts in product(*starts_per_dim):
        slice_obj = tuple(slice(st, st + r) for st, r in zip(starts, roi_size))
        slice_list.append(slice_obj)
    return slice_list


def _resize_importance(
    imp: torch.Tensor,
    new_spatial: Sequence[int],
) -> torch.Tensor:
    """Nearest‑neighbour resize keeping dtype & device."""
    if list(imp.shape[2:]) == list(new_spatial):
        return imp
    return F.interpolate(imp, size=new_spatial, mode="nearest")


def _preprocess_inputs(
    inputs: torch.Tensor,
    roi_size: Sequence[int] | int,
    padding_mode: str,
    cval: float,
) -> Tuple[torch.Tensor, Tuple[int, ...], List[int], List[int], List[int]]:
    """Pad input if ROI bigger than image; return padded tensor & meta."""
    batch_sz, _, *img_size = inputs.shape
    roi_size = _fallback_roi_size(roi_size, img_size)
    need_pad = [max(r - s, 0) for r, s in zip(roi_size, img_size)]
    if any(need_pad):
        pad_seq: List[int] = []  # torch.pad pads last dim first
        for diff in reversed(need_pad):
            half = diff // 2
            pad_seq.extend([half, diff - half])
        inputs = F.pad(inputs, pad=pad_seq, mode=padding_mode, value=cval)
        img_size = list(inputs.shape[2:])
    else:
        pad_seq = [0] * (len(img_size) * 2)
    return inputs, roi_size, need_pad, pad_seq, img_size


def _plan_patches(
    img_size: Sequence[int],
    roi_size: Sequence[int],
    overlap: float,
    sw_batch_size: int,
    batch_sz: int,
    progress_unit: str,
) -> Tuple[List[Tuple[slice, ...]], int, int]:
    intervals = _scan_intervals(roi_size, overlap)
    slices = _dense_patch_slices(img_size, roi_size, intervals)
    total_patches = len(slices) * batch_sz
    total_units = (
        total_patches
        if progress_unit == "patch"
        else math.ceil(total_patches / sw_batch_size)
    )
    return slices, total_patches, total_units


def _allocate_buffers(
    pred_tuple: Tuple[torch.Tensor, ...],
    batch_sz: int,
    img_size: Sequence[int],
    roi_size: Sequence[int],
    device: torch.device | str,
    dtype: torch.dtype,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    out_buffers: List[torch.Tensor] = []
    cnt_buffers: List[torch.Tensor] = []
    ndim = len(img_size)
    for seg in pred_tuple:
        zoom = [seg.shape[2 + d] / roi_size[d] for d in range(ndim)]
        out_shape = [batch_sz, seg.shape[1]] + [
            int(img_size[d] * zoom[d]) for d in range(ndim)
        ]
        out_buffers.append(torch.zeros(out_shape, dtype=dtype, device=device))
        cnt_buffers.append(
            torch.zeros([1, 1] + out_shape[2:], dtype=dtype, device=device)
        )
    return out_buffers, cnt_buffers


def _accumulate_batch(
    pred_tuple: Tuple[torch.Tensor, ...],
    patch_slices: List[Tuple[int, Tuple[slice, ...]]],
    out_buffers: List[torch.Tensor],
    cnt_buffers: List[torch.Tensor],
    importance_map: torch.Tensor,
    roi_size: Sequence[int],
):
    ndim = len(roi_size)
    for buf_idx, seg in enumerate(pred_tuple):
        zoom = [seg.shape[2 + d] / roi_size[d] for d in range(ndim)]
        resized_imp = _resize_importance(importance_map, seg.shape[2:]).to(seg.dtype)
        for local_i, (b, s) in enumerate(patch_slices):
            canvas_slice: List[slice] = [slice(b, b + 1), slice(None)]
            for d, sl in enumerate(s):
                zs, ze = int(sl.start * zoom[d]), int(sl.stop * zoom[d])
                canvas_slice.append(slice(zs, ze))
            cs = tuple(canvas_slice)
            out_buffers[buf_idx][cs] += seg[local_i] * resized_imp
            cnt_buffers[buf_idx][(slice(None), slice(None)) + cs[2:]] += resized_imp


def _finalise_outputs(
    out_buffers: List[torch.Tensor],
    cnt_buffers: List[torch.Tensor],
    need_pad: List[int],
    img_size_orig: Sequence[int],
    pad_seq: List[int],
    dict_keys: Optional[List[Any]],
    tensor_only: bool,
):
    # average overlaps
    outputs = [
        o / torch.clamp_min(c, torch.finfo(o.dtype).eps)
        for o, c in zip(out_buffers, cnt_buffers)
    ]
    # remove padding
    if any(need_pad):
        crop_slices: List[slice] = []
        for d, diff in enumerate(need_pad):
            half = diff // 2
            end = half + img_size_orig[d]  # original size
            crop_slices.append(slice(half, end))
        crop_full = (slice(None), slice(None), *crop_slices)
        outputs = [o[crop_full] for o in outputs]
    # restore output structure
    if dict_keys is not None:
        return dict(zip(dict_keys, outputs))
    return outputs[0] if tensor_only else tuple(outputs)


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
    """Run sliding‑window inference over *N*D images.

    Args:
        inputs: Input tensor of shape ``(B, C, *spatial)``.
        roi_size: ROI size used during sliding.  ``≤ 0`` or ``None`` means *full
            image* in that dimension.
        sw_batch_size: How many patches to process per forward pass.
        predictor: Callable like ``lambda x: model(x)``.
        overlap: Fraction between [0, 1) determining stride.
        mode: Importance map mode – ``"constant"`` or ``"gaussian"``.
        sigma_scale: Gaussian σ as a fraction of ROI (if ``mode=='gaussian'``).
        padding_mode: Padding strategy forwarded to :func:`torch.nn.functional.pad`.
        cval: Constant pad value (if ``padding_mode=='constant'``).
        sw_device: Device used for *patch* inference (defaults to ``inputs.device``).
        device: Device hosting accumulation buffers (defaults to ``inputs.device``).
        progress: Whether to show a tqdm progress bar.
        roi_weight_map: Pre‑computed importance map matching ``roi_size``.
        process_fn: Optional callable ``(pred_tuple, patch, importance) -> (new_pred, new_imp)``.
        progress_callback: ``f(done: int, total: int)`` for GUI updates.
        progress_unit: Defines callback granularity – per‑"patch" or per‑"batch".

    Returns:
        Same structure (Tensor / tuple / dict) as returned by ``predictor``.
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

    batch_sz = inputs.shape[0]

    # roi_size fallback & pad──────────────────────────────────────────────
    inputs, roi_size, need_pad, pad_seq, img_size = _preprocess_inputs(
        inputs, roi_size, padding_mode, cval
    )

    # ──Sliding‑window coordinates─────────────────────────
    slices, total_patches, total_units = _plan_patches(
        img_size, roi_size, overlap, sw_batch_size, batch_sz, progress_unit
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
    tensor_only = True  # predictor 是否僅回傳 Tensor
    buffers_ready = False

    # ──主迴圈───────────────────────────────────────────────────────────────
    done_units = 0
    for g in range(0, total_patches, sw_batch_size):
        prange = range(g, min(g + sw_batch_size, total_patches))

        # 取得每個切片 (batch_idx, roi_slice)
        patch_slices: List[Tuple[int, Tuple[slice, ...]]] = [
            (i // len(slices), slices[i % len(slices)]) for i in prange
        ]
        patches = torch.stack(
            [inputs[b, :, *s].to(sw_device, non_blocking=True) for b, s in patch_slices]
        )

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
            tensor_only = False
        else:  # Sequence
            pred_tuple = tuple(pred_out)
            tensor_only = False

        # 自訂 process_fn（與 MONAI 同介面）
        imp_curr = imp_map
        if process_fn is not None:
            pred_tuple, imp_curr = process_fn(pred_tuple, patches, imp_map)

        # 建立 / 檢查 buffers（針對多輸出）
        if not buffers_ready:
            out_buffers, cnt_buffers = _allocate_buffers(
                pred_tuple, batch_sz, img_size, roi_size, device, inputs.dtype
            )
            buffers_ready = True

        # Accumulate predictions
        _accumulate_batch(
            pred_tuple, patch_slices, out_buffers, cnt_buffers, imp_curr, roi_size
        )

        # ──進度更新─────────────────────────────────────────────────────────
        done_units += len(prange) if progress_unit == "patch" else 1
        if pbar is not None:
            pbar.update(len(prange) if progress_unit == "patch" else 1)
        if progress_callback is not None:
            progress_callback(done_units, total_units)

    if pbar is not None:
        pbar.close()

    return _finalise_outputs(
        out_buffers, cnt_buffers, need_pad, img_size, pad_seq, dict_keys, tensor_only
    )
