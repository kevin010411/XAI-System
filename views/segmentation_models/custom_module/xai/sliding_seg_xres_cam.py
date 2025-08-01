from logging import warning
from collections import defaultdict
from typing import Callable, Dict, List, Sequence, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from ..utils import mem_watch, XAI

# 目前是計算後馬上計算heatmap，所以pooling大小、channel是固定的。


@XAI.register_module()
class SlidingSegXResCAM:

    def __init__(
        self,
        model: torch.nn.Module = None,
        target_layers: Sequence[torch.nn.Module] = None,
        *,
        class_selector: Callable[[torch.Tensor], torch.LongTensor] | None = None,
        upsample_mode: str | None = None,
        pool_size: int | Tuple[int, ...] | None = None,  # 新增此參數
    ) -> None:
        self.model = model
        self.layers = list(target_layers or [])
        self.class_selector = class_selector or (
            lambda p: p.flatten(2).mean(-1).topk(1, dim=1).indices
        )
        self.upsample_mode = upsample_mode
        self.pool_size = pool_size  # 新增此行，儲存 pooling 大小設定

        self._acts: Dict[str, torch.Tensor] = {}
        self._grads: Dict[str, torch.Tensor] = {}
        self._handles = {}
        self._stash: Dict[str, List[Tuple[int, Tuple[slice, ...], torch.Tensor]]] = (
            defaultdict(list)
        )

        if self.layers is not None:
            self.layers = [self._resolve_layer(layer) for layer in self.layers]
            for layer in self.layers:
                self._register_one(layer)

        self.output: Tuple[torch.Tensor, Dict[str, torch.Tensor]] | None = None

    def _spatial_mean(self, t: torch.Tensor) -> torch.Tensor:
        if self.pool_size is None:
            return t.mean(dim=tuple(range(2, t.ndim)), keepdim=True)
        else:
            # 使用 F.avg_poolNd 根據維度自動決定
            spatial_dims = t.ndim - 2
            if isinstance(self.pool_size, int):
                kernel = (self.pool_size,) * spatial_dims
            else:
                kernel = self.pool_size

            if spatial_dims == 2:
                return F.avg_pool2d(
                    t, kernel_size=kernel, stride=1, padding=kernel[0] // 2
                )
            elif spatial_dims == 3:
                return F.avg_pool3d(
                    t, kernel_size=kernel, stride=1, padding=kernel[0] // 2
                )
            else:
                raise ValueError("Unsupported tensor dimension for spatial pooling")

    # ------------------------------------------------------------------
    # Sliding‑window XAI hooks
    # ------------------------------------------------------------------
    def pre(
        self, patches: torch.Tensor, patch_slices, step: int
    ):  # noqa: D401 – simple hook
        """Clear per‑batch grad/act cache."""
        self._acts.clear()
        self._grads.clear()

    def post(self, pred_tuple, patch_slices, step: int):  # noqa: D401
        if self.model is None:
            warning("SlidingGradCAM3D did not have model to observe")
        preds: torch.Tensor = pred_tuple[0]
        sel = self.class_selector(preds)  # (B, N_cls)
        B, n_cls = sel.shape  # variable n_cls allowed

        # Flatten classes so we can backprop once per *class*
        for ci in range(n_cls):
            targets = sel[:, ci]  # (B,)
            self.model.zero_grad(set_to_none=True)
            # Build scalar loss = sum_i logit[b, target]
            loss = preds[torch.arange(B), targets].sum()
            loss.backward(retain_graph=True)

            for lname, act in self._acts.items():
                grad: Tensor = self._grads[lname]
                w: Tensor = self._spatial_mean(grad)
                heat: Tensor = F.relu((w * act).sum(dim=1, keepdim=True))  # (B,1,*)

                # Spatial upsample to match pred resolution
                mode = self.upsample_mode
                if mode is None:
                    mode = "trilinear" if heat.ndim == 5 else "bilinear"

                heat = F.interpolate(
                    heat,
                    size=preds.shape[2:],
                    mode=mode,
                    align_corners=False,
                )
                # normalise per‑sample
                axes = tuple(range(2, heat.ndim))
                h_min = heat.amin(dim=axes, keepdim=True)
                h_max = heat.amax(dim=axes, keepdim=True)
                heat_norm = (heat - h_min) / (h_max - h_min + 1e-8)

                # stash each sample‑heatmap + slice
                for i, (_, slc) in enumerate(patch_slices):
                    self._stash[lname].append(
                        (i, slc, heat_norm[i].detach().cpu())
                    )  # append (patch_idx,slice,heatmap)

    def final(self, results):  # noqa: D401 – simple hook
        """Stitch heat‑maps & expose ``self.output``."""
        if isinstance(results, torch.Tensor):
            B, _, *spatial = results.shape
        elif isinstance(results, (list, tuple)):
            B, _, *spatial = results[0].shape
        else:
            any_tensor = next(iter(results.values()))
            B, _, *spatial = any_tensor.shape

        stitched: Dict[str, torch.Tensor] = {}
        for lname, items in self._stash.items():
            canvas = torch.zeros((B, 1, *spatial))
            count = torch.zeros_like(canvas)
            for b_idx, slc, heat in items:
                canvas[(b_idx, slice(None), *slc)] += heat
                count[(b_idx, slice(None), *slc)] += 1
            stitched[lname] = canvas / count.clamp_min(1e-6)

        # provide combined output
        self.output = (results, stitched)

    def set_model(self, model: torch.nn.Module):
        self.model = model

    def set_class(self, class_selector):
        self.class_selector = class_selector

    def set_target_layers(self, targets):
        """
        targets: str / nn.Module / list / tuple
        - 若傳入單一元素，也會自動轉 list。
        - 會先解除舊 hooks，再註冊新 hooks。
        """
        if self.model is None:
            warning("沒有model不能設定target_layers")
            return

        # 1) 解除舊 hooks
        self.clear_hooks()

        # 2) 標準化成 list
        if not isinstance(targets, (list, tuple)):
            targets = [targets]

        self.layers = [self._resolve_layer(target) for target in targets]
        # 3) 逐一註冊
        for layer in self.layers:
            self._register_one(layer)

    def clear_hooks(self):
        """移除所有 hooks 並清空暫存。"""
        for h in self._handles.values():
            h.remove()
        self._handles.clear()
        self._acts.clear()
        self._grads.clear()

    def _register_one(self, layer: torch.nn.Module):
        lname = self._layer_name(layer)

        def _fwd_hook(_, __, out, lname=lname):
            self._acts[lname] = out.detach()
            out.register_hook(lambda g, lname=lname: self._grads.__setitem__(lname, g))

        h = layer.register_forward_hook(_fwd_hook)
        self._handles[lname] = h

    def _resolve_layer(self, target):
        # 允許 nn.Module 或字串名稱
        if isinstance(target, torch.nn.Module):
            return target
        elif isinstance(target, str):
            try:
                return dict(self.model.named_modules())[target]
            except KeyError:
                raise ValueError(f"No layer named {target}")
        else:
            raise TypeError("target 必須是 nn.Module 或 str")

    # ----- Utility -----
    @staticmethod
    def _layer_name(layer: torch.nn.Module) -> str:
        return layer.__class__.__name__ + f"@{id(layer):x}"
