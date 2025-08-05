import torch.nn.functional as F
import torch

from ..utils import mem_watch, XAI
from .sliding_seg_grad_cam import SlidingSegGradCAM

# 目前是計算後馬上計算heatmap，所以pooling大小、channel是固定的。


@XAI.register_module()
class SlidingSegXResCAM(SlidingSegGradCAM):

    def __init__(
        self, pool_size: int | tuple[int, ...] | None = None, *kwargs  # 新增此參數
    ) -> None:
        super().__init__(
            *kwargs,
        )
        self.pool_size = pool_size

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
