from typing import Any
import numpy as np
import torch
from torchvision.transforms.v2 import Compose, RandomAffine
from torchvision.transforms import v2
from ..utils import TRANSFORMS, build_transform


@TRANSFORMS.register_module()
class Compose(v2.Compose):
    def __init__(self, transforms) -> None:
        super().__init__(transforms)
        new_transforms = []
        for transform in self.transforms:
            if isinstance(transform, dict):
                transform = build_transform(transform)
            elif not isinstance(transform, v2.Transform):
                raise TypeError(
                    f"Expected a dict or a torchvision transform, but got {type(transform)}"
                )
            new_transforms.append(transform)
        self.transforms = new_transforms

    def inverse(self, *inputs: Any) -> Any:
        needs_unpacking = len(inputs) > 1
        for transform in self.transforms[::-1]:
            outputs = transform.inverse(*inputs)
            inputs = outputs if needs_unpacking else (outputs,)
        return outputs


@TRANSFORMS.register_module()
class ToTensor(v2.ToTensor):
    def forward(self, *data):
        out = []
        for d in data:
            if isinstance(d, np.ndarray):
                out.append(super().forward(d).permute(1, 2, 0))
            else:
                out.append(d)
        return tuple(out) if len(out) > 1 else out[0]

    def inverse(self, *data):
        out = []
        for d in data:
            if isinstance(d, torch.Tensor):
                out.append(d.cpu().numpy())
            else:
                out.append(d)
        return tuple(out) if len(out) > 1 else out[0]


@TRANSFORMS.register_module()
class ToImage(v2.ToImage):
    pass


@TRANSFORMS.register_module()
class ToDtype(v2.ToDtype):
    pass
