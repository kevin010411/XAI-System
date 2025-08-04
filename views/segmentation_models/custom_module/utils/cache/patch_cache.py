import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch


Slice = Tuple[slice, ...]  # e.g. (slice(z0,z1), slice(y0,y1), slice(x0,x1))
Item = Tuple[int, Slice, torch.Tensor]  # (batch_idx, slice, tensor)


class PatchCache(ABC):
    """抽象快取介面。任何子類都必須實作下列方法。"""

    # ---------- 寫入 ----------
    @abstractmethod
    def add(
        self,
        layer: str,
        b_idx: int,
        slc: Slice,
        tensor: torch.Tensor,
        shape: tuple[int, int],
    ) -> None: ...

    # ---------- 讀取 ----------
    @abstractmethod
    def iter(self, layer: str) -> Iterable[Item]: ...
    @abstractmethod
    def layers(self) -> List[str]: ...

    # ---------- 管理 ----------
    @abstractmethod
    def clear(self) -> None: ...

    # ---------- 序列化 ----------
    @abstractmethod
    def save(self, path: os.PathLike) -> None: ...
    @classmethod
    @abstractmethod
    def load(cls, path: os.PathLike) -> "PatchCache": ...
